"""RL-ZVP + F-GRPO + NSR patches on top of TRL 1.2's GRPOTrainer.

The v2 training run (phase1_v2_20260423_2250) plateaued at 85.98% HumanEval+
(—1.22 pp below base) across gens 10–15 with frac_reward_zero_std ≈ 1 in nearly
every step. This module addresses the two root causes identified in
docs/research-scan.md scan #7:

    1. Zero-variance groups in GRPO produce zero advantage → no gradient.
       RL-ZVP (arXiv:2509.21880) replaces the zero advantage with a
       reward-sign-modulated constant, preserving directional learning
       signal from degenerate groups.

    2. Small group sizes (num_generations=2 on a 3090) concentrate updates
       on common solutions and miss rare-correct trajectories.
       F-GRPO (arXiv:2602.06717) focal-loss-style downweights advantages
       on high-success prompts, amplifying rare-correct learning.

Implementation: subclass GRPOTrainer; intercept `_calculate_rewards` to
capture per-func rewards, then post-process the `advantages` tensor in the
dict returned by `_generate_and_score_completions`. No need to re-run
reward functions (which would double sandbox load) and no copy of TRL's
650-line core method.
"""

from __future__ import annotations

import logging

import torch
from trl import GRPOTrainer

log = logging.getLogger(__name__)


class RLZVPGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with RL-ZVP zero-variance rescue + F-GRPO focal scaling.

    Parameters pulled from kwargs before super().__init__:
        rlzvp_enabled: bool — apply RL-ZVP zero-variance advantage rescue. Default True.
        rlzvp_magnitude: float — |advantage| assigned to zero-variance samples. 0.5 matches paper.
        rlzvp_pos_threshold: float — reward above this counts as "positive" for sign. 0.5.
        fgrpo_gamma: float | None — focal gamma. 2.0 = standard focal loss.
            None disables F-GRPO. Default 2.0.
        zero_var_tol: float — rewards within this of group mean count as zero-variance. 1e-6.
    """

    def __init__(self, *args, **kwargs):
        self.rlzvp_enabled = kwargs.pop("rlzvp_enabled", True)
        self.rlzvp_magnitude = float(kwargs.pop("rlzvp_magnitude", 0.5))
        self.rlzvp_pos_threshold = float(kwargs.pop("rlzvp_pos_threshold", 0.5))
        self.fgrpo_gamma: float | None = kwargs.pop("fgrpo_gamma", 2.0)
        self.zero_var_tol = float(kwargs.pop("zero_var_tol", 1e-6))
        super().__init__(*args, **kwargs)
        self._last_rewards_per_func: torch.Tensor | None = None

    # --- capture per-func rewards so we don't re-run sandbox --------------
    def _calculate_rewards(self, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._last_rewards_per_func = rewards_per_func
        return rewards_per_func

    # --- advantage post-processor -----------------------------------------
    def _generate_and_score_completions(self, generation_batch):
        out = super()._generate_and_score_completions(generation_batch)
        advantages = out["advantages"]
        rpf = self._last_rewards_per_func
        if rpf is None or advantages.numel() == 0:
            return out
        weights = self.reward_weights.to(rpf.device)
        rewards = (rpf * weights.unsqueeze(0)).nansum(dim=1)
        # Slice to local process. Single-GPU case: whole tensor.
        n_total = rewards.shape[0]
        n_local = advantages.shape[0]
        offset = self.accelerator.process_index * n_local
        rewards_local = rewards[offset : offset + n_local]
        advantages = _apply_rlzvp_and_fgrpo(
            advantages=advantages,
            rewards=rewards_local.to(advantages.device, dtype=advantages.dtype),
            num_generations=self.num_generations,
            rlzvp_enabled=self.rlzvp_enabled,
            rlzvp_magnitude=self.rlzvp_magnitude,
            rlzvp_pos_threshold=self.rlzvp_pos_threshold,
            fgrpo_gamma=self.fgrpo_gamma,
            zero_var_tol=self.zero_var_tol,
        )
        out["advantages"] = advantages
        return out


def _apply_rlzvp_and_fgrpo(
    *,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    num_generations: int,
    rlzvp_enabled: bool,
    rlzvp_magnitude: float,
    rlzvp_pos_threshold: float,
    fgrpo_gamma: float | None,
    zero_var_tol: float,
) -> torch.Tensor:
    """Pure-function advantage transform. Split out for unit testing.

    Shapes:
        advantages, rewards: [N] where N = batch * num_generations
        returns: [N]
    """
    assert advantages.shape == rewards.shape, (advantages.shape, rewards.shape)
    n = advantages.shape[0]
    if n == 0 or num_generations <= 0 or n % num_generations != 0:
        return advantages
    grouped = rewards.view(-1, num_generations)
    group_mean = grouped.mean(dim=1, keepdim=True).expand_as(grouped).reshape(-1)
    group_std = grouped.std(dim=1, unbiased=False, keepdim=True).expand_as(grouped).reshape(-1)
    is_zero_var = group_std < zero_var_tol

    if rlzvp_enabled:
        sign = torch.where(
            rewards > rlzvp_pos_threshold,
            torch.ones_like(rewards),
            -torch.ones_like(rewards),
        )
        zvp_adv = (sign * rlzvp_magnitude).to(advantages.dtype)
        advantages = torch.where(is_zero_var, zvp_adv, advantages)

    if fgrpo_gamma is not None:
        # focal weight = (1 - p_success)^gamma. p_success = group mean reward, clipped to [0,1].
        p_success = group_mean.clamp(0.0, 1.0)
        focal_weight = (1.0 - p_success) ** float(fgrpo_gamma)
        advantages = advantages * focal_weight.to(advantages.dtype)

    return advantages
