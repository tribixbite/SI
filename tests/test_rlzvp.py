"""Unit tests for the RL-ZVP + F-GRPO advantage transform.

Pure tensor math — no GPU, no sandbox, no TRL runtime.
"""

from __future__ import annotations

import pytest
import torch

from si.trainer_rlzvp import _apply_rlzvp_and_fgrpo


def _apply(advantages, rewards, num_gen, **kw):
    return _apply_rlzvp_and_fgrpo(
        advantages=torch.tensor(advantages, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        num_generations=num_gen,
        rlzvp_enabled=kw.get("rlzvp_enabled", True),
        rlzvp_magnitude=kw.get("rlzvp_magnitude", 0.5),
        rlzvp_pos_threshold=kw.get("rlzvp_pos_threshold", 0.5),
        fgrpo_gamma=kw.get("fgrpo_gamma", None),  # disable F-GRPO unless asked
        zero_var_tol=kw.get("zero_var_tol", 1e-6),
    )


def test_zero_var_all_positive_gets_positive_advantage():
    # Group of 2, both rewards = 1.0. Standard GRPO → advantage 0.
    out = _apply([0.0, 0.0], [1.0, 1.0], num_gen=2)
    assert torch.allclose(out, torch.tensor([0.5, 0.5]))


def test_zero_var_all_negative_gets_negative_advantage():
    # Group of 2, both rewards = 0.0 (below threshold 0.5) → -0.5 each.
    out = _apply([0.0, 0.0], [0.0, 0.0], num_gen=2)
    assert torch.allclose(out, torch.tensor([-0.5, -0.5]))


def test_non_zero_var_group_is_not_touched():
    # Group of 2, one passes one fails → non-degenerate.
    # Standard advantages (computed upstream): [+0.5, -0.5].
    # RL-ZVP should NOT overwrite them.
    out = _apply([0.5, -0.5], [1.0, 0.0], num_gen=2)
    assert torch.allclose(out, torch.tensor([0.5, -0.5]))


def test_mixed_groups_only_zero_var_ones_patched():
    # First group (0,1): both 1.0 → zero-var. Adv was 0 → becomes +0.5 each.
    # Second group (2,3): 1.0 vs 0.0 → non-zero var. Adv [+0.5,-0.5] unchanged.
    out = _apply(
        advantages=[0.0, 0.0, 0.5, -0.5],
        rewards=[1.0, 1.0, 1.0, 0.0],
        num_gen=2,
    )
    expected = torch.tensor([0.5, 0.5, 0.5, -0.5])
    assert torch.allclose(out, expected)


def test_rlzvp_disabled_leaves_zero_advantages_zero():
    out = _apply(
        advantages=[0.0, 0.0],
        rewards=[1.0, 1.0],
        num_gen=2,
        rlzvp_enabled=False,
    )
    assert torch.allclose(out, torch.tensor([0.0, 0.0]))


def test_fgrpo_downweights_easy_groups():
    # Group of 2 with reward mean = 0.9 (easy). gamma=2 → focal_weight = 0.01.
    # Advantage [0.5, -0.5] × 0.01 → [0.005, -0.005].
    out = _apply(
        advantages=[0.5, -0.5],
        rewards=[1.0, 0.8],
        num_gen=2,
        rlzvp_enabled=False,
        fgrpo_gamma=2.0,
    )
    # p_success = 0.9, (1-0.9)^2 = 0.01
    assert torch.allclose(out, torch.tensor([0.005, -0.005]), atol=1e-6)


def test_fgrpo_amplifies_hard_groups_less():
    # Mean reward = 0.1, focal = 0.81 — close to 1 (most weight preserved).
    out = _apply(
        advantages=[0.5, -0.5],
        rewards=[0.2, 0.0],
        num_gen=2,
        rlzvp_enabled=False,
        fgrpo_gamma=2.0,
    )
    # p_success = 0.1, (1-0.1)^2 = 0.81
    assert torch.allclose(out, torch.tensor([0.405, -0.405]), atol=1e-6)


def test_rlzvp_plus_fgrpo_bundle():
    # Zero-var group with mean 0.0 (all failed). RL-ZVP sets adv to -0.5.
    # F-GRPO weight = (1-0)^2 = 1.0, so adv stays -0.5.
    out = _apply(
        advantages=[0.0, 0.0],
        rewards=[0.0, 0.0],
        num_gen=2,
        rlzvp_enabled=True,
        fgrpo_gamma=2.0,
    )
    assert torch.allclose(out, torch.tensor([-0.5, -0.5]))


def test_rlzvp_plus_fgrpo_on_easy_zero_var():
    # Zero-var easy group (all pass). RL-ZVP gives +0.5, F-GRPO weight (1-1)^2 = 0 → zeroed.
    # This is the intentional trade: when the model already solves it, don't push further.
    out = _apply(
        advantages=[0.0, 0.0],
        rewards=[1.0, 1.0],
        num_gen=2,
        rlzvp_enabled=True,
        fgrpo_gamma=2.0,
    )
    assert torch.allclose(out, torch.tensor([0.0, 0.0]))


def test_empty_input_returns_empty():
    out = _apply([], [], num_gen=2)
    assert out.numel() == 0


def test_mismatched_group_size_returns_unchanged():
    # N=3 with num_gen=2 → 3 % 2 != 0 → return advantages unchanged.
    out = _apply([0.1, 0.2, 0.3], [1.0, 0.0, 1.0], num_gen=2)
    assert torch.allclose(out, torch.tensor([0.1, 0.2, 0.3]))
