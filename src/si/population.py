"""Branch population manager — the Phase 2 Elo-tournament substrate.

Owns the N LoRA branches that compete each generation. Two responsibilities:

  1. Apply the selector's ReplacementPlan: each bottom-quartile ("dead") branch
     is reseeded from a top-quartile parent's LoRA plus a small Gaussian
     perturbation (docs/04-implementation.md §2.3.5, sigma from EloConfig).
  2. Snapshot the population at each committed generation and restore it on
     revert, so the anchor's revert rule (the non-negotiable anti-collapse
     defense, see si.anchor.HeldOutAnchor.should_revert) has a concrete state
     to roll back to.

The only operation that needs PEFT/torch + disk — producing a perturbed copy of
a LoRA adapter — is injected as a callback (`reseed_fn`), the same idiom
islands.py uses for `lora_merge_fn`. Everything else (who reseeds from whom,
per-generation path versioning, experience snapshot/restore) is pure and is
unit-tested without a GPU.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path

from si.contracts import Branch, EloState, Experience, ReplacementPlan

log = logging.getLogger(__name__)


ReseedFn = Callable[[str, str, float], None]
"""(parent_lora_path, dest_lora_path, sigma) -> None.

Write a copy of the parent adapter's weights, with N(0, sigma) added per
weight, to dest_lora_path. The Phase-2 wire-up supplies a PEFT-backed
implementation; tests supply a recording stub.
"""


class BranchManager:
    """Owns branch objects + their on-disk LoRA lineage across generations.

    Branch objects are mutated in place, so any `Island` holding the same
    references sees replacements/reverts without rebuilding island membership.
    """

    def __init__(
        self,
        branches: list[Branch],
        *,
        lora_root: str,
        reseed_fn: ReseedFn,
        perturb_sigma: float = 0.001,
    ) -> None:
        if not branches:
            raise ValueError("BranchManager needs at least one branch")
        self.branches: dict[str, Branch] = {b.branch_id: b for b in branches}
        self.lora_root = Path(lora_root)
        self.reseed = reseed_fn
        self.sigma = perturb_sigma
        # gen -> {branch_id: (lora_path, deep-copied Experience)}
        self._snapshots: dict[int, dict[str, tuple[str, Experience]]] = {}

    def lora_path(self, branch_id: str, gen: int) -> str:
        """Deterministic, per-generation adapter path. Versioning by generation
        means a reseed never overwrites a committed generation's weights, which
        is what makes revert_to sound."""
        return str(self.lora_root / branch_id / f"gen{gen}")

    def commit(self, gen: int) -> None:
        """Record the population state for a committed generation so a later
        revert can restore exactly these adapters + experience buffers."""
        self._snapshots[gen] = {
            bid: (b.lora_path, copy.deepcopy(b.experience)) for bid, b in self.branches.items()
        }
        log.info("population: committed snapshot for gen %d (%d branches)", gen, len(self.branches))

    def apply_replacement(
        self, plan: ReplacementPlan, gen: int, elo: EloState | None = None
    ) -> list[str]:
        """Reseed each dead branch from its parent, writing the perturbed copy to
        the branch's gen-versioned path. Returns the reseeded branch ids.

        A reseeded branch is a fresh individual: empty experience, and (if `elo`
        is given) its rating reset to the default so it re-earns standing rather
        than inheriting the dead branch's depressed rating.
        """
        reseeded: list[str] = []
        for dead_id, parent_id in plan.replace:
            dead = self.branches.get(dead_id)
            parent = self.branches.get(parent_id)
            if dead is None or parent is None:
                log.warning(
                    "population: skip replace (%s<-%s); unknown branch id", dead_id, parent_id
                )
                continue
            # reseed_fn owns disk writes (incl. creating dest's parent dir),
            # keeping this manager pure and GPU/disk-free for testing.
            dest = self.lora_path(dead_id, gen)
            self.reseed(parent.lora_path, dest, self.sigma)
            dead.lora_path = dest
            dead.experience = Experience()
            if elo is not None:
                dead.elo = elo.default_rating
                elo.ratings[dead_id] = elo.default_rating
            reseeded.append(dead_id)
        log.info("population: reseeded %d branches at gen %d", len(reseeded), gen)
        return reseeded

    def revert_to(self, gen: int) -> None:
        """Restore the population to a previously committed generation."""
        snap = self._snapshots.get(gen)
        if snap is None:
            raise KeyError(
                f"no committed snapshot for gen {gen}; "
                f"have {sorted(self._snapshots)}. Cannot revert."
            )
        for bid, (lora_path, exp) in snap.items():
            b = self.branches[bid]
            b.lora_path = lora_path
            b.experience = copy.deepcopy(exp)
        log.warning("population: reverted to gen %d snapshot", gen)
