"""Ring-topology island migration (OpenEvolve pattern).

Every N generations, each island's top branch exports its LoRA delta and
top-K experience entries to the next island clockwise.

Per docs/03-stack.md §'From OpenEvolve':
- Migrant's delta is a starting point for next-gen mutation, NOT merged.
- Migrant experiences enter receiver's replay buffer at 2× priority for 1 gen.

This file is implementable without the rest of the stack; the tricky part
(LoRA-delta merge) is delegated to a callback passed in at construction, so
the topology logic stays testable in isolation.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Callable

from si.contracts import Branch, Experience, Island, Migrator


LoRAMergeFn = Callable[[str, str], str]
"""(receiver_lora_path, donor_lora_path) -> new_lora_path."""


class RingMigrator(Migrator):
    """Clockwise ring migration across islands."""

    def __init__(
        self,
        migration_every: int,
        migrant_experience_priority: float,
        lora_merge_fn: LoRAMergeFn,
        top_k_experiences: int = 32,
    ) -> None:
        self.migration_every = migration_every
        self.priority = migrant_experience_priority
        self.merge = lora_merge_fn
        self.top_k = top_k_experiences

    def migrate(self, islands: list[Island], gen: int) -> list[Island]:
        if gen == 0 or gen % self.migration_every != 0:
            return islands
        if len(islands) < 2:
            return islands

        new_islands: list[Island] = []
        n = len(islands)

        for i, island in enumerate(islands):
            donor_island = islands[(i - 1) % n]  # clockwise source
            top_donor = self._top_branch_by_elo(donor_island)
            if top_donor is None:
                new_islands.append(island)
                continue

            # Receiver = current island's LOWEST-Elo branch
            low_branch = self._low_branch_by_elo(island)
            if low_branch is None:
                new_islands.append(island)
                continue

            merged_lora_path = self.merge(low_branch.lora_path, top_donor.lora_path)
            donor_experience = self._top_k_experience(top_donor.experience, self.top_k)
            seeded_experience = Experience(
                recent_wins=list(low_branch.experience.recent_wins) + donor_experience,
                recent_losses=list(low_branch.experience.recent_losses),
                proposer_seeds=list(low_branch.experience.proposer_seeds)
                + list(top_donor.experience.proposer_seeds[: self.top_k // 2]),
            )

            updated_low = dataclass_replace(
                low_branch,
                lora_path=merged_lora_path,
                experience=seeded_experience,
            )

            new_branches = [updated_low if b.branch_id == low_branch.branch_id else b for b in island.branches]
            new_islands.append(dataclass_replace(island, branches=new_branches))

        return new_islands

    @staticmethod
    def _top_branch_by_elo(island: Island) -> Branch | None:
        if not island.branches:
            return None
        return max(
            island.branches,
            key=lambda b: island.elo_state.ratings.get(b.branch_id, island.elo_state.default_rating),
        )

    @staticmethod
    def _low_branch_by_elo(island: Island) -> Branch | None:
        if not island.branches:
            return None
        return min(
            island.branches,
            key=lambda b: island.elo_state.ratings.get(b.branch_id, island.elo_state.default_rating),
        )

    @staticmethod
    def _top_k_experience(exp: Experience, k: int) -> list:
        """Prefer recent wins; losses are noise for migration."""
        return list(exp.recent_wins[-k:])
