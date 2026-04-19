"""Top-level self-improvement loop.

Orchestrates the four clocks defined in docs/00-overview.md:

  inner loop  (seconds)  — AZR self-play, per problem
  middle loop (minutes)  — Elo matches + GRPO updates, per generation
  outer loop  (~hours)   — island ring migration, every N generations
  anchor      (~2h)      — held-out eval + commit/revert, every M generations

This file is a skeleton. Each method raises NotImplementedError with a pointer
to the spec section that defines its contract. Implementations land in Phase 1
through Phase 4 per docs/04-implementation.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from si.config import RunConfig
from si.contracts import (
    Anchor,
    AnchorResult,
    Branch,
    EloState,
    Experience,
    Island,
    Match,
    Migrator,
    Proposer,
    Selector,
    Solver,
    Task,
    TaskType,
    Verifier,
)

log = logging.getLogger(__name__)


@dataclass
class LoopState:
    gen: int = 0
    elo: EloState | None = None
    last_anchor: AnchorResult | None = None
    committed_gen: int = 0


class Loop:
    """Orchestrator for the SI self-improvement loop.

    Composition over inheritance: the six contracts (proposer/solver/verifier/
    selector/migrator/anchor) are passed in. Swap any one for ablation without
    touching this file.
    """

    def __init__(
        self,
        config: RunConfig,
        proposer: Proposer,
        solver: Solver,
        verifier: Verifier,
        selector: Selector,
        migrator: Migrator,
        anchor: Anchor,
    ) -> None:
        self.config = config
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.selector = selector
        self.migrator = migrator
        self.anchor = anchor
        self.state = LoopState()

    # ---- inner loop ---------------------------------------------------------

    def propose_batch(self) -> list[Task]:
        """AZR §3.3 proposer — generate proposals across task types."""
        tasks: list[Task] = []
        n_per_type = self.config.proposer.proposals_per_generation // 3
        for task_type in (TaskType.DEDUCTION, TaskType.ABDUCTION, TaskType.INDUCTION):
            exp = self._aggregate_experience()
            tasks.extend(self.proposer.propose(task_type, exp, n_per_type))
        return tasks

    def solve_and_verify(self, tasks: list[Task], branch: Branch) -> list[Match]:
        """Single-branch rollout: solve every task, verify, produce matches later."""
        raise NotImplementedError(
            "Phase 2. See docs/04-implementation.md §2.3. "
            "Must: (a) load branch LoRA, (b) batch-solve, (c) verify each, "
            "(d) append to branch.experience, (e) emit one Match per pairing "
            "against other branches solving the same task."
        )

    # ---- middle loop --------------------------------------------------------

    def run_matches(self, tasks: list[Task], islands: list[Island]) -> list[Match]:
        """Round-robin pairs of branches on shared tasks. See docs/03-stack.md §'From RoboPhD'."""
        raise NotImplementedError("Phase 2. Pair selection: random pairs within island.")

    def grpo_update(self, islands: list[Island]) -> None:
        """Per-branch GRPO update on its own experience buffer.

        Uses verl as the training backend; see docs/01-sources.md 'verl'.
        Top quartile: update. Middle half: update. Bottom quartile: replaced
        by mutated copy (handled in selector.replacement_plan).
        """
        raise NotImplementedError("Phase 2. Hand off to verl GRPO trainer per-branch.")

    # ---- outer loop ---------------------------------------------------------

    def maybe_migrate(self, islands: list[Island]) -> list[Island]:
        if not self.config.islands.enabled:
            return islands
        if self.state.gen % self.config.islands.migration_every != 0:
            return islands
        log.info("MIGRATION at gen %d", self.state.gen)
        return self.migrator.migrate(islands, self.state.gen)

    # ---- anchor -------------------------------------------------------------

    def maybe_anchor_check(self, islands: list[Island]) -> bool:
        """Return True if generation is committed, False if reverted."""
        if self.state.gen == 0 or self.state.gen % self.config.anchor.anchor_every != 0:
            return True  # no-op gen, always "committed"
        population = [b for island in islands for b in island.branches]
        curr = self.anchor.evaluate(population)
        if self.anchor.should_revert(self.state.last_anchor, curr):
            log.warning(
                "ANCHOR REVERT at gen %d (curr=%.3f, prev=%.3f)",
                self.state.gen,
                curr.aggregate,
                self.state.last_anchor.aggregate if self.state.last_anchor else float("nan"),
            )
            self._revert_to(self.state.committed_gen)
            return False
        self.state.last_anchor = curr
        self.state.committed_gen = self.state.gen
        log.info("ANCHOR COMMIT at gen %d score=%.3f", self.state.gen, curr.aggregate)
        return True

    # ---- main driver --------------------------------------------------------

    def run(self, islands: list[Island]) -> list[Island]:
        """Run the loop until max_generations or anchor stabilization."""
        while self.state.gen < self.config.max_generations:
            self.state.gen += 1
            log.info("=== gen %d ===", self.state.gen)

            # Inner: generate tasks
            tasks = self.propose_batch()

            # Middle: matches, Elo, GRPO
            matches = self.run_matches(tasks, islands)
            self.state.elo = self.selector.update(matches, self.state.elo or EloState())
            self.grpo_update(islands)
            plan = self.selector.replacement_plan(self.state.elo)
            islands = self._apply_replacement(islands, plan)

            # Outer: migration
            islands = self.maybe_migrate(islands)

            # Anchor
            self.maybe_anchor_check(islands)

        return islands

    # ---- helpers ------------------------------------------------------------

    def _aggregate_experience(self) -> Experience:
        """For Phase 1: a shared experience pool. Phase 2+: per-island or per-branch."""
        return Experience()

    def _apply_replacement(self, islands: list[Island], plan) -> list[Island]:
        raise NotImplementedError("Phase 2. Apply ReplacementPlan: delete LoRAs, re-seed from parent + noise.")

    def _revert_to(self, gen: int) -> None:
        raise NotImplementedError(
            "Phase 2. Revert LoRA checkpoints and experience buffers to committed gen. "
            "Use VERSIONS.lock to also revert dependency state if needed (docs/01-sources.md)."
        )
