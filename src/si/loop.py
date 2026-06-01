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
import random
from collections.abc import Callable
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
    ReplacementPlan,
    Selector,
    Solver,
    Task,
    TaskType,
    Verifier,
)
from si.population import BranchManager

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
        branch_manager: BranchManager | None = None,
        solver_factory: Callable[[Branch], Solver] | None = None,
        trainer_fn: Callable[[Branch], None] | None = None,
    ) -> None:
        self.config = config
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.selector = selector
        self.migrator = migrator
        self.anchor = anchor
        # Owns LoRA-branch lineage + snapshot/revert. Required for Phase 2;
        # left optional so Phase 1 single-branch entrypoints can omit it.
        self.branch_manager = branch_manager
        # GPU ops, injected so the orchestration below stays testable:
        #   solver_factory(branch) -> a Solver bound to the branch's LoRA
        #   trainer_fn(branch)     -> run one GRPO step on branch.experience,
        #                             writing the updated adapter to branch.lora_path
        # If solver_factory is None we fall back to the single shared solver.
        self._solver_factory = solver_factory
        self._trainer_fn = trainer_fn
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

    def solve_and_verify(self, tasks: list[Task], branch: Branch) -> dict[str, bool]:
        """One branch solves every task (with its own LoRA), each is verified,
        and the (task, solution, passed) triple is appended to the branch's
        experience buffer. Returns {task_id: passed} so run_matches can pair
        branches without re-solving. See docs/04-implementation.md §2.3."""
        solver = self._solver_for(branch)
        solutions = solver.solve_batch(tasks)
        passed: dict[str, bool] = {}
        for task, sol in zip(tasks, solutions, strict=True):
            result = self.verifier.verify(task, sol)
            branch.experience.append_match(task, sol, result.passed)
            passed[task.task_id] = result.passed
        return passed

    # ---- middle loop --------------------------------------------------------

    def run_matches(self, tasks: list[Task], islands: list[Island]) -> list[Match]:
        """Random branch pairs compete on shared tasks within each island
        (docs/04 §2.3). Each branch solves the task pool once (one LoRA load per
        branch), then ~`matches_per_generation_multiplier`*population pairings
        are sampled from the cached pass/fail. Winner = the branch that passed
        when the other didn't; equal outcomes are draws (winner=None)."""
        if not tasks:
            return []
        matches: list[Match] = []
        rng = random.Random(self.state.gen)  # reproducible per generation
        mult = self.config.elo.matches_per_generation_multiplier
        for island in islands:
            branches = island.branches
            if len(branches) < 2:
                continue
            results = {b.branch_id: self.solve_and_verify(tasks, b) for b in branches}
            n_matches = max(1, round(mult * len(branches)))
            for _ in range(n_matches):
                task = rng.choice(tasks)
                a, b = rng.sample(branches, 2)
                pa = results[a.branch_id][task.task_id]
                pb = results[b.branch_id][task.task_id]
                if pa == pb:
                    winner: str | None = None
                else:
                    winner = a.branch_id if pa else b.branch_id
                matches.append(
                    Match(
                        task_id=task.task_id,
                        branch_a=a.branch_id,
                        branch_b=b.branch_id,
                        winner=winner,
                        walltime_ms=0,
                    )
                )
        return matches

    def grpo_update(self, islands: list[Island]) -> None:
        """Per-branch GRPO update on each branch's own experience buffer.

        Top quartile + middle half are trained; the bottom quartile is skipped
        because it gets reseeded by _apply_replacement immediately after, so
        training it would be wasted compute (docs/04 §2.3.5-7)."""
        if self._trainer_fn is None:
            raise RuntimeError(
                "Phase 2 grpo_update requires a trainer_fn; "
                "construct Loop(..., trainer_fn=...). See docs/04-implementation.md §2.3."
            )
        plan = self.selector.replacement_plan(self.state.elo or EloState())
        train_ids = set(plan.keep) | set(plan.mutate)
        for island in islands:
            for branch in island.branches:
                if branch.branch_id in train_ids:
                    self._trainer_fn(branch)

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
        if self.branch_manager is not None:
            self.branch_manager.commit(self.state.gen)
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

    def _solver_for(self, branch: Branch) -> Solver:
        """A Solver bound to this branch's LoRA. Falls back to the shared solver
        when no factory is injected (single-branch / test paths)."""
        if self._solver_factory is not None:
            return self._solver_factory(branch)
        return self.solver

    def _apply_replacement(self, islands: list[Island], plan: ReplacementPlan) -> list[Island]:
        """Reseed bottom-quartile branches from top-quartile parents + Gaussian
        noise (docs/04 §2.3.5). The BranchManager mutates the Branch objects in
        place; islands hold the same references, so membership is preserved and
        the same list is returned."""
        if self.branch_manager is None:
            raise RuntimeError(
                "Phase 2 requires a BranchManager; construct Loop(..., branch_manager=...) "
                "from the population's branches. See docs/04-implementation.md §2.2."
            )
        self.branch_manager.apply_replacement(plan, self.state.gen, self.state.elo)
        return islands

    def _revert_to(self, gen: int) -> None:
        """Restore LoRA checkpoints + experience buffers to a committed gen.
        Pairs with Anchor.should_revert — the anti-collapse non-negotiable."""
        if self.branch_manager is None:
            raise RuntimeError(
                "Phase 2 requires a BranchManager to revert; see docs/04-implementation.md §2.2."
            )
        self.branch_manager.revert_to(gen)
