"""Match loop: proposer → solver (K rollouts) → verifier → rewards.

For Phase 1 (single branch) there are no pairwise matches — we just need
proposer and solver rewards for the GRPO update. Phase 2+ will extend this
module with Match records for Elo.

Per AZR §3.3:
    proposer_reward = -|0.5 - pass_rate_k|
    solver_reward   = 1 if passed else 0   (per rollout)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from si.contracts import Solution, Task, TaskType, VerifyResult
from si.proposer import AZRProposer
from si.solver import GemmaSolver
from si.verifier import FINAL_REPR_MARKER, SandboxVerifier, _parse_final_repr, _run

log = logging.getLogger(__name__)


@dataclass
class Rollout:
    solution: Solution
    result: VerifyResult

    @property
    def passed(self) -> bool:
        return self.result.passed


@dataclass
class ProposalOutcome:
    """All data generated for a single proposal: the task, K solver attempts,
    their verify results, and the derived rewards."""

    task: Task
    rollouts: list[Rollout] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.rollouts:
            return 0.0
        return sum(r.passed for r in self.rollouts) / len(self.rollouts)

    @property
    def proposer_reward(self) -> float:
        """AZR §3.3: -|0.5 - pass_rate|. Peaks at 0 when solver is at the edge."""
        return -abs(0.5 - self.pass_rate)

    def solver_rewards(self) -> list[float]:
        return [1.0 if r.passed else 0.0 for r in self.rollouts]


@dataclass
class GenerationResults:
    """Everything produced by one generation's match loop."""

    outcomes: list[ProposalOutcome]
    failed_proposals: int  # count of proposer outputs that didn't parse or didn't execute

    @property
    def n_tasks(self) -> int:
        return len(self.outcomes)

    @property
    def aggregate_pass_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.pass_rate for o in self.outcomes) / len(self.outcomes)

    @property
    def aggregate_proposer_reward(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.proposer_reward for o in self.outcomes) / len(self.outcomes)

    def mc_difficulty_histogram(self, bins: int = 10) -> list[int]:
        """For instrumentation (docs/05-evaluation.md 'proposer_mc_difficulty')."""
        hist = [0] * bins
        for o in self.outcomes:
            idx = min(bins - 1, max(0, math.floor(o.pass_rate * bins)))
            hist[idx] += 1
        return hist


class MatchRunner:
    """Runs one generation's worth of self-play matches."""

    def __init__(
        self,
        proposer: AZRProposer,
        solver: GemmaSolver,
        verifier: SandboxVerifier,
        *,
        mc_rollouts: int = 8,
        proposals_per_type: int = 16,
        probe_timeout_s: float = 5.0,
    ) -> None:
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.mc_rollouts = mc_rollouts
        self.proposals_per_type = proposals_per_type
        self.probe_timeout_s = probe_timeout_s

    def run_generation(self, task_types: list[TaskType] | None = None) -> GenerationResults:
        task_types = task_types or [TaskType.DEDUCTION, TaskType.ABDUCTION]
        outcomes: list[ProposalOutcome] = []
        failed = 0
        for tt in task_types:
            raw_tasks = self.proposer.propose(tt, experience=None, n=self.proposals_per_type)  # type: ignore[arg-type]
            tasks: list[Task] = []
            for task in raw_tasks:
                resolved = self._resolve_task(task)
                if resolved is None:
                    failed += 1
                else:
                    tasks.append(resolved)
            log.info(
                "match_runner: %s — %d/%d proposals became valid tasks",
                tt.value,
                len(tasks),
                self.proposals_per_type,
            )
            for task in tasks:
                outcomes.append(self._play(task))
        return GenerationResults(outcomes=outcomes, failed_proposals=failed)

    def _resolve_task(self, task: Task) -> Task | None:
        """For abduction, the proposer gives (P, I) but the verifier needs
        (P, O) where O=P(I). Run P(I) in the sandbox to fill O. For deduction,
        the input is already what the solver needs; leave output=None."""
        if task.task_type is TaskType.DEDUCTION:
            if task.program is None or task.input is None:
                return None
            return task
        if task.task_type is TaskType.ABDUCTION:
            if task.program is None or task.input is None:
                return None
            code = f"{task.program}\nprint({FINAL_REPR_MARKER!r}, repr(f({task.input})))"
            ok, stdout, _, _ = _run(code, self.probe_timeout_s)
            if not ok:
                return None
            target_out = _parse_final_repr(stdout)
            if target_out is None:
                return None
            return Task(
                task_type=TaskType.ABDUCTION,
                program=task.program,
                input=None,
                output=target_out,
                proposer_branch_id=task.proposer_branch_id,
                gen=task.gen,
                task_id=task.task_id,
            )
        return None

    def _play(self, task: Task) -> ProposalOutcome:
        solutions = self.solver.solve_rollouts(task, self.mc_rollouts)
        rollouts: list[Rollout] = []
        for sol in solutions:
            result = self.verifier.verify(task, sol)
            rollouts.append(Rollout(solution=sol, result=result))
        return ProposalOutcome(task=task, rollouts=rollouts)
