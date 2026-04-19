"""Narrow interface contracts between SI components.

Each component (proposer, solver, verifier, selector, migrator, anchor) is
defined as a Protocol so concrete implementations can be swapped for ablation
without touching the loop.

This file is the load-bearing part of the architecture. Changes here ripple
through the whole stack; changes elsewhere usually don't need to touch this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class TaskType(str, Enum):
    """AZR's three task types (Zhao et al. 2025, §3.2)."""

    DEDUCTION = "deduction"  # given (program, input) → predict output
    ABDUCTION = "abduction"  # given (program, output) → predict input
    INDUCTION = "induction"  # given (input, output) pairs → predict program


@dataclass(frozen=True)
class Task:
    """A self-generated reasoning task. Triple (P, I, O); two fields given."""

    task_type: TaskType
    program: str | None
    input: str | None
    output: str | None
    proposer_branch_id: str
    gen: int
    task_id: str  # content hash; stable across runs

    def missing_field(self) -> str:
        if self.task_type is TaskType.DEDUCTION:
            return "output"
        if self.task_type is TaskType.ABDUCTION:
            return "input"
        return "program"


@dataclass(frozen=True)
class Solution:
    """A solver's answer to a Task."""

    task_id: str
    solver_branch_id: str
    body: str  # the predicted missing field; code for induction, value otherwise
    trace: str  # the solver's chain-of-thought, for In-Place TTT + analysis
    walltime_ms: int


@dataclass(frozen=True)
class VerifyResult:
    """Binary verifier outcome plus diagnostic output."""

    task_id: str
    solution_id: str
    passed: bool
    stdout: str
    stderr: str
    walltime_ms: int
    exit_code: int


@dataclass
class Experience:
    """A branch's rolling experience buffer, used for GRPO replay and proposer priming."""

    recent_wins: list[tuple[Task, Solution]] = field(default_factory=list)
    recent_losses: list[tuple[Task, Solution]] = field(default_factory=list)
    proposer_seeds: list[Task] = field(default_factory=list)

    def append_match(self, task: Task, solution: Solution, passed: bool) -> None:
        (self.recent_wins if passed else self.recent_losses).append((task, solution))


@dataclass
class Match:
    """Head-to-head outcome between two branches on one task."""

    task_id: str
    branch_a: str
    branch_b: str
    winner: str | None  # None → draw (both pass or both fail)
    walltime_ms: int


@dataclass
class EloState:
    ratings: dict[str, float] = field(default_factory=dict)
    k: float = 32.0
    default_rating: float = 1500.0


@dataclass
class AnchorResult:
    gen: int
    aggregate: float  # mean pass@1 across branches on anchor set
    per_branch: dict[str, float] = field(default_factory=dict)


# ---- Protocols ---------------------------------------------------------------


@runtime_checkable
class Proposer(Protocol):
    """Generates tasks with optimal difficulty (AZR §3.3 MC-rollout reward)."""

    def propose(self, task_type: TaskType, experience: Experience, n: int) -> list[Task]: ...


@runtime_checkable
class Solver(Protocol):
    """Solves tasks. May use In-Place TTT internally; interface unchanged."""

    def solve(self, task: Task) -> Solution: ...

    def solve_batch(self, tasks: list[Task]) -> list[Solution]: ...


@runtime_checkable
class Verifier(Protocol):
    """Sandboxed, deterministic execution of a solution against a task.

    Implementations MUST be memory-bounded, CPU-bounded, network-disabled, and
    filesystem-isolated. See docs/04-implementation.md §3.
    """

    def verify(self, task: Task, solution: Solution) -> VerifyResult: ...


@runtime_checkable
class Selector(Protocol):
    """Ranks branches based on match history and produces replacement plan."""

    def update(self, matches: list[Match], state: EloState) -> EloState: ...

    def replacement_plan(self, state: EloState) -> "ReplacementPlan": ...


@runtime_checkable
class Migrator(Protocol):
    """Ring-topology migration across islands (OpenEvolve pattern)."""

    def migrate(self, islands: list["Island"], gen: int) -> list["Island"]: ...


@runtime_checkable
class Anchor(Protocol):
    """Held-out benchmark evaluator with reversion decision."""

    def evaluate(self, population: list["Branch"]) -> AnchorResult: ...

    def should_revert(self, prev: AnchorResult | None, curr: AnchorResult) -> bool: ...


# ---- Placeholder types referenced above --------------------------------------


@dataclass
class ReplacementPlan:
    keep: list[str]  # branch ids to keep unchanged (top quartile)
    mutate: list[str]  # GRPO update (middle half)
    replace: list[tuple[str, str]]  # (dead_branch, parent_branch) pairs


@dataclass
class Branch:
    branch_id: str
    lora_path: str  # path to LoRA adapter on disk
    elo: float
    experience: Experience


@dataclass
class Island:
    island_id: str
    branches: list[Branch]
    elo_state: EloState


# ---- Fingerprint contract ----------------------------------------------------
#
# Every Task, Solution, VerifyResult and Match gets fingerprinted and written
# to an append-only log. Reversion = truncate log to last committed gen.
#
# This is the audit trail. If the system starts winning by cheating, the diff
# of the log before and after the cheating-gen will reveal it.
