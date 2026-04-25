"""Mix 1 — proposer co-training data construction.

Closes AZR's full loop: train the proposer to emit tasks at the solver's
MC-difficulty edge so the solver's training distribution targets its own
learning frontier.

Pipeline:
    1. Use current proposer to generate N candidate tasks.
    2. Score each task: run K solver MC rollouts → compute pass_rate.
    3. Filter to medium-difficulty tasks (pass_rate ∈ [low, high], default [0.3, 0.7]).
    4. SFT proposer on (system_prompt, generated_task_text) pairs that survived.

This module supplies the data side. The actual SFT reuses src/si/trainer_ssd.py
since the schema matches: chat-format prompt + completion text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from si.contracts import Task, TaskType
from si.match import MatchRunner
from si.proposer import AZRProposer
from si.prompts import (
    proposer_abduction_prompt,
    proposer_deduction_prompt,
)
from si.solver import GemmaSolver
from si.ssd import SSDSample
from si.verifier import SandboxVerifier

log = logging.getLogger(__name__)

_SYSTEM_PROPOSER = (
    "You are a careful Python task designer for an AI reasoning benchmark. "
    "Produce tasks that are crisp, deterministic, and solvable with reasoning. "
    "Follow the output format exactly."
)


def _task_to_text(task: Task) -> str:
    """Reconstruct the canonical fenced output a proposer would emit for a task."""
    if task.task_type is TaskType.DEDUCTION:
        program = task.program or ""
        inp = task.input or ""
        return f"```python\n{program}\n```\n\n```input\n{inp}\n```"
    elif task.task_type is TaskType.ABDUCTION:
        program = task.program or ""
        inp = task.input or ""  # the proposer's emitted input; verified-runnable
        return f"```python\n{program}\n```\n\n```input\n{inp}\n```"
    raise ValueError(f"Unsupported task type {task.task_type}")


def _proposer_user_prompt(task_type: TaskType) -> str:
    if task_type is TaskType.DEDUCTION:
        return proposer_deduction_prompt()
    return proposer_abduction_prompt()


def collect_proposer_training_pairs(
    runner: MatchRunner,
    *,
    proposals_per_type: int,
    mc_rollouts: int,
    min_pass_rate: float,
    max_pass_rate: float,
    task_types: list[TaskType] | None = None,
) -> list[SSDSample]:
    """Run a generation of proposer→solver→verify, score by MC pass_rate, filter,
    and return SSDSample-shaped (prompt, completion) pairs ready for SFT.

    Each surviving task becomes one training example: prompt is the proposer's
    system+user message that produced it; completion is the canonical fenced
    rendering of the task itself.
    """
    runner.proposals_per_type = proposals_per_type
    runner.mc_rollouts = mc_rollouts

    types = task_types or [TaskType.DEDUCTION, TaskType.ABDUCTION]
    results = runner.run_generation(types)
    pairs: list[SSDSample] = []
    kept = 0
    for outcome in results.outcomes:
        pr = outcome.pass_rate
        if pr < min_pass_rate or pr > max_pass_rate:
            continue
        kept += 1
        task = outcome.task
        user_prompt = _proposer_user_prompt(task.task_type)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROPOSER}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        pairs.append(
            SSDSample(
                task_id=f"proposer_{task.task_id}",
                task_type=f"proposer_{task.task_type.value}",
                prompt_messages=messages,
                completion_text=_task_to_text(task),
            )
        )
    log.info(
        "proposer_train: %d tasks generated, %d failed-resolution, %d in difficulty band, %d pairs",
        results.n_tasks,
        results.failed_proposals,
        kept,
        len(pairs),
    )
    return pairs


def build_match_runner(
    *,
    llm,
    verifier: SandboxVerifier,
    branch_id: str = "p_train",
    gen: int = 0,
    proposer_temperature: float = 1.0,
    solver_temperature: float = 0.7,
) -> MatchRunner:
    """One-stop helper: assemble Proposer + Solver + Verifier into a MatchRunner."""
    proposer = AZRProposer(
        llm, branch_id=branch_id, temperature=proposer_temperature, gen=gen
    )
    solver = GemmaSolver(llm, branch_id=f"s_{branch_id}", temperature=solver_temperature)
    return MatchRunner(
        proposer=proposer,
        solver=solver,
        verifier=verifier,
        proposals_per_type=8,  # caller may override
        mc_rollouts=4,         # caller may override
    )
