"""GemmaSolver — predicts the missing field of a Task via Gemma 4.

Phase 1: deduction (predict output) and abduction (predict input).

The solver is stateless; it takes a Task and emits a Solution. MC rollouts
(K completions per task) for the proposer reward are implemented by setting
GenParams.n; that batching keeps vLLM fully utilized.
"""

from __future__ import annotations

import logging
import time

from si.contracts import Solution, Solver, Task, TaskType
from si.llm import GemmaLLM, GenParams
from si.parsers import extract_input, extract_output
from si.prompts import solver_abduction_prompt, solver_deduction_prompt

log = logging.getLogger(__name__)

_SYSTEM_SOLVER = (
    "You are a careful Python reasoning assistant. Given a function and one of "
    "(input, output), predict the missing value exactly. Output only the "
    "requested fenced block — no explanation."
)


class GemmaSolver(Solver):
    """Batched solver.

    solve_batch([task1, task2, ...]) -> [Solution for task1, Solution for task2, ...]

    With GenParams.n > 1, solve_rollouts(task, k) produces k candidate
    solutions — used by the match loop for MC proposer reward.
    """

    def __init__(
        self,
        llm: GemmaLLM,
        *,
        branch_id: str = "s0",
        temperature: float = 0.5,
        max_tokens: int = 1024,
        lora_path: str | None = None,
    ) -> None:
        self.llm = llm
        self.branch_id = branch_id
        self.lora_path = lora_path  # branch adapter swapped in per chat_batch call
        self.params = GenParams(temperature=temperature, max_tokens=max_tokens)

    # ---- Solver protocol ----------------------------------------------------

    def solve(self, task: Task) -> Solution:
        return self.solve_batch([task])[0]

    def solve_batch(self, tasks: list[Task]) -> list[Solution]:
        prompts = [self._prompt_for(t) for t in tasks]
        t0 = time.time()
        completions = self.llm.chat_batch(
            prompts, self.params, system=_SYSTEM_SOLVER, lora_path=self.lora_path
        )
        wall_ms = int((time.time() - t0) * 1000)
        results: list[Solution] = []
        for task, comp_list in zip(tasks, completions, strict=True):
            body = self._parse_solution(task, comp_list[0])
            results.append(
                Solution(
                    task_id=task.task_id,
                    solver_branch_id=self.branch_id,
                    body=body or "",
                    trace=comp_list[0],
                    walltime_ms=wall_ms // max(1, len(tasks)),
                )
            )
        return results

    # ---- MC rollout for proposer reward -------------------------------------

    def solve_rollouts(self, task: Task, k: int) -> list[Solution]:
        """K independent solutions for the same task. Used for MC proposer reward."""
        rollout_params = GenParams(
            temperature=max(self.params.temperature, 0.7),  # diversity for MC
            top_p=self.params.top_p,
            max_tokens=self.params.max_tokens,
            n=k,
        )
        t0 = time.time()
        completions = self.llm.chat_batch(
            [self._prompt_for(task)], rollout_params, system=_SYSTEM_SOLVER, lora_path=self.lora_path
        )
        wall_ms = int((time.time() - t0) * 1000)
        return [
            Solution(
                task_id=task.task_id,
                solver_branch_id=self.branch_id,
                body=self._parse_solution(task, text) or "",
                trace=text,
                walltime_ms=wall_ms // max(1, k),
            )
            for text in completions[0]
        ]

    # ---- internals ----------------------------------------------------------

    def _prompt_for(self, task: Task) -> str:
        if task.task_type is TaskType.DEDUCTION:
            assert task.program is not None and task.input is not None
            return solver_deduction_prompt(task.program, task.input)
        if task.task_type is TaskType.ABDUCTION:
            assert task.program is not None and task.output is not None
            return solver_abduction_prompt(task.program, task.output)
        raise NotImplementedError(f"solver prompt for {task.task_type!r} not built yet")

    def _parse_solution(self, task: Task, text: str) -> str | None:
        if task.task_type is TaskType.DEDUCTION:
            return extract_output(text)
        return extract_input(text)
