"""AZRProposer — generates (program, input-or-output) triples via Gemma 4.

Phase 1 supports deduction and abduction. Induction deferred (needs seed
programs).

Per AZR §3.3: proposer reward uses MC rollout to target pass_rate ≈ 0.5 on the
current solver. That computation lives in the match loop, not here — this
module only generates candidate tasks.
"""

from __future__ import annotations

import hashlib
import logging

from si.contracts import Experience, Proposer, Task, TaskType
from si.llm import GemmaLLM, GenParams
from si.parsers import contains_banned_import, extract_input, extract_program
from si.prompts import (
    BANNED_IMPORTS,
    proposer_abduction_prompt,
    proposer_deduction_prompt,
)

log = logging.getLogger(__name__)

_SYSTEM_PROPOSER = (
    "You are a careful Python task designer for an AI reasoning benchmark. "
    "Produce tasks that are crisp, deterministic, and solvable with reasoning. "
    "Follow the output format exactly."
)


def _task_id(task_type: TaskType, program: str, field: str) -> str:
    digest = hashlib.sha256()
    digest.update(task_type.value.encode())
    digest.update(b"\x00")
    digest.update(program.encode())
    digest.update(b"\x00")
    digest.update(field.encode())
    return digest.hexdigest()[:16]


class AZRProposer(Proposer):
    """Generates tasks by prompting Gemma 4 to write program+input pairs."""

    def __init__(
        self,
        llm: GemmaLLM,
        *,
        branch_id: str = "p0",
        temperature: float = 0.8,
        max_tokens: int = 1024,
        gen: int = 0,
    ) -> None:
        self.llm = llm
        self.branch_id = branch_id
        self.params = GenParams(temperature=temperature, max_tokens=max_tokens)
        self.gen = gen

    def propose(self, task_type: TaskType, experience: Experience, n: int) -> list[Task]:
        if task_type is TaskType.INDUCTION:
            log.debug("Induction proposer not implemented in Phase 1; returning [].")
            return []
        prompt = self._prompt_for(task_type)
        completions = self.llm.chat_batch(
            user_prompts=[prompt] * n,
            params=self.params,
            system=_SYSTEM_PROPOSER,
        )
        tasks: list[Task] = []
        for batch in completions:
            for text in batch:
                task = self._parse(task_type, text)
                if task is not None:
                    tasks.append(task)
        log.info("proposer %s: %d/%d parseable tasks", task_type.value, len(tasks), n)
        return tasks

    def _prompt_for(self, task_type: TaskType) -> str:
        if task_type is TaskType.DEDUCTION:
            return proposer_deduction_prompt()
        return proposer_abduction_prompt()

    def _parse(self, task_type: TaskType, text: str) -> Task | None:
        program = extract_program(text)
        if program is None or contains_banned_import(program, BANNED_IMPORTS):
            return None
        input_lit = extract_input(text)
        if input_lit is None:
            return None
        # For deduction we keep the input and leave output None (solver fills it).
        # For abduction we still need to *derive* the target output by running the
        # program — that happens in the match loop, not here. We stash the input
        # in task.input and leave output None; the match loop will run the program
        # and overwrite task.output before handing to the solver.
        tid = _task_id(task_type, program, input_lit)
        return Task(
            task_type=task_type,
            program=program,
            input=input_lit,
            output=None,
            proposer_branch_id=self.branch_id,
            gen=self.gen,
            task_id=tid,
        )
