"""SSD — Simple Self-Distillation (arXiv:2604.01193).

The idea: sample N candidate solutions per task from the current policy, keep
those the verifier passes, and fine-tune the model on (prompt, passing-response)
pairs. No RL, no advantage, no zero-variance problem. Pure SFT on the model's
own successful samples.

Two reasons to try this after the v2/v3 plateau:
    1. SSD paper reports +12.9 pp on Qwen3-30B LiveCodeBench v6 where GRPO
       fails — directly applicable to our setup.
    2. Our diagnosis from v2/v3 is that within-batch advantage methods can't
       extract direction from all-fail groups. SSD sidesteps the issue by
       only training on passing samples, so every gradient step has a
       positive target.

The stack:
    ssd-sample: vLLM generates N candidates per task; SandboxVerifier filters.
                Writes `samples.jsonl` with (task_id, prompt, passing_body).
    ssd-train:  Unsloth + TRL SFTTrainer fine-tunes on that dataset.

Two-process design because Gemma 4 E4B bf16 (vLLM) + 4-bit QLoRA (Unsloth) +
activations don't coexist on one 3090 — same ping-pong rationale as Phase 1.

Not included in this module: the CLI wiring (see si.cli), the Unsloth SFT
trainer (see si.trainer_ssd), or the sample-orchestration at CLI level.
This module only provides the pure-Python functions they compose.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from si.contracts import Solution, Task, TaskType
from si.parsers import extract_input, extract_output
from si.verifier import SandboxVerifier

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSDSample:
    """One verifier-passing sample: the prompt-as-chat-messages + the body we want to imitate."""

    task_id: str
    task_type: str
    prompt_messages: list[dict]
    completion_text: str


def load_task_pool(outcomes_files: list[str]) -> list[Task]:
    """Load tasks from outcomes*.jsonl files, dedupe by task_id."""
    seen: dict[str, Task] = {}
    for path in outcomes_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t = rec["task"]
                if t["task_id"] in seen:
                    continue
                seen[t["task_id"]] = Task(
                    task_type=TaskType(t["task_type"]),
                    program=t["program"],
                    input=t["input"],
                    output=t["output"],
                    proposer_branch_id=t["proposer_branch_id"],
                    gen=t["gen"],
                    task_id=t["task_id"],
                )
    tasks = list(seen.values())
    log.info("load_task_pool: %d unique tasks across %d files", len(tasks), len(outcomes_files))
    return tasks


def extract_body(task: Task, completion_text: str) -> str | None:
    """Pull the expected fenced body out of a solver completion."""
    if task.task_type is TaskType.DEDUCTION:
        return extract_output(completion_text)
    return extract_input(completion_text)


def verify_and_pack(
    *,
    task: Task,
    candidates: list[str],
    verifier: SandboxVerifier,
    system_prompt: str,
    user_prompt: str,
    keep_failing: bool = False,
) -> list[SSDSample] | tuple[list[SSDSample], list[SSDSample]]:
    """Run the verifier against each candidate, keep passing ones.

    With keep_failing=True, returns (passing, failing) where failing samples
    are usable for DPO preference pairs.
    """
    passing: list[SSDSample] = []
    failing: list[SSDSample] = []
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    for text in candidates:
        body = extract_body(task, text)
        sample = SSDSample(
            task_id=task.task_id,
            task_type=task.task_type.value,
            prompt_messages=messages,
            completion_text=text.rstrip(),
        )
        if not body:
            if keep_failing:
                failing.append(sample)
            continue
        sol = Solution(
            task_id=task.task_id, solver_branch_id="ssd", body=body, trace=text, walltime_ms=0
        )
        try:
            result = verifier.verify(task, sol)
        except Exception as e:
            log.warning("verifier error on %s: %s", task.task_id, e)
            if keep_failing:
                failing.append(sample)
            continue
        if result.passed:
            passing.append(sample)
        elif keep_failing:
            failing.append(sample)
    return (passing, failing) if keep_failing else passing


def write_samples(samples: list[SSDSample], out_path: str) -> None:
    """Emit SSDSample records as JSONL, one per line."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "task_id": s.task_id,
                        "task_type": s.task_type,
                        "prompt_messages": s.prompt_messages,
                        "completion_text": s.completion_text,
                    }
                )
                + "\n"
            )


def read_samples(path: str) -> list[SSDSample]:
    out: list[SSDSample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(
                SSDSample(
                    task_id=rec["task_id"],
                    task_type=rec["task_type"],
                    prompt_messages=rec["prompt_messages"],
                    completion_text=rec["completion_text"],
                )
            )
    return out
