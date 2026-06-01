"""JSONL (de)serialization for Task / match results.

Shared by the Phase 2 subprocess primitives (scripts/phase2_propose.py writes
resolved tasks, scripts/phase2_solve.py reads them and writes per-branch
results). Kept tiny and torch-free so it is unit-tested without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from si.contracts import EloState, Task, TaskType


def task_to_dict(t: Task) -> dict[str, Any]:
    return {
        "task_type": t.task_type.value,
        "program": t.program,
        "input": t.input,
        "output": t.output,
        "proposer_branch_id": t.proposer_branch_id,
        "gen": t.gen,
        "task_id": t.task_id,
    }


def task_from_dict(d: dict[str, Any]) -> Task:
    return Task(
        task_type=TaskType(d["task_type"]),
        program=d["program"],
        input=d["input"],
        output=d["output"],
        proposer_branch_id=d["proposer_branch_id"],
        gen=d["gen"],
        task_id=d["task_id"],
    )


def write_tasks(path: str | Path, tasks: list[Task]) -> None:
    with Path(path).open("w") as f:
        for t in tasks:
            f.write(json.dumps(task_to_dict(t)) + "\n")


def read_tasks(path: str | Path) -> list[Task]:
    with Path(path).open() as f:
        return [task_from_dict(json.loads(line)) for line in f if line.strip()]


def write_elo_state(path: str | Path, state: EloState) -> None:
    Path(path).write_text(
        json.dumps({"ratings": state.ratings, "k": state.k, "default_rating": state.default_rating})
    )


def read_elo_state(path: str | Path) -> EloState:
    """Load persisted Elo (carried across generations). Fresh state if absent."""
    p = Path(path)
    if not p.exists():
        return EloState()
    d = json.loads(p.read_text())
    return EloState(ratings=dict(d["ratings"]), k=d["k"], default_rating=d["default_rating"])
