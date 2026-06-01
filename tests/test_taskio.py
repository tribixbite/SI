"""Round-trip tests for Task JSONL (de)serialization."""

from si.contracts import EloState, Task, TaskType
from si.taskio import (
    read_elo_state,
    read_tasks,
    task_from_dict,
    task_to_dict,
    write_elo_state,
    write_tasks,
)


def _task(tid: str, tt: TaskType = TaskType.DEDUCTION) -> Task:
    return Task(task_type=tt, program="def f(x): return x", input="1", output=None,
                proposer_branch_id="p", gen=3, task_id=tid)


def test_dict_round_trip_preserves_all_fields():
    t = _task("t1", TaskType.ABDUCTION)
    t2 = task_from_dict(task_to_dict(t))
    assert t2 == t


def test_jsonl_round_trip(tmp_path):
    tasks = [_task("a", TaskType.DEDUCTION), _task("b", TaskType.ABDUCTION)]
    p = tmp_path / "tasks.jsonl"
    write_tasks(p, tasks)
    assert read_tasks(p) == tasks


def test_elo_state_round_trip(tmp_path):
    s = EloState(ratings={"a": 1612.5, "b": 1487.5}, k=24.0, default_rating=1500.0)
    p = tmp_path / "elo.json"
    write_elo_state(p, s)
    s2 = read_elo_state(p)
    assert s2.ratings == s.ratings and s2.k == s.k and s2.default_rating == s.default_rating


def test_read_elo_state_absent_returns_fresh(tmp_path):
    s = read_elo_state(tmp_path / "nope.json")
    assert s.ratings == {} and s.default_rating == 1500.0


def test_read_skips_blank_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join([
        '{"task_type": "deduction", "program": "p", "input": "i", "output": null, '
        '"proposer_branch_id": "x", "gen": 0, "task_id": "z"}',
        "",
        "",
    ]))
    out = read_tasks(p)
    assert len(out) == 1 and out[0].task_id == "z"
