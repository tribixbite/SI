"""Round-trip tests for Task JSONL (de)serialization."""

from si.contracts import Task, TaskType
from si.taskio import read_tasks, task_from_dict, task_to_dict, write_tasks


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
