"""Unit tests for SSD pure functions. No GPU, no sandbox."""

import json
import tempfile
from pathlib import Path

from si.contracts import TaskType
from si.ssd import SSDSample, extract_body, load_task_pool, read_samples, write_samples


def _mk_outcomes_file(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_load_task_pool_dedupes_by_task_id():
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.jsonl"
        p2 = Path(d) / "b.jsonl"
        rec = lambda tid, inp: {
            "task": {
                "task_type": "deduction", "program": "def f(x): return x", "input": inp,
                "output": None, "proposer_branch_id": "p", "gen": 0, "task_id": tid,
            },
        }
        _mk_outcomes_file(p1, [rec("t1", "1"), rec("t2", "2")])
        _mk_outcomes_file(p2, [rec("t2", "2"), rec("t3", "3")])  # t2 duplicated across files
        tasks = load_task_pool([str(p1), str(p2)])
        assert len(tasks) == 3  # t1, t2, t3 (not t2 twice)
        assert {t.task_id for t in tasks} == {"t1", "t2", "t3"}


def test_load_task_pool_empty_and_blank_lines():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.jsonl"
        p.write_text("\n\n")  # just blank lines
        tasks = load_task_pool([str(p)])
        assert tasks == []


def test_extract_body_deduction_uses_output_block():
    from si.contracts import Task

    task = Task(
        task_type=TaskType.DEDUCTION, program="def f(x): return x", input="1", output=None,
        proposer_branch_id="p", gen=0, task_id="t",
    )
    assert extract_body(task, "```output\n42\n```") == "42"
    assert extract_body(task, "no fence") is None


def test_extract_body_abduction_uses_input_block():
    from si.contracts import Task

    task = Task(
        task_type=TaskType.ABDUCTION, program="def f(x): return x", input=None, output="42",
        proposer_branch_id="p", gen=0, task_id="t",
    )
    assert extract_body(task, "```input\n5\n```") == "5"


def test_samples_roundtrip_jsonl():
    samples = [
        SSDSample(
            task_id="t1", task_type="deduction",
            prompt_messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            completion_text="```output\n42\n```",
        ),
        SSDSample(
            task_id="t2", task_type="abduction",
            prompt_messages=[{"role": "user", "content": [{"type": "text", "text": "find"}]}],
            completion_text="```input\n5\n```",
        ),
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.jsonl"
        write_samples(samples, str(p))
        back = read_samples(str(p))
        assert len(back) == 2
        assert back[0].task_id == "t1"
        assert back[1].completion_text == "```input\n5\n```"
        assert back[0].prompt_messages == samples[0].prompt_messages
