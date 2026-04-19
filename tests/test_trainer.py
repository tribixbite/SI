"""Unit tests for trainer scaffolding — no GPU, no sandbox."""

from si.contracts import Task, TaskType
from si.match import ProposalOutcome, Rollout
from si.trainer import (
    _extract_body,
    _task_from_row,
    _to_text,
    outcomes_to_dataset,
)


def _mk_task(task_type: TaskType, **kwargs) -> Task:
    return Task(
        task_type=task_type,
        program=kwargs.get("program"),
        input=kwargs.get("input"),
        output=kwargs.get("output"),
        proposer_branch_id="p0",
        gen=0,
        task_id=kwargs.get("task_id", "t0"),
    )


def test_outcomes_to_dataset_roundtrips_tasks():
    ded = _mk_task(TaskType.DEDUCTION, program="def f(x): return x", input="5")
    abd = _mk_task(TaskType.ABDUCTION, program="def f(x): return x*2", output="10")
    ds = outcomes_to_dataset(
        [ProposalOutcome(task=ded, rollouts=[]), ProposalOutcome(task=abd, rollouts=[])]
    )
    assert len(ds) == 2
    assert ds[0]["task_type"] == "deduction"
    assert ds[1]["task_type"] == "abduction"
    # Chat prompt format
    assert isinstance(ds[0]["prompt"], list)
    assert ds[0]["prompt"][0]["role"] == "system"
    assert ds[0]["prompt"][1]["role"] == "user"


def test_task_from_row_recovers_task():
    task = _mk_task(TaskType.DEDUCTION, program="def f(x): return x", input="5")
    ds = outcomes_to_dataset([ProposalOutcome(task=task, rollouts=[])])
    recovered = _task_from_row(ds[0])
    assert recovered.task_type is TaskType.DEDUCTION
    assert recovered.program == "def f(x): return x"
    assert recovered.input == "5"


def test_to_text_chat_format():
    completion = [{"role": "assistant", "content": "hello"}]
    assert _to_text(completion) == "hello"


def test_to_text_string_format():
    assert _to_text("hello") == "hello"


def test_to_text_empty_chat():
    assert _to_text([{"role": "user", "content": "hi"}]) == ""


def test_extract_body_deduction_looks_for_output_block():
    task = _mk_task(TaskType.DEDUCTION, program="def f(x): return x", input="5")
    assert _extract_body(task, "```output\n42\n```") == "42"
    assert _extract_body(task, "no fence") is None


def test_extract_body_abduction_looks_for_input_block():
    task = _mk_task(TaskType.ABDUCTION, program="def f(x): return x", output="5")
    assert _extract_body(task, "```input\n5\n```") == "5"
