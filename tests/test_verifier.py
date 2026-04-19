"""Integration tests for SandboxVerifier against a real sandbox-fusion container.

These tests are slow (~60s cold start) and require docker + the sandbox-fusion
image locally. Skip gracefully when either is missing.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from si.contracts import Solution, Task, TaskType
from si.verifier import SANDBOX_IMAGE, SandboxContainer, SandboxVerifier


def _docker_image_present() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", SANDBOX_IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_image_present(),
    reason=f"docker or {SANDBOX_IMAGE} not available",
)


@pytest.fixture(scope="module")
def verifier():
    v = SandboxVerifier(container=SandboxContainer())
    yield v
    v.close()


def _make_task(task_type: TaskType, *, program=None, inp=None, out=None) -> Task:
    return Task(
        task_type=task_type,
        program=program,
        input=inp,
        output=out,
        proposer_branch_id="p0",
        gen=0,
        task_id="t",
    )


def _make_solution(body: str) -> Solution:
    return Solution(task_id="t", solver_branch_id="s0", body=body, trace="", walltime_ms=0)


def test_deduction_correct_answer(verifier):
    task = _make_task(
        TaskType.DEDUCTION,
        program="def f(x):\n    return x * 2",
        inp="5",
    )
    result = verifier.verify(task, _make_solution("10"))
    assert result.passed


def test_deduction_wrong_answer(verifier):
    task = _make_task(
        TaskType.DEDUCTION,
        program="def f(x):\n    return x * 2",
        inp="5",
    )
    result = verifier.verify(task, _make_solution("11"))
    assert not result.passed


def test_deduction_timeout_fails_gracefully(verifier):
    task = _make_task(
        TaskType.DEDUCTION,
        program="def f(x):\n    while True: pass",
        inp="1",
    )
    result = verifier.verify(task, _make_solution("1"))
    assert not result.passed


def test_abduction_correct_input(verifier):
    task = _make_task(
        TaskType.ABDUCTION,
        program="def f(x):\n    return x * 2",
        out="10",
    )
    result = verifier.verify(task, _make_solution("5"))
    assert result.passed


def test_abduction_wrong_input(verifier):
    task = _make_task(
        TaskType.ABDUCTION,
        program="def f(x):\n    return x * 2",
        out="10",
    )
    result = verifier.verify(task, _make_solution("6"))
    assert not result.passed


def test_induction_correct_program(verifier):
    task = _make_task(
        TaskType.INDUCTION,
        inp="[1, 2, 3]",
        out="[2, 4, 6]",
    )
    result = verifier.verify(task, _make_solution("def f(x):\n    return x * 2"))
    assert result.passed


def test_induction_partially_correct_fails(verifier):
    task = _make_task(
        TaskType.INDUCTION,
        inp="[1, 2, 3]",
        out="[2, 4, 7]",
    )
    result = verifier.verify(task, _make_solution("def f(x):\n    return x * 2"))
    assert not result.passed


def test_syntactically_invalid_solution_fails(verifier):
    task = _make_task(
        TaskType.DEDUCTION,
        program="def f(x):\n    return x",
        inp="1",
    )
    result = verifier.verify(task, _make_solution("(((("))  # not a valid literal
    assert not result.passed
