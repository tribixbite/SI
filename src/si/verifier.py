"""Sandboxed code execution for AZR verification.

Manages a volcengine/sandbox-fusion container and implements the Verifier
protocol. One container per run; dispatched by task type.

Safety model (docs/03-stack.md §"The trust model"):
    - Proposer and solver output strings that get executed.
    - The sandbox container is the isolation boundary.
    - This module NEVER calls subprocess.run on model output; every execution
      goes through sandbox_fusion.run_code.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass

import docker
import requests
from docker.models.containers import Container
from sandbox_fusion import RunCodeRequest, RunStatus, run_code, set_endpoint

from si.contracts import Solution, Task, TaskType, VerifyResult, Verifier

log = logging.getLogger(__name__)

SANDBOX_IMAGE = "volcengine/sandbox-fusion:server-20250609"
SANDBOX_INTERNAL_PORT = "8080/tcp"
FINAL_REPR_MARKER = "<FINAL_REPR_SYMBOL>"


@dataclass
class SandboxContainer:
    """Lifecycle manager for a sandbox-fusion docker container on a dynamic host port."""

    image: str = SANDBOX_IMAGE
    ready_timeout_s: int = 120
    container: Container | None = None
    host_port: int = 0

    def start(self) -> None:
        client = docker.from_env()
        self.host_port = _find_free_port()
        log.info("Starting sandbox container (image=%s, host_port=%d)", self.image, self.host_port)
        self.container = client.containers.run(
            self.image,
            ports={SANDBOX_INTERNAL_PORT: self.host_port},
            detach=True,
            remove=True,
        )
        self._wait_ready()
        set_endpoint(f"http://localhost:{self.host_port}")

    def _wait_ready(self) -> None:
        assert self.container is not None
        deadline = time.time() + self.ready_timeout_s
        endpoint = f"http://localhost:{self.host_port}/"
        while time.time() < deadline:
            self.container.reload()
            if self.container.status in ("exited", "dead"):
                logs = self.container.logs().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"Sandbox container died during startup: {logs}")
            if self.container.status == "running":
                try:
                    if requests.get(endpoint, timeout=2).status_code < 500:
                        log.info("Sandbox ready on :%d", self.host_port)
                        return
                except requests.RequestException:
                    pass
            time.sleep(1.0)
        raise TimeoutError(f"Sandbox container not ready after {self.ready_timeout_s}s")

    def stop(self) -> None:
        if self.container is None:
            return
        try:
            self.container.stop(timeout=5)
        except Exception as e:
            log.warning("sandbox stop failed: %s", e)
        finally:
            self.container = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _run(code: str, timeout_s: float = 5.0) -> tuple[bool, str, str, int]:
    """Execute python code in the sandbox. Returns (passed_status, stdout, stderr, walltime_ms)."""
    t0 = time.time()
    req = RunCodeRequest(code=code, language="python", run_timeout=timeout_s)
    resp = run_code(req)
    walltime_ms = int((time.time() - t0) * 1000)
    ok = resp.status == RunStatus.Success
    stdout = resp.run_result.stdout if resp.run_result else ""
    stderr = resp.run_result.stderr if resp.run_result else ""
    return ok, stdout or "", stderr or "", walltime_ms


def _parse_final_repr(stdout: str) -> str | None:
    """Extract the printed repr value after the final marker."""
    marker_idx = stdout.rfind(FINAL_REPR_MARKER)
    if marker_idx < 0:
        return None
    tail = stdout[marker_idx + len(FINAL_REPR_MARKER) :].strip()
    return tail or None


def _py_literal_eq(a: str, b: str) -> bool:
    """Compare two Python-literal strings by eval'ing both; falls back to str ==."""
    try:
        return eval(a) == eval(b)  # noqa: S307 — controlled inputs from sandbox
    except Exception:
        return a.strip() == b.strip()


class SandboxVerifier(Verifier):
    """AZR-style verifier. Owns a single sandbox container; dispatches by task type.

    Contracts (from docs/03-stack.md):
        DEDUCTION (P, I given; predict O):
            run P(I) in sandbox; compare computed O to solution.body.
        ABDUCTION (P, O given; predict I):
            run P(solution.body) in sandbox; compare to O.
        INDUCTION (I/O pairs given; predict P):
            run solution.body (the predicted P) on each I; compare to each O.
            Pass iff all pairs match.

    For induction, task.input and task.output are expected to be string reprs
    of Python lists of equal length.
    """

    def __init__(self, container: SandboxContainer | None = None, timeout_s: float = 5.0) -> None:
        self.container = container or SandboxContainer()
        if self.container.container is None:
            self.container.start()
        self.timeout_s = timeout_s

    def close(self) -> None:
        self.container.stop()

    def __enter__(self) -> "SandboxVerifier":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def verify(self, task: Task, solution: Solution) -> VerifyResult:
        if task.task_type is TaskType.DEDUCTION:
            return self._verify_deduction(task, solution)
        if task.task_type is TaskType.ABDUCTION:
            return self._verify_abduction(task, solution)
        return self._verify_induction(task, solution)

    def _verify_deduction(self, task: Task, solution: Solution) -> VerifyResult:
        assert task.program is not None and task.input is not None
        code = f"{task.program}\nprint({FINAL_REPR_MARKER!r}, repr(f({task.input})))"
        ok, stdout, stderr, wt = _run(code, self.timeout_s)
        passed = False
        if ok:
            computed = _parse_final_repr(stdout)
            if computed is not None:
                passed = _py_literal_eq(computed, solution.body)
        return VerifyResult(
            task_id=task.task_id,
            solution_id=solution.task_id,  # Solution is keyed by task_id
            passed=passed,
            stdout=stdout,
            stderr=stderr,
            walltime_ms=wt,
            exit_code=0 if ok else 1,
        )

    def _verify_abduction(self, task: Task, solution: Solution) -> VerifyResult:
        assert task.program is not None and task.output is not None
        code = (
            f"{task.program}\n"
            f"print({FINAL_REPR_MARKER!r}, repr(f({solution.body}) == {task.output}))"
        )
        ok, stdout, stderr, wt = _run(code, self.timeout_s)
        passed = False
        if ok:
            result = _parse_final_repr(stdout)
            passed = result == "True"
        return VerifyResult(
            task_id=task.task_id,
            solution_id=solution.task_id,
            passed=passed,
            stdout=stdout,
            stderr=stderr,
            walltime_ms=wt,
            exit_code=0 if ok else 1,
        )

    def _verify_induction(self, task: Task, solution: Solution) -> VerifyResult:
        assert task.input is not None and task.output is not None
        # Solver predicted the program; task carries I/O pairs as list literals.
        code = (
            f"{solution.body}\n"
            f"__inputs = {task.input}\n"
            f"__outputs = {task.output}\n"
            f"__acc = []\n"
            f"for __i, __o in zip(__inputs, __outputs):\n"
            f"    try: __acc.append(f(*__i) == __o if isinstance(__i, tuple) else f(__i) == __o)\n"
            f"    except Exception: __acc.append(False)\n"
            f"print({FINAL_REPR_MARKER!r}, repr(all(__acc) and len(__acc) > 0))"
        )
        ok, stdout, stderr, wt = _run(code, self.timeout_s)
        passed = False
        if ok:
            result = _parse_final_repr(stdout)
            passed = result == "True"
        return VerifyResult(
            task_id=task.task_id,
            solution_id=solution.task_id,
            passed=passed,
            stdout=stdout,
            stderr=stderr,
            walltime_ms=wt,
            exit_code=0 if ok else 1,
        )
