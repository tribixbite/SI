"""Phase 1 smoke: load Gemma 4 E4B, propose 6 small tasks, solve + verify each.

Success = we can parse most proposer outputs into tasks, the solver produces
plausible answers, and the verifier's pass/fail signal is coherent. Not a
measure of pass-rate at this stage — that takes training.
"""

from __future__ import annotations

import logging
import os
import time

from si.contracts import Task, TaskType
from si.llm import GemmaLLM
from si.proposer import AZRProposer
from si.solver import GemmaSolver
from si.verifier import SandboxVerifier


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    model_path = os.environ.get(
        "SI_MODEL_PATH", "/home/matilda/git/SI/cache/gemma-4-E4B-hf"
    )
    llm = GemmaLLM(model_path, cuda_visible_devices="1")

    proposer = AZRProposer(llm, branch_id="p0", temperature=0.8, max_tokens=800)
    solver = GemmaSolver(llm, branch_id="s0", temperature=0.5, max_tokens=400)

    print("\n=== Step 1: propose 3 deduction + 3 abduction tasks ===")
    t0 = time.time()
    ded_tasks = proposer.propose(TaskType.DEDUCTION, experience=None, n=3)  # type: ignore[arg-type]
    abd_tasks_pre = proposer.propose(TaskType.ABDUCTION, experience=None, n=3)  # type: ignore[arg-type]
    print(f"proposer walltime: {time.time()-t0:.1f}s")
    print(f"deduction parsed: {len(ded_tasks)}/3, abduction parsed: {len(abd_tasks_pre)}/3")

    # For abduction the proposer emits (program, input). We need the matching
    # target output by running the program. Do this via the verifier's sandbox.
    verifier = SandboxVerifier()
    abd_tasks: list[Task] = []
    for t in abd_tasks_pre:
        if t.program is None or t.input is None:
            continue
        probe = Task(
            task_type=TaskType.DEDUCTION,
            program=t.program,
            input=t.input,
            output=None,
            proposer_branch_id=t.proposer_branch_id,
            gen=t.gen,
            task_id=t.task_id,
        )
        # Hack: use the deduction verifier path with a dummy "matches" solution
        # to just execute P(I) and capture stdout; easier to do a one-off sandbox call.
        from si.contracts import Solution
        from si.verifier import _parse_final_repr, _run, FINAL_REPR_MARKER

        code = f"{t.program}\nprint({FINAL_REPR_MARKER!r}, repr(f({t.input})))"
        ok, stdout, stderr, _ = _run(code, timeout_s=5.0)
        target_out = _parse_final_repr(stdout) if ok else None
        if target_out is None:
            continue
        abd_tasks.append(
            Task(
                task_type=TaskType.ABDUCTION,
                program=t.program,
                input=None,
                output=target_out,
                proposer_branch_id=t.proposer_branch_id,
                gen=t.gen,
                task_id=t.task_id,
            )
        )

    print(f"abduction tasks with runnable programs: {len(abd_tasks)}/{len(abd_tasks_pre)}")

    all_tasks = ded_tasks + abd_tasks
    if not all_tasks:
        print("!! No tasks to solve. Aborting smoke.")
        verifier.close()
        return

    print(f"\n=== Step 2: solve {len(all_tasks)} tasks (single rollout each) ===")
    t0 = time.time()
    solutions = solver.solve_batch(all_tasks)
    print(f"solver walltime: {time.time()-t0:.1f}s")

    print("\n=== Step 3: verify ===")
    passed = 0
    for task, sol in zip(all_tasks, solutions, strict=True):
        result = verifier.verify(task, sol)
        status = "PASS" if result.passed else "FAIL"
        body_preview = (sol.body[:60] + "...") if len(sol.body) > 60 else sol.body
        print(f"  [{status}] {task.task_type.value:>10} id={task.task_id[:8]} sol={body_preview!r}")
        if not result.passed:
            # Show proposer-supplied context for failed ones
            inp_or_out = task.input if task.task_type is TaskType.ABDUCTION else task.output
            print(f"         program[:80]: {task.program[:80] if task.program else '?'!r}")
            print(f"         task {'output' if task.task_type is TaskType.ABDUCTION else 'input'}: {task.input if task.task_type is TaskType.DEDUCTION else task.output!r}")
        passed += int(result.passed)

    print(f"\n=== Summary ===")
    print(f"proposals parseable: {len(all_tasks)}/6")
    print(f"solver pass@1:       {passed}/{len(all_tasks)}")

    verifier.close()


if __name__ == "__main__":
    main()
