"""Phase 2 per-branch SOLVE subprocess.

One branch solves a shared task pool and the results are verified. Run as a
fresh process per branch (the Phase 1 isolation pattern): vLLM can't serve
Gemma4 LoRA, so the branch's adapter is PEFT-merged into the base and the
merged dir is loaded; a fresh process also sidesteps vLLM teardown flakiness
between branch swaps.

Writes results.jsonl: one {task_id, passed, body} per task — the orchestrator
reads all branches' results to score Elo matches.

    .venv/bin/python scripts/phase2_solve.py \\
        --adapter runs/<branch>/adapter --tasks tasks.jsonl --out b0_results.jsonl \\
        --branch-id b0
Use --adapter base to solve with the unmodified base model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from si.llm import GemmaLLM  # noqa: E402
from si.phase2_ops import ensure_merged_model  # noqa: E402
from si.solver import GemmaSolver  # noqa: E402
from si.taskio import read_tasks  # noqa: E402
from si.verifier import SandboxContainer, SandboxVerifier  # noqa: E402

DEFAULT_MODEL = str(REPO / "cache/gemma-4-E4B-hf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="branch adapter dir, or 'base' for no adapter")
    p.add_argument("--tasks", required=True, help="resolved tasks JSONL (from phase2_propose.py)")
    p.add_argument("--out", required=True, help="results JSONL output")
    p.add_argument("--branch-id", default="b0")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base", default=DEFAULT_MODEL, help="base model for merging the adapter")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--cuda-device", default="1")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    load_path = args.model if args.adapter.lower() == "base" else ensure_merged_model(args.adapter, args.base)

    container = SandboxContainer()
    container.start()
    try:
        llm = GemmaLLM(load_path, cuda_visible_devices=args.cuda_device)
        solver = GemmaSolver(
            llm, branch_id=args.branch_id, temperature=args.temperature, max_tokens=args.max_tokens
        )
        verifier = SandboxVerifier(container=container)
        tasks = read_tasks(args.tasks)
        solutions = solver.solve_batch(tasks)
        passed = 0
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w") as f:
            for task, sol in zip(tasks, solutions, strict=True):
                vr = verifier.verify(task, sol)
                passed += int(vr.passed)
                f.write(json.dumps({"task_id": task.task_id, "passed": vr.passed, "body": sol.body}) + "\n")
    finally:
        container.stop()

    print(f"phase2_solve[{args.branch_id}]: {passed}/{len(tasks)} passed -> {args.out}")


if __name__ == "__main__":
    main()
