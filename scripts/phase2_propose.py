"""Phase 2 PROPOSE subprocess.

The shared proposer (base model) generates deduction + abduction proposals,
resolves them into solver-ready tasks (abduction outputs filled by running the
program in the sandbox), and writes them to a tasks JSONL that every branch's
solve subprocess then consumes.

    .venv/bin/python scripts/phase2_propose.py --out tasks.jsonl --n-per-type 8 --gen 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from si.contracts import Experience, TaskType  # noqa: E402
from si.llm import GemmaLLM  # noqa: E402
from si.match import resolve_proposed_task  # noqa: E402
from si.proposer import AZRProposer  # noqa: E402
from si.taskio import write_tasks  # noqa: E402
from si.verifier import SandboxContainer  # noqa: E402

DEFAULT_MODEL = str(REPO / "cache/gemma-4-E4B-hf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="resolved tasks JSONL output")
    p.add_argument("--n-per-type", type=int, default=8)
    p.add_argument("--gen", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--cuda-device", default="1")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Abduction resolution runs the proposed program in the sandbox.
    container = SandboxContainer()
    container.start()
    try:
        llm = GemmaLLM(args.model, cuda_visible_devices=args.cuda_device)
        proposer = AZRProposer(llm, branch_id="prop", gen=args.gen, temperature=args.temperature)
        resolved = []
        for tt in (TaskType.DEDUCTION, TaskType.ABDUCTION):
            for raw in proposer.propose(tt, Experience(), args.n_per_type):
                task = resolve_proposed_task(raw)
                if task is not None:
                    resolved.append(task)
    finally:
        container.stop()

    write_tasks(args.out, resolved)
    print(f"phase2_propose: {len(resolved)} resolved tasks -> {args.out}")


if __name__ == "__main__":
    main()
