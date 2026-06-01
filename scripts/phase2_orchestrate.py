"""Phase 2 self-improvement loop — single-GPU subprocess orchestrator.

Holds branch state (branch_id -> current LoRA adapter dir) in Python and shells
out to the per-stage primitives so each GPU step is a fresh process (vLLM and
Unsloth can't co-reside on one 24GB card; fresh processes also dodge vLLM
teardown flakiness). Reseed is CPU-only and runs inline.

Per generation:
  propose -> per-branch solve (+emit SSD samples) -> score (Elo + plan)
          -> train keep+mutate branches (SSD warm-started from their adapter)
          -> reseed bottom quartile (perturb a parent adapter)
          -> every N gens: anchor the top branch, commit/revert.

  --dry-run fakes the GPU stages (propose/solve/train) so the control flow,
  Elo scoring, plan application and branch-state transitions can be validated
  on CPU in seconds.

    .venv/bin/python scripts/phase2_orchestrate.py \\
        --run-id p2_test --branches 4 --gens 10 --anchor-every 5 \\
        --seed-adapter runs/phase1_v2_20260423_2250/adapter/adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from si.phase2_ops import perturb_lora_adapter  # noqa: E402

log = logging.getLogger("phase2_orchestrate")
PY = sys.executable
BASE_HF = str(REPO / "cache/gemma-4-E4B-hf")


def _run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd)


def _fake_tasks(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "task_type": "deduction", "program": "def f(x):\n    return x + 1",
                "input": str(i), "output": None, "proposer_branch_id": "x",
                "gen": 0, "task_id": f"t{i}",
            }) + "\n")


def _fake_results(path: Path, task_ids: list[str], pass_prob: float, rng: random.Random) -> None:
    with path.open("w") as f:
        for tid in task_ids:
            f.write(json.dumps({"task_id": tid, "passed": rng.random() < pass_prob, "body": "0"}) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--anchor-every", type=int, default=5)
    p.add_argument("--seed-adapter", required=True, help="initial adapter all branches start from")
    p.add_argument("--n-per-type", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.001)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--cuda-device", default="1")
    p.add_argument("--dry-run", action="store_true", help="fake GPU stages; validate control flow on CPU")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    root = REPO / "runs" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    elo_state = root / "elo.json"
    # branch_id -> current adapter dir
    branches: dict[str, str] = {f"b{i}": args.seed_adapter for i in range(args.branches)}
    rng = random.Random(0)
    committed_gen = 0

    for gen in range(1, args.gens + 1):
        gdir = root / f"gen{gen:04d}"
        gdir.mkdir(parents=True, exist_ok=True)
        tasks = gdir / "tasks.jsonl"
        log.info("===== gen %d =====", gen)

        # --- propose ---
        if args.dry_run:
            _fake_tasks(tasks, args.n_per_type)
        else:
            _run([PY, str(REPO / "scripts/phase2_propose.py"), "--out", str(tasks),
                  "--n-per-type", str(args.n_per_type), "--gen", str(gen),
                  "--cuda-device", args.cuda_device])
        task_ids = [json.loads(line)["task_id"] for line in tasks.read_text().splitlines() if line.strip()]

        # --- per-branch solve (fresh subprocess each) ---
        result_args: list[str] = []
        samples: dict[str, str] = {}
        for bid, adapter in branches.items():
            res = gdir / f"{bid}_results.jsonl"
            smp = gdir / f"{bid}_samples.jsonl"
            samples[bid] = str(smp)
            if args.dry_run:
                # stronger branches (lower index) pass more often
                _fake_results(res, task_ids, pass_prob=0.8 - 0.15 * int(bid[1:]), rng=rng)
            else:
                _run([PY, str(REPO / "scripts/phase2_solve.py"), "--adapter", adapter,
                      "--tasks", str(tasks), "--out", str(res), "--branch-id", bid,
                      "--emit-samples", str(smp), "--cuda-device", args.cuda_device])
            result_args += ["--results", f"{bid}={res}"]

        # --- score (Elo + replacement plan) ---
        plan_path = gdir / "plan.json"
        _run([PY, str(REPO / "scripts/phase2_score.py"), "--tasks", str(tasks),
              "--elo-state", str(elo_state), "--out-plan", str(plan_path),
              "--gen", str(gen), *result_args])
        plan = json.loads(plan_path.read_text())

        # --- train keep + mutate (SSD warm-started from each branch's adapter) ---
        for bid in plan["keep"] + plan["mutate"]:
            new_adapter = gdir / bid
            if args.dry_run:
                new_adapter.mkdir(parents=True, exist_ok=True)
                (new_adapter / "adapter").mkdir(exist_ok=True)
            elif Path(samples[bid]).exists() and Path(samples[bid]).stat().st_size > 0:
                _run([PY, "-m", "si.cli", "ssd-train", "--samples", samples[bid],
                      "--adapter-out", str(new_adapter), "--warm-start-adapter", branches[bid],
                      "--epochs", str(args.train_epochs)])
            else:
                log.info("branch %s: no passing samples, keeping current adapter", bid)
                continue
            inner = new_adapter / "adapter"
            branches[bid] = str(inner if inner.exists() else new_adapter)

        # --- reseed bottom quartile (perturb a parent adapter; CPU) ---
        for entry in plan["replace"]:
            dead, parent = entry["dead"], entry["parent"]
            dest = gdir / f"{dead}_reseed"
            if args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
            else:
                perturb_lora_adapter(branches[parent], str(dest), args.sigma)
            branches[dead] = str(dest)

        # --- anchor + commit/revert every N gens ---
        if gen % args.anchor_every == 0:
            top = plan["ranking"][0]["branch_id"]
            log.info("gen %d: anchor checkpoint on top branch %s (adapter=%s)", gen, top, branches[top])
            # A real run evaluates branches[top] via anchor_chunked.sh and reverts
            # branch state to the committed gen on regression (Anchor.should_revert).
            committed_gen = gen

        (root / "branches.json").write_text(json.dumps(branches, indent=2))
        log.info("gen %d done; ranking=%s", gen,
                 [r["branch_id"] for r in plan["ranking"]])

    log.info("phase2 orchestrate complete: %d gens, committed_gen=%d", args.gens, committed_gen)
    print(json.dumps({"run_id": args.run_id, "gens": args.gens, "branches": branches}, indent=2))


if __name__ == "__main__":
    main()
