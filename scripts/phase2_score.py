"""Phase 2 SCORE step (no GPU).

Reads every branch's results.jsonl (from phase2_solve.py), samples Elo matches
with the shared winner rule (si.match.sample_matches), updates the persisted
EloState, and writes a replacement plan JSON the orchestrator acts on:
top quartile = keep, middle = train (mutate), bottom quartile = reseed.

    .venv/bin/python scripts/phase2_score.py \\
        --tasks tasks.jsonl --gen 3 \\
        --results b0=b0_results.jsonl --results b1=b1_results.jsonl \\
        --elo-state elo.json --out-plan plan_gen3.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from si.elo import EloSelector, ranked  # noqa: E402
from si.match import sample_matches  # noqa: E402
from si.taskio import read_elo_state, read_tasks, write_elo_state  # noqa: E402


def _load_results(path: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["task_id"]] = bool(d["passed"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--results", action="append", required=True,
                   help="branch_id=results.jsonl (repeatable)")
    p.add_argument("--elo-state", required=True, help="persisted EloState JSON (created if absent)")
    p.add_argument("--out-plan", required=True)
    p.add_argument("--gen", type=int, default=0, help="seeds the match RNG + plan metadata")
    p.add_argument("--matches-mult", type=float, default=3.0)
    p.add_argument("--quartile", type=float, default=0.25)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    task_ids = [t.task_id for t in read_tasks(args.tasks)]
    passed: dict[str, dict[str, bool]] = {}
    for spec in args.results:
        bid, _, path = spec.partition("=")
        passed[bid] = _load_results(path)
    branch_ids = list(passed)
    if len(branch_ids) < 2:
        raise SystemExit("need >= 2 branches to score matches")

    state = read_elo_state(args.elo_state)
    rng = random.Random(args.gen)
    n_matches = max(1, round(args.matches_mult * len(branch_ids)))
    matches = sample_matches(branch_ids, task_ids, passed, n_matches, rng)

    selector = EloSelector(replacement_quartile_size=args.quartile)
    state = selector.update(matches, state)
    plan = selector.replacement_plan(state)
    write_elo_state(args.elo_state, state)

    out = {
        "gen": args.gen,
        "n_matches": len(matches),
        "ranking": [{"branch_id": b, "elo": round(r, 1)} for b, r in ranked(state)],
        "keep": plan.keep,
        "mutate": plan.mutate,
        "replace": [{"dead": d, "parent": par} for d, par in plan.replace],
        "per_branch_pass": {b: sum(v.values()) for b, v in passed.items()},
    }
    Path(args.out_plan).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_plan).write_text(json.dumps(out, indent=2))
    print(f"phase2_score gen{args.gen}: {len(matches)} matches; "
          f"keep={plan.keep} mutate={plan.mutate} replace={[d for d, _ in plan.replace]}")


if __name__ == "__main__":
    main()
