"""Held-out anchor evaluation + commit/revert rule.

The decision logic is exact and testable. The benchmark runner is swappable
(HumanEval+ / LiveCodeBench / MBPP+) — see concrete_runners below.

Per docs/05-evaluation.md §'Detection rules':
- reversion fires on anchor regression beyond tolerance
- tolerance tightens after gen 50 (strict mode)
- multi-anchor reporting is the defense against anchor-memorization
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from si.contracts import Anchor, AnchorResult, Branch

if TYPE_CHECKING:
    from si.llm import GemmaLLM

log = logging.getLogger(__name__)


BenchmarkRunner = Callable[[Branch], float]
"""(branch) -> pass@1 score in [0, 1]."""


class HeldOutAnchor(Anchor):
    def __init__(
        self,
        runner: BenchmarkRunner,
        benchmark_name: str,
        tolerance: float = 0.02,
        strict_tolerance: float = 0.01,
        strict_after_gen: int = 50,
    ) -> None:
        self.runner = runner
        self.benchmark_name = benchmark_name
        self.tolerance = tolerance
        self.strict_tolerance = strict_tolerance
        self.strict_after_gen = strict_after_gen

    def evaluate(self, population: list[Branch]) -> AnchorResult:
        per_branch = {b.branch_id: float(self.runner(b)) for b in population}
        aggregate = sum(per_branch.values()) / max(1, len(per_branch))
        # gen is not on AnchorResult's positional args — callers set it after,
        # but we include it in evaluate() via a sentinel.
        return AnchorResult(gen=-1, aggregate=aggregate, per_branch=per_branch)

    def should_revert(self, prev: AnchorResult | None, curr: AnchorResult) -> bool:
        if prev is None:
            return False
        tol = self.strict_tolerance if curr.gen >= self.strict_after_gen else self.tolerance
        drop = prev.aggregate - curr.aggregate
        if drop > (prev.aggregate * tol):
            log.warning(
                "Anchor regression: prev=%.4f curr=%.4f drop=%.4f tolerance=%.4f",
                prev.aggregate,
                curr.aggregate,
                drop,
                tol,
            )
            return True
        return False


def hash_anchor_set(problem_paths: list[Path]) -> str:
    """Content-hash the anchor set so drift is detectable between runs."""
    h = hashlib.sha256()
    for p in sorted(problem_paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def anchor_hash_check(expected_hash: str, problem_paths: list[Path]) -> None:
    """Raise if the anchor set on disk has drifted since run start.

    Call this at run startup and before every anchor evaluation.
    """
    actual = hash_anchor_set(problem_paths)
    if actual != expected_hash:
        raise RuntimeError(
            f"ANCHOR DRIFT DETECTED. expected={expected_hash} actual={actual}. "
            "The benchmark set on disk has changed since run start. "
            "This invalidates all commit/revert decisions for this run."
        )


def write_anchor_log(path: Path, result: AnchorResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"gen": result.gen, "aggregate": result.aggregate, "per_branch": result.per_branch}))
        f.write("\n")


# --- Concrete benchmark runners (stubs — implement in Phase 1 per docs/04) ----


def humaneval_plus_runner(branch: Branch) -> float:
    """Run HumanEval+ against the branch's model checkpoint.

    This is a factory-style stub: the real runner closes over a loaded GemmaLLM
    that knows about the branch's LoRA adapter. Use :func:`make_humaneval_runner`
    below at the Phase 1 wire-up site.
    """
    raise NotImplementedError(
        "Phase 1. Use make_humaneval_runner(llm) from si.anchor instead — this "
        "stub exists to satisfy docs/01 and is replaced at runtime."
    )


def make_humaneval_runner(
    llm: "GemmaLLM",
    *,
    max_problems: int | None = None,
    timeout_s: float = 10.0,
) -> "BenchmarkRunner":
    """Build a branch-runner closed over a loaded GemmaLLM.

    Per-branch LoRA is the caller's concern — they load the adapter onto `llm`
    before running this. For Phase 1 single-branch, the base model is the only
    one loaded and the runner is invoked directly.
    """
    from si.humaneval import humaneval_plus_pass_at_1  # local import keeps anchor importable w/o deps

    def _runner(_branch: Branch) -> float:
        result = humaneval_plus_pass_at_1(
            llm, max_problems=max_problems, timeout_s=timeout_s
        )
        return result.pass_at_1

    return _runner


def livecodebench_runner(branch: Branch) -> float:
    """Run LiveCodeBench v6 frozen subset."""
    raise NotImplementedError("Phase 2. See docs/01-sources.md LiveCodeBench section.")


def mbpp_plus_runner(branch: Branch) -> float:
    """Meta-anchor: MBPP+. Never evaluated during training; report only."""
    raise NotImplementedError("Phase 1 report-only. Same mechanism as humaneval_plus_runner.")
