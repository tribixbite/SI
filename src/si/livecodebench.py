"""LiveCodeBench v6 anchor runner (docs/05-evaluation.md §"Secondary anchor").

Loads the livecodebench/code_generation_lite dataset at release_v6 (= v1..v6
cumulative, 1055 problems). Two testtypes:
    - functional (444 problems) — starter_code defines a function; tests call it.
    - stdin      (610 problems) — code reads from stdin, writes to stdout.

The sandbox wrapper handles both: functional is the same pattern as HumanEval+
(paste completion, run tests), stdin uses sandbox-fusion's `stdin` field.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from sandbox_fusion import RunCodeRequest, RunStatus, run_code

from si.llm import GemmaLLM, GenParams

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/home/matilda/git/SI/cache/livecodebench"

# release_v<N> = cumulative through testN.jsonl.
VERSION_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)

_SYSTEM_LCB = (
    "You solve programming contest problems. Read the problem carefully. "
    "Output a complete Python 3 solution inside a single ```python fenced "
    "block. The code must run standalone: use input() for stdin and print() "
    "for stdout. No prose outside the code block."
)


@dataclass
class LCBProblem:
    problem_id: str
    title: str
    content: str
    platform: str
    difficulty: str
    starter_code: str
    public_tests: list[dict]   # [{'input', 'output', 'testtype'}]
    private_tests: list[dict]  # same shape; decoded from pickle
    testtype: str              # 'functional' | 'stdin'


@dataclass
class LCBResult:
    passed: int
    total: int
    per_problem: dict[str, bool]
    wall_s: float
    per_difficulty: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def pass_at_1(self) -> float:
        return self.passed / max(1, self.total)


def _decode_private(priv: str | None) -> list[dict]:
    """LCB's private_tests encoding is base64( zlib( pickle(str(JSON)) ) )."""
    if not priv:
        return []
    try:
        import pickle
        raw = zlib.decompress(base64.b64decode(priv))
        # The bytes may be pickled (holds a JSON string) or raw utf-8 JSON.
        try:
            inner = pickle.loads(raw)
        except Exception:
            inner = raw.decode("utf-8", errors="replace")
        if isinstance(inner, bytes):
            inner = inner.decode("utf-8", errors="replace")
        return json.loads(inner)
    except Exception as e:
        log.debug("private_tests decode failed: %s", e)
        return []


def load_lcb(
    version: str = "release_v6", data_dir: str = DEFAULT_DATA_DIR
) -> list[LCBProblem]:
    """Load LCB problems for a given version. Problems with public_tests missing
    (malformed records) are skipped."""
    if version not in VERSION_FILES:
        raise ValueError(f"Unknown LCB version {version!r}; choices: {list(VERSION_FILES)}")
    problems: list[LCBProblem] = []
    for fname in VERSION_FILES[version]:
        path = Path(data_dir) / fname
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ptc_raw = rec.get("public_test_cases")
                if not ptc_raw:
                    continue
                public_tests = json.loads(ptc_raw) if isinstance(ptc_raw, str) else ptc_raw
                if not public_tests:
                    continue
                testtype = public_tests[0].get("testtype", "stdin")
                problems.append(
                    LCBProblem(
                        problem_id=rec["question_id"],
                        title=rec.get("question_title", ""),
                        content=rec["question_content"],
                        platform=rec.get("platform", ""),
                        difficulty=rec.get("difficulty", "unknown"),
                        starter_code=rec.get("starter_code") or "",
                        public_tests=public_tests,
                        private_tests=_decode_private(rec.get("private_test_cases")),
                        testtype=testtype,
                    )
                )
    log.info("loaded %d LCB problems (%s)", len(problems), version)
    return problems


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    if m is None:
        return text.strip()
    return m.group(1).strip()


def _build_user_prompt(prob: LCBProblem) -> str:
    if prob.starter_code:
        return (
            f"{prob.content}\n\n"
            f"Complete the following solution (keep the signature; only fill in the body):\n"
            f"```python\n{prob.starter_code}\n```\n"
        )
    return prob.content


def _run_one_test(prob_id: str, code: str, test: dict, timeout_s: float) -> bool:
    stdin = test.get("input", "")
    expected = (test.get("output") or "").strip()
    req = RunCodeRequest(code=code, language="python", stdin=stdin, run_timeout=timeout_s)
    try:
        resp = run_code(req)
    except Exception as e:
        log.debug("%s sandbox error: %s", prob_id, e)
        return False
    if resp.status != RunStatus.Success:
        return False
    stdout = ""
    if resp.run_result and resp.run_result.stdout is not None:
        stdout = resp.run_result.stdout.strip()
    return stdout == expected


def _check_problem(prob: LCBProblem, code: str, timeout_s: float = 10.0) -> bool:
    """Run completion against every public + private test, sequentially with
    early-exit on first failure. Most failed candidates trip on the public test
    in 1 sandbox call; parallelizing tests within a candidate dispatches all of
    them and removes the short-circuit, which is a net loss when most candidates
    fail (measured: 2.2× slower on base LCB at parallel_tests=8)."""
    tests = list(prob.public_tests) + list(prob.private_tests)
    if not tests:
        return False
    for t in tests:
        if not _run_one_test(prob.problem_id, code, t, timeout_s):
            return False
    return True


def _solve_problem(
    prob: LCBProblem, candidates: list[str], timeout_s: float
) -> tuple[str, bool]:
    """Run candidates in order with sequential tests; first to pass wins."""
    for cand in candidates:
        code = _extract_code(cand)
        if _check_problem(prob, code, timeout_s=timeout_s):
            return prob.problem_id, True
    return prob.problem_id, False


def lcb_pass_at_1(
    llm: GemmaLLM,
    *,
    version: str = "release_v6",
    data_dir: str = DEFAULT_DATA_DIR,
    max_problems: int | None = None,
    testtype_filter: str | None = None,  # 'functional' | 'stdin' | None
    temperature: float = 0.2,
    max_completion_tokens: int = 1024,
    timeout_s: float = 10.0,
    n_candidates: int = 1,
    parallel_problems: int = 8,
) -> LCBResult:
    """Generate `n_candidates` completions per LCB problem; pass = ANY candidate passes.

    n_candidates>1 implements Best-of-N test-time compute scaling. With our
    sandbox verifier, "best" = "first that passes all tests" — equivalent to
    BoN with a hard verifier. Inference-only; no training cost.

    parallel_problems > 1 fans out across problems (each problem still runs
    its candidates sequentially with test-level early-exit). This preserves
    the short-circuit on the common-case fail path while parallelizing the
    independent across-problem verification work.
    """
    problems = load_lcb(version, data_dir)
    if testtype_filter:
        problems = [p for p in problems if p.testtype == testtype_filter]
    if max_problems is not None:
        problems = problems[:max_problems]

    t0 = time.time()
    log.info("LCB %s: generating %d × %d completions...", version, len(problems), n_candidates)
    user_prompts = [_build_user_prompt(p) for p in problems]
    # For n>1 we want diverse samples; raise temp from 0.2 to 0.8 unless caller overrode.
    sampling_temp = temperature if n_candidates == 1 else max(temperature, 0.8)
    params = GenParams(
        temperature=sampling_temp, top_p=0.95, max_tokens=max_completion_tokens, n=n_candidates
    )
    nested = llm.chat_batch(user_prompts, params, system=_SYSTEM_LCB)
    log.info("LCB: generation done in %.1fs", time.time() - t0)

    per_problem: dict[str, bool] = {}
    per_diff: dict[str, list[int]] = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0], "unknown": [0, 0]}
    diff_by_pid = {p.problem_id: p.difficulty for p in problems}
    if parallel_problems <= 1:
        results = [_solve_problem(p, c, timeout_s) for p, c in zip(problems, nested, strict=True)]
    else:
        with ThreadPoolExecutor(max_workers=parallel_problems) as ex:
            futures = [
                ex.submit(_solve_problem, p, c, timeout_s)
                for p, c in zip(problems, nested, strict=True)
            ]
            results = [f.result() for f in futures]
    passed = 0
    for pid, ok in results:
        per_problem[pid] = ok
        d = diff_by_pid.get(pid, "unknown")
        if d not in per_diff:
            d = "unknown"
        per_diff[d][1] += 1
        if ok:
            passed += 1
            per_diff[d][0] += 1

    wall_s = time.time() - t0
    log.info(
        "LCB: %d/%d passed (%.1f%%) in %.1fs (BoN=%d, temp=%.2f)",
        passed, len(problems), 100.0 * passed / max(1, len(problems)), wall_s, n_candidates, sampling_temp,
    )
    return LCBResult(
        passed=passed,
        total=len(problems),
        per_problem=per_problem,
        wall_s=wall_s,
        per_difficulty={k: (v[0], v[1]) for k, v in per_diff.items() if v[1] > 0},
    )
