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


def _check_problem(prob: LCBProblem, code: str, timeout_s: float = 10.0) -> bool:
    """Run the completion against every public + private test. Pass = all match."""
    tests = list(prob.public_tests) + list(prob.private_tests)
    if not tests:
        return False
    for t in tests:
        stdin = t.get("input", "")
        expected = (t.get("output") or "").strip()
        req = RunCodeRequest(
            code=code,
            language="python",
            stdin=stdin,
            run_timeout=timeout_s,
        )
        try:
            resp = run_code(req)
        except Exception as e:
            log.debug("%s sandbox error: %s", prob.problem_id, e)
            return False
        if resp.status != RunStatus.Success:
            return False
        stdout = ""
        if resp.run_result and resp.run_result.stdout is not None:
            stdout = resp.run_result.stdout.strip()
        if stdout != expected:
            return False
    return True


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
) -> LCBResult:
    """Generate one completion per LCB problem, execute in sandbox, return pass@1."""
    problems = load_lcb(version, data_dir)
    if testtype_filter:
        problems = [p for p in problems if p.testtype == testtype_filter]
    if max_problems is not None:
        problems = problems[:max_problems]

    t0 = time.time()
    log.info("LCB %s: generating %d completions...", version, len(problems))
    user_prompts = [_build_user_prompt(p) for p in problems]
    params = GenParams(temperature=temperature, top_p=0.95, max_tokens=max_completion_tokens)
    nested = llm.chat_batch(user_prompts, params, system=_SYSTEM_LCB)
    log.info("LCB: generation done in %.1fs", time.time() - t0)

    per_problem: dict[str, bool] = {}
    per_diff: dict[str, list[int]] = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0], "unknown": [0, 0]}
    passed = 0
    for prob, comp_list in zip(problems, nested, strict=True):
        code = _extract_code(comp_list[0])
        ok = _check_problem(prob, code, timeout_s=timeout_s)
        per_problem[prob.problem_id] = ok
        d = prob.difficulty if prob.difficulty in per_diff else "unknown"
        per_diff[d][1] += 1
        if ok:
            passed += 1
            per_diff[d][0] += 1

    wall_s = time.time() - t0
    log.info("LCB: %d/%d passed (%.1f%%) in %.1fs", passed, len(problems), 100.0 * passed / max(1, len(problems)), wall_s)
    return LCBResult(
        passed=passed,
        total=len(problems),
        per_problem=per_problem,
        wall_s=wall_s,
        per_difficulty={k: (v[0], v[1]) for k, v in per_diff.items() if v[1] > 0},
    )
