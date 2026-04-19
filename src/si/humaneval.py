"""HumanEval+ anchor runner — computes pass@1 against a model checkpoint.

Uses the pinned evalplus dataset (commit
d362e933265c3e7e3df8101c930a89c3c470cd9f per docs/01-sources.md) and our
SandboxVerifier for sandboxed execution. We do NOT shell out to evalplus's
CLI; the dataset is the authoritative anchor and the pass/fail logic is
tiny enough to keep inline.

Pass@1 definition (matches evalplus):
    For each problem, generate 1 completion, paste it after the problem's
    prompt, append the problem's test harness + check() call, run in the
    sandbox. Pass iff no uncaught exception / non-zero exit.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash

from si.llm import GemmaLLM, GenParams
from si.verifier import _run

log = logging.getLogger(__name__)

EXPECTED_HUMANEVAL_PLUS_HASH = "fe585eb4df8c88d844eeb463ea4d0302"

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)

_SYSTEM_HE = (
    "You complete a Python function. Output ONLY the function body (no prose, "
    "no additional tests). Wrap your code in a single fenced ```python block."
)


@dataclass
class HumanEvalResult:
    passed: int
    total: int
    per_problem: dict[str, bool]
    wall_s: float

    @property
    def pass_at_1(self) -> float:
        return self.passed / max(1, self.total)


def _extract_completion(text: str) -> str:
    """Pull the function body out of a fenced block, or return raw if no fence."""
    m = _CODE_FENCE.search(text)
    if m is not None:
        return m.group(1)
    return text


def _build_test_code(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Combine problem prompt + model completion + problem tests into one snippet.

    Two candidate layouts because models vary on what they emit after the fence:
        A) completion already contains the full function: use as-is.
        B) completion contains only the body after the signature: paste after prompt.

    Heuristic: if completion contains `def {entry_point}`, use variant A. Else B.
    """
    completion_rstripped = completion.rstrip()
    if f"def {entry_point}" in completion_rstripped:
        # Full function redefined: use the completion verbatim, discarding the prompt.
        code = completion_rstripped.lstrip()
    else:
        # Body-only completion: preserve leading indentation so it sits under the signature.
        code = prompt + completion_rstripped
    return f"{code}\n\n{test}\n\ncheck({entry_point})"


def humaneval_plus_pass_at_1(
    llm: GemmaLLM,
    *,
    max_problems: int | None = None,
    timeout_s: float = 10.0,
    temperature: float = 0.2,
    max_completion_tokens: int = 800,
    verify_dataset_hash: bool = True,
) -> HumanEvalResult:
    """Generate one completion per HumanEval+ problem, execute in sandbox, return pass@1."""
    problems = get_human_eval_plus()
    if verify_dataset_hash:
        actual = get_human_eval_plus_hash()
        if actual != EXPECTED_HUMANEVAL_PLUS_HASH:
            raise RuntimeError(
                f"HumanEval+ dataset hash drift: expected {EXPECTED_HUMANEVAL_PLUS_HASH!r}, "
                f"got {actual!r}. Anchor drift invalidates the revert rule."
            )
    items = list(problems.items())
    if max_problems is not None:
        items = items[:max_problems]

    # Batch all completions through one vLLM call for throughput.
    user_prompts = [
        f"Complete this Python function (do not redefine it):\n\n```python\n{prob['prompt']}```"
        for _, prob in items
    ]
    gen_params = GenParams(
        temperature=temperature, top_p=0.95, max_tokens=max_completion_tokens
    )
    t0 = time.time()
    log.info("HumanEval+: generating %d completions...", len(items))
    completions_nested = llm.chat_batch(user_prompts, gen_params, system=_SYSTEM_HE)
    log.info("HumanEval+: generation done in %.1fs", time.time() - t0)

    per_problem: dict[str, bool] = {}
    passed = 0
    for (tid, prob), comp_list in zip(items, completions_nested, strict=True):
        raw = comp_list[0]
        completion = _extract_completion(raw)
        code = _build_test_code(prob["prompt"], completion, prob["test"], prob["entry_point"])
        ok, _stdout, stderr, _ = _run(code, timeout_s)
        passed_problem = ok and not stderr.strip()
        per_problem[tid] = passed_problem
        if passed_problem:
            passed += 1

    wall_s = time.time() - t0
    log.info("HumanEval+: %d/%d passed (%.1f%%) in %.1fs", passed, len(items), 100.0 * passed / max(1, len(items)), wall_s)
    return HumanEvalResult(passed=passed, total=len(items), per_problem=per_problem, wall_s=wall_s)
