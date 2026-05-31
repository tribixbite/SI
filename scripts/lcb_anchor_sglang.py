"""LCB v6 anchor using SGLang backend (parallel to si.cli anchor).

Why a separate script: SGLang lives in a parallel venv (.venv-sglang) without
Unsloth/TRL/vLLM, and `si.cli` imports those. Running this directly avoids
pulling in unwanted heavy deps; the only SI imports are GenParams and the
LCB-side helpers, which are vLLM-free.

Usage from the SGLang venv:
    .venv-sglang/bin/python scripts/lcb_anchor_sglang.py \\
        --model /home/matilda/git/SI/cache/qwen3.6-27b-awq-int4 \\
        --quantization awq_marlin \\
        --bon 8 \\
        --parallel-problems 4 \\
        --max-completion-tokens 4096 \\
        --problem-offset 0 \\
        --problem-limit 76 \\
        --out /home/matilda/git/SI/runs/qwen36_27b_lcb_v6_bon1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from si.llm import GenParams  # noqa: E402  — must come after sys.path tweak
from si.livecodebench import lcb_pass_at_1  # noqa: E402
from si.llm_sglang import SGLangLLM  # noqa: E402
from si.verifier import SandboxContainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", default=None, help="awq | awq_marlin | gptq | fp8 | None")
    p.add_argument("--bon", type=int, default=1)
    p.add_argument("--parallel-problems", type=int, default=8)
    p.add_argument("--max-completion-tokens", type=int, default=1024)
    p.add_argument("--problem-offset", type=int, default=0)
    p.add_argument("--problem-limit", type=int, default=None)
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--cuda-device", default="1")
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Suppress Qwen3-family reasoning trace (apples-to-apples with non-reasoning baselines)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    container = SandboxContainer()
    container.start()
    try:
        llm = SGLangLLM(
            args.model,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            cuda_visible_devices=args.cuda_device,
            enable_thinking=False if args.no_thinking else None,
        )
        try:
            result = lcb_pass_at_1(
                llm,
                version="release_v6",
                max_problems=args.max_problems,
                temperature=0.2,
                n_candidates=args.bon,
                parallel_problems=args.parallel_problems,
                max_completion_tokens=args.max_completion_tokens,
                problem_offset=args.problem_offset,
                problem_limit=args.problem_limit,
            )
        finally:
            llm.close()

        out_d = {
            "adapter": None,
            "benchmark": "lcb",
            "backend": "sglang",
            "model": args.model,
            "passed": result.passed,
            "total": result.total,
            "pass_at_1": result.pass_at_1,
            "wall_s": result.wall_s,
            "per_problem": result.per_problem,
            "per_difficulty": result.per_difficulty,
            "bon": args.bon,
            "thinking": not args.no_thinking,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out_d, open(args.out, "w"), indent=2)
        print(
            f"LCB v6 (BoN={args.bon}, sglang): "
            f"{result.passed}/{result.total} = {result.pass_at_1:.2%}  wall={result.wall_s:.0f}s"
        )
    finally:
        container.stop()


if __name__ == "__main__":
    main()
