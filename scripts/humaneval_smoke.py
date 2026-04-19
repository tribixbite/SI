"""Quick HumanEval+ sanity: 20 problems, base Gemma 4 E4B, report pass@1."""

from __future__ import annotations

import logging
import os

from si.humaneval import humaneval_plus_pass_at_1
from si.llm import GemmaLLM
from si.verifier import SandboxContainer
from sandbox_fusion import set_endpoint


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    model_path = os.environ.get(
        "SI_MODEL_PATH", "/home/matilda/git/SI/cache/gemma-4-E4B-hf"
    )
    # Ensure sandbox is up
    container = SandboxContainer()
    container.start()
    try:
        llm = GemmaLLM(model_path, cuda_visible_devices="1")
        result = humaneval_plus_pass_at_1(
            llm,
            max_problems=20,
            timeout_s=10.0,
            temperature=0.2,
            max_completion_tokens=512,
        )
        print(f"\nHumanEval+ (first 20, base E4B): pass@1 = {result.passed}/{result.total} = {result.pass_at_1:.1%}")
        failing = [tid for tid, ok in result.per_problem.items() if not ok]
        print(f"failed: {failing[:10]}{'...' if len(failing) > 10 else ''}")
    finally:
        container.stop()


if __name__ == "__main__":
    main()
