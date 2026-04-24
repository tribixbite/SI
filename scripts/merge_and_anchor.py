"""Merge a LoRA adapter into the base Gemma 4 model and run HumanEval+ anchor.

Needed because vLLM 0.19.1 doesn't support LoRA on Gemma4ForConditionalGeneration
(the multimodal arch). Merge-then-load sidesteps the missing LoRA kernel.

Usage:
    python scripts/merge_and_anchor.py \
        --adapter runs/<run_id>/adapter/adapter \
        --merged-out runs/<run_id>/merged \
        --anchor-out runs/<run_id>/anchor_final.json

Leaves `merged-out` on disk (~15 GB) so you can repeat anchor eval without
re-merging. Safe to delete after anchoring if disk is tight.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL = "/home/matilda/git/SI/cache/gemma-4-E4B-hf"


def merge(adapter_path: str, merged_out: str, base: str = BASE_MODEL) -> str:
    """Load base bf16 + apply adapter + merge + save."""
    log.info("loading base %s", base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    log.info("applying adapter %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    log.info("merging LoRA into base weights")
    merged = model.merge_and_unload()
    log.info("saving merged model to %s", merged_out)
    Path(merged_out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_out, safe_serialization=True, max_shard_size="5GB")
    # Copy tokenizer + processor so vLLM can load it standalone.
    tok = AutoTokenizer.from_pretrained(base)
    tok.save_pretrained(merged_out)
    try:
        proc = AutoProcessor.from_pretrained(base)
        proc.save_pretrained(merged_out)
    except Exception as e:
        log.info("no processor to save (%s) — text-only anchor still works", e)
    return merged_out


def run_anchor(merged_path: str, anchor_out: str | None) -> dict:
    # Import lazily so we don't initialize CUDA before the merge frees memory.
    from si.humaneval import humaneval_plus_pass_at_1
    from si.llm import GemmaLLM
    from si.verifier import SandboxContainer

    container = SandboxContainer()
    container.start()
    try:
        llm = GemmaLLM(merged_path, cuda_visible_devices="1")
        result = humaneval_plus_pass_at_1(llm, timeout_s=10.0, temperature=0.2)
    finally:
        container.stop()
    record = {
        "merged": merged_path,
        "passed": result.passed,
        "total": result.total,
        "pass_at_1": result.pass_at_1,
        "wall_s": result.wall_s,
        "per_problem": result.per_problem,
    }
    if anchor_out:
        Path(anchor_out).parent.mkdir(parents=True, exist_ok=True)
        with open(anchor_out, "w") as f:
            json.dump(record, f, indent=2)
    log.info("pass@1 = %d/%d = %.4f", result.passed, result.total, result.pass_at_1)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="path to PEFT adapter dir (containing adapter_config.json)")
    p.add_argument("--merged-out", required=True, help="where to write the merged model")
    p.add_argument("--anchor-out", default=None, help="JSON output path for anchor result")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--skip-merge", action="store_true", help="re-run anchor against an already-merged dir")
    p.add_argument("--skip-anchor", action="store_true", help="merge only; caller will run anchor separately")
    args = p.parse_args()

    if not args.skip_merge:
        merge(args.adapter, args.merged_out, base=args.base)
    if not args.skip_anchor:
        run_anchor(args.merged_out, args.anchor_out)


if __name__ == "__main__":
    main()
