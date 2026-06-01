"""Concrete GPU/disk operations for the Phase 2 loop.

These are the callbacks the orchestration in loop.py / population.py delegates
to (reseed_fn, and later solver_factory / trainer_fn). Kept out of those
modules so the orchestration stays import-light and unit-testable without
torch.

`perturb_lora_adapter` is the real `ReseedFn`: it produces a mutated copy of a
LoRA adapter by adding N(0, sigma) noise to each weight tensor. It works
directly on the adapter's safetensors file — no base model load — so it is
cheap and runs on CPU.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

log = logging.getLogger(__name__)

_WEIGHTS_FILE = "adapter_model.safetensors"
_REPO = Path(__file__).resolve().parent.parent.parent
_MERGE_SCRIPT = _REPO / "scripts" / "merge_and_anchor.py"
_MERGED_ROOT = _REPO / "cache" / "_merged"


def resolve_adapter_dir(adapter_path: str) -> str:
    """Trainers save the PEFT adapter at <out>/adapter/; accept either level."""
    p = Path(adapter_path)
    if (p / "adapter_config.json").exists():
        return str(p)
    if (p / "adapter" / "adapter_config.json").exists():
        return str(p / "adapter")
    raise FileNotFoundError(f"no adapter_config.json at {p} or {p}/adapter")


def merged_dir_for(adapter_path: str) -> Path:
    """The cache dir ensure_merged_model would use for this adapter (may not
    exist). Lets the orchestrator delete regenerable merges between generations
    so per-branch-per-gen merged models don't fill the disk (~15 GB each)."""
    adapter_dir = resolve_adapter_dir(adapter_path)
    h = sha256(Path(adapter_dir).resolve().as_posix().encode()).hexdigest()[:16]
    return _MERGED_ROOT / h


def ensure_merged_model(adapter_path: str, base_model: str) -> str:
    """Return a vLLM-loadable merged-model dir for this adapter, merging it into
    the base on first use and caching by adapter-path hash.

    vLLM 0.19.1 can't serve LoRA on Gemma4ForConditionalGeneration, so per-branch
    inference goes through a merged checkpoint. Shared by cli.anchor and the
    Phase 2 per-branch solve subprocess. CPU merge (~15 GB RAM, no GPU)."""
    adapter_dir = resolve_adapter_dir(adapter_path)
    merged = merged_dir_for(adapter_path)
    if not (merged / "config.json").exists():
        log.info("merging adapter %s -> %s", adapter_dir, merged)
        subprocess.check_call(
            [sys.executable, str(_MERGE_SCRIPT),
             "--adapter", adapter_dir, "--merged-out", str(merged),
             "--base", base_model, "--skip-anchor"]
        )
    return str(merged)


def perturb_lora_adapter(
    parent_path: str,
    dest_path: str,
    sigma: float,
    *,
    generator: torch.Generator | None = None,
) -> None:
    """Write a copy of the LoRA adapter at `parent_path` to `dest_path`, with
    N(0, sigma) added to every floating-point weight (docs/04 §2.3.5).

    Matches the `population.ReseedFn` signature (the `generator` kwarg is for
    deterministic tests; the loop calls this positionally with 3 args). All
    non-weight files (adapter_config.json, etc.) are copied verbatim, and the
    safetensors metadata is preserved so PEFT can reload the adapter.
    """
    parent = Path(parent_path)
    dest = Path(dest_path)
    weights = parent / _WEIGHTS_FILE
    if not weights.exists():
        raise FileNotFoundError(f"no {_WEIGHTS_FILE} under {parent}")
    dest.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(weights), framework="pt") as f:  # type: ignore[no-untyped-call]
        metadata = f.metadata()
        for key in f.keys():  # noqa: SIM118 — safe_open has no __iter__
            t = f.get_tensor(key)
            if t.is_floating_point():
                noise = torch.randn(t.shape, generator=generator).to(t.dtype) * sigma
                t = t + noise
            tensors[key] = t
    save_file(tensors, str(dest / _WEIGHTS_FILE), metadata=metadata)

    for p in parent.iterdir():
        if p.is_file() and p.name != _WEIGHTS_FILE:
            shutil.copy2(p, dest / p.name)
    log.info("perturb_lora_adapter: %s -> %s (sigma=%g)", parent, dest, sigma)
