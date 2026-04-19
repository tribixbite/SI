# SI — Claude Instructions

Home-scale compounding self-improvement: AZR + Elo tournament + island migration + in-place TTT on Gemma 4. Pre-alpha scaffolding; no trained checkpoints yet.

## Where things live in this repo

```
src/si/
  contracts.py   Protocols + dataclasses (the architectural contract — load-bearing)
  config.py      RunConfig dataclasses; YAML overlays live in configs/
  elo.py         Elo selector — IMPLEMENTED and tested (math has to be exact)
  islands.py     Ring-topology migrator — IMPLEMENTED (LoRA merge is a callback)
  anchor.py      Held-out eval + revert rule — decision logic done, benchmark runners are stubs
  loop.py        Top-level orchestrator — skeleton with NotImplementedError per phase
  cli.py         `si` entry point (typer) — all subcommands stubbed
configs/         tier1_e4b.yaml (MVP, 1×3090), tier2_26b.yaml (5×3090)
docs/            00-overview → 06-risks. Read 00 first. §numbers below reference these.
scripts/         bootstrap.sh, 00_smoke_test.sh, 01_phase1_run.sh, 02_anchor_eval.sh
tests/test_elo.py  Only pytest file today; must stay green (pure math, no GPU)
```

Implementation status: `elo.py` + `islands.py` + `anchor.py` decision logic are real; everything in `loop.py` raises NotImplementedError pointing to the relevant phase in `docs/04-implementation.md`. Match that idiom when adding stubs.

## Hardware context (this machine)

- 2× RTX 3090 24GB, Ryzen 9 5900X, 47GB RAM, ~16GB free on `/`.
- **GPU 0 is occupied** by the wsl-llm inference server (Qwen3.5 GGUF via ik_llama.cpp, ~23GB resident). Do not reallocate without stopping that service.
- **GPU 1 is free** (~24GB). All SI work targets GPU 1 unless explicitly told otherwise. Set `CUDA_VISIBLE_DEVICES=1` in any runner.
- CUDA toolkit 12.6 at `/usr/local/cuda-12.6`. **Do not upgrade to CUDA 13.2** — breaks Gemma 4 GGUFs per `docs/01-sources.md`. 13.1 driver is fine; 12.6 toolkit is what we build against.
- Sibling repo `../wsl-llm/` owns the server/model stack (ik_llama.cpp, llama.cpp, vLLM venv at `~/bench_env`, model files at `~/models/`). Read `../wsl-llm/CLAUDE.md` before touching models, services, or GPU allocation.
- Disk budget is tight (16GB free). Phase 0 model download alone (E4B-HF, ~10GB) nearly fills it. Before downloading 26B/31B HF weights, free space or stage to a different volume.

## How to run

Phase 0 bootstrap expects `SI_ROOT=$HOME/SI` by default; this repo lives at `~/git/SI` instead. Override before running:

```bash
export SI_ROOT=$(pwd)
export SI_DEPS=$SI_ROOT/deps
export SI_CACHE=$SI_ROOT/cache
bash scripts/bootstrap.sh     # clones AZR/verl/RoboPhD/openevolve, installs deps, starts sandbox
bash scripts/00_smoke_test.sh # pytest + vLLM sanity + sandbox health
bash scripts/01_phase1_run.sh # AZR self-play on Gemma 4 E4B
bash scripts/02_anchor_eval.sh <run_id>
```

Tests only (no GPU, fast): `pytest tests/ -q`.

## Non-negotiables (baked into the design — don't quietly violate)

- **Anchor reversion on regression.** `docs/00-overview.md` §"What's non-negotiable". This is the single defense against mutual-hallucination collapse. If code bypasses `Anchor.should_revert`, the whole premise breaks.
- **Zero external training data.** No MBPP, no HumanEval problems, no CodeAlpaca in training. Anchor set is eval-only, hash-verified at startup, kept in memory (see `anchor.anchor_hash_check`).
- **Verifier sandboxing.** Never call `subprocess.run` on proposer/solver output outside the sandbox. Proposer/solver output is *data*, not code. `docs/03-stack.md` §"The trust model".
- **`use_cache=True` for Gemma 4.** Flipping to False corrupts attention on 31B / 26B-A4B. Already the default in `ModelConfig`; don't override.
- **Uncommitted config = no run.** `scripts/01_phase1_run.sh` refuses to start if `configs/` has uncommitted changes. Keep that discipline.
- **`SI_CACHE`, `$SI_DEPS` are sacred paths.** Everything regenerable (weights, traces, checkpoints) lives there and is gitignored. Never write run artifacts under `src/`.

## Editing conventions

- Typed strict (`mypy strict` in pyproject). New modules get `from __future__ import annotations` and full annotations.
- Ruff with the config in pyproject; line length 100; prefer `SIM`/`RUF` fixes over manual rewrites.
- Protocols in `contracts.py` are load-bearing — if you change a method signature there, plan to update `loop.py`, the concrete classes, and tests in the same commit.
- Default comment count: zero. Only write a comment for a non-obvious *why* (a known Gemma 4 bug, a deliberate deviation from a paper, an invariant that isn't visible from the code).
- When stubbing a new phase method, raise `NotImplementedError` with a concrete pointer (`"Phase 2. See docs/04-implementation.md §2.3"`) — match the existing idiom in `loop.py`.

## Phase gating

Do not start Phase N+1 until Phase N's success criterion in `docs/04-implementation.md` passes. Criteria are deliberately quantitative (≥5pp over base on HumanEval+, Elo stability ≥60% head-to-head, etc.). If a criterion fails, debug the current phase — do not paper over it at the next layer.

## External anchors

- AZR — `github.com/LeapLabTHU/Absolute-Zero-Reasoner` — substrate we fork from
- verl — `github.com/verl-project/verl` — RL training backend (GRPO)
- RoboPhD — `github.com/andborth/RoboPhD` — Elo selection reference
- openevolve — `github.com/codelion/openevolve` — island pattern reference
- EvalPlus pinned commit `d362e933265c3e7e3df8101c930a89c3c470cd9f` — anchor stability depends on this
