# SI — Claude Instructions

Home-scale compounding self-improvement: AZR + Elo tournament + island migration + in-place TTT on Gemma 4. **Phase 1 implemented and actively training**; Phases 2–4 still scaffolding. Read `docs/07-architecture.md` for the as-built description before editing anything under `src/si/`.

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

Implementation status (2026-04-24):
- **Phase 1 built + validated** — AZRProposer, GemmaSolver, SandboxVerifier, MatchRunner, UnslothSITrainer (RL-ZVP+F-GRPO), SSD trainer (arXiv:2604.01193), HumanEval+ runner, **LCB v6 runner** (1054 problems), CLI subcommands (`si rollout/train/anchor/ssd-sample/ssd-train`). 49/49 tests green.
- **Primary anchor = LiveCodeBench v6** (21.92% base, lots of headroom). HumanEval+ is secondary/format-regression-detector only (87.20% base = 8B-class zero-RL capacity ceiling per scan #3).
- **Key finding (research-scan.md §2026-04-24 14:20)**: all three training approaches (vanilla GRPO, RL-ZVP, SSD) teach the same per-difficulty signature on LCB v6 — lose easy, gain medium, no effect on hard. SSD is the cleanest (+10 medium, −2 easy, 0 hard). That's real AZR-paradigm signal; the "plateau" on HumanEval+ was anchor-mismatch, not a training failure.
- **Phase 2+ scaffolded only** — `elo.py` + `islands.py` + `anchor.py` decision logic are real; `loop.py`'s multi-branch paths still raise NotImplementedError. When implementing Phase 2, match that idiom for still-unbuilt pieces.
- Full module map is in `docs/07-architecture.md`; training trajectory in `docs/research-scan.md`.

### Current scores (2026-04-24)

| Model | HumanEval+ | LCB v6 | LCB easy / medium / hard |
|---|---|---|---|
| Base (`google/gemma-4-E4B-it`) | 143/164 = 87.20% | 231/1054 = 21.92% | 52.17 / 11.78 / 5.14 |
| v2 gen_10 (vanilla GRPO) | 141/164 = 85.98% | 232/1054 = 22.01% | 50.93 / 13.61 / 4.57 |
| v3 gen_10 (RL-ZVP+F-GRPO) | 140/164 = 85.37% | 231/1054 = 21.92% | 50.31 / 13.87 / 4.57 |
| SSD v1 (500 tasks × 8 cand) | 139/164 = 84.76% | 239/1054 = 22.68% | 51.55 / 14.40 / 5.14 |

Target: +5 pp on LCB v6 (21.92 → 26.92). ssd_v2 (792 tasks × 16 cand, running) is the next data point.

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
export SI_ROOT=$(pwd); export SI_DEPS=$SI_ROOT/deps; export SI_CACHE=$SI_ROOT/cache
bash scripts/bootstrap.sh         # venv + deps + sandbox-fusion image pull
pytest tests/ -q                  # 38/38 should pass; no GPU needed for most

# Baseline anchor (~45s on one 3090):
python -m si.cli anchor --out runs/base_humaneval_plus.json

# Full Phase 1 training cycle (~12–16h on one 3090):
bash scripts/phase1_loop.sh phase1_v2_$(date +%Y%m%d_%H%M) 50 5 32 8
#                                    ^id                      ^gens ^anchor_every ^proposals/type ^mc_rollouts

# Monitor:
tail -f runs/<id>/phase1.log       # heartbeat lines (gpu_used, ram, step transitions)
ls -t runs/<id>/anchor_gen*.json   # anchor trajectory
```

The legacy `scripts/01_phase1_run.sh` / `02_anchor_eval.sh` are stubs from the original spec; the working orchestrator is `scripts/phase1_loop.sh`.

## Non-negotiables (baked into the design — don't quietly violate)

- **Anchor reversion on regression.** `docs/00-overview.md` §"What's non-negotiable". This is the single defense against mutual-hallucination collapse. If code bypasses `Anchor.should_revert`, the whole premise breaks.
- **Zero external training data.** No MBPP, no HumanEval problems, no CodeAlpaca in training. Anchor set is eval-only, hash-verified at startup, kept in memory (see `anchor.anchor_hash_check`).
- **Verifier sandboxing.** Never call `subprocess.run` on proposer/solver output outside the sandbox. Proposer/solver output is *data*, not code. `docs/03-stack.md` §"The trust model".
- **`use_cache=True` for Gemma 4.** Flipping to False corrupts attention on 31B / 26B-A4B. Already the default in `ModelConfig`; don't override.
- **Gemma 4 chat template is mandatory** for the -it variants. Raw completion produces repetition loops. `GemmaLLM.chat_batch` always wraps in chat format; don't bypass.
- **LoRA scope = language_model only.** Gemma 4 is multimodal — its q/k/v/o/gate/up/down_proj names appear in vision_tower and audio_tower too. Use the regex in `trainer_unsloth.py` to scope. See `memory/project_gemma4_eager_attn_training.md` + the Phase 1-v2 commit message for the full story.
- **Uncommitted config/code = no run.** `scripts/phase1_loop.sh` refuses to start if `configs/` or `src/si/` have uncommitted changes. Keep that discipline.
- **`SI_CACHE`, `$SI_DEPS` are sacred paths.** Everything regenerable (weights, traces, checkpoints, merged model dirs) lives there and is gitignored. Never write run artifacts under `src/`.

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
