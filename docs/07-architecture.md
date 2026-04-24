# 07. Architecture (as built)

`docs/00-overview.md` through `docs/06-risks.md` describe the *spec of intent*. This document describes what is *actually built* in `src/si/` after the Phase 1 path-C implementation landed on 2026-04-19/20. Read 00 first for goals; read this to see what exists in code.

## One-screen picture of the implementation

```
┌────────────────────────────────────────────────────────────────────────────┐
│ scripts/phase1_loop.sh — ping-pong orchestrator                            │
│                                                                            │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐    │
│  │ si rollout       │   │ si train         │   │ si anchor            │    │
│  │ (vLLM 0.19 + HF) │ → │ (Unsloth + TRL   │ → │ (vLLM + evalplus     │    │
│  │                  │   │  1.2 + PEFT LoRA)│   │  HumanEval+ pass@1)  │    │
│  └──────────────────┘   └──────────────────┘   └──────────────────────┘    │
│          │                       │                        │                │
│          ▼                       ▼                        ▼                │
│   outcomes_gen####.jsonl   runs/<id>/adapter/       anchor_gen####.json    │
│   metrics_gen####.json                                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

Subprocess ping-pong because vLLM (for rollout, ~19 GB VRAM) and HF+Unsloth (for GRPO training, ~11 GB VRAM + activations) don't coexist on one 3090. Each gen boots one, tears down, boots the other.

## Module map

| File | Role | Status |
|---|---|---|
| `src/si/contracts.py` | Protocol interfaces — Proposer/Solver/Verifier/Selector/Migrator/Anchor | existed, untouched |
| `src/si/config.py` | RunConfig dataclasses + YAML overlays | existed, untouched |
| `src/si/elo.py` | Elo selector (Phase 2) | existed, tested |
| `src/si/islands.py` | Ring-topology migrator (Phase 3) | existed, tested |
| `src/si/anchor.py` | Commit/revert decision logic + `make_humaneval_runner` | factory wired |
| `src/si/loop.py` | Full-population loop (Phase 2+) | still stubbed; Phase 1 orchestrated via shell |
| `src/si/prompts.py` | Condensed AZR prompts, Gemma chat-template ready | new |
| `src/si/parsers.py` | Fenced block extractors + banned-import AST check | new |
| `src/si/llm.py` | Shared `GemmaLLM` vLLM wrapper — chat-only | new |
| `src/si/proposer.py` | `AZRProposer` for deduction + abduction | new |
| `src/si/solver.py` | `GemmaSolver` — batched + MC rollouts | new |
| `src/si/verifier.py` | `SandboxContainer` + `SandboxVerifier` (sandbox-fusion) | new |
| `src/si/match.py` | MC proposer reward (−\|0.5 − pass\_rate\|), generation metrics | new |
| `src/si/humaneval.py` | HumanEval+ pass@1 driver via sandbox | new |
| `src/si/trainer.py` | Vanilla HF + TRL 1.2 GRPOTrainer (fallback) | new |
| `src/si/trainer_unsloth.py` | **Default trainer** — Unsloth FastVisionModel + TRL GRPO | new |
| `src/si/cli.py` | `si rollout`, `si train`, `si anchor`, `si smoke`, `si status` | new |

## Key implementation decisions (with rationale)

### Base model: `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`

We load the Unsloth-packaged 4-bit QLoRA weights for training. Unsloth's Gemma 4 patches handle the heterogeneous head-dim NaN bug (`head_dim=256` local, `global_head_dim=512` global) automatically; vanilla HF SDPA produces NaN logits under `torch.multinomial` without `attn_implementation="eager"`. See `memory/project_gemma4_eager_attn_training.md`.

For vLLM rollouts, we use the bf16 `google/gemma-4-E4B-it` (15 GB) because vLLM's quantized loader can't share weights with the HF trainer and the 4-bit Unsloth weights are bnb-formatted (vLLM prefers GPTQ/AWQ/int8). Rollout and training use separate copies; they never coexist on GPU.

### Trainer: Unsloth FastVisionModel + TRL 1.2

- **FastVisionModel** (not `FastLanguageModel`) because Gemma 4 is multimodal (text + vision + audio towers). Unsloth detects this and routes prompts through the text tower.
- **TRL 1.2** (not 0.22.2 per Unsloth's own notebook) because TRL 0.22 imports `GuidedDecodingParams` from vLLM 0.15 that's removed in 0.19. TRL 1.2 uses `beta` (KL coef) instead of 0.22's `epsilon`/`epsilon_high`/`delta`/`loss_type='bnpo'`.
- **Local patch** to `trl/import_utils.py` — transformers 5.x's `_is_package_available` returns a tuple, TRL bare-bools it; the `vllm_ascend` import fires incorrectly. Patched to unwrap.
- **LoRA rank 32, alpha 64** on `q/k/v/o/gate/up/down_proj` only (Unsloth's default target modules for the text tower).
- **`use_gradient_checkpointing="unsloth"`** — Unsloth's custom checkpointing saves ~60% VRAM vs HF's.
- **`optim="adamw_8bit"`** — bitsandbytes 8-bit AdamW.
- **`num_generations=2`** (GRPO group size) — higher values OOM on 3090 with our max_completion_length=512.

### AZR specifics

- **Deduction + abduction only** in Phase 1. Induction needs a seed pool of programs we haven't built yet; deferred.
- **Two reward signals combined** at training time:
  - `make_verifier_reward_fn` — 1.0 if SandboxVerifier passes, 0.0 else, −0.2 if output unparseable. This is the AZR binary solver reward.
  - `make_format_reward_fn` — +0.1 shaping bonus if the expected fenced block is present. Keeps low-reward steps from collapsing completely.
- **MC proposer reward** (`-|0.5 − pass_rate_k|`) is computed by `MatchRunner._play` during rollout. Currently only logged; not yet used for proposer training (Phase 1.5).

### Sandbox: `volcengine/sandbox-fusion:server-20250609`

- 26.2 GB container, dynamic host port. One container per training run.
- Python 3.11 sandbox with CPU/mem/time caps. Network disabled.
- Our `SandboxVerifier` dispatches deduction → `f(I) == solver.O`, abduction → `f(solver.I) == O`, induction → all-pairs check on list literals.

### HumanEval+ anchor

- Pinned to evalplus commit `d362e933265c3e7e3df8101c930a89c3c470cd9f` (matches AZR paper).
- Dataset hash verified at run start (`EXPECTED_HUMANEVAL_PLUS_HASH = fe585eb4df8c88d844eeb463ea4d0302`) — drift invalidates the revert rule.
- Executed in-sandbox via `SandboxVerifier._run`, not evalplus's CLI (skipped pulling its extras).
- Base Gemma 4 E4B: **143/164 = 87.20%**. Phase 1 target: **≥92.20%** (+5 pp).

## What's NOT built yet (and where it goes)

| Missing | Goes where | Notes |
|---|---|---|
| Induction task type | `proposer.py` + `solver.py` + prompts | Needs seed program pool |
| Proposer training | `trainer_unsloth.py` extended reward fn | Phase 1.5 — train proposer with MC reward |
| Elo selection | `loop.py` wired to `elo.py` | Phase 2 |
| Multi-branch LoRA swap | `loop.py` + vLLM LoRA adapter mounting | Phase 2 |
| Island migration | `loop.py` wired to `islands.py` | Phase 3 |
| In-Place TTT | new module `ttt.py` | Phase 4 |
| Same-process rollout + train (no ping-pong) | `trainer_unsloth.py` with `fast_inference=True` | Phase 1.5 optimization |

## Non-obvious invariants

- `contracts.py` Protocols are load-bearing. Changing any signature there must ripple into concrete classes + `loop.py` in the same commit.
- Configs in `configs/` must be committed before `scripts/phase1_loop.sh` starts — it refuses to begin with a dirty tree. Reproducibility > convenience.
- `SI_CACHE` and `$SI_DEPS` are sacred. Everything regenerable (weights, traces, checkpoints) lives there; `src/` never holds artifacts.
- Chat template is mandatory for Gemma 4 -it. Raw completion produces repetition loops. See `memory/project_gemma4_e4b_inference.md`.
- `attn_implementation="eager"` is auto-applied by Unsloth's patches; do not override.

## Validation status (as of 2026-04-20)

- 38/38 unit + integration tests green (`tests/test_{elo,verifier,parsers,match,trainer}.py`).
- End-to-end Phase 1 smoke succeeded: rollout → Unsloth GRPO → anchor with the correct VRAM ping-pong.
- Base Gemma 4 E4B anchor: 87.20% on HumanEval+ (164 problems).
- First trained adapter anchor (1 GRPO step on 6 tasks): 87.20% — no drift, but also no compounding. Expected; AZR's signal emerges over hundreds of steps.
