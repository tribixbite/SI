# SI — Self-Improving

A home-scale research stack for compounding self-improvement of a language-model-based agent, assembled from four published components that have not been combined before:

1. **AZR** — zero-data self-play with code executor as verifier (arXiv [2505.03335](https://arxiv.org/abs/2505.03335), [LeapLabTHU/Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner))
2. **RoboPhD** — Elo-tournament agent evolution under tight evaluation budgets (arXiv [2604.04347](https://arxiv.org/abs/2604.04347), [andborth/RoboPhD](https://github.com/andborth/RoboPhD))
3. **In-Place TTT** — fast-weight adaptation of MLP projections at inference time (arXiv [2604.06169](https://arxiv.org/abs/2604.06169))
4. **OpenEvolve** — island-based evolutionary coding agent (AlphaEvolve reimplementation, [codelion/openevolve](https://github.com/codelion/openevolve))

Base model: **Gemma 4 26B A4B** (MoE, 3.8B active) or **Gemma 4 31B Dense**, via Unsloth GGUFs or Hugging Face weights (Apache 2.0).

Target hardware: 1× 3090 (MVP), 5× 3090 (working), 10× 3090 (exploration).

## Status (2026-04-24)

**Phase 1 MVP implemented and training.** The AZR self-play loop runs end-to-end on Gemma 4 E4B with Unsloth QLoRA + TRL GRPO + sandbox-fusion verifier + HumanEval+ anchor. 38/38 unit + integration tests green.

- **Base model anchor (HumanEval+):** 143/164 = **87.20%** on `google/gemma-4-E4B-it`.
- **Phase 1 target:** ≥92.20% (+5 pp) on the same anchor.
- **First full training run** (`phase1_full_20260420`, 144 gens, 12 h walltime) exposed three compounding bugs: `si anchor` didn't mount the adapter, vLLM 0.19.1 can't hot-mount LoRA on `Gemma4ForConditionalGeneration`, and `FastVisionModel` applied LoRA to vision + audio towers diluting the signal. All three are fixed; **Phase 1-v2** re-runs with text-tower-only LoRA, auto-merge anchor, 4× denser rollouts.
- **Phases 2–4** (Elo tournament, island migration, In-Place TTT) are still scaffolded — the spec docs remain load-bearing for them.

## Reading order

- **"What's built today and how does it run?"** → [`docs/07-architecture.md`](docs/07-architecture.md), then [`CLAUDE.md`](CLAUDE.md) for hardware notes.
- **"How do I run or monitor a Phase 1 training cycle?"** → the [Running Phase 1](#running-phase-1) section below.
- **"What's the original design spec?"** → [`docs/00-overview.md`](docs/00-overview.md) → 06-risks.md for intent / risks / evaluation rules.
- **"What changed in the last N hours?"** → [`docs/research-scan.md`](docs/research-scan.md) (hourly scan log during training runs).

## Running Phase 1

One-time setup (see [`scripts/bootstrap.sh`](scripts/bootstrap.sh) for the full pin list):

```bash
export SI_ROOT=$(pwd); export SI_DEPS=$SI_ROOT/deps; export SI_CACHE=$SI_ROOT/cache
bash scripts/bootstrap.sh       # venv + deps + sandbox-fusion image pull
```

Baseline anchor (~45 s on one 3090):

```bash
python -m si.cli anchor --out runs/base_humaneval_plus.json
```

Full Phase 1 training cycle (~12–16 h wall on one 3090):

```bash
bash scripts/phase1_loop.sh phase1_v2_$(date +%Y%m%d_%H%M) 50 5 32 8
#                                    ^run_id                  ^gens ^anchor_every ^proposals/type ^mc_rollouts
```

Monitor a live run (or after the fact):

```bash
tail -f runs/<run_id>/phase1.log      # heartbeat with GPU + CPU vitals per step
ls -t runs/<run_id>/anchor_gen*.json  # anchor trajectory
```

A full description of what the orchestrator does is in [`docs/07-architecture.md`](docs/07-architecture.md).

## One-screen summary

## One-screen summary

A single Gemma 4 instance plays two roles — **Proposer** generates code tasks with optimal difficulty (Monte Carlo rollout reward); **Solver** attempts them against a Python executor (binary reward). This is AZR's loop. We run N such Solver instances as LoRA branches competing in an **Elo tournament** (RoboPhD's selection), where match outcomes on shared problems drive replacement of bottom-quartile branches each generation. Cross-branch knowledge transfers via **OpenEvolve's island-migration** pattern — periodic migration of top programs between branches' experience buffers. **In-Place TTT** supplies a per-problem fast-weight adaptation layer on the final MLP projection, amortizing context without checkpoint growth.

Everything is anchored against a fixed held-out benchmark (HumanEval+ or LiveCodeBench frozen snapshot) with **mandatory generation reversion on anchor regression** — the non-negotiable defense against mutual-hallucination collapse.

## Success criterion for MVP

After 48 hours on 1× 3090 with Gemma 4 E4B: the AZR+Elo stack beats base Gemma 4 E4B by ≥5 percentage points on held-out HumanEval+, with zero external training data, and the Elo branch ranking is stable (top-ranked branch wins ≥60% of head-to-head matches against bottom-ranked). If either condition fails, the bottleneck is diagnosable from the instrumentation — see [`docs/05-evaluation.md`](docs/05-evaluation.md).

## License

Apache 2.0, matching all four upstream components. See [`LICENSE`](LICENSE).

## Honest disclaimers

This is not a path to AGI. It is a reproducible platform for compounding self-improvement of a coding/reasoning agent in domains with exact verifiers, running at hobbyist scale, using four published mechanisms that haven't been combined before. If it compounds further than expected, that's the upside; if it plateaus at generation N, that's still a publishable negative result with reusable infrastructure.

Trading, open-ended generation, and any domain without a cheap exact verifier remain out of scope. See [`docs/06-risks.md`](docs/06-risks.md) for specific failure modes and detection rules.
