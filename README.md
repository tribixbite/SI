# SI — Self-Improving

A home-scale research stack for compounding self-improvement of a language-model-based agent, assembled from four published components that have not been combined before:

1. **AZR** — zero-data self-play with code executor as verifier (arXiv [2505.03335](https://arxiv.org/abs/2505.03335), [LeapLabTHU/Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner))
2. **RoboPhD** — Elo-tournament agent evolution under tight evaluation budgets (arXiv [2604.04347](https://arxiv.org/abs/2604.04347), [andborth/RoboPhD](https://github.com/andborth/RoboPhD))
3. **In-Place TTT** — fast-weight adaptation of MLP projections at inference time (arXiv [2604.06169](https://arxiv.org/abs/2604.06169))
4. **OpenEvolve** — island-based evolutionary coding agent (AlphaEvolve reimplementation, [codelion/openevolve](https://github.com/codelion/openevolve))

Base model: **Gemma 4 26B A4B** (MoE, 3.8B active) or **Gemma 4 31B Dense**, via Unsloth GGUFs or Hugging Face weights (Apache 2.0).

Target hardware: 1× 3090 (MVP), 5× 3090 (working), 10× 3090 (exploration).

## Status

Pre-alpha scaffolding and specification. No trained checkpoints yet. This repo exists to specify the system precisely enough that the MVP can be built deterministically in a week.

## Reading order

Start with [`docs/00-overview.md`](docs/00-overview.md). Then, depending on what you want:

- **"Is this worth building?"** → [`docs/06-risks.md`](docs/06-risks.md)
- **"What exactly is being combined?"** → [`docs/03-stack.md`](docs/03-stack.md)
- **"How do I run the MVP this weekend?"** → [`docs/04-implementation.md`](docs/04-implementation.md) §1
- **"What are the exact source commits and versions?"** → [`docs/01-sources.md`](docs/01-sources.md)

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
