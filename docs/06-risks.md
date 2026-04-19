# 06. Risks, Failure Modes, and Honest Limits

## What will probably work

The single-branch AZR reproduction (Phase 1). The AZR paper has public code, 1.7k GitHub stars, and published results on Qwen2.5. Substituting Gemma 4 E4B is a model-swap, not a research risk. Expect it to work.

## What's genuinely uncertain

### 1. Whether Elo selection compounds on top of AZR

RoboPhD shows Elo-tournament evolution beats Pareto/greedy at fixed eval budget, but the evolved objects in RoboPhD are *prompts and Python scripts*, not LoRA-adapted LLM weights. The mechanism is orthogonal to the substrate; there's no published result confirming it transfers to gradient-mutated LoRAs.

Probability compounds: ~50%. Probability provides at least some signal even if it doesn't compound: ~80%.

**If it doesn't compound:** still worth publishing as a negative. "Elo tournament does not add value to AZR self-play in the LoRA-on-Gemma-4 regime" is a real result.

### 2. Whether island migration helps

OpenEvolve's migration preserves diversity in program-space evolution. In LoRA-space evolution, two LoRAs of the same rank live on a manifold where naive migration (copying weights) might be meaningless — the receiving branch's base representation may have drifted in ways that make the migrant's deltas nonsensical.

Probability helps: ~40%. Our migration mechanism (use migrant as initialization for next-gen mutation, not direct weight transfer) is designed to mitigate this but is not validated.

**If it doesn't help:** fall back to Elo without migration. The stack degrades gracefully.

### 3. Whether In-Place TTT integrates cleanly

Paper is 12 days old at time of spec (April 7, 2026). No public code. Mechanism may interact pathologically with GRPO rollouts — the fast-weight update during rollout could alter solver behavior in ways that invalidate the advantage estimate.

Probability clean integration: ~30%. This is why it's Phase 4, not Phase 1.

**If it breaks GRPO:** disable during rollouts; use only at final inference time for anchor eval. A weaker but safer integration.

### 4. Whether any of this generalizes beyond code

AZR authors report cross-domain transfer (coding → math). We haven't confirmed it. If your goal is a "self-improving reasoning agent" broadly, the code-domain specialization might not transfer.

Probability of cross-domain transfer: ~60% for math (AZR shows it); ~20% for broader domains without adding domain-specific verifiers.

## What won't work

### Trading / market prediction / non-stationary domains

No cheap exact verifier. Any backtest is a noisy, overfittable, non-stationary proxy. Running the self-improvement loop against a backtest produces a population of models that are maximally confident in strategies that worked on one draw of history. This failure mode is well-established; no amount of cleverness in the self-improvement loop fixes a broken reward signal.

If trading is the goal, the path is not AI research — it's building the market-exposure infrastructure first, with small live capital, and letting the RL loop train against actual fills. That's a different project in a different risk category (money at stake), and at least a year of infrastructure work before any model-training is appropriate.

### Open-ended creative generation

"Write me a good novel" has no verifier. Soft verifiers (reward models) produce mutual hallucination. This is a hard limit, not a current limitation.

### AGI

This stack compounds self-improvement in bounded, exact-verifier domains. It does not address general intelligence, agency, or transfer learning across arbitrary domains. If "self-improving AGI" is the success condition, this project will not reach it. If "novel research artifact demonstrating compounding self-improvement at hobbyist scale" is the success condition, this is a credible path.

## The long-conversation risk

This project spans weeks at minimum. Two specific failure modes show up over long runs that don't show up in the first 24 hours:

### Config drift

You will tweak configs. Some tweaks will help, some won't. Three weeks in, you won't remember which tweak did what, and your run state will be a Frankenstein of a dozen small changes with no clean ablation history.

**Mitigation:** every config change gets a git commit. Every run logs its config hash to W&B. `scripts/run.sh` refuses to start if there are uncommitted changes. Painful up front, invaluable at week 4.

### Verifier rot

The sandbox keeps working, but Python versions drift, dependencies update, new security patches land. Test suites that passed last week fail for unrelated reasons. Anchor scores drop; you chase a phantom regression.

**Mitigation:** `scripts/verify_anchor_env.sh` runs the anchor evaluator against a **frozen model checkpoint** before each run. If the frozen model's anchor score has changed from last run, the environment has drifted. Fix the environment, don't fix the model.

## The genuinely uncomfortable risk

The system might appear to work. Scores might compound for 50 generations, we might write a paper, and the whole improvement might turn out to be a subtle anchor-memorization artifact we didn't catch. Self-improving systems are exceptionally good at finding ways to hack their own evaluator that look like learning from the inside.

**Defense in depth:**
- Multi-anchor (primary + secondary + meta). Different benchmarks, different suppliers, different problem distributions.
- Automatic diff between winning solutions across generations (are we producing meaningfully different code, or the same code with cosmetic variations?).
- Third-party eval at generation 50, 100, 200 — have someone else run the model on a private held-out set.

If we publish a positive result, the headline number is meta-anchor score, not primary. Primary can always be hacked; the one the system has never seen, less so.

## What to cut if resources are constrained

If you have one weekend and one 3090:
- Phase 1 only. Confirm AZR + Gemma 4 E4B improves on HumanEval+.
- Skip Phases 2–4.
- If Phase 1 works at all, you've validated the single most important premise and can decide whether to invest further.

If you have a month and 5 GPUs:
- Phases 1–3. Skip Phase 4 (In-Place TTT).
- This is the strong-minimum publishable configuration.

If you have three months and 10 GPUs:
- All four phases plus the anchor-rotation and multi-anchor disciplines.
- This is the "write a real paper" configuration.

## Honest expectation

Best realistic outcome: a working demonstration that the AZR+Elo+Islands stack compounds for 30–80 generations on Gemma 4, producing a measurable improvement on held-out anchors with zero external data, before plateauing. A publishable result, a reusable platform, and genuine novel ideas (specifically the cross-component combination and the in-place-TTT-as-amortization framing) that would survive peer review.

Worst realistic outcome: Phase 1 works (AZR reproduces on Gemma 4), Phase 2 fails to add signal (Elo doesn't compound on LoRAs), Phase 3 and 4 abandoned. Still a publishable negative and a tool (the single-branch AZR-on-Gemma-4 trained model, which is itself a useful open-source artifact).

Unexpectedly good outcome: sustained compounding past generation 200, cross-domain transfer to math, meta-anchor score continuing to rise after primary saturates. Paper, conference talk, reference implementation for the field.

Unexpectedly bad outcome: sandbox escape during Phase 2. This is the one risk that merits serious paranoia, which is why the verifier discipline in `04-implementation.md` §3 and the trust model in `03-stack.md` are non-negotiable.

Neither the best nor the worst is the most likely. The most likely outcome is "Phase 1 works, Phase 2 partly works, paper is written, field moves slightly" — which is the normal shape of useful research.
