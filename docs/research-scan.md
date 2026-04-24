# Research scan log

Kept per the `feedback_research_scan_cadence.md` memory: each hourly monitor check of a long SI run includes a pass over new HF papers and upstream releases. Entries are appended chronologically; each has a clear **adopt / defer / skip** call.

## 2026-04-20 10:10 UTC — scan #1 (Phase 1 run start)

Trigger: Phase 1 run `phase1_full_20260420` launched ~09:59 UTC (gen 1 rollout in progress).

### TurboQuant (Google Research, ICLR 2026, arXiv:2504.19874)

- **What:** Training-free, data-oblivious KV-cache quantizer. Random-rotate → per-slot quantize → 1-bit residual QJL. 3-bit keys, 2-bit values. 6× KV shrink, 8× attention speedup on H100.
- **Why it matters to SI:** Would free ~3 GB of VRAM during vLLM rollouts (KV cache 3.9 → 0.65 GB at 4K context on a 3090). Enables longer contexts or bigger batches at anchor eval.
- **Integration effort:** Low, once vLLM ships a release containing it. Just set `kv_cache_dtype="turboquant_3bit"` on `GemmaLLM` init.
- **Status of upstream:** Not in vLLM 0.19.1. Merged upstream in PR #39890 on 2026-04-15. First tagged release should include it within weeks.
- **Decision:** **Defer.** Revisit when `vllm>=0.19.2`/`0.20` lands. Not the bottleneck for Phase 1 on a 3090 with Gemma 4 E4B (our limit is weights, not KV cache).
- **Reference:** [Google blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), [vLLM issue](https://github.com/vllm-project/vllm/issues/38171).

### NanoQuant (arXiv:2602.06694, Feb 2026)

- **What:** Sub-1-bit PTQ via low-rank binary factorization + ADMM. Llama2-70B compressed 25.8× on one H100 in 13 h.
- **Why it matters:** Could let SI run 26B-A4B at Tier 2 on a single 3090. Research-grade though.
- **Integration effort:** High. No open training integration; would be a separate compression pass. Breaks Unsloth's patch stack.
- **Decision:** **Skip for Phase 1.** Track for Tier 2 exploration post-Phase-1 results.

### GuidedQuant (arXiv:2505.07004, May 2025)

- **What:** PTQ that uses gradient info from end loss. Boosts state-of-the-art across weight-only scalar / vector / W+A quantization.
- **Why it matters:** Better accuracy at fixed bit budget than GPTQ/AWQ/SmoothQuant.
- **Integration effort:** Medium. Would apply to vLLM rollout weights, not training.
- **Decision:** **Skip.** Unsloth/bnb 4-bit is already good enough for Phase 1; not worth fighting the tool stack yet.

### OSTQuant, SpinQuant, QuIP#, FOEM, CrossQuant, AffineQuant

- Well-known PTQ family, all in the same bucket as GuidedQuant.
- **Decision:** **Skip.** Consider when we need better-than-4-bit accuracy on 31B.

### Summary of scan #1

Nothing to pull into the Phase 1 run in flight. TurboQuant is the one to adopt on next vLLM bump. Full PTQ family is parked for Tier 2.

---

## 2026-04-24 02:58 UTC — scan #2 (Phase 1-v2 gen 1 in progress)

Trigger: `phase1_v2_20260423_2250` started 02:50Z, currently mid-train on gen 1. Searching for new GRPO / RLVR / self-play work that might explain or fix the `frac_reward_zero_std=1` problem we hit in phase1_full_20260420.

### ERPO — Explore Residual Prompts in Policy Optimization (arXiv:2511.04800, Nov 2025)

- **What:** Directly targets GRPO's residual-prompts problem — prompts where every rollout in a group gets the same reward, producing zero advantage and zero gradient. Maintains a per-prompt history tracker; when a prompt has all-correct or all-incorrect outputs in the last K appearances, adaptively **raises the sampling temperature** for that prompt to revive exploration.
- **Why it matters to SI:** This is exactly our observed failure mode. The v1 run logged `frac_reward_zero_std=1` in ~95% of steps. ERPO's fix is a small addition to the rollout loop, not a new loss function. Tested on Qwen2.5 math reasoning; consistent improvement over vanilla GRPO.
- **Integration effort:** Low. Add a `task_id → (history, temperature)` dict inside `MatchRunner`. Before each `solver.solve_rollouts`, look up the temperature override for the task. Update history from the verify results.
- **Decision:** **Adopt in Phase 1.5.** Not safe to hot-patch into the running v2 process — wait for this run to finish, evaluate, then ship ERPO as `src/si/erpo.py` and re-run.
- **Reference:** https://hf.co/papers/2511.04800

### GRESO — GRPO with Efficient Selective Rollout (arXiv:2506.02177, Jun 2025)

- **What:** Pre-rollout filter that predicts and skips uninformative prompts using reward dynamics. Prompts uninformative in one epoch tend to stay uninformative. 2.0–2.4× wall-clock speedup on rollout with no accuracy loss.
- **Why it matters to SI:** Phase 1-v2 rollout takes ~4 min/gen at 32 prompts/type × 8 MC; 50 gens ≈ 3.3 h just in rollout. GRESO would cut this meaningfully.
- **Integration effort:** Medium. Needs per-prompt reward dynamics history (compatible with ERPO's tracker). Filter runs before `MatchRunner._play`.
- **Decision:** **Adopt in Phase 1.5 or Phase 2.** Bundles well with ERPO (shares the history tracker). Skip for the current v2 run.

### Scaf-GRPO (arXiv:2510.19807, Oct 2025)

- **What:** Tackles the "learning cliff" where models can't solve problems far above current ability → zero reward → no signal. Progressive scaffolding: inject tiered in-prompt hints (from abstract to concrete) only when the model plateaus. +44.3% on AIME24 with Qwen2.5-Math-7B.
- **Why it matters:** AZR's MC proposer reward is supposed to prevent learning cliffs by targeting pass_rate≈0.5, but on the untrained base Gemma 4 E4B our pass@1 was 1/6 ≈ 17% — the curriculum may never land in the sweet spot without help.
- **Integration effort:** Medium-high. Requires distinguishing "stagnated" tasks and generating graded hints. Our proposer already emits (P, I) pairs; Scaf-GRPO would need hint templates added to the solver prompts.
- **Decision:** **Defer.** ERPO is a simpler first attack on the same underlying problem.

### ExPO, DUMP, MEML-GRPO, GRPO-CARE, GRPO-Verif, Training-Free GRPO

- Each has merit but addresses problems we haven't hit yet (multi-expert sampling, multi-domain curricula, verification self-training, API-only deployment). Parked until Phase 2+.
- **Decision:** **Skip for Phase 1/1.5.**

### Summary of scan #2

Two clear adoption candidates for **Phase 1.5**: ERPO (direct fix for zero-variance groups) + GRESO (throughput). Both share a per-prompt history tracker so they bundle naturally. Estimate ~1 day of work for both. Do NOT hot-patch into the live Phase 1-v2 process — reproducibility first.

---

## 2026-04-24 04:00 UTC — scan #3 (hourly monitor, gen 5/50 train-start)

Healthy run: 5 gens in ~68 min, per-gen 19 min (3 min rollout + 16 min train). First trained anchor (gen_0005) lands in ~15 min.

### SD-Zero — Self-Distillation Zero (arXiv:2604.12002, 13 Apr 2026) 🎯 TOP HIT

- **What:** Trains one model to play two roles — Generator (produces initial response) and Reviser (conditions on generator's response + its binary reward, produces improved response). Then on-policy self-distillation distills the reviser into the generator. Turns sparse binary rewards into dense token-level self-supervision. **>10 % gain over base**, outperforms RFT / GRPO / SDFT on math + code at same sample budget (Qwen3-4B-Instruct, Olmo-3-7B-Instruct).
- **Why it matters to SI — directly:** This is the principled fix for the problem we've been fighting. AZR's binary verifier reward is exactly the sparse-reward setting SD-Zero was designed for. Instead of playing tricks with temperature (ERPO) or pre-filtering prompts (GRESO), SD-Zero converts the binary into dense token-level supervision — no gradient starvation. The reviser can localize which tokens to revise given the reward, then teach that back to the generator.
- **Integration effort:** Medium-high. Needs a new role prompt ("given this response and reward, produce an improved response"), a second forward pass per trained step, and on-policy distillation loss replacing part of the GRPO loss. Probably a `src/si/reviser.py` + `src/si/trainer_sdzero.py`. Still uses Unsloth + TRL 1.2 infra.
- **Decision:** **Adopt as Phase 1.5 alternative trainer.** Compare SD-Zero directly against GRPO+ERPO on the same anchor. The paper's results suggest SD-Zero could give us our +5 pp target alone. Highest-priority research integration now.
- **Reference:** https://hf.co/papers/2604.12002

### PSV — Propose, Solve, Verify (arXiv:2512.18160, 20 Dec 2025)

- **What:** Same framework shape as AZR (proposer + solver), but uses **formal verification** (Verus) instead of Python executor. 9.6 × pass@1 improvement over inference-only / expert-iteration baselines. Ablations identify formal verification and **difficulty-aware proposal** as essential ingredients.
- **Why it matters:** Validates our MC-difficulty proposer reward design — the paper explicitly names difficulty-aware proposal as necessary for self-play to succeed.
- **Integration effort:** Formal verification substitute is out of scope for Python self-play (we already have a cheap executor). Nothing new to adopt directly.
- **Decision:** **Reference / cite.** Their "difficulty-aware proposal" result is a good ammunition for why AZR's MC reward matters.

### ReflexiCoder (arXiv:2603.05863, Mar 2026)

- **What:** RL-zero framework that internalizes generate → reflect → correct into weights. ReflexiCoder-8B gets **87.20 % on HumanEval+**, **94.51 % on HumanEval**, 81.80 % MBPP+, rivaling GPT-5.1.
- **Why it matters:** Their 87.20 % HumanEval+ matches our base Gemma 4 E4B exactly — suggests that number is near-ceiling for strong small-model zero-RL on HumanEval+. Our +5 pp target may be harder than it seemed. Secondary benchmarks (MBPP+, BigCodeBench, LiveCodeBench) would show larger deltas.
- **Integration effort:** Direct port is research-scale; their granular reward functions are code-specific.
- **Decision:** **Defer.** Lesson: add MBPP+ or LiveCodeBench as a secondary anchor so we're not bottlenecked on HumanEval+ ceiling.

### Socratic-Zero (arXiv:2509.24726, Sep 2025)

- **What:** Teacher / Solver / Generator co-evolution from 100 seed questions. +20.2 pp over data-synthesis methods on seven math benchmarks.
- **Why it matters:** Adds a third role (Generator = distills Teacher's question-design). Our AZR variant has only Proposer + Solver; Socratic-Zero's Generator could be our Phase 2+ Elo selection mechanism.
- **Decision:** **Consider for Phase 2.** Skip for now.

### SimpleRL-Zoo (arXiv:2503.18892, Mar 2025)

- **What:** Zero-RL across 10 diverse base models. Key design lessons: format-reward + query-difficulty control.
- **Why it matters:** Validates our shaping-reward approach (`make_format_reward_fn`). Also notes "aha moment" varies across base models — Gemma 4 may or may not exhibit it.
- **Decision:** **Reference.** Read before Phase 2 scaling.

### Summary of scan #3

Strongest hit: **SD-Zero (Apr 13, 2026)** — published 10 days ago. Self-distilled reviser turns binary reward into dense supervision; addresses our core problem at the mechanism level instead of via temperature tricks. Promote to top adoption candidate above ERPO/GRESO. Caveat: untested on 4-bit QLoRA + Unsloth stack; integration is real work. Adds `src/si/trainer_sdzero.py` and a reviser prompt path.

Also note: ReflexiCoder-8B hit 87.20% HumanEval+ (our base!). Our +5 pp target on that single anchor may be the wrong KPI. Recommend adding MBPP+ or LiveCodeBench v6 as secondary anchor before Phase 1-v3.

---

## 2026-04-24 05:00 UTC — scan #4 (hourly check; gen 7 train-start)

Run status: gen 5 anchor **failed** due to nested adapter-path bug (fixed in commit 766adb6; takes effect at gen 10 anchor). Gen 6 completed in 29 min; gen 7 train running. Run still healthy.

Queried for generic "self-improvement efficient" — mostly resurfacing older papers. Two worth noting:

### IMM — Iterative Model Merging (arXiv:2503.02103, Mar 2025)

- **What:** Self-improved models exhibit "superficial self-improved reasoners" — ID accuracy up but OOD down, because weight updates concentrate in less reasoning-critical layers. IMM merges original + self-improved weights to preserve generalization.
- **Why it matters to SI:** Directly relevant to our **anchor-revert rule**. Our current revert is a hard rollback. IMM gives a softer recovery: merge back toward base when the anchor regresses, preserving ID gains while recovering OOD. Could replace the binary commit/revert with a continuous "drift control."
- **Integration effort:** Low — a merge function called when `should_revert` fires. Fits neatly into `anchor.py`.
- **Decision:** **Defer, consider for Phase 2.** Revert rule is a sharper defense for now; IMM is a sophistication worth adding once Phase 1 shows any genuine gain to preserve.

### S²R — Teach Self-Verify + Self-Correct via RL (arXiv:2502.12853, Feb 2025)

- **What:** SFT on 3.1k curated self-verification samples, then outcome + process RL. Qwen2.5-math-7B: 51.0% → 81.6% on math benchmarks.
- **Why it matters:** Related to SD-Zero's dense-supervision angle; specifically targets verification + correction. Our AZR has a verifier (sandbox) but no *self*-verification signal.
- **Decision:** **Skip for now.** SD-Zero covers the same territory more directly and is newer.

### Generation-Verification Gap (arXiv:2412.02674, Dec 2024)

- **What:** Formalizes when self-improvement is possible. Scales monotonically with pretraining FLOPs.
- **Decision:** **Cite.** Theoretical backing for why we expect Gemma 4 E4B (a relatively small pretrain) to plateau quickly — matches our observation.

### Summary of scan #4

No new top-tier adoption candidate. SD-Zero remains #1. IMM is a promising replacement for hard anchor revert; park for Phase 2.

---

## 2026-04-24 06:12 UTC — scan #5 (gen 10 anchor REGRESSED to 85.98%)

Trigger: first trained anchor came in at 141/164 = 85.98% vs base 143/164 = 87.20% (drop of 1.22 pp, within the 2% tolerance but directionally wrong). Queried for GRPO collapse / small-group-size / reward-noise literature specifically.

**Mother lode.** Multiple recent papers describe our exact failure mode:

### EBPO — Empirical Bayes Policy Optimization (arXiv:2602.05165, 5 Feb 2026) 🎯 NEW TOP HIT

- **What:** Directly names the failure modes we're seeing: "GRPO suffers from high estimator variance under computational constraints (**small group sizes**) and vanishing gradient signals in saturated failure regimes where all responses yield identical zero rewards." Proposes a shrinkage estimator (Welford's online algorithm) that borrows strength from the policy's **global** accumulated statistics when the local group is degenerate. **Theoretically guarantees non-vanishing penalty signals in failure scenarios.**
- **Why it matters to SI:** We have `num_generations=2` (small group), and `frac_reward_zero_std=1` in nearly every step (saturated failure). EBPO was literally designed for this. Benchmarks: outperforms GRPO across AIME + OlympiadBench, specifically strong at small group sizes.
- **Integration effort:** Low-medium. It's a loss-function change, no new forward passes. Patches TRL's GRPO advantage computation with a shrinkage term. Shares infra with current trainer.
- **Decision:** **Promote to #1 Phase 1.5 candidate, above SD-Zero.** Lighter-weight and targets our exact constraint. SD-Zero stays as #2.
- **Reference:** https://hf.co/papers/2602.05165

### LLD / LLDS — Lazy Likelihood Displacement (arXiv:2512.04220, 3 Dec 2025)

- **What:** Characterizes GRPO collapse as **Lazy Likelihood Displacement**: systematic reduction in likelihood of both correct and incorrect responses. Three-phase trajectory: "early stagnation → steady decay → accelerated collapse." Proposes LLDS, a lightweight likelihood-preserving regularization that activates only when a trajectory's likelihood decreases and touches only the responsible tokens. **+37.8 % on Qwen2.5-3B, +32 % on Qwen2.5-7B.**
- **Why it matters:** Our gen 0 → 10 trajectory (87.20% → 85.98%) IS early stagnation. LLDS is a preemptive defense against the decay phase.
- **Integration effort:** Low. Regularization term on top of existing GRPO loss. Complements EBPO (they fix different aspects).
- **Decision:** **Adopt alongside EBPO in Phase 1.5.** Bundle.

### Prompt Augmentation for GRPO (arXiv:2602.03190, 3 Feb 2026)

- **What:** Fixed reasoning prompts during training cause entropy collapse. Augmenting with diverse templates/formats prevents collapse, stable scaling to long training without KL regularization.
- **Why it matters:** Our system prompt is fixed (`_SYSTEM_SOLVER` / `_SYSTEM_PROPOSER`). Paper argues this contributes to the very instability we're seeing.
- **Integration effort:** Very low. Randomize the system prompt among a small pool of templates for each rollout.
- **Decision:** **Adopt in Phase 1.5 — cheap win.** Add a template pool to `prompts.py`.

### S-GRPO, GMPO, GTPO, Dr.DPO, NoisyGRPO

- Each proposes a different correction to GRPO's instability: noise-aware advantage weights (S-GRPO), geometric-mean policy loss (GMPO), trajectory-level conflict-token protection (GTPO), distributional robustness (Dr.DPO), noise injection + Bayesian estimation (NoisyGRPO).
- **Decision:** **Skip for Phase 1.5.** EBPO + LLDS + prompt augmentation cover the same failure modes with lower integration cost. Reconsider if EBPO doesn't fix things.

### Summary of scan #5 — rankings after gen 10 regression

Adoption priority for **Phase 1.5** (after the v2 run concludes):

| # | Method | Effort | Targets | Expected gain vs v2 |
|---|---|---|---|---|
| 1 | **EBPO** + Welford shrinkage | Low-med | Small-group + zero-std degeneracy | High — direct fix |
| 2 | **LLDS** likelihood regularizer | Low | Prevents early-stagnation → decay | Medium — bundle with EBPO |
| 3 | **Prompt augmentation** (template pool) | Very low | Entropy collapse from fixed system prompt | Small but cheap |
| 4 | **SD-Zero** reviser | High | Binary → dense supervision | High if ceiling is the real problem |
| 5 | **MBPP+ secondary anchor** | Low | HumanEval+ ceiling at 8B ~87% | Unblocks measurement |

**Plan:** Let v2 finish OR stop at gen 15 if still regressing. Then Phase 1.5 = EBPO + LLDS + prompt aug (bundled into one release) → measure → if still insufficient, add SD-Zero.

---

## 2026-04-24 08:35 UTC — scan #7 (deeper sweep after v2 stopped)

User requested a thorough review before committing to SSD. Searched four more angles: expert iteration, small-model coding recipes, April 2026 dense-supervision, zero-variance GRPO specifically. Found **two papers that target our exact failure mode more directly than any previous candidate.**

### RL-ZVP — No Prompt Left Behind (arXiv:2509.21880, Sep 2025) 🎯 EXACT-MATCH

- **What (verbatim from abstract):** "current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward — so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. … RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics."
- **Why it matters to SI:** This is **literally our problem** named in the title and targeted by the method. Our `frac_reward_zero_std=1` in nearly every step = zero-variance prompts being silently discarded. RL-ZVP extracts the signal instead.
- **Results:** **+8.61 pp accuracy and +7.77 pp pass rate over GRPO** across six math reasoning benchmarks. Beats baselines that filter zero-variance prompts (which is effectively what we're doing by getting no gradient from them).
- **Integration effort:** Low. It's an alternative advantage computation inside the GRPO loss — patches `trainer_unsloth.py`. No new forward passes, no new prompt formats.
- **Decision:** **New #1 Phase 1.5 candidate.** More targeted than EBPO (which uses a global shrinkage prior; RL-ZVP uses token-level characteristics of the zero-variance group directly).
- **Reference:** https://hf.co/papers/2509.21880

### NSR — Surprising Effectiveness of Negative Reinforcement (arXiv:2506.01347, Jun 2025)

- **What:** Decomposes GRPO learning signal into Positive (reward correct) and Negative (penalize incorrect). Key finding: **training with only negative samples — no reinforcement of correct responses — often matches or exceeds GRPO.** Reinforcing only correct improves pass@1 but **degrades pass@k due to reduced diversity.**
- **Why it matters:** Our current reward = `{1.0 if verified else 0.0}` + `+0.1` format bonus. The 1.0 branch positively reinforces correct answers, which per this paper **reduces diversity and hurts pass@k**. The 3-improved / 5-regressed pattern we saw (narrowing of what the model "can do") is exactly this diversity collapse.
- **Integration effort:** Tiny. Change reward to `{0.0, -1.0}` instead of `{1.0, 0.0}` and reinterpret.
- **Decision:** **Adopt as a reward-shape ablation in Phase 1.5.** Simple A/B.
- **Reference:** https://hf.co/papers/2506.01347

### PACS — Implicit Actor-Critic via Supervised Learning (arXiv:2509.02522, Sep 2025)

- **What:** Reformulates RLVR as supervised learning over a score function parameterized by the policy, optimized with cross-entropy. Implicit actor-critic. Stable where GRPO collapses.
- **Results:** +13 pp over PPO, +14 pp over GRPO on AIME 2025 pass@256.
- **Decision:** **#3 candidate, deferred.** More invasive than RL-ZVP; same-order integration to SD-Zero.

### F-GRPO — Difficulty-Aware Advantage Scaling (arXiv:2602.06717, 6 Feb 2026)

- **What:** Small-groups-miss-rare-correct problem explicitly identified: "Smaller groups often miss rare-correct trajectories while still containing mixed rewards, concentrating probability on common solutions." Focal-loss-style downweighting of high-success prompts. +6.2 pp pass@256 on Qwen2.5-7B with GRPO.
- **Why it matters:** Validates that our `num_generations=2` is a real problem. F-GRPO's fix is orthogonal to RL-ZVP (theirs scales advantage by difficulty; RL-ZVP extracts signal from degenerate groups). Bundle-friendly.
- **Decision:** **Adopt alongside RL-ZVP.** Complementary.

### OpenCodeReasoning insight (arXiv:2504.01943, Apr 2025)

- **Key finding:** "execution filtering negatively affected benchmark accuracy, leading us to prioritize instruction diversity over solution correctness." Qwen3-style distilled SFT model reaches 61.8 % on LiveCodeBench — **without RL**, purely SFT.
- **Why it matters to SSD plan:** If we go the SSD route, the paper explicitly says **don't filter by verifier pass** — diversity matters more than correctness for generalization. This flips my original SSD design.
- **Decision:** **Adjust SSD spec if we use it.** Keep even failing samples (or at least sample diversely before filtering).

### NExt — Nonlinear LoRA Trajectory Extrapolation (arXiv:2604.11446, 13 Apr 2026)

- **What:** Predicts LoRA parameter trajectory from rank-1 subspace; skips intermediate steps. 37.5 % compute reduction.
- **Decision:** **Park for Phase 2.** Compute-saving, not capability-increasing.

### Multiple Ticket Hypothesis (arXiv:2602.01599, Feb 2026)

- **What:** Training only a randomly selected 1 % of parameters matches full-RLVR finetuning. Arbitrary sparse masks succeed.
- **Why it matters:** Our LoRA rank 32 on text tower = 73 M trainable params = ~0.91 % of model. We're already at that sparsity. Suggests rank 16 or 8 would perform similarly; worth an A/B if we're constrained.
- **Decision:** **Possible Phase 1.5 ablation** — try rank 8 for speed.

### Revised Phase 1.5 ranking (final)

Given the new findings, **two candidates are better than SSD for our exact failure mode**:

| # | Method | Targets | Integration cost | Expected gain |
|---|---|---|---|---|
| 1 | **RL-ZVP** (arXiv:2509.21880) | Zero-variance groups directly | Low — GRPO advantage patch | +8.6 pp reported |
| 2 | **F-GRPO** focal scaling | Small-group rare-correct misses | Low — bundle with RL-ZVP | +6.2 pp pass@256 |
| 3 | **Prompt augmentation** (2602.03190) | Fixed-prompt entropy collapse | Very low | Small but cheap |
| 4 | **NSR** reward shape ablation (2506.01347) | Diversity collapse from positive reinforcement | Tiny — flip reward sign | A/B test |
| 5 | **SSD** self-distillation bootstrap (2604.01193) | Plateau breakthrough via SFT | Medium — new trainer | +12.9 pp on LiveCodeBench |
| 6 | **EBPO** Welford shrinkage (2602.05165) | Global prior for degenerate groups | Medium | Overlapping with RL-ZVP |
| 7 | **SD-Zero** reviser (2604.12002) | Binary → dense supervision | High | +10 pp baseline-relative |

### Recommendation

**Start with RL-ZVP instead of SSD.** Three reasons:

1. **More targeted.** RL-ZVP's paper explicitly targets the exact failure mode we measured (zero-variance prompts causing no gradient). SSD is more general.
2. **Lower integration cost.** RL-ZVP is a loss-function patch inside our existing Unsloth + TRL 1.2 pipeline. SSD needs a new trainer module.
3. **Keeps AZR paradigm intact.** Doesn't force us away from self-play. If RL-ZVP fails, SSD is still available as a fallback. If we do SSD first we lose the ability to attribute the gain to any specific RL fix.

Bundle RL-ZVP with F-GRPO focal scaling + prompt augmentation (same trainer patch) for Phase 1.5.

If that doesn't hit +5 pp on HumanEval+ (and especially if HumanEval+ ceiling at ~87 % for 8B is the real bottleneck), then either layer SSD on top or switch to a harder anchor (MBPP+ / LiveCodeBench).

---

## 2026-04-24 07:11 UTC — scan #6 (gen 14 train-start; gen 15 anchor ~25 min out)

No change in trajectory since scan #5 (still just base=87.20% and gen_10=85.98%). Pace accelerated — gens 11-13 completed in ~10 min each thanks to Unsloth caching. Targeted search for AZR-adjacent work.

### SSD — Embarrassingly Simple Self-Distillation (arXiv:2604.01193, 1 Apr 2026) 🎯

- **What:** Sample N candidate solutions from the model with tuned temperature + truncation, then run plain supervised fine-tuning on those samples. No verifier, no teacher, no RL. Qwen3-30B-Instruct: **42.4 % → 55.3 %** on LiveCodeBench v6 (+12.9 pp). Gains concentrate on harder problems. Works on Qwen + Llama at 4B/8B/30B.
- **Why it matters to SI:** A complementary bootstrap step we could run **before** (or interleaved with) AZR. If the raw model can improve itself by simple self-distillation, we don't need GRPO to do all the work — that breaks the "GRPO-or-nothing" assumption I've been operating under. SSD by itself could give us the +5 pp target on HumanEval+ (the paper shows even larger gains on LiveCodeBench).
- **Integration effort:** Tiny. A `si ssd-bootstrap` subcommand that uses `GemmaLLM` to generate N candidates per HumanEval+-style prompt, filters by the verifier if available, then SFT via the existing Unsloth path.
- **Decision:** **Adopt as Phase 1.5 opener.** Run SSD first; it's a strictly additive bootstrap that doesn't fight GRPO. If SSD alone hits the +5 pp target, Phase 1 succeeds and we skip EBPO. If not, layer EBPO/LLDS on top.
- **Reference:** https://hf.co/papers/2604.01193

### R-Zero (arXiv:2508.05004, Aug 2025)

- **What:** Challenger/Solver co-evolution from zero data. Two independent models (not role-switched). Qwen3-4B-Base: +6.49 pp math, +7.54 pp general reasoning.
- **Why it matters:** Architectural variant of AZR — separate models diverge more. Could inform Phase 2 multi-branch design where each island holds a specialized challenger-solver pair.
- **Decision:** **Defer to Phase 2.** Our current single-model role switching (per AZR's §3.2) is simpler to debug first.

### Code-A1 (arXiv:2603.15611, Mar 2026) + Sol-Ver (arXiv:2502.14948, Feb 2025)

- **What:** Adversarial code-vs-test generation. Code LLM rewarded for passing tests; Test LLM rewarded for exposing bugs. Matches/exceeds human-test-trained models on Qwen2.5-Coder.
- **Why it matters:** Induction tasks in AZR need test generation. Code-A1's adversarial separation is a clean template for that.
- **Decision:** **Park for Phase 2** (when we add induction). Won't help the current regression.

### Summary of scan #6 — update to the plan

New ordering for **Phase 1.5**:

1. **SSD self-distillation bootstrap** — simplest, gain is additive, no GRPO coupling. Go first.
2. **EBPO** Welford shrinkage — fix GRPO's vanishing-gradient problem in the residual (post-SSD) training.
3. **LLDS** likelihood regularizer — bundle with EBPO.
4. **Prompt augmentation** — cheap add-on.
5. **SD-Zero** (Apr 13) — bigger lift, fallback if the above don't hit target.

Rationale: SSD is decoupled from everything else and has demonstrated double-digit gains on LiveCodeBench. If SSD alone gives us +5 pp on HumanEval+, Phase 1 is done. If it doesn't, we still have the trained adapter as a better starting point for GRPO (EBPO etc.).

---

## 2026-04-24 08:13 UTC — v2 run stopped (gen 18 train-done; decision trigger)

gen_15 anchor = 141/164 = 85.98% — **identical per-problem to gen_10**. Five generations of GRPO produced zero net change:
```
gen_0000  143/164 = 87.20%  (base)
gen_0010  141/164 = 85.98%  (-1.22pp)
gen_0015  141/164 = 85.98%  (plateau confirmed, per-problem == gen_10)
```

Diff vs base:
- Regressed (base pass → gen_10 fail): HumanEval/65, 84, 86, 93, 137
- Improved (base fail → gen_10 pass): HumanEval/91, 126, 140
- Net -2.

The LoRA converged to a fixed point at temp=0.2 (deterministic) distinct from base but slightly worse on pass@1. Further GRPO steps do not move it — matches scan #5's "Lazy Likelihood Displacement: early stagnation" phase precisely.

**Stopped `phase1_v2_20260423_2250` at gen 18** (killed PID 634686 + reaped vLLM workers 692994, 692995). Saved ~6 h of GPU-idle compute we would have spent on more of the same plateau.

### Phase 1.5 plan (pending user approval)

New direction — SSD self-distillation bootstrap instead of more GRPO first:

1. **SSD bootstrap** (arXiv:2604.01193) — 1-2 days impl. Generate N candidates per held-out `si`-style prompt, SFT on the verifier-passing ones. Additive; doesn't fight GRPO; doesn't require AZR proposer. Target: +3–5 pp on HumanEval+ alone.
2. **Anchor check** — if +5 pp met, Phase 1 done. If not, add:
3. **EBPO** (arXiv:2602.05165) — Welford shrinkage in GRPO to fix `frac_reward_zero_std=1`. Run AZR-GRPO on top of the SSD-initialized adapter.
4. **LLDS + prompt augmentation** — bundle with EBPO.
5. **SD-Zero** if still short.

Why this order over original EBPO-first plan: gen 10/15 plateau at 85.98% with identical per-problem pattern shows the problem isn't just gradient variance — the model is stuck in a local optimum where binary rewards produce no useful signal. SSD's dense SFT supervision should break the plateau from a different angle before we reintroduce the unstable GRPO loop.

---
