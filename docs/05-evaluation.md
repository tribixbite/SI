# 05. Evaluation — Anchors, Metrics, Detection Rules

## The anchor set

**Primary anchor:** HumanEval+ (100 problems), with test set frozen at EvalPlus commit `d362e933265c3e7e3df8101c930a89c3c470cd9f`.

**Secondary anchor (Tier 2+):** LiveCodeBench v6 held-out subset, 50 problems, frozen at a specific repo SHA captured at project start and stored in `benchmarks/livecodebench.sha`.

**Meta-anchor:** a third, smaller set (20 problems) drawn from MBPP+ that is **never** evaluated during training. Used only for final reporting when a run is considered complete. Protects against anchor-memorization — even if the primary anchor somehow leaks into training (it shouldn't, by the trust model in §3 of `03-stack.md`), the meta-anchor remains clean.

### Why these and not others

- HumanEval+ is the community benchmark with the most robust test suites (averaging ~30 tests per problem, vs ~5 in stock HumanEval). Overfitting is harder.
- LiveCodeBench is the 2024+ frontier benchmark, less contaminated in Gemma 4's pretrain data.
- MBPP+ for meta-anchor because it's broadly correlated with the others but distinct enough to detect narrow overfitting.

### What's forbidden as anchor

- Anything in the model's training data (we can't confirm this for Gemma 4, but we can avoid obvious contaminants — Codeforces problems pre-2024, LeetCode top problems, AoC challenges).
- Anything solver-editable. The anchor set is read-only on disk; hash-verified at startup; held in memory for the run.
- Anything the proposer might have seen. The proposer starts only from AZR's format seeds, not from the anchor set. We do not let the anchor set bleed into prompts.

## The primary metrics

Per generation, logged to W&B and to `$SI_ROOT/runs/<run_id>/metrics.jsonl`:

- **`aggregate_pass_rate`**: mean pass@1 across anchor, population-averaged.
- **`top_branch_pass_rate`**: pass@1 of the branch with highest current Elo.
- **`elo_range`**: top Elo minus bottom Elo. Widening range = healthy selection; collapsing range = convergence.
- **`proposer_mc_difficulty`**: distribution of AZR proposer's own difficulty score. Healthy: centered near 0.5, spread wide. Unhealthy: bimodal (all trivial or all impossible) or spike at 0 (proposer collapse).
- **`verification_rate`**: fraction of proposer outputs that yield a valid task at all (executor doesn't error on the reference). Should stabilize above 0.8. Declining = proposer degradation.
- **`sandbox_timeouts_per_gen`**: count of verifier timeouts. Rising sharply = solver outputs exploiting timeout attack (hangs) or proposer generating pathological programs.
- **`lora_weight_drift`**: Frobenius norm of LoRA delta vs previous generation, per branch. Collapsing to zero = dead branch. Diverging unboundedly = instability, consider LR decrease.

## Detection rules — failure modes and tripwires

### Mutual hallucination
**Symptom:** `aggregate_pass_rate` (anchor) flat or declining while `solver/pass_rate` on self-generated tasks rises. Solver and proposer have agreed on meaningless tasks.
**Detection:** automatic — the anchor reversion rule catches this. Required response: identify which generation started the drift (find the last gen where anchor improved); revert; tighten proposer's MC difficulty reward (penalize tasks with pass_rate > 0.9 or < 0.1 more aggressively).

### Specialization cascade
**Symptom:** `elo_range` collapsing rapidly; all branches score similarly; no signal for selection.
**Detection:** when `elo_range < 50` for 5 consecutive generations.
**Response:** increase migration rate (every 3 gens instead of 5); perturb low-Elo branches more aggressively (Gaussian σ=0.01 instead of 0.001); consider spawning a "wildcard" branch initialized from base model every 20 gens.

### Verifier exploit
**Symptom:** one branch achieves anomalously high pass rate; its outputs look structurally unusual.
**Detection:** when a branch's pass rate exceeds population mean + 3σ for 2+ generations; auto-trigger a diff of its outputs against others' on the same problems.
**Response:** suspected problem entries are blacklisted; verifier hardening pass (add test cases, add runtime checks); suspected branch's LoRA delta is discarded, branch re-seeded from median branch.

### Catastrophic forgetting
**Symptom:** anchor score drops after long stretches of self-play even without any obvious distribution shift.
**Detection:** anchor reversion fires without population score change (self-play scores stayed level but anchor dropped).
**Response:** add KL penalty against base model to GRPO (if not already present); shorten GRPO step horizon; reduce LR.

### Sandbox escape (critical)
**Symptom:** any evidence of filesystem access outside sandbox, network activity, or process persistence across verifier calls.
**Detection:** periodic external audit — check sandbox-external filesystem timestamps, network logs, process list.
**Response:** **halt immediately.** Do not attempt to patch the running system. Examine the escaped output, identify the mechanism, upgrade sandbox (Firecracker with seccomp in restrictive mode), re-start from last clean checkpoint. This is the one failure mode where "revert and continue" is not acceptable.

### Proposer collapse
**Symptom:** proposer output diversity (token-level entropy over recent proposals) dropping; all proposals become variants of the same template.
**Detection:** `proposer_output_entropy` < 3.0 bits/token over 5-gen window (baseline is typically 5–7).
**Response:** increase proposer temperature; refresh proposer from frozen base (keep solver LoRAs; reset proposer-side adapter to base); audit recent proposals for degenerate patterns.

### Numerical instability
**Symptom:** loss NaN, LoRA weights → NaN, verl errors during backward pass.
**Detection:** any NaN in LoRA weight Frobenius norm.
**Response:** reduce LR 10x; enable gradient clipping (max_norm=1.0 in verl config); check for mixed-precision issues (try pure bfloat16 instead of fp16).

## Reporting

At end of a run:
- Final anchor score (primary, secondary, meta).
- Delta from base Gemma 4 on all three anchors.
- Training FLOPs (verl logs; approximate via `active_params × tokens_trained × 6`).
- Total problems generated (proposer output count).
- Fraction of anchor improvement attributable to each phase (requires ablation runs).
- Stability: max anchor drawdown during run; number of reversions fired.

Publishable negative result: if the stack plateaus at generation N, report N, the metric at plateau, and the dominant detected failure mode. "Self-improvement plateaus at generation N due to specialization cascade" is a real contribution; pretending otherwise is not.
