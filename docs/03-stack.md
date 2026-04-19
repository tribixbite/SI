# 03. The Stack — How the Four Components Compose

This is the design document. What talks to what, where the boundaries are, which paper each piece comes from.

## The layer diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     ANCHOR LAYER (frozen)                        │
│   HumanEval+ (100 problems, locked)                              │
│   LiveCodeBench v6 held-out (50 problems, locked)                │
│   Reversion rule: population-aggregate score must not decrease   │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                    (evaluated every K=10 generations)
                              │
┌──────────────────────────────────────────────────────────────────┐
│                     OUTER LOOP — ISLANDS                         │
│            (OpenEvolve pattern — ring topology)                  │
│                                                                  │
│   Island 0: branches 0-2     Island 1: branches 3-5              │
│   Island 2: branches 6-8     Island 3: branches 9-11             │
│                                                                  │
│   Every 5 generations: top-1 program per island migrates         │
│   clockwise. Experience buffers merge. Proposer priors swap.     │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌──────────────────────────────────────────────────────────────────┐
│                  MIDDLE LOOP — ELO TOURNAMENT                    │
│                  (RoboPhD pattern — K=32)                        │
│                                                                  │
│   Match: same AZR problem, two branches solve, verifier decides  │
│   Elo update after each match. Bottom quartile replaced every    │
│   generation by mutated top-quartile LoRAs.                      │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌──────────────────────────────────────────────────────────────────┐
│                    INNER LOOP — AZR SELF-PLAY                    │
│               (Zhao et al. 2025, arXiv:2505.03335)               │
│                                                                  │
│   Single Gemma 4 instance, role-switched via system prompt.      │
│                                                                  │
│   Proposer:                        Solver:                       │
│     emits (P,I,O) triples            takes 2 of 3, outputs       │
│     reward = MC-rollout difficulty   the 3rd                     │
│                                      reward = verifier binary    │
│                                                                  │
│   Three task types: deduction (P,I→O), abduction (P,O→I),        │
│   induction (I/O pairs → P).                                     │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌──────────────────────────────────────────────────────────────────┐
│               INFERENCE LAYER — IN-PLACE TTT (Phase 2)           │
│               (Feng et al. 2026, arXiv:2604.06169)               │
│                                                                  │
│   Final MLP projection matrix becomes fast-weight.               │
│   Adapts to current context via NTP objective.                   │
│   Reverts between problems (fresh fast-weights each task).       │
│   Orthogonal to LoRA branch specialization.                      │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌──────────────────────────────────────────────────────────────────┐
│                   BASE — GEMMA 4 (FROZEN)                        │
│               26B-A4B MoE (Tier 2) or 31B Dense (Tier 3)         │
│                          Apache 2.0                              │
└──────────────────────────────────────────────────────────────────┘
```

## Time-scale separation

| Layer | Timescale | What changes |
|---|---|---|
| In-Place TTT | per problem (seconds) | final-MLP projection only, reverted after each problem |
| AZR self-play | per rollout (seconds) | nothing persists, but experience buffers grow |
| GRPO on LoRA | per generation (minutes) | LoRA adapter weights |
| Elo selection | per generation (minutes) | branch identity (PBT replacement) |
| Island migration | per 5 generations (~hour) | cross-branch experience |
| Anchor check | per 10 generations (~2h) | commit / revert whole generation |

These separate cleanly. No gradient flows between layers. Everything below the Anchor layer is reversible per-generation if anchor eval regresses.

## What each component contributes, precisely

### From AZR

- **Proposer task-type taxonomy:** deduction, abduction, induction. All three tasks reduce to `(program, input, output)` triples; two fields given, one unknown. Python executor verifies exactly.
- **Proposer reward via Monte Carlo rollout:** the proposer is rewarded for proposing tasks the current solver sometimes solves and sometimes fails. This naturally produces a curriculum at the edge of solver ability. Exact formula: `reward_proposer = -|0.5 - pass_rate_k|` where `pass_rate_k` is fraction of k=8 independent solver attempts that pass. Tasks trivially easy or impossible score zero.
- **Solver reward:** binary. Passes verifier → 1. Fails → 0. No partial credit.
- **Seed data:** small bootstrap set (hundreds of examples) to start the proposer. Generated by prompting the base Gemma 4 per AZR's `scripts/seeding/` template. Not training data — only initial format examples.
- **Training algorithm:** GRPO on veRL. Direct inheritance from AZR codebase.

### From RoboPhD

- **Elo selection mechanism:** K=32 rating updates after head-to-head matches. Asymmetric update for upsets (low-rated beating high-rated moves both scores more).
- **Validation-free principle:** we do not reserve a separate validation set for ranking. Rankings come from head-to-head matches on training problems. The anchor is used only for go/no-go reversion decisions, not for ranking.
- **Self-instrumenting agents:** each LoRA branch maintains its own rollout trace log. Over generations, branches that produce more informative traces (for their own future-generation successors via migration) win more Elo matches implicitly, because their mutations are better-informed.
- **What we don't adopt from RoboPhD:** the "evolve via LLM writing Python code" mechanism. Our branches evolve via GRPO gradient updates, not via prompt mutation. RoboPhD's Elo mechanism is orthogonal to the substrate it ranks; we're porting just the ranking.

### From OpenEvolve

- **Island topology:** N branches partitioned into M islands (M=4 at Tier 2, with 3 branches each). Ring-topology migration every 5 generations. Top-1 branch per island exports its LoRA delta and top-K experience buffer entries to the next island clockwise.
- **Migration mechanics:** the migrant's LoRA delta is applied as a **starting point** for the receiving island's next-generation mutation, not merged directly. Experience buffer entries (proposed tasks + solutions that passed) enter the receiving island's replay buffer with priority weight 2× native entries for 1 generation, then normal weight.
- **Why this prevents convergence:** without migration, Elo selection drives every branch toward the current population mean. Migration injects off-distribution gradients; diversity is preserved because islands are rankings-separated (each island runs its own Elo rankings, with inter-island matches held only at the anchor check).
- **What we don't adopt from OpenEvolve:** the entire LLM-driven-mutation pipeline. OpenEvolve mutates code; we mutate LoRA weights via gradient steps. The *evolutionary topology* is what we borrow, not the mutation operator.

### From In-Place TTT (Phase 2)

- **Fast-weight layer:** the final Linear projection in each Gemma 4 MLP block (`mlp.down_proj` in HF naming). Treated as fast-weight; base weights frozen.
- **Adaptation objective:** next-token-prediction on the current rollout trajectory. Loss: cross-entropy on the solver's own output tokens, applied chunk-wise every 256 tokens.
- **Reversion:** after each problem, fast-weights reset to the branch's LoRA-modified base. Intent: TTT amortizes the per-problem context, doesn't accumulate across problems (that accumulation is what LoRA + AZR handle at the slow timescale).
- **Implementation note:** the paper's core mechanism is under 100 lines. If upstream code isn't released by MVP time, re-implement from §3 of the paper. Test against a trivial context-adaptation task (next-token prediction on a held-out corpus) before integrating with the self-play loop.

## The integration contract

Each component is called through a narrow interface so it can be swapped or disabled:

```python
# src/si/contracts.py (sketch)

class Proposer(Protocol):
    def propose(self, task_type: TaskType, context: Experience) -> Task: ...

class Solver(Protocol):
    def solve(self, task: Task) -> Solution: ...
    # Solver may use In-Place TTT internally; interface is unchanged.

class Verifier(Protocol):
    def verify(self, task: Task, solution: Solution) -> VerifyResult: ...
    # sandboxed, deterministic, returns (passed: bool, stdout, stderr, walltime_ms)

class Selector(Protocol):
    def rank(self, branches: List[Branch], match_history: MatchLog) -> Ranking: ...

class Migrator(Protocol):
    def migrate(self, islands: List[Island], gen: int) -> List[Island]: ...

class Anchor(Protocol):
    def evaluate(self, population: List[Branch]) -> AnchorResult: ...
    def should_revert(self, prev: AnchorResult, curr: AnchorResult) -> bool: ...
```

Concrete types live in `src/si/contracts.py`. Each of the six interfaces has a default implementation (AZRProposer, GemmaSolver, SandboxVerifier, EloSelector, RingMigrator, HumanEvalPlusAnchor) and can be replaced independently for ablation.

## The trust model

Every untrusted input is scored, never executed as code. Specifically:

- **Proposer output is not code to us, it's data.** The proposer emits strings; the verifier runs them in sandbox. If the proposer learns to output a shell escape, it gets zero reward and the sandbox contains the damage.
- **Solver output is not code to us, it's data.** Same rule. The solver can only "win" by producing output that the verifier runs in sandbox and confirms correct. There is no text path from solver output to loop-control logic.
- **Verifier output flows in one direction only.** Verifier result → reward. Never verifier result → code path. We explicitly refuse the pattern `if verifier_output_contains("TRUSTED"): ...`. All reward signals are binary or bounded numeric; no structured data from the verifier influences control flow.
- **The anchor is immutable during a run.** Loaded once at startup, hash verified, kept in memory. Disk copy never re-read mid-run.

This is the "mutual hallucination" defense at the code level. Even if solver and proposer converge on a shared delusion, the delusion can never modify the runner that evaluates them.
