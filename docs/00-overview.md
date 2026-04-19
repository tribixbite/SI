# 00. Overview

## Thesis

A self-improving agent at home is achievable in 2026 if you stop trying to invent the loop and start assembling published, working loops in the right way.

Four mechanisms have each been shown to work independently in the last twelve months. None have been combined. Each covers a gap the others leave:

| Component | Covers | Gap it leaves |
|---|---|---|
| AZR | zero-data curriculum generation + self-verification | single model, no population dynamics |
| RoboPhD Elo | robust selection under tight eval budgets | no knowledge transfer between agents |
| OpenEvolve islands | population diversity, migration, crossover | prompts/programs only, no weight updates |
| In-Place TTT | per-problem fast-weight adaptation | no long-horizon curriculum |

Combine them and each gap is covered by another component. That is the SI stack.

## The loop at three scales

**Inner loop (seconds-to-minutes).** A single Gemma 4 instance, role-switched, runs AZR's propose→solve→verify cycle. Proposer emits a (program, input) triple or a (program, output) triple; solver tries to deduce/abduce/induce the missing piece; Python executor provides binary reward. This is the core of AZR (Zhao et al. 2025, arXiv [2505.03335](https://arxiv.org/abs/2505.03335)).

**Middle loop (minutes-to-hours).** N solver LoRA branches play an Elo tournament on a shared stream of AZR-generated problems. Each problem is a match; the branch whose solution passes the verifier wins. Elo ratings update after each match with K=32. Bottom-quartile branches are replaced each generation by mutated copies of top-quartile branches. This is RoboPhD's selection mechanism (Borthwick & Ash 2026, arXiv [2604.04347](https://arxiv.org/abs/2604.04347) and [2601.01126](https://arxiv.org/abs/2601.01126)).

**Outer loop (hours-to-days).** Branches are arranged as **islands** with ring-topology migration every M generations — top programs and proposer strategies migrate to neighboring branches' experience buffers, preserving diversity against convergence. This is OpenEvolve's island pattern ([codelion/openevolve](https://github.com/codelion/openevolve) README §"island-based evolutionary architecture").

**Orthogonal to the loops:** each forward pass during solving runs with **In-Place TTT** active — the final projection matrix of MLP blocks adapts to the current problem's context via a next-token-prediction objective, then reverts for the next problem (Feng et al. 2026, arXiv [2604.06169](https://arxiv.org/abs/2604.06169)). This is amortized context: the model sees its own reasoning and updates the final-MLP weights mid-inference rather than stuffing the full trajectory into the context window every time.

## What's non-negotiable

**Held-out anchor with reversion.** Every K generations (default K=10), the population is evaluated against a frozen benchmark snapshot (HumanEval+ with a locked test set, or LiveCodeBench v6 frozen at commit X). If aggregate population score on the anchor drops vs the previous anchor check, the generation is reverted. No exceptions. This is the single most important mechanism — it is the hard wall against the failure mode that has sunk every prior co-evolutionary system: solvers and proposers converging on meaningless tasks they can mutually verify but that carry no real information.

**Verifier sandboxing.** The Python executor is not optional discipline. A solver that discovers it can `os.system("cat /proc/self/mem")` or hang the runner with a fork bomb has "won" the match per the verifier. Run every verification in Firecracker or gVisor with CPU/memory/time caps. The [AZR repo's own warning](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner#python-executor) says its executor is research-grade only; we do not inherit that risk.

**Zero external training data.** AZR's core claim is that external data is not needed. We preserve this. The anchor is eval-only, never seen during training. No MBPP, no HumanEval problems, no CodeAlpaca. Keep the claim or don't claim it.

## What's explicitly out of scope

- Trading, market prediction, or any non-stationary adversarial domain. No exact verifier → no compounding self-improvement, only compounding overfitting. See [`06-risks.md`](06-risks.md) §3.
- Open-ended creative generation. Soft verifiers produce mutual hallucination.
- General AGI. This stack compounds on code synthesis and mathematical reasoning specifically because those have cheap exact verifiers. Transfer to other domains is open empirically — expected positive, not guaranteed.

## What makes this different from "just run AZR"

Three things:

1. **Population over single-model.** AZR as published trains one model. We train N LoRA branches in competition. Elo selection + island migration give diversity that one-model AZR cannot produce.
2. **Elo replaces verified-reward alone.** Branches aren't ranked only by absolute pass-rate (which saturates and gives no signal) but by head-to-head outcomes on contested problems. Elo handles the non-transitivity that emerges when different branches specialize.
3. **Fast weights on top of LoRA.** LoRA branches are the slow-weight specialization; In-Place TTT is the per-problem fast-weight adaptation. This separation of time-scales matches what actually works in biological learning and, more concretely, what the 2026 literature says is the missing piece in LLM RL.

That combination is the project. Every piece is published and working; the assembly is new.
