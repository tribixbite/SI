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
