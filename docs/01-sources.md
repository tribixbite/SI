# 01. Sources — Papers, Repositories, Versions

All sources verified current as of April 19, 2026. Pin to the versions below unless upgrading with a deliberate test run.

## Core papers (mechanism sources)

### AZR — Absolute Zero Reasoner
- **Paper:** Zhao, Wu, Yue et al. "Absolute Zero: Reinforced Self-play Reasoning with Zero Data." arXiv:2505.03335, May 2025.
- **arXiv:** https://arxiv.org/abs/2505.03335
- **HF paper page:** https://hf.co/papers/2505.03335
- **Project page:** https://andrewzh112.github.io/absolute-zero-reasoner/
- **Code:** https://github.com/LeapLabTHU/Absolute-Zero-Reasoner
- **License:** (check repo; Apache-compatible per upstream)
- **Why it matters:** Zero-data curriculum, SOTA on coding + math benchmarks with no external training data.
- **Citation (BibTeX):**
  ```
  @misc{zhao2025absolutezeroreinforcedselfplay,
    title={Absolute Zero: Reinforced Self-play Reasoning with Zero Data},
    author={Andrew Zhao and Yiran Wu and Yang Yue and Tong Wu and Quentin Xu
            and Yang Yue and Matthieu Lin and Shenzhi Wang and Qingyun Wu
            and Zilong Zheng and Gao Huang},
    year={2025}, eprint={2505.03335}, archivePrefix={arXiv}, primaryClass={cs.LG},
    url={https://arxiv.org/abs/2505.03335}
  }
  ```

### RoboPhD — Elo Tournament Agent Evolution
- **Paper (primary, evolution mechanics):** Borthwick, Ash, Galczak. "RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets." arXiv:2604.04347, April 2026.
- **Paper (original, Text-to-SQL version with Elo details):** Borthwick & Ash. "RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution." arXiv:2601.01126, January 2026.
- **arXiv:** https://arxiv.org/abs/2604.04347 and https://arxiv.org/abs/2601.01126
- **Code:** https://github.com/andborth/RoboPhD
- **License:** MIT
- **Why it matters:** Elo-based selection beats Pareto/greedy alternatives under a fixed eval budget. Validation-free — the evaluations that drive evolution are the same ones that rank agents. Critical for 3090-scale where eval budget is the bottleneck.
- **Key API:** `optimize_anything()` — the toolkit's entry point. Inspect and adapt rather than depend on.

### In-Place TTT — Fast-Weight Test-Time Training
- **Paper:** Feng, Luo, Hua et al. "In-Place Test-Time Training." arXiv:2604.06169, April 7, 2026.
- **arXiv:** https://arxiv.org/abs/2604.06169
- **HF paper page:** https://hf.co/papers/2604.06169
- **Code:** Not yet publicly released as of 2026-04-19. Monitor HF paper page and authors' GitHub.
- **Authors' affiliations:** ByteDance Seed, Peking University.
- **Why it matters:** Modifies the final projection matrix of MLP blocks during inference with a next-token-prediction objective. Cheaper than LoRA fine-tuning, preserves base weights, compatible with context parallelism. Closest published work to "self-editing weights during inference."
- **Implementation risk:** Since the paper is 12 days old at time of this spec and no code is public, we treat In-Place TTT as **Phase 2** work. MVP runs without it. If no code is released by the time the MVP is working, we implement from the paper — the mechanism is described precisely enough (§3 of the paper) to re-derive.

### OpenEvolve — Evolutionary Coding Agent Framework
- **Reference paper (AlphaEvolve):** Novikov et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." arXiv:2506.13131, June 2025. https://arxiv.org/abs/2506.13131
- **Implementation paper (optional):** Assumpção et al. "CodeEvolve." arXiv:2510.14150, October 2025. https://arxiv.org/abs/2510.14150
- **Primary code we use:** https://github.com/codelion/openevolve
- **Alternate (also good):** https://github.com/inter-co/science-codeevolve (CodeEvolve's own repo)
- **PyPI:** `openevolve` — latest 0.2.x series as of April 2026
- **License:** Apache 2.0
- **Why it matters:** Island-based genetic algorithm with ring-topology migration. Battle-tested on circle packing, function minimization, symbolic regression. We borrow the island-migration pattern, not the whole framework.

## Training framework

### veRL — Volcano Engine RL for LLMs
- **Repo:** https://github.com/verl-project/verl (migrated from `volcengine/verl` in Jan 2026)
- **Latest major version at time of spec:** v0.5+ (AgentLoop abstraction, server-based async rollout)
- **License:** Apache 2.0
- **PyPI:** `verl`
- **Docs:** https://verl.readthedocs.io/
- **Supports:** PPO, GRPO, GSPO, DAPO, DrGRPO, ReMax, REINFORCE++, RLOO, PRIME. LoRA RL. FSDP2. vLLM + SGLang rollout backends.
- **Why it matters:** AZR's own codebase is a veRL fork. We inherit that choice to stay close to upstream and to benefit from veRL's mature multi-GPU infrastructure. GRPO is our default RL algorithm for the LoRA branches; DAPO is our fallback if GRPO instability shows up (see [`06-risks.md`](06-risks.md) §2).

## Base model

### Gemma 4 (Google DeepMind, April 2, 2026)
- **License:** Apache 2.0
- **Family:** E2B, E4B, 26B-A4B (MoE, 3.8B active), 31B Dense
- **Benchmarks (31B):** MMLU Pro 85.2, AIME 2026 89.2, GPQA Diamond 84.3, LiveCodeBench v6 80.0, Codeforces ELO 2150, Arena AI rank #3 (ELO 1452), τ2-bench 86.4 (up from 6.6 on Gemma 3 27B)
- **Context:** 128K (E2B/E4B), 256K (26B/31B)
- **Hugging Face (official):** https://huggingface.co/google/gemma-4-31B-it and siblings
- **Unsloth GGUFs (recommended for local):**
  - https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
  - https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
  - https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF (sweet spot for 1× 3090)
  - https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
- **Unsloth docs:** https://unsloth.ai/docs/models/gemma-4
- **Memory footprint:**
  - E2B Q4: ~1.5GB | E4B Q4: ~5GB
  - 26B MoE Q4_0: ~15.6GB | 31B Q4_0: ~17.4GB (fits single 3090)
  - 26B 8-bit: ~28GB | 31B 8-bit: ~34GB
- **Known issue:** Set `use_cache=True` explicitly. Bug where `use_cache=False` corrupts attention on 31B and 26B-A4B (documented in Unsloth's fine-tuning guide). Also avoid CUDA 13.2 runtime for GGUFs (causes bad outputs per Unsloth). Pin CUDA 12.x.
- **Thinking mode:** Gemma 4 has a native `<|think|>` channel. We **disable** it in the proposer and **enable** it in the solver by default. Rationale: proposer needs fast task generation; solver benefits from chain-of-thought. Override per-role in `configs/roles.yaml`.

## Verifier / benchmarks

### EvalPlus (HumanEval+ and MBPP+)
- **Repo:** https://github.com/evalplus/evalplus
- **Pinned commit (from AZR README):** `d362e933265c3e7e3df8101c930a89c3c470cd9f`
- **Install:** `pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus@d362e933265c3e7e3df8101c930a89c3c470cd9f"`
- **Why pinned:** Benchmarks mutate. Reversion-on-anchor-regression is meaningless if the anchor itself drifts. We freeze at the exact commit AZR used, making our results directly comparable to AZR's published baselines.

### LiveCodeBench (held-out anchor candidate)
- **Repo:** https://github.com/LiveCodeBench/LiveCodeBench
- **Mirror (code_generation_lite):** https://hf-mirror.com/datasets/livecodebench/code_generation_lite
- **Use:** Frozen subset as anchor. See [`05-evaluation.md`](05-evaluation.md).

## Python executor / sandboxing

### sandbox-fusion (recommended by AZR)
- Set `azr.executor=sandboxfusion` in AZR config
- Run in Docker per AZR docs
- **Alternative we recommend for production discipline:** Firecracker microVMs. ~50ms cold, sub-ms hot, true isolation. See [`04-implementation.md`](04-implementation.md) §3.2.

## Supporting infrastructure

- **vLLM:** https://github.com/vllm-project/vllm — batched inference engine. Pin `>=0.8.0` for FSDP compatibility (per verl docs).
- **SGLang:** https://github.com/sgl-project/sglang — alternative rollout backend with better multi-turn support.
- **PEFT (LoRA):** https://github.com/huggingface/peft — for LoRA branch adapters.
- **Unsloth:** https://github.com/unslothai/unsloth — fine-tuning library, ~1.5x faster than FA2, 60% less VRAM. Gemma 4 E2B trains on 8GB VRAM per Unsloth.
- **Ray:** https://github.com/ray-project/ray — verl's orchestration layer.

## How we track version drift

A `scripts/freeze_versions.sh` (see [`scripts/`](../scripts/)) emits a `VERSIONS.lock` with exact git commits and pip versions at the moment of any successful anchor-passing generation. Reverting a generation = reverting to the previous `VERSIONS.lock`. This is the operational companion to the anchor-reversion rule.
