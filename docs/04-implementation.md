# 04. Implementation — Step by Step

Four phases. Each phase has a clear success criterion. Do not proceed to phase N+1 until phase N passes.

## Phase 0 — Environment (half a day)

### 0.1 System

```bash
# Ubuntu 22.04 or 24.04. Verify CUDA is 12.4, not 13.2.
nvidia-smi  # expect driver >= 550, CUDA <= 12.4
nvcc --version  # if nvcc shows 13.x, pin to 12.4 via conda-toolkit

# Python
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
```

### 0.2 Clone dependencies in a pinned layout

```bash
export SI_ROOT=$HOME/SI
export SI_DEPS=$SI_ROOT/deps
export SI_CACHE=$SI_ROOT/cache
mkdir -p $SI_DEPS $SI_CACHE

cd $SI_DEPS
git clone https://github.com/LeapLabTHU/Absolute-Zero-Reasoner.git
git clone https://github.com/verl-project/verl.git
git clone https://github.com/andborth/RoboPhD.git
git clone https://github.com/codelion/openevolve.git

# Pin versions
( cd verl && git checkout v0.5.0 )  # adjust to latest tagged stable
( cd openevolve && git checkout v0.2.19 )  # latest per PyPI at spec time
# AZR and RoboPhD: pin to the commit at clone time; check git log.
```

### 0.3 Python environment

```bash
cd $SI_ROOT
uv venv --python 3.11
source .venv/bin/activate

# Core
uv pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu124
uv pip install "vllm>=0.8.0,<0.9"
uv pip install transformers==4.46.*  # verify Gemma 4 compatible
uv pip install peft accelerate datasets
uv pip install ray[default]

# veRL
uv pip install -e $SI_DEPS/verl

# OpenEvolve (optional at Phase 0)
uv pip install -e $SI_DEPS/openevolve

# Evaluation (PINNED — matches AZR)
uv pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus@d362e933265c3e7e3df8101c930a89c3c470cd9f"

# Utilities
uv pip install wandb rich typer pydantic
```

### 0.4 Download base model

```bash
cd $SI_CACHE
# For Tier 1 (MVP, 1× 3090): E4B
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-Q4_K_M.gguf --local-dir ./gemma-4-E4B

# For Tier 2 (5× 3090): 26B MoE
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
    gemma-4-26B-A4B-it-Q4_K_M.gguf --local-dir ./gemma-4-26B

# For LoRA training, we need full HF weights (not just GGUF):
huggingface-cli download google/gemma-4-E4B-it --local-dir ./gemma-4-E4B-hf
```

### 0.5 Verify inference works

```bash
# Quick sanity check
python -c "
from vllm import LLM, SamplingParams
llm = LLM('$SI_CACHE/gemma-4-E4B-hf', dtype='bfloat16', gpu_memory_utilization=0.85)
out = llm.generate(['def fibonacci(n):'], SamplingParams(temperature=0.2, max_tokens=64))
print(out[0].outputs[0].text)
"
```

**Success criterion for Phase 0:** the sanity check returns a plausible `fibonacci` implementation. If not, fix before proceeding.

---

## Phase 1 — AZR single-branch reproduction (Weekend 1)

**Goal:** run AZR as published, Gemma 4 E4B instead of Qwen2.5. Replace the base model, keep everything else. Confirm the AZR loop actually improves the model on HumanEval+.

### 1.1 Configure AZR for Gemma 4

AZR's configs live in `$SI_DEPS/Absolute-Zero-Reasoner/configs/`. Copy the 7B config as a template:

```bash
cp $SI_DEPS/Absolute-Zero-Reasoner/configs/azr_7b.yaml \
   $SI_ROOT/configs/azr_gemma4_e4b.yaml
```

Edit `azr_gemma4_e4b.yaml`:
- `model.path`: point to `$SI_CACHE/gemma-4-E4B-hf`
- `model.use_cache`: `true` (explicit — known bug when false)
- `rollout.chat_template`: use Gemma 4's template (`gemma-4` or `gemma-4-thinking`)
- `rollout.temperature`: 0.8 for proposer, 0.5 for solver (per AZR §4.1)
- `rollout.n`: 8 (K for MC proposer reward)
- `actor.optim.lr`: 1e-6 (verl default for GRPO, safe starting point)
- `azr.executor`: `sandboxfusion`

### 1.2 Generate seed data

```bash
cd $SI_DEPS/Absolute-Zero-Reasoner
export OUTPUT_SEED_PATH=$SI_ROOT/data/gemma4_ded_abd_seed.jsonl
export OUTPUT_CODE_F_SEED_PATH=$SI_ROOT/data/gemma4_ind_seed.jsonl

# No 'gemma4' script exists upstream yet; adapt from 'coder7b':
cp scripts/seeding/coder7b.sh scripts/seeding/gemma4_e4b.sh
# Edit gemma4_e4b.sh: change model path, keep format prompts.
bash scripts/seeding/gemma4_e4b.sh
```

### 1.3 Run AZR self-play

```bash
# Tier 1 (1× 3090): E4B fits comfortably.
bash scripts/selfplay/gemma4_e4b.sh 2>&1 | tee $SI_ROOT/logs/phase1_run1.log
```

Expect ~200–500 training steps before visible improvement. Watch W&B for:
- `solver/pass_rate` trending up
- `proposer/mc_difficulty` centered near 0.5 (healthy curriculum)
- `solver/response_length` growing (emergent chain-of-thought)

### 1.4 Evaluate on anchor

```bash
# Convert veRL checkpoint to HF:
python -m absolute_zero_reasoner.utils.convert2hf \
    $SI_ROOT/runs/phase1/actor \
    $SI_ROOT/runs/phase1/actor/huggingface/ \
    $SI_ROOT/checkpoints/phase1_gen_N

# Run HumanEval+
conda activate evalplus  # or ensure evalplus is in current env
bash $SI_DEPS/Absolute-Zero-Reasoner/evaluation/code_eval/scripts/run_evalplus.sh \
    0 humaneval $SI_ROOT/checkpoints/phase1_gen_N
```

**Phase 1 success criterion:**
- Trained model scores ≥5pp higher than base Gemma 4 E4B on HumanEval+.
- Zero external training data used (only AZR self-generated).
- Training ran to completion without sandbox escape or OOM.

**If it fails:** the AZR loop itself has a config issue. Do not move to Phase 2. Debug single-branch AZR first — it's the substrate everything else sits on.

---

## Phase 2 — Multi-branch Elo tournament (Week 2–3)

**Goal:** add the SI differentiator — N LoRA branches competing via Elo, replacing bottom-quartile each generation.

### 2.1 Fork AZR into SI-controlled codebase

```bash
cd $SI_ROOT
# We copy the relevant AZR modules and modify, rather than vendoring the whole fork.
mkdir -p src/si/azr
cp -r $SI_DEPS/Absolute-Zero-Reasoner/absolute_zero_reasoner/rewards src/si/azr/
cp -r $SI_DEPS/Absolute-Zero-Reasoner/absolute_zero_reasoner/prompts src/si/azr/
# Loop control + population logic is ours.
```

### 2.2 Implement branch manager

Create `src/si/population.py` (the key new code, see scaffolding in that file). Responsibilities:
- Maintain N LoRA adapters on disk, one per branch.
- Swap adapters in/out of a single base model for rollout (via PEFT's `model.set_adapter()`).
- Track Elo rating per branch.
- Log head-to-head matches into `$SI_ROOT/matches.jsonl`.

### 2.3 Match orchestration

Per generation:

1. Proposer (single, shared across branches — not yet specialized per branch) generates B proposals.
2. For each proposal, select 2 random branches. Both solve. Verifier decides. Elo update.
3. (Matches = 3× population size per generation is a good heuristic.)
4. After all matches: compute generation Elo ranking.
5. Bottom quartile branches: delete LoRA, re-initialize from top-quartile LoRA + small Gaussian perturbation (σ=0.001 per weight).
6. Top quartile branches: apply GRPO update on their own experience buffer.
7. Middle half: GRPO update, no replacement.

### 2.4 Anchor check every 10 generations

```python
# src/si/anchor.py (see scaffolding)
def check_anchor(population, gen):
    if gen % 10 != 0: return None
    scores = [run_humaneval_plus(b) for b in population]
    curr = {"gen": gen, "aggregate": sum(scores), "per_branch": scores}
    prev = load_last_anchor()
    if prev and curr["aggregate"] < prev["aggregate"] * 0.98:  # 2% tolerance
        revert_to_gen(prev["gen"])
        return "REVERTED"
    save_anchor(curr)
    return "COMMITTED"
```

The 2% tolerance is for noise; exact thresholds require tuning. Err conservative (smaller tolerance) — reversion is safer than drift.

### 2.5 Evaluate

Same as Phase 1, but compare:
- Base Gemma 4
- Phase 1 (single-branch AZR)
- Phase 2 top-Elo branch

**Phase 2 success criterion:**
- Phase 2 top-Elo branch ≥3pp above Phase 1.
- Elo rankings stable across last 10 generations (top-ranked branch wins ≥60% of matches).
- No anchor reversion in last 20 generations.

---

## Phase 3 — Island migration (Week 4)

Add OpenEvolve-style ring-topology migration. This is ~200 lines of code in `src/si/islands.py`.

- Group N branches into M islands of N/M each (e.g., 12 branches, 4 islands of 3).
- Each island runs its own Elo ranking.
- Every 5 generations: top-1 branch per island exports (LoRA delta, top-32 experience entries) to next island clockwise.
- Migrant's delta seeds the next generation's lowest-Elo branch in receiving island.
- Migrant experiences enter receiving island's replay buffer at 2× priority for 1 generation.

**Phase 3 success criterion:**
- Gemini diversity metric (pairwise Elo-rating variance) sustained across 20 generations vs. monotonically decreasing in Phase 2.
- Top-island top-branch continues to compound on anchor (≥2pp additional gain over Phase 2).

---

## Phase 4 — In-Place TTT integration (Week 5+)

Implement In-Place TTT from the paper. As of the spec date, no public code from the authors exists.

- Identify `mlp.down_proj` modules across all Gemma 4 transformer blocks.
- Register them as "fast-weight" layers with their own optimizer state (AdamW, small LR 1e-4).
- Before each problem: snapshot current fast-weights.
- During problem: after every chunk of 256 tokens generated, compute NTP loss on the most recent 256-token chunk, apply one gradient step to fast-weights only.
- After problem: restore snapshot.

**Phase 4 success criterion:**
- Per-problem fast-weight adaptation either improves anchor score by ≥1pp at fixed LoRA checkpoint, or the ablation is clean (TTT-off beats TTT-on by <1pp, confirming null result worth publishing).

## Top-level orchestrator

The whole thing is coordinated by `src/si/loop.py`. See that file for the skeleton. Pseudocode:

```python
while not converged and generations < MAX_GENERATIONS:
    gen = current_gen()

    # Inner loop: generate problems via AZR
    problems = proposer.batch_propose(
        batch_size=config.problems_per_gen,
        seed_from=experience.recent_wins,
    )

    # Middle loop: Elo matches
    matches = run_matches(problems, population, pairs_per_problem=3)
    elo.update_from(matches)

    # GRPO update for all branches on their own experience
    for branch in population:
        branch.grpo_step(branch.replay_buffer.sample(config.batch_size))

    # PBT replacement
    population = elo_replace_bottom_quartile(population, elo.ranks())

    # Outer loop: migration every 5 gens
    if gen % config.migration_every == 0:
        islands = migrator.ring_migrate(islands)

    # Anchor: every 10 gens
    if gen % config.anchor_every == 0:
        result = anchor.evaluate(population)
        if anchor.should_revert(result):
            revert_to_last_committed_gen()
            log.warn(f"REVERTED at gen {gen}")
        else:
            commit_gen(gen)
```

That is the complete loop. Everything else in the repo is infrastructure around this.

## Verifier setup (in detail, because it matters most)

### Minimum-viable (MVP Phase 1–2)

```bash
# Use AZR's sandboxfusion via Docker.
docker pull sandboxfusion/sandbox:latest  # adjust image per upstream
docker run -d -p 8765:8765 --name si-sandbox \
    --cpus=2 --memory=2g --pids-limit=256 \
    sandboxfusion/sandbox:latest
```

Set `azr.executor=sandboxfusion` in config. Point to `http://localhost:8765`.

### Production (Tier 2+, Phase 3)

Firecracker microVM pool. Ballpark 16 VMs pre-warmed, each with:
- 512MB RAM
- 1 vCPU
- Python 3.11 + numpy + nothing else (no network, no disk outside VM)
- 5-second wall-clock timeout per verification
- Reset to snapshot on each use

Expected throughput: ~100 verifications/second per host, bottlenecked on microVM spawn if not pre-pooled.

Reference implementation (not SI-specific, but pattern): [`firecracker-containerd`](https://github.com/firecracker-microvm/firecracker-containerd), or the simpler approach with [`gVisor`](https://gvisor.dev/) if you don't need the full VM isolation.

**Do not ship a homegrown `subprocess` + `chroot` + `seccomp` sandbox.** Every team that does this eventually gets escaped. Use battle-tested isolation.
