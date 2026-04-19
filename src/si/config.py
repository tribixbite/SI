"""Configuration dataclasses for SI runs.

Configs are loaded from YAML in configs/, overlaid with CLI args, and hashed
to wandb at run start. A run refuses to start with uncommitted config changes
(see scripts/run.sh). This is deliberate: reproducibility > convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    base_model_path: str  # HF repo id or local path
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.85
    use_cache: bool = True  # MUST be True for Gemma 4 26B/31B (known bug)
    max_model_len: int = 8192
    trust_remote_code: bool = False


@dataclass
class ProposerConfig:
    # AZR §3.3: MC-rollout-based difficulty reward
    mc_rollouts: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 1024
    proposals_per_generation: int = 64
    enable_thinking: bool = False  # Gemma 4's <|think|> channel
    seed_mix_ratio: float = 0.1  # fraction of prompts drawn from seed pool


@dataclass
class SolverConfig:
    temperature: float = 0.5
    top_p: float = 0.95
    max_new_tokens: int = 2048
    enable_thinking: bool = True
    n_rollouts_per_task: int = 1  # raise for exploration phases


@dataclass
class VerifierConfig:
    backend: str = "sandboxfusion"  # "sandboxfusion" | "firecracker" | "gvisor"
    endpoint: str = "http://localhost:8765"
    timeout_s: int = 5
    memory_mb: int = 512
    cpu_count: int = 1
    network_enabled: bool = False  # never flip to True outside explicit audit


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class GRPOConfig:
    lr: float = 1e-6
    kl_coef: float = 1e-3
    kl_type: str = "low_var_kl"  # matches verl default
    clip_range: float = 0.2
    group_size: int = 8
    ppo_mini_batch_size: int = 32
    grad_clip_norm: float = 1.0


@dataclass
class EloConfig:
    k: float = 32.0
    default_rating: float = 1500.0
    matches_per_generation_multiplier: float = 3.0  # matches = mult × population_size
    replacement_quartile_size: float = 0.25
    mutation_sigma: float = 0.001  # Gaussian perturbation on replacement


@dataclass
class IslandConfig:
    enabled: bool = True
    n_islands: int = 4
    migration_every: int = 5  # generations between migrations
    migrant_experience_priority: float = 2.0
    migrant_priority_decay_gens: int = 1


@dataclass
class AnchorConfig:
    primary: str = "humaneval_plus"  # "humaneval_plus" | "livecodebench" | "mbpp_plus"
    secondary: str | None = None
    meta: str = "mbpp_plus"  # never used during training; report-only
    anchor_every: int = 10  # check every N generations
    reversion_tolerance: float = 0.02  # 2% noise budget
    reversion_strict_after_gen: int = 50  # after gen 50, tolerance halves


@dataclass
class InPlaceTTTConfig:
    enabled: bool = False  # Phase 4 only
    target_module_name: str = "mlp.down_proj"
    lr: float = 1e-4
    chunk_tokens: int = 256
    reset_between_problems: bool = True


@dataclass
class RunConfig:
    run_id: str
    seed: int = 42
    tier: int = 1  # 1 | 2 | 3 — drives population size defaults
    max_generations: int = 500

    model: ModelConfig = field(default_factory=lambda: ModelConfig(base_model_path=""))
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    islands: IslandConfig = field(default_factory=IslandConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    ttt: InPlaceTTTConfig = field(default_factory=InPlaceTTTConfig)

    n_solver_branches: int = 4
    n_proposer_variants: int = 1

    def population_size(self) -> int:
        return self.n_solver_branches + self.n_proposer_variants
