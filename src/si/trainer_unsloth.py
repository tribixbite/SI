"""Unsloth-based GRPO LoRA trainer for Gemma 4 E4B.

Replaces src/si/trainer.py for Phase 1. Uses Unsloth's FastVisionModel
(Gemma 4 is multimodal) + TRL 0.24 GRPOTrainer. Recipe sourced from Unsloth's
Gemma4_(E2B)_GRPO.ipynb (Auto Kernel Creation pattern).

Differences from the vanilla TRL trainer:
    - FastVisionModel patches HF internals to avoid Gemma 4's
      heterogeneous-head-dim NaN bug (no attn_implementation='eager' needed).
    - 4-bit QLoRA base (model_path points to unsloth/gemma-4-E4B-it-unsloth-bnb-4bit).
    - Multimodal prompts: content is a list of typed blocks, not a raw string.
    - Sampling params per Unsloth guide: temp=1.0, top_p=0.95, top_k=64.
    - GRPO loss = 'bnpo' with epsilon/epsilon_high/delta.

IMPORTANT: this module MUST be imported (or `import unsloth` run) before any
transformers/trl/peft imports in the caller, or Unsloth's patches won't apply.
"""

from __future__ import annotations

# Unsloth patches HF internals on import; do this first.
import unsloth  # noqa: F401, I001
from unsloth import FastModel

import logging
from dataclasses import dataclass
from pathlib import Path

import random

from datasets import Dataset
from trl import GRPOConfig

from si.contracts import Solution, Task, TaskType
from si.match import ProposalOutcome
from si.parsers import extract_input, extract_output
from si.prompts import (
    SOLVER_SYSTEM_TEMPLATES,
    solver_abduction_prompt,
    solver_deduction_prompt,
)
from si.trainer_rlzvp import RLZVPGRPOTrainer
from si.verifier import SandboxVerifier

log = logging.getLogger(__name__)

DEFAULT_UNSLOTH_MODEL = "/home/matilda/git/SI/cache/gemma-4-E4B-unsloth-4bit"

_SYSTEM_SOLVER = SOLVER_SYSTEM_TEMPLATES[0]  # default; dataset may randomize per row


@dataclass
class UnslothTrainerConfig:
    model_path: str = DEFAULT_UNSLOTH_MODEL
    output_dir: str = "runs/_trainer"
    max_seq_length: int = 4096
    max_completion_length: int = 512
    lora_rank: int = 32
    load_in_4bit: bool = True
    lr: float = 5e-5
    weight_decay: float = 0.001
    warmup_ratio: float = 0.1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_generations: int = 2  # small group; higher values OOM on 3090 with Gemma 4 E4B
    max_steps: int = -1  # -1 means use num_train_epochs
    epochs: int = 1
    beta: float = 0.01  # KL coefficient; 10x v2 default because RL-ZVP injects signal on every step
    grad_clip_norm: float = 1.0
    random_state: int = 3407
    # RL-ZVP / F-GRPO knobs (docs/research-scan.md scan #7).
    # rlzvp_magnitude=0.05 chosen after a 64-step smoke on stale outcomes blew up
    # (kl→11790, grad→3216) at magnitude=0.5. With frac_reward_zero_std≈1 every
    # sample gets injected advantage, so a magnitude one order of magnitude below
    # normal GRPO advantage (|adv|~1) is the safe starting point.
    rlzvp_enabled: bool = True
    rlzvp_magnitude: float = 0.05
    rlzvp_pos_threshold: float = 0.5  # reward above this is "positive" for sign
    fgrpo_gamma: float | None = 2.0  # None disables F-GRPO focal scaling
    randomize_system_prompt: bool = True
    # MicroCoder-GRPO recipe (arXiv:2603.07777, Li et al. Mar 2026).
    # Tuned specifically for LiveCodeBench v6: removes KL pull (beta=0),
    # widens the asymmetric clip range (epsilon_high > epsilon), and keeps
    # truncation masking on. Reports +17.6% relative over standard GRPO
    # baselines on LCB v6 in 300 steps. Set microcoder_grpo=True to apply
    # these knobs (overrides beta and TRL clip defaults).
    microcoder_grpo: bool = False
    microcoder_epsilon: float = 0.2
    microcoder_epsilon_high: float = 0.5  # paper: 0.4-0.6 sweet spot
    microcoder_temperature: float = 1.2   # paper: 1.0-1.3; static substitute for diversity-determined


def _user_prompt_for(task: Task) -> str:
    if task.task_type is TaskType.DEDUCTION:
        assert task.program is not None and task.input is not None
        return solver_deduction_prompt(task.program, task.input)
    if task.task_type is TaskType.ABDUCTION:
        assert task.program is not None and task.output is not None
        return solver_abduction_prompt(task.program, task.output)
    raise NotImplementedError(f"no prompt builder for {task.task_type}")


def outcomes_to_unsloth_dataset(
    outcomes: list[ProposalOutcome], *, randomize_system: bool = True, seed: int = 3407
) -> Dataset:
    """Build a HF dataset with multimodal-style typed content blocks.

    Gemma 4's processor rejects plain string content — must be a list of
    {"type": "text", "text": "..."} blocks even for text-only tasks.

    `randomize_system=True` cycles across SOLVER_SYSTEM_TEMPLATES (prompt
    augmentation per arXiv:2602.03190) to prevent entropy collapse from a
    fixed system prompt.
    """
    rng = random.Random(seed)
    rows = []
    for o in outcomes:
        task = o.task
        prompt_user = _user_prompt_for(task)
        sys_prompt = rng.choice(SOLVER_SYSTEM_TEMPLATES) if randomize_system else _SYSTEM_SOLVER
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt_user}]},
        ]
        rows.append(
            {
                "prompt": messages,
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "program": task.program or "",
                "input": task.input or "",
                "output": task.output or "",
            }
        )
    return Dataset.from_list(rows)


def _task_from_row(row: dict) -> Task:
    return Task(
        task_type=TaskType(row["task_type"]),
        program=row["program"] or None,
        input=row["input"] or None,
        output=row["output"] or None,
        proposer_branch_id="p0",
        gen=0,
        task_id=row["task_id"],
    )


def _completion_text(completion) -> str:
    """GRPOTrainer passes completions as list of chat turns (multimodal-style)."""
    if isinstance(completion, list):
        for turn in reversed(completion):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                content = turn.get("content", "")
                if isinstance(content, list):
                    # multimodal: concat text blocks
                    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    return "".join(parts)
                return str(content)
        return ""
    return str(completion)


def _extract_body(task: Task, completion_text: str) -> str | None:
    if task.task_type is TaskType.DEDUCTION:
        return extract_output(completion_text)
    return extract_input(completion_text)


def make_verifier_reward_fn(verifier: SandboxVerifier):
    """AZR verifier reward: 1.0 if the sandbox verifies the solver's answer, else 0.0."""

    def reward_fn(completions, **kwargs) -> list[float]:
        task_ids = kwargs.get("task_id", [])
        task_types = kwargs.get("task_type", [])
        programs = kwargs.get("program", [])
        inputs = kwargs.get("input", [])
        outputs = kwargs.get("output", [])
        rewards: list[float] = []
        for i, comp in enumerate(completions):
            row = {
                "task_id": task_ids[i],
                "task_type": task_types[i],
                "program": programs[i],
                "input": inputs[i],
                "output": outputs[i],
            }
            task = _task_from_row(row)
            text = _completion_text(comp)
            body = _extract_body(task, text)
            if not body:
                rewards.append(-0.2)  # unparseable
                continue
            sol = Solution(task_id=task.task_id, solver_branch_id="s0", body=body, trace=text, walltime_ms=0)
            try:
                result = verifier.verify(task, sol)
                rewards.append(1.0 if result.passed else 0.0)
            except Exception as e:
                log.warning("verifier error in reward_fn: %s", e)
                rewards.append(0.0)
        return rewards

    return reward_fn


def make_format_reward_fn():
    """Small shaping reward: +0.1 if the completion contains the expected fenced block."""

    def reward_fn(completions, **kwargs) -> list[float]:
        task_types = kwargs.get("task_type", [])
        rewards: list[float] = []
        for i, comp in enumerate(completions):
            text = _completion_text(comp)
            tt = TaskType(task_types[i])
            if tt is TaskType.DEDUCTION:
                ok = "```output" in text
            else:
                ok = "```input" in text
            rewards.append(0.1 if ok else 0.0)
        return rewards

    return reward_fn


class UnslothSITrainer:
    """Phase 1 single-branch solver GRPO trainer on Unsloth + TRL 0.24."""

    def __init__(self, cfg: UnslothTrainerConfig, verifier: SandboxVerifier) -> None:
        self.cfg = cfg
        self.verifier = verifier
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        log.info("Loading Unsloth FastModel from %s", cfg.model_path)
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=cfg.model_path,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            fast_inference=False,
            full_finetuning=False,
        )
        # Restrict LoRA to the text tower. Gemma 4's multimodal tree has
        # q_proj/k_proj/... in vision_tower and audio_tower too; the earlier
        # FastVisionModel run applied LoRA to all three and diluted the signal
        # across modalities the AZR task never exercised.
        self.model = FastModel.get_peft_model(
            self.model,
            r=cfg.lora_rank,
            target_modules=r"^.*\blanguage_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            lora_alpha=cfg.lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.random_state,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )

    def _grpo_args(self) -> GRPOConfig:
        # TRL 1.2 uses `beta` (KL coefficient) instead of Unsloth notebook's
        # `epsilon`/`epsilon_high`/`delta`/`loss_type='bnpo'` (those are from
        # TRL 0.22). The sampling knobs (temperature/top_p/top_k) are still
        # available. Other Unsloth-recommended defaults preserved.
        beta = 0.0 if self.cfg.microcoder_grpo else self.cfg.beta
        temperature = (
            self.cfg.microcoder_temperature if self.cfg.microcoder_grpo else 1.0
        )
        kwargs = dict(
            output_dir=self.cfg.output_dir,
            temperature=temperature,
            top_p=0.95,
            top_k=64,
            learning_rate=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            num_generations=self.cfg.num_generations,
            max_completion_length=self.cfg.max_completion_length,
            max_steps=self.cfg.max_steps,
            num_train_epochs=self.cfg.epochs,
            max_grad_norm=self.cfg.grad_clip_norm,
            beta=beta,
            mask_truncated_completions=True,
            save_strategy="no",
            logging_steps=1,
            report_to="none",
            bf16=True,
        )
        if self.cfg.microcoder_grpo:
            # Asymmetric clip: allow larger positive updates than negative.
            # Paper sets epsilon=0.2, epsilon_high=0.4-0.6.
            kwargs["epsilon"] = self.cfg.microcoder_epsilon
            kwargs["epsilon_high"] = self.cfg.microcoder_epsilon_high
        return GRPOConfig(**kwargs)

    def train_on_generation(self, outcomes: list[ProposalOutcome]) -> str:
        if not outcomes:
            log.warning("train_on_generation called with no outcomes; skipping")
            return ""
        dataset = outcomes_to_unsloth_dataset(
            outcomes,
            randomize_system=self.cfg.randomize_system_prompt,
            seed=self.cfg.random_state,
        )
        log.info(
            "building RLZVP GRPO trainer for %d tasks (rlzvp=%s fgrpo_gamma=%s aug_prompt=%s)",
            len(dataset),
            self.cfg.rlzvp_enabled,
            self.cfg.fgrpo_gamma,
            self.cfg.randomize_system_prompt,
        )
        trainer = RLZVPGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                make_verifier_reward_fn(self.verifier),
                make_format_reward_fn(),
            ],
            args=self._grpo_args(),
            train_dataset=dataset,
            rlzvp_enabled=self.cfg.rlzvp_enabled,
            rlzvp_magnitude=self.cfg.rlzvp_magnitude,
            rlzvp_pos_threshold=self.cfg.rlzvp_pos_threshold,
            fgrpo_gamma=self.cfg.fgrpo_gamma,
        )
        trainer.train()
        adapter_path = str(Path(self.cfg.output_dir) / "adapter")
        trainer.model.save_pretrained(adapter_path)
        log.info("saved adapter to %s", adapter_path)
        return adapter_path
