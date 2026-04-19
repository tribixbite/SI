"""Phase 1 GRPO LoRA trainer for the solver on Gemma 4 E4B.

Trains a LoRA adapter on Gemma 4 using TRL's GRPOTrainer. The AZR-style
reward closes over our SandboxVerifier: for each generated completion, we
parse the solver output, verify against the corresponding task, and emit
binary reward {0, 1}.

Scope:
    - Phase 1 MVP: solver-only training. Proposer stays at the base model.
    - Single LoRA branch (Phase 2 will add branch swapping).
    - Prompts are built by si.prompts.solver_deduction_prompt /
      solver_abduction_prompt; tasks attached as dataset metadata so the
      reward fn can recover the right Task to verify against.

Non-goals for Phase 1:
    - proposer training (adds per-sample reward heterogeneity; deferred).
    - induction (needs seed programs).
    - multi-branch + Elo (Phase 2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from si.contracts import Solution, Task, TaskType
from si.match import ProposalOutcome
from si.parsers import extract_input, extract_output
from si.prompts import solver_abduction_prompt, solver_deduction_prompt
from si.verifier import SandboxVerifier

log = logging.getLogger(__name__)


_SYSTEM_SOLVER = (
    "You are a careful Python reasoning assistant. Given a function and one of "
    "(input, output), predict the missing value exactly. Output only the "
    "requested fenced block — no explanation."
)


@dataclass
class SITrainerConfig:
    model_path: str
    output_dir: str
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-6
    num_generations: int = 4  # GRPO group size; generation_batch_size must be divisible by this
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 1536
    max_completion_length: int = 512
    beta: float = 1e-3
    grad_clip_norm: float = 1.0


def outcomes_to_dataset(outcomes: list[ProposalOutcome]) -> Dataset:
    """Build a HF Dataset of prompts from a generation's outcomes.

    Each row carries:
        - prompt: chat messages list (system + user) ready for GRPOTrainer
        - task_id: for reward-fn lookup
        - task_type, program, input, output: for verifier reconstruction
    """
    rows = []
    for o in outcomes:
        task = o.task
        prompt_user = _user_prompt_for(task)
        messages = [
            {"role": "system", "content": _SYSTEM_SOLVER},
            {"role": "user", "content": prompt_user},
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


def _user_prompt_for(task: Task) -> str:
    if task.task_type is TaskType.DEDUCTION:
        assert task.program is not None and task.input is not None
        return solver_deduction_prompt(task.program, task.input)
    if task.task_type is TaskType.ABDUCTION:
        assert task.program is not None and task.output is not None
        return solver_abduction_prompt(task.program, task.output)
    raise NotImplementedError(f"no prompt builder for {task.task_type}")


def _task_from_row(row: dict) -> Task:
    tt = TaskType(row["task_type"])
    return Task(
        task_type=tt,
        program=row["program"] or None,
        input=row["input"] or None,
        output=row["output"] or None,
        proposer_branch_id="p0",
        gen=0,
        task_id=row["task_id"],
    )


def make_reward_fn(verifier: SandboxVerifier):
    """Build a GRPO-compatible reward function closed over the verifier.

    TRL 1.x calls: reward_fn(completions, **kwargs) -> list[float]
    where kwargs contains dataset columns as equal-length lists.
    """

    def reward_fn(completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        task_ids = kwargs.get("task_id", [])
        task_types = kwargs.get("task_type", [])
        programs = kwargs.get("program", [])
        inputs = kwargs.get("input", [])
        outputs = kwargs.get("output", [])
        for i, comp in enumerate(completions):
            row = {
                "task_id": task_ids[i],
                "task_type": task_types[i],
                "program": programs[i],
                "input": inputs[i],
                "output": outputs[i],
            }
            task = _task_from_row(row)
            body = _extract_body(task, _to_text(comp))
            sol = Solution(task_id=task.task_id, solver_branch_id="s0", body=body or "", trace=_to_text(comp), walltime_ms=0)
            try:
                result = verifier.verify(task, sol)
                rewards.append(1.0 if result.passed else 0.0)
            except Exception as e:
                log.warning("verifier error in reward_fn: %s", e)
                rewards.append(0.0)
        return rewards

    return reward_fn


def _to_text(completion) -> str:
    """Completions may be list[dict] (chat) or str (raw)."""
    if isinstance(completion, list):
        # Chat-format: list of turn dicts; take the assistant turn content.
        for turn in reversed(completion):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return str(turn.get("content", ""))
        return ""
    return str(completion)


def _extract_body(task: Task, completion_text: str) -> str | None:
    if task.task_type is TaskType.DEDUCTION:
        return extract_output(completion_text)
    return extract_input(completion_text)


class SITrainer:
    """Phase 1 single-branch solver GRPO trainer."""

    def __init__(self, cfg: SITrainerConfig, verifier: SandboxVerifier) -> None:
        self.cfg = cfg
        self.verifier = verifier
        # Gemma 4 is multimodal; restrict LoRA to the text tower to avoid PEFT
        # tripping on the vision tower's Gemma4ClippableLinear wrappers.
        # Suffix-based target_modules matches from both trees, so use a regex.
        self.lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=r"^model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            bias="none",
            task_type="CAUSAL_LM",
        )
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _grpo_args(self, num_train_epochs: int) -> GRPOConfig:
        # TRL 1.2 dropped max_prompt_length; tokenizer handles truncation via
        # the model_max_length setting. max_completion_length remains.
        #
        # Gemma 4 has heterogeneous head dimensions (head_dim=256 local,
        # global_head_dim=512) — vLLM forces TRITON_ATTN to avoid mixed-backend
        # numerical divergence. HF's SDPA implementation has the same failure
        # mode (NaN logits → torch.multinomial CUDA assert). model_init_kwargs
        # with attn_implementation='eager' works around it.
        return GRPOConfig(
            output_dir=self.cfg.output_dir,
            learning_rate=self.cfg.lr,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            num_generations=self.cfg.num_generations,
            max_completion_length=self.cfg.max_completion_length,
            num_train_epochs=num_train_epochs,
            beta=self.cfg.beta,
            max_grad_norm=self.cfg.grad_clip_norm,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            model_init_kwargs={"attn_implementation": "eager", "dtype": "bfloat16"},
        )

    def train_on_generation(self, outcomes: list[ProposalOutcome], epochs: int = 1) -> str:
        """One GRPO pass over a generation's tasks. Returns the saved adapter path."""
        if not outcomes:
            log.warning("train_on_generation called with no outcomes; skipping")
            return ""
        dataset = outcomes_to_dataset(outcomes)
        log.info("building GRPO trainer for %d tasks (epochs=%d)", len(dataset), epochs)
        trainer = GRPOTrainer(
            model=self.cfg.model_path,
            reward_funcs=make_reward_fn(self.verifier),
            args=self._grpo_args(num_train_epochs=epochs),
            train_dataset=dataset,
            peft_config=self.lora_config,
        )
        trainer.train()
        adapter_path = str(Path(self.cfg.output_dir) / "adapter")
        trainer.model.save_pretrained(adapter_path)
        log.info("saved adapter to %s", adapter_path)
        return adapter_path
