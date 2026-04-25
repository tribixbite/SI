"""Unsloth + TRL DPOTrainer wrapper for Mix 2.

DPO uses (prompt, chosen, rejected) triples. The current policy is the LoRA-
adapted model; the reference policy is the same model with the LoRA adapter
*disabled* (Unsloth's `set_adapter("__base__")` or PEFT's `disable_adapter()`).
TRL's DPOTrainer detects this when `ref_model=None` and the model has PEFT.
"""

from __future__ import annotations

import unsloth  # noqa: F401, I001
from unsloth import FastModel

import logging
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

log = logging.getLogger(__name__)

DEFAULT_UNSLOTH_MODEL = "/home/matilda/git/SI/cache/gemma-4-E4B-unsloth-4bit"


@dataclass
class DPOTrainerConfig:
    model_path: str = DEFAULT_UNSLOTH_MODEL
    output_dir: str = "runs/_dpo"
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    lora_rank: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    lr: float = 5e-6  # DPO is sensitive — keep LR small
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    max_steps: int = -1
    grad_clip_norm: float = 1.0
    beta: float = 0.1  # DPO temperature parameter
    random_state: int = 3407


def pairs_to_dataset(pairs: list[dict], tokenizer) -> Dataset:
    """Render each preference pair through the chat template.

    DPO expects 'prompt', 'chosen', 'rejected' fields. We render the prompt
    through the tokenizer's chat template so it includes Gemma 4's generation
    prompt; chosen/rejected are kept as raw strings (DPO appends them).
    """
    rows = []
    for p in pairs:
        prompt_text = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True
        )
        rows.append(
            {
                "prompt": prompt_text,
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
        )
    return Dataset.from_list(rows)


class DPOSITrainer:
    """Mix 2: DPO refinement on (passing, failing) preference pairs."""

    def __init__(self, cfg: DPOTrainerConfig, *, warm_start_adapter: str | None = None) -> None:
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        log.info("Loading FastModel from %s (warm_start_adapter=%s)", cfg.model_path, warm_start_adapter)
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=cfg.model_path,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            fast_inference=False,
            full_finetuning=False,
        )
        self.model = FastModel.get_peft_model(
            self.model,
            r=cfg.lora_rank,
            target_modules=r"^.*\blanguage_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            lora_alpha=cfg.lora_rank * 2,
            lora_dropout=cfg.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.random_state,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
        # If warm-starting, load existing adapter weights into the freshly
        # initialized PEFT model (matches target_modules from above).
        if warm_start_adapter:
            from peft import PeftModel
            log.info("Loading warm-start adapter weights from %s", warm_start_adapter)
            # Replace the base PEFT module with the saved adapter
            self.model = PeftModel.from_pretrained(
                self.model.get_base_model(), warm_start_adapter, is_trainable=True
            )

    def _dpo_args(self) -> DPOConfig:
        return DPOConfig(
            output_dir=self.cfg.output_dir,
            learning_rate=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            num_train_epochs=self.cfg.epochs,
            max_steps=self.cfg.max_steps,
            max_grad_norm=self.cfg.grad_clip_norm,
            max_length=self.cfg.max_seq_length,
            max_prompt_length=self.cfg.max_prompt_length,
            beta=self.cfg.beta,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=True,
            seed=self.cfg.random_state,
        )

    def train_on_pairs(self, pairs: list[dict]) -> str:
        if not pairs:
            log.warning("train_on_pairs called with no pairs; skipping")
            return ""
        dataset = pairs_to_dataset(pairs, self.tokenizer)
        log.info("DPO: %d preference pairs", len(dataset))
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # PEFT model: ref = adapter-disabled base (TRL handles)
            args=self._dpo_args(),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()
        adapter_path = str(Path(self.cfg.output_dir) / "adapter")
        trainer.model.save_pretrained(adapter_path)
        log.info("saved DPO adapter to %s", adapter_path)
        return adapter_path
