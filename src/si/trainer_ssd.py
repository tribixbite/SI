"""Unsloth + TRL SFT trainer for SSD (arXiv:2604.01193).

Pure SFT on (prompt, completion) pairs. Much simpler than GRPO since there's
no advantage math, no reward function, no reviser — just cross-entropy on the
completion tokens conditioned on the chat-templated prompt.
"""

from __future__ import annotations

# Unsloth first: patches HF internals before the other imports.
import unsloth  # noqa: F401, I001
from unsloth import FastModel

import logging
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from si.ssd import SSDSample

log = logging.getLogger(__name__)

DEFAULT_UNSLOTH_MODEL = "/home/matilda/git/SI/cache/gemma-4-E4B-unsloth-4bit"


@dataclass
class SSDTrainerConfig:
    model_path: str = DEFAULT_UNSLOTH_MODEL
    output_dir: str = "runs/_ssd"
    max_seq_length: int = 2048
    lora_rank: int = 32
    load_in_4bit: bool = True
    lr: float = 2e-5  # SFT is stable at higher LR than GRPO
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 2
    grad_clip_norm: float = 1.0
    random_state: int = 3407
    packing: bool = True  # packs multiple samples into max_seq_length for efficiency
    lora_dropout: float = 0.05  # ssd_v2 overfit at 0.0; regularize LoRA deltas


def samples_to_dataset(samples: list[SSDSample], tokenizer) -> Dataset:
    """Render each sample through the chat template. The 'messages' field is the
    conversation in Gemma 4's multimodal typed-blocks format; we append the
    assistant turn with the passing completion.

    Unsloth's SFTTrainer accepts a dataset with a single `text` column where each
    row is the full chat-rendered string. The loss masks the prompt tokens and
    trains only on the completion tokens.
    """
    rows = []
    for s in samples:
        full_messages = list(s.prompt_messages) + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": s.completion_text}],
            }
        ]
        rendered = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        rows.append({"text": rendered})
    return Dataset.from_list(rows)


class SSDTrainer:
    """Unsloth FastModel + TRL SFTTrainer wrapper for SSD training."""

    def __init__(self, cfg: SSDTrainerConfig) -> None:
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        log.info("Loading Unsloth FastModel from %s", cfg.model_path)
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

    def _sft_args(self) -> SFTConfig:
        return SFTConfig(
            output_dir=self.cfg.output_dir,
            learning_rate=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            num_train_epochs=self.cfg.epochs,
            max_grad_norm=self.cfg.grad_clip_norm,
            max_seq_length=self.cfg.max_seq_length,
            packing=self.cfg.packing,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=True,
            seed=self.cfg.random_state,
        )

    def train_on_samples(self, samples: list[SSDSample]) -> str:
        if not samples:
            log.warning("train_on_samples called with no samples; skipping")
            return ""
        dataset = samples_to_dataset(samples, self.tokenizer)
        log.info("SSD SFT: %d samples", len(dataset))
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=self._sft_args(),
        )
        trainer.train()
        adapter_path = str(Path(self.cfg.output_dir) / "adapter")
        trainer.model.save_pretrained(adapter_path)
        log.info("saved SSD adapter to %s", adapter_path)
        return adapter_path
