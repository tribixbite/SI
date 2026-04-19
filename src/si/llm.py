"""Shared Gemma 4 vLLM wrapper used by Proposer + Solver.

One vLLM engine per process. Role-switching (proposer vs solver) happens at the
prompt level, not the model level — same weights, different system prompt /
sampling params. This matches the spec (docs/00-overview.md §"inner loop").
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)


@dataclass
class GenParams:
    temperature: float
    top_p: float = 0.95
    max_tokens: int = 2048
    n: int = 1
    stop: list[str] | None = None

    def to_vllm(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=self.n,
            stop=self.stop,
        )


class GemmaLLM:
    """Thin wrapper around vllm.LLM for chat-mode batched generation.

    Always uses the tokenizer's chat template; never raw completion
    (see memory: Gemma 4 E4B inference baseline — -it variant demands chat
    template or it goes into repetition loops).
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
        cuda_visible_devices: str | None = "1",
    ) -> None:
        if cuda_visible_devices is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
        log.info("Loading Gemma LLM from %s", model_path)
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )

    def chat_batch(
        self,
        user_prompts: list[str],
        params: GenParams,
        system: str | None = None,
    ) -> list[list[str]]:
        """Generate N=params.n completions per prompt. Returns [[comp1, comp2, ...], ...]."""
        conversations = []
        for p in user_prompts:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            conversations.append(msgs)
        outs = self.llm.chat(conversations, params.to_vllm())
        return [[completion.text for completion in o.outputs] for o in outs]
