"""Shared Gemma 4 vLLM wrapper used by Proposer + Solver.

One vLLM engine per process. Role-switching (proposer vs solver) happens at the
prompt level, not the model level — same weights, different system prompt /
sampling params. This matches the spec (docs/00-overview.md §"inner loop").
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

# vLLM imports are lazy (inside GemmaLLM.__init__) so this module can be
# imported in environments without vLLM installed — e.g. the .venv-sglang
# parallel environment we use for the SGLang backend. The GenParams dataclass
# below is the only thing si.livecodebench needs from this module before it
# picks an LLM backend.

log = logging.getLogger(__name__)


@dataclass
class GenParams:
    temperature: float
    top_p: float = 0.95
    max_tokens: int = 2048
    n: int = 1
    stop: list[str] | None = None

    def to_vllm(self):
        from vllm import SamplingParams
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
        max_model_len: int = 8192,
        enforce_eager: bool = True,
        cuda_visible_devices: str | None = "1",
        lora_path: str | None = None,
        max_lora_rank: int = 64,
        enable_lora: bool = False,
    ) -> None:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        if cuda_visible_devices is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
        # LoRA is enabled if an initial adapter is given OR enable_lora is set
        # (Phase 2 swaps a different branch adapter in per chat_batch call).
        lora_enabled = enable_lora or lora_path is not None
        log.info("Loading Gemma LLM from %s  lora=%s enable_lora=%s", model_path, lora_path, lora_enabled)
        llm_kwargs = {
            "model": model_path,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
        }
        if lora_enabled:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank
            llm_kwargs["max_loras"] = 1
        self.llm = LLM(**llm_kwargs)
        self._LoRARequest = LoRARequest
        # path -> LoRARequest, with a stable unique int id per path so vLLM can
        # cache adapters across swaps.
        self._lora_registry: dict[str, object] = {}
        self._lora_request = None
        if lora_path is not None:
            self._lora_request = self._resolve_lora(lora_path)

    def _resolve_lora(self, lora_path: str | None):
        """Return a cached LoRARequest for this adapter path (None if no path)."""
        if lora_path is None:
            return None
        req = self._lora_registry.get(lora_path)
        if req is None:
            req = self._LoRARequest(
                lora_name=lora_path,
                lora_int_id=len(self._lora_registry) + 1,
                lora_path=lora_path,
            )
            self._lora_registry[lora_path] = req
        return req

    def chat_batch(
        self,
        user_prompts: list[str],
        params: GenParams,
        system: str | None = None,
        lora_path: str | None = None,
    ) -> list[list[str]]:
        """Generate N=params.n completions per prompt. Returns [[comp1, comp2, ...], ...].

        `lora_path` swaps in a specific branch adapter for this call (Phase 2);
        when omitted, the adapter passed at construction (if any) is used."""
        conversations = []
        for p in user_prompts:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            conversations.append(msgs)
        chat_kwargs = {}
        lora_request = self._resolve_lora(lora_path) if lora_path is not None else self._lora_request
        if lora_request is not None:
            chat_kwargs["lora_request"] = lora_request
        outs = self.llm.chat(conversations, params.to_vllm(), **chat_kwargs)
        return [[completion.text for completion in o.outputs] for o in outs]
