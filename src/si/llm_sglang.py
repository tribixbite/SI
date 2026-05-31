"""SGLang inference wrapper — drop-in replacement for `si.llm.GemmaLLM`.

Why a sibling module: SGLang 0.5.12+ has day-0 support for Qwen3.6 / Gemma 4
/ GLM-5.1 and a confirmed WSL2 recipe. Crucially, it does NOT hit the
vLLM 0.19.1 `cudaErrorUnknown` async-event-sync bug that blocks Qwen3-Coder
runs past ~2h cumulative session time (see memory:
project_qwen3coder_blocked).

Interface contract mirrors `GemmaLLM.chat_batch`:
    chat_batch(user_prompts, GenParams, system=None) -> list[list[str]]

Run from the parallel venv `.venv-sglang` (sglang ships its own torch/vLLM-
free stack). The existing `.venv` keeps Unsloth + TRL for training.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from si.llm import GenParams

log = logging.getLogger(__name__)


class SGLangLLM:
    """Thin wrapper around `sglang.Engine` for chat-mode batched generation.

    Engine vs server: we use the in-process Engine (lower latency, no HTTP
    overhead). For multi-process fault isolation (analogous to the
    subprocess-chunked anchor we ship), launch the CLI's `anchor_chunked.sh`
    around this — each chunk gets a fresh Engine in a fresh Python process.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        cuda_visible_devices: str | None = "1",
        quantization: str | None = None,  # "awq", "awq_marlin", "gptq", "fp8", None
        chat_template: str | None = None,  # path or built-in name; None = auto-detect
    ) -> None:
        if cuda_visible_devices is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
        log.info("Loading SGLangLLM from %s  quant=%s", model_path, quantization)
        from sglang.srt.server_args import ServerArgs
        from sglang import Engine

        engine_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "dtype": dtype,
            "mem_fraction_static": gpu_memory_utilization,
            "context_length": max_model_len,
            "disable_cuda_graph": True,  # mirrors vLLM enforce_eager=True; safer on WSL2
            "skip_tokenizer_init": False,
        }
        if quantization is not None:
            engine_kwargs["quantization"] = quantization
        if chat_template is not None:
            engine_kwargs["chat_template"] = chat_template
        # SGLang's ServerArgs accepts more kwargs than Engine; pass via dict.
        self.engine = Engine(**engine_kwargs)

    def chat_batch(
        self,
        user_prompts: list[str],
        params: GenParams,
        system: str | None = None,
    ) -> list[list[str]]:
        """Generate N=params.n completions per prompt. Returns [[c1, c2, ...], ...].

        Matches GemmaLLM.chat_batch's return shape exactly so it slots into
        livecodebench.py without changes.
        """
        conversations = []
        for p in user_prompts:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            conversations.append(msgs)
        sampling = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_new_tokens": params.max_tokens,
            "n": params.n,
        }
        if params.stop:
            sampling["stop"] = params.stop
        # SGLang's chat endpoint takes prompts as the list of messages and
        # returns a dict per prompt with `output_ids`/`text`/`meta_info`.
        # When n>1, the response includes a list of outputs per prompt.
        outs = self.engine.generate(
            input_ids=None,
            sampling_params=sampling,
            prompt=conversations if isinstance(conversations[0], list) else conversations,
        )
        # SGLang returns a list of dicts (one per prompt) when n=1, or a
        # list of lists when n>1. Normalize to list[list[str]].
        normalized: list[list[str]] = []
        for out in outs:
            if isinstance(out, list):
                normalized.append([o["text"] for o in out])
            else:
                normalized.append([out["text"]])
        return normalized

    def close(self) -> None:
        """Release GPU memory. Important to call between chunked runs."""
        if hasattr(self, "engine"):
            try:
                self.engine.shutdown()
            except Exception as e:
                log.warning("SGLangLLM shutdown error: %s", e)
