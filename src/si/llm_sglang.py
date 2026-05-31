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
        enable_thinking: bool | None = None,  # Qwen3-family reasoning toggle; None = template default
    ) -> None:
        self.enable_thinking = enable_thinking
        if cuda_visible_devices is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
        log.info("Loading SGLangLLM from %s  quant=%s", model_path, quantization)
        from sglang import Engine
        from transformers import AutoTokenizer

        # SGLang's Engine.generate(text=...) takes raw strings — it does NOT
        # apply a chat template (that only happens in the OpenAI server
        # endpoint). We render the chat template ourselves with the HF
        # tokenizer, matching how GemmaLLM relies on vLLM's .chat() helper.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
        rendered: list[str] = []
        for p in user_prompts:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            tmpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if self.enable_thinking is not None:
                tmpl_kwargs["enable_thinking"] = self.enable_thinking
            rendered.append(self.tokenizer.apply_chat_template(msgs, **tmpl_kwargs))
        sampling = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_new_tokens": params.max_tokens,
            "n": params.n,
        }
        if params.stop:
            sampling["stop"] = params.stop
        outs = self.engine.generate(prompt=rendered, sampling_params=sampling)
        if isinstance(outs, dict):  # single-prompt edge case
            outs = [outs]
        # For n>1, SGLang expands the batch block-major (`text = text * n`,
        # see io_struct.py:_expand_inputs) and returns a FLAT list of B*n
        # dicts: outs[i + k*B] is the k-th sample of prompt i. Regroup.
        b = len(rendered)
        n = params.n
        return [[outs[i + k * b]["text"] for k in range(n)] for i in range(b)]

    def close(self) -> None:
        """Release GPU memory. Important to call between chunked runs."""
        if hasattr(self, "engine"):
            try:
                self.engine.shutdown()
            except Exception as e:
                log.warning("SGLangLLM shutdown error: %s", e)
