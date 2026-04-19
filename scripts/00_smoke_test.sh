#!/usr/bin/env bash
# SI — Phase 0 smoke test. Verifies base model inference + sandbox + evalplus.
# Runs in ~5 minutes on 1× 3090. No training, no LoRA, no loop.
set -euo pipefail

: "${SI_ROOT:=$(pwd)}"
: "${SI_CACHE:=${SI_ROOT}/cache}"

if [[ ! -d "${SI_ROOT}/.venv" ]]; then
    echo "ERROR: .venv not found. Run scripts/bootstrap.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${SI_ROOT}/.venv/bin/activate"

echo "=== SI smoke test ==="
echo

# --- 1. Unit tests that don't need GPU ---------------------------------------

echo "[1/4] pytest (pure-Python modules)"
cd "${SI_ROOT}"
pytest tests/ -q || { echo "FAIL: unit tests"; exit 1; }
echo "  OK"
echo

# --- 2. Base model generates plausible output --------------------------------

MODEL_PATH="${SI_CACHE}/gemma-4-E4B-hf"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Base model not found at ${MODEL_PATH}" >&2
    echo "       Download with: huggingface-cli download google/gemma-4-E4B-it --local-dir ${MODEL_PATH}" >&2
    exit 1
fi

echo "[2/4] vLLM inference on Gemma 4 E4B"
python - <<'PY'
import sys, os
from vllm import LLM, SamplingParams

model_path = os.environ["SI_CACHE"] + "/gemma-4-E4B-hf"
llm = LLM(model_path, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
params = SamplingParams(temperature=0.2, max_tokens=128, stop=["\n\n"])
prompts = [
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "# Python function that returns True iff x is prime\ndef is_prime(x):\n",
]
for p in prompts:
    out = llm.generate([p], params)
    gen = out[0].outputs[0].text.strip()
    if len(gen) < 10:
        print(f"FAIL: empty/short generation for prompt: {p!r}", file=sys.stderr)
        sys.exit(1)
    print(f"  prompt[:30]={p[:30]!r}...")
    print(f"  gen[:80]   ={gen[:80]!r}...")
print("  OK")
PY
echo

# --- 3. Sandbox image is pulled (AZR will launch container at runtime) -------

echo "[3/4] sandbox image check"
SANDBOX_IMAGE="volcengine/sandbox-fusion:server-20250609"
if docker image inspect "${SANDBOX_IMAGE}" >/dev/null 2>&1; then
    echo "  OK (${SANDBOX_IMAGE} present locally)"
else
    echo "  WARN: ${SANDBOX_IMAGE} not pulled; AZR will pull at runtime (slow first run)." >&2
    echo "        Pre-pull with: docker pull ${SANDBOX_IMAGE}"
fi
echo

# --- 4. evalplus loads (fast static check, no model eval) --------------------

echo "[4/4] evalplus importable"
python -c "import evalplus; import evalplus.data; print('  OK (evalplus', evalplus.__version__ if hasattr(evalplus,'__version__') else '?', ')')"
echo

echo "=== smoke test passed ==="
echo "Next: bash scripts/01_phase1_run.sh"
