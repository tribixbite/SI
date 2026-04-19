#!/usr/bin/env bash
# SI — anchor eval for a completed or mid-run checkpoint.
# Usage: bash scripts/02_anchor_eval.sh <run_id> [humaneval|mbpp|livecodebench]
set -euo pipefail

: "${SI_ROOT:=$(pwd)}"
: "${SI_DEPS:=${SI_ROOT}/deps}"

RUN_ID="${1:?usage: $0 <run_id> [benchmark]}"
BENCH="${2:-humaneval}"
RUN_DIR="${SI_ROOT}/runs/${RUN_ID}"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: no run at ${RUN_DIR}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${SI_ROOT}/.venv/bin/activate"

# Find latest actor checkpoint.
ACTOR_CKPT=$(find "${RUN_DIR}" -type d -name "actor" | head -1)
if [[ -z "${ACTOR_CKPT}" ]]; then
    ACTOR_CKPT=$(find "${SI_DEPS}/Absolute-Zero-Reasoner/checkpoints" -type d -name "actor" | head -1 || true)
fi
if [[ -z "${ACTOR_CKPT}" ]]; then
    echo "ERROR: no actor checkpoint found. Train Phase 1 first." >&2
    exit 1
fi

HF_CKPT="${RUN_DIR}/hf_ckpt"
echo "Converting veRL checkpoint to HF format..."
python -m absolute_zero_reasoner.utils.convert2hf \
    "${ACTOR_CKPT}" \
    "${ACTOR_CKPT}/huggingface/" \
    "${HF_CKPT}"

echo "Running ${BENCH} evaluation..."
bash "${SI_DEPS}/Absolute-Zero-Reasoner/evaluation/code_eval/scripts/run_evalplus.sh" \
    0 "${BENCH}" "${HF_CKPT}" \
    2>&1 | tee "${RUN_DIR}/anchor_${BENCH}.log"

# Extract pass@1
PASS1=$(grep -oP 'pass@1:\s*\K[0-9.]+' "${RUN_DIR}/anchor_${BENCH}.log" | tail -1 || echo "?")
echo "Anchor result: ${BENCH} pass@1 = ${PASS1}"
echo "${RUN_ID},${BENCH},${PASS1},$(date -u +%FT%TZ)" >> "${SI_ROOT}/anchor_log.csv"
