#!/usr/bin/env bash
# SI Phase 1 orchestrator. Ping-pongs rollout (vLLM) <-> train (HF+GRPO)
# subprocesses because Gemma 4 E4B can't fit both trainers on one 3090.
#
# Usage:
#   bash scripts/phase1_loop.sh <run_id> <n_generations> [anchor_every] [proposals_per_type]
#
# Writes everything under: runs/<run_id>/
#   outcomes_gen0001.jsonl, outcomes_gen0002.jsonl, ...
#   metrics_gen0001.json, ...
#   adapter/                (latest LoRA adapter; overwritten each gen)
#   anchor_gen0010.json     (every anchor_every gens)
#   VERSIONS.lock           (snapshot of env versions)
#
# Safety: refuses to start with uncommitted changes under configs/ or src/si/.

set -euo pipefail

RUN_ID="${1:?usage: $0 <run_id> <n_generations> [anchor_every=10] [proposals_per_type=8]}"
N_GEN="${2:?missing n_generations}"
ANCHOR_EVERY="${3:-10}"
PROPOSALS_PER_TYPE="${4:-8}"

: "${SI_ROOT:=/home/matilda/git/SI}"
: "${SI_MODEL_PATH:=${SI_ROOT}/cache/gemma-4-E4B-hf}"
export SI_ROOT SI_MODEL_PATH

# Reproducibility gate: no uncommitted code/config changes mid-run.
if ! git -C "${SI_ROOT}" diff --quiet configs/ src/si/; then
    echo "ERROR: uncommitted changes under configs/ or src/si/ — commit or stash first." >&2
    git -C "${SI_ROOT}" status --short configs/ src/si/ >&2
    exit 1
fi

RUN_DIR="${SI_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
VENV_PY="${SI_ROOT}/.venv/bin/python"

echo "=== SI Phase 1 — run_id=${RUN_ID} gens=${N_GEN} anchor_every=${ANCHOR_EVERY} ==="
echo "  SI_ROOT       = ${SI_ROOT}"
echo "  SI_MODEL_PATH = ${SI_MODEL_PATH}"
echo "  RUN_DIR       = ${RUN_DIR}"

# Snapshot versions
{
    echo "# Phase 1 run ${RUN_ID} — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    git -C "${SI_ROOT}" log -1 --format='code: %H %s'
    echo "python: $(${VENV_PY} --version 2>&1)"
    ${VENV_PY} -m pip freeze 2>/dev/null | grep -iE "^(torch|transformers|vllm|peft|trl|evalplus|accelerate|sandbox-fusion)" || true
} > "${RUN_DIR}/VERSIONS.lock"

# Optional: base-model anchor before training (gen 0 eval)
if [[ ! -f "${RUN_DIR}/anchor_gen0000.json" ]]; then
    echo "--- anchor(base) at gen 0 ---"
    ${VENV_PY} -m si.cli anchor --out "${RUN_DIR}/anchor_gen0000.json" || true
fi

ADAPTER_PATH="${RUN_DIR}/adapter"
for ((g=1; g<=N_GEN; g++)); do
    GEN4=$(printf "%04d" "${g}")
    OUTCOMES="${RUN_DIR}/outcomes_gen${GEN4}.jsonl"

    echo ""
    echo "=== generation ${g}/${N_GEN} ==="

    if [[ -f "${OUTCOMES}" ]]; then
        echo "  skip rollout — ${OUTCOMES} exists"
    else
        echo "--- rollout ---"
        if [[ -d "${ADAPTER_PATH}" ]]; then
            ${VENV_PY} -m si.cli rollout --out-dir "${RUN_DIR}" --gen "${g}" \
                --proposals-per-type "${PROPOSALS_PER_TYPE}" --adapter "${ADAPTER_PATH}"
        else
            ${VENV_PY} -m si.cli rollout --out-dir "${RUN_DIR}" --gen "${g}" \
                --proposals-per-type "${PROPOSALS_PER_TYPE}"
        fi
    fi

    echo "--- train ---"
    TRAIN_ARGS=(--outcomes "${OUTCOMES}" --adapter-out "${ADAPTER_PATH}")
    if [[ -d "${ADAPTER_PATH}" ]]; then
        TRAIN_ARGS+=(--adapter-in "${ADAPTER_PATH}")
    fi
    ${VENV_PY} -m si.cli train "${TRAIN_ARGS[@]}"

    if (( g % ANCHOR_EVERY == 0 )); then
        echo "--- anchor(gen${g}) ---"
        ${VENV_PY} -m si.cli anchor --adapter "${ADAPTER_PATH}" \
            --out "${RUN_DIR}/anchor_gen${GEN4}.json" || echo "anchor failed — continuing"
    fi
done

echo ""
echo "=== done ${RUN_ID} — see ${RUN_DIR} ==="
