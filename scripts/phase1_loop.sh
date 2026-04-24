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

RUN_ID="${1:?usage: $0 <run_id> <n_generations> [anchor_every=5] [proposals_per_type=32] [mc_rollouts=8]}"
N_GEN="${2:?missing n_generations}"
ANCHOR_EVERY="${3:-5}"
PROPOSALS_PER_TYPE="${4:-32}"
MC_ROLLOUTS="${5:-8}"

: "${SI_ROOT:=/home/matilda/git/SI}"
: "${SI_MODEL_PATH:=${SI_ROOT}/cache/gemma-4-E4B-hf}"
export SI_ROOT SI_MODEL_PATH
# Reduce OOM risk during GRPO on 3090 by letting allocator expand segments.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

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
    ${VENV_PY} -m pip freeze 2>/dev/null | grep -iE "^(torch|transformers|vllm|peft|trl|evalplus|accelerate|sandbox-fusion|unsloth)" || true
} > "${RUN_DIR}/VERSIONS.lock"

# Per-gen log aggregator so a monitor can tail runs/<id>/phase1.log.
# Also appends GPU mem / process liveness stamps per step.
MASTER_LOG="${RUN_DIR}/phase1.log"
_stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
_heartbeat() {
    # gen=$1 step=$2 (rollout|train|anchor)
    local gen="$1" step="$2" meta="${3:-}"
    local free used
    read -r used free < <(nvidia-smi --query-gpu=memory.used,memory.free \
        --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 | tr ',' ' ')
    local cpu_mem
    cpu_mem=$(free -m | awk '/^Mem:/{print $3"/"$2"MB"}')
    printf '%s  gen=%s  step=%s  gpu_used=%sMiB  gpu_free=%sMiB  ram=%s  %s\n' \
        "$(_stamp)" "${gen}" "${step}" "${used:-?}" "${free:-?}" "${cpu_mem}" "${meta}" \
        >> "${MASTER_LOG}"
}
echo "# phase1.log for ${RUN_ID}" > "${MASTER_LOG}"
_heartbeat 0 start "n_gen=${N_GEN} anchor_every=${ANCHOR_EVERY} proposals_per_type=${PROPOSALS_PER_TYPE}"

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

    ROLLOUT_LOG="${RUN_DIR}/rollout_gen${GEN4}.log"
    TRAIN_LOG="${RUN_DIR}/train_gen${GEN4}.log"
    ANCHOR_LOG="${RUN_DIR}/anchor_gen${GEN4}.log"

    if [[ -f "${OUTCOMES}" ]]; then
        echo "  skip rollout — ${OUTCOMES} exists"
        _heartbeat "${g}" rollout-skip
    else
        echo "--- rollout ---"
        _heartbeat "${g}" rollout-start
        if [[ -d "${ADAPTER_PATH}" ]]; then
            ${VENV_PY} -m si.cli rollout --out-dir "${RUN_DIR}" --gen "${g}" \
                --proposals-per-type "${PROPOSALS_PER_TYPE}" --mc-rollouts "${MC_ROLLOUTS}" \
                --adapter "${ADAPTER_PATH}" \
                &> "${ROLLOUT_LOG}"
        else
            ${VENV_PY} -m si.cli rollout --out-dir "${RUN_DIR}" --gen "${g}" \
                --proposals-per-type "${PROPOSALS_PER_TYPE}" --mc-rollouts "${MC_ROLLOUTS}" \
                &> "${ROLLOUT_LOG}"
        fi
        _heartbeat "${g}" rollout-done "log=${ROLLOUT_LOG}"
    fi

    echo "--- train ---"
    _heartbeat "${g}" train-start
    TRAIN_ARGS=(--outcomes "${OUTCOMES}" --adapter-out "${ADAPTER_PATH}")
    if [[ -d "${ADAPTER_PATH}" ]]; then
        TRAIN_ARGS+=(--adapter-in "${ADAPTER_PATH}")
    fi
    ${VENV_PY} -m si.cli train "${TRAIN_ARGS[@]}" &> "${TRAIN_LOG}"
    _heartbeat "${g}" train-done "log=${TRAIN_LOG}"

    # Snapshot adapter per generation so we can compare trajectories + support
    # revert on anchor regression. Point-in-time copy; rolling adapter still
    # lives at ${ADAPTER_PATH} for next gen's resume.
    SNAPSHOT_DIR="${RUN_DIR}/adapter_gen${GEN4}"
    if [[ -d "${ADAPTER_PATH}" && ! -d "${SNAPSHOT_DIR}" ]]; then
        cp -r "${ADAPTER_PATH}" "${SNAPSHOT_DIR}"
        _heartbeat "${g}" snapshot "path=${SNAPSHOT_DIR}"
    fi

    if (( g % ANCHOR_EVERY == 0 )); then
        echo "--- anchor(gen${g}) ---"
        _heartbeat "${g}" anchor-start
        if ${VENV_PY} -m si.cli anchor --adapter "${ADAPTER_PATH}" \
            --out "${RUN_DIR}/anchor_gen${GEN4}.json" &> "${ANCHOR_LOG}"; then
            PASS=$(python3 -c "import json; d=json.load(open('${RUN_DIR}/anchor_gen${GEN4}.json')); print(f\"{d['pass_at_1']:.4f}\")" 2>/dev/null || echo "?")
            _heartbeat "${g}" anchor-done "pass@1=${PASS}"
        else
            _heartbeat "${g}" anchor-fail "log=${ANCHOR_LOG}"
            echo "anchor failed — continuing"
        fi
    fi
done

echo ""
echo "=== done ${RUN_ID} — see ${RUN_DIR} ==="
