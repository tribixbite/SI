#!/usr/bin/env bash
# SI — Phase 1 runner. Reproduces AZR on Gemma 4 E4B.
# See docs/04-implementation.md §Phase 1.
set -euo pipefail

: "${SI_ROOT:=$(pwd)}"
: "${SI_DEPS:=${SI_ROOT}/deps}"
: "${SI_CACHE:=${SI_ROOT}/cache}"

# Refuse to start with uncommitted config changes. Reproducibility > convenience.
if ! git -C "${SI_ROOT}" diff --quiet configs/; then
    echo "ERROR: uncommitted changes under configs/. Commit or stash before running." >&2
    git -C "${SI_ROOT}" status --short configs/
    exit 1
fi

CONFIG_HASH=$(git -C "${SI_ROOT}" log -1 --format=%H -- configs/tier1_e4b.yaml || echo "UNTRACKED")
RUN_ID="phase1_$(date -u +"%Y%m%dT%H%M%SZ")_${CONFIG_HASH:0:8}"
RUN_DIR="${SI_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "SI Phase 1 — run_id=${RUN_ID}"
cp "${SI_ROOT}/configs/tier1_e4b.yaml" "${RUN_DIR}/config.yaml"
cp "${SI_ROOT}/VERSIONS.lock" "${RUN_DIR}/VERSIONS.lock" 2>/dev/null || \
    echo "WARN: no VERSIONS.lock; run scripts/bootstrap.sh first" >&2

# shellcheck disable=SC1091
source "${SI_ROOT}/.venv/bin/activate"

# 1. Seed data generation (AZR §B)
if [[ ! -f "${SI_ROOT}/data/gemma4_ded_abd_seed.jsonl" ]]; then
    echo "Generating AZR seed data..."
    cd "${SI_DEPS}/Absolute-Zero-Reasoner"
    export OUTPUT_SEED_PATH="${SI_ROOT}/data/gemma4_ded_abd_seed.jsonl"
    export OUTPUT_CODE_F_SEED_PATH="${SI_ROOT}/data/gemma4_ind_seed.jsonl"
    mkdir -p "${SI_ROOT}/data"
    # Adapt coder7b template for Gemma 4 E4B; AZR ships no gemma4 seeder yet.
    if [[ ! -f scripts/seeding/gemma4_e4b.sh ]]; then
        sed -e "s|Qwen2.5-7B-Coder|${SI_CACHE}/gemma-4-E4B-hf|g" \
            scripts/seeding/coder7b.sh > scripts/seeding/gemma4_e4b.sh
        chmod +x scripts/seeding/gemma4_e4b.sh
    fi
    bash scripts/seeding/gemma4_e4b.sh 2>&1 | tee "${RUN_DIR}/seed.log"
    cd "${SI_ROOT}"
fi

# 2. Self-play training
echo "Starting AZR self-play..."
cd "${SI_DEPS}/Absolute-Zero-Reasoner"
if [[ ! -f scripts/selfplay/gemma4_e4b.sh ]]; then
    sed -e "s|Qwen2.5-7B-Coder|${SI_CACHE}/gemma-4-E4B-hf|g" \
        scripts/selfplay/coder7b.sh > scripts/selfplay/gemma4_e4b.sh
    chmod +x scripts/selfplay/gemma4_e4b.sh
fi
bash scripts/selfplay/gemma4_e4b.sh 2>&1 | tee "${RUN_DIR}/train.log"

echo "Phase 1 complete. Run anchor eval with scripts/02_anchor_eval.sh ${RUN_ID}"
