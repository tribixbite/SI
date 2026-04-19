#!/usr/bin/env bash
# SI — environment bootstrap. Run once per machine.
# See docs/04-implementation.md §Phase 0 for rationale.
set -euo pipefail

: "${SI_ROOT:=$(pwd)}"
: "${SI_DEPS:=${SI_ROOT}/deps}"
: "${SI_CACHE:=${SI_ROOT}/cache}"

echo "SI bootstrap"
echo "  SI_ROOT  = ${SI_ROOT}"
echo "  SI_DEPS  = ${SI_DEPS}"
echo "  SI_CACHE = ${SI_CACHE}"

mkdir -p "${SI_DEPS}" "${SI_CACHE}" "${SI_ROOT}/logs" "${SI_ROOT}/runs"

# --- Sanity checks ------------------------------------------------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi not found. SI requires NVIDIA GPUs." >&2
fi

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "")
if [[ "${CUDA_VERSION}" == "13.2" ]]; then
    echo "ERROR: CUDA 13.2 detected. Gemma 4 GGUFs produce bad outputs on 13.2." >&2
    echo "       Pin to CUDA 12.4 (see docs/01-sources.md#base-model)." >&2
    exit 1
fi

# --- Clone dependency repositories -------------------------------------------

clone_or_update() {
    local url="$1"
    local dir="$2"
    local pin="${3:-}"
    if [[ -d "${SI_DEPS}/${dir}/.git" ]]; then
        echo "  [skip] ${dir} already cloned"
    else
        echo "  [clone] ${url}"
        git clone "${url}" "${SI_DEPS}/${dir}"
    fi
    if [[ -n "${pin}" ]]; then
        ( cd "${SI_DEPS}/${dir}" && git checkout "${pin}" )
    fi
}

echo "Cloning dependencies..."
clone_or_update https://github.com/LeapLabTHU/Absolute-Zero-Reasoner.git Absolute-Zero-Reasoner
clone_or_update https://github.com/verl-project/verl.git verl
clone_or_update https://github.com/andborth/RoboPhD.git RoboPhD
clone_or_update https://github.com/codelion/openevolve.git openevolve

# --- Python environment ------------------------------------------------------

if [[ ! -d "${SI_ROOT}/.venv" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    echo "Creating venv (Python 3.11)..."
    uv python install 3.11
    uv venv --python 3.11 "${SI_ROOT}/.venv"
fi

# shellcheck disable=SC1091
source "${SI_ROOT}/.venv/bin/activate"

echo "Installing AZR's pinned requirements (authoritative for torch/vllm/transformers versions)..."
# AZR's requirements.txt is a full freeze; installing it first pins the whole stack.
# Later installs use --no-deps to avoid clobbering these versions.
uv pip install -r "${SI_DEPS}/Absolute-Zero-Reasoner/requirements.txt"

echo "Installing verl from source (no-deps: keep AZR pins)..."
uv pip install -e "${SI_DEPS}/verl" --no-deps

echo "Installing SI (this repo) in editable mode (no-deps)..."
uv pip install -e "${SI_ROOT}" --no-deps

echo "Installing EvalPlus at pinned commit (no-deps to preserve torch/vllm pins)..."
uv pip install --no-deps \
    "evalplus @ git+https://github.com/evalplus/evalplus@d362e933265c3e7e3df8101c930a89c3c470cd9f"

echo "Installing SI dev extras..."
uv pip install pytest pytest-asyncio ruff mypy rich typer

# --- Sandbox -----------------------------------------------------------------

if command -v docker >/dev/null 2>&1; then
    # AZR spawns sandbox-fusion containers programmatically with dynamic host ports
    # (see absolute_zero_reasoner/utils/code_utils/sandboxfusion_executor.py).
    # We only need the image pre-pulled locally.
    SANDBOX_IMAGE="volcengine/sandbox-fusion:server-20250609"
    echo "Pre-pulling sandbox-fusion image (${SANDBOX_IMAGE})..."
    docker image inspect "${SANDBOX_IMAGE}" >/dev/null 2>&1 || \
        docker pull "${SANDBOX_IMAGE}" || \
        echo "  WARN: pull failed; AZR will attempt to pull at runtime." >&2
else
    echo "WARN: Docker not found. Install Docker before running Phase 1." >&2
fi

# --- Freeze version lock -----------------------------------------------------

{
    echo "# SI VERSIONS.lock — generated $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "python: $(python --version 2>&1)"
    echo "cuda: ${CUDA_VERSION}"
    for dep in Absolute-Zero-Reasoner verl RoboPhD openevolve; do
        if [[ -d "${SI_DEPS}/${dep}/.git" ]]; then
            sha=$( cd "${SI_DEPS}/${dep}" && git rev-parse HEAD )
            echo "${dep}: ${sha}"
        fi
    done
    echo ""
    echo "# pip freeze"
    pip freeze
} > "${SI_ROOT}/VERSIONS.lock"

echo ""
echo "Bootstrap complete."
echo "  VERSIONS.lock written."
echo "  Next: set SI_CACHE and run 'bash scripts/00_smoke_test.sh'."
