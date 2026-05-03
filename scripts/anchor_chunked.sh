#!/bin/bash
# Subprocess-chunked LCB anchor — each chunk runs in its own Python/vLLM
# process to avoid the cumulative cudaErrorUnknown bug on vLLM 0.19.1 + AWQ.
#
# Robustness:
#   - per-chunk timeout (default 5400s = 90min) catches hung vLLM that didn't
#     crash but is stuck in async_copy_ready_event.synchronize() (the WSL2
#     CUDA async-event proxy bug, vllm/v1/worker/gpu_model_runner.py:251).
#   - up to MAX_RETRIES=2 attempts per chunk; each retry is a fresh Python
#     process and therefore a fresh CUDA context (resets the allocator and
#     event/stream state).
#   - chunk results are persisted as soon as a chunk succeeds, so the script
#     is resumable if killed externally.
#
# Usage:
#   anchor_chunked.sh <model_path> <bon> <max_tokens> <chunks> <out_path> [extra anchor args...]
# Env overrides:
#   CHUNK_TIMEOUT_S=5400  per-chunk wallclock timeout (seconds)
#   MAX_RETRIES=2         retries per chunk after timeout/failure
set -euo pipefail

MODEL=${1:?model path}
BON=${2:?bon}
MAX_TOK=${3:?max_completion_tokens}
N_CHUNKS=${4:?n_chunks}
OUT=${5:?out merged json}
shift 5
EXTRA="$@"

CHUNK_TIMEOUT_S=${CHUNK_TIMEOUT_S:-5400}
MAX_RETRIES=${MAX_RETRIES:-2}

WORKDIR=$(dirname "$OUT")/$(basename "$OUT" .json)_chunks
mkdir -p "$WORKDIR"

# LCB v6 has 1054 problems.
N_PROBLEMS=1054
CHUNK=$(( (N_PROBLEMS + N_CHUNKS - 1) / N_CHUNKS ))
echo "chunked anchor: model=$MODEL bon=$BON max_tok=$MAX_TOK n_chunks=$N_CHUNKS chunk_size=$CHUNK timeout=${CHUNK_TIMEOUT_S}s retries=$MAX_RETRIES"

run_chunk() {
    local i=$1 off=$2 out_i=$3 log_i=$4 attempt=$5
    echo "$(date -Is) chunk $i attempt $attempt (offset=$off, limit=$CHUNK, timeout=${CHUNK_TIMEOUT_S}s)"
    # `timeout --foreground --kill-after=30s` sends SIGTERM at the deadline,
    # then SIGKILL 30s later if the process didn't exit. --foreground lets
    # Ctrl-C reach us during interactive runs.
    if timeout --foreground --kill-after=30s "${CHUNK_TIMEOUT_S}s" \
        env CUDA_VISIBLE_DEVICES=1 /home/matilda/git/SI/.venv/bin/python -m si.cli anchor \
            --benchmark lcb --bon "$BON" --parallel-problems 4 \
            --max-completion-tokens "$MAX_TOK" \
            --problem-offset "$off" --problem-limit "$CHUNK" \
            --model "$MODEL" --out "$out_i" $EXTRA \
            >> "$log_i" 2>&1
    then
        echo "$(date -Is) chunk $i attempt $attempt OK"
        return 0
    fi
    local rc=$?
    echo "$(date -Is) chunk $i attempt $attempt FAILED rc=$rc"
    return $rc
}

INPUTS=""
for i in $(seq 0 $((N_CHUNKS - 1))); do
    OFF=$((i * CHUNK))
    OUT_I="$WORKDIR/chunk_$i.json"
    LOG_I="$WORKDIR/chunk_$i.log"
    if [ -f "$OUT_I" ]; then
        echo "$(date -Is) chunk $i exists, skipping"
        INPUTS="$INPUTS --input $OUT_I"
        continue
    fi
    : > "$LOG_I"  # truncate log for fresh chunk
    success=0
    for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
        if run_chunk "$i" "$OFF" "$OUT_I" "$LOG_I" "$attempt"; then
            success=1
            break
        fi
        # brief pause between retries — lets the kernel reap the dead CUDA
        # context cleanly before the next process tries to grab the GPU.
        sleep 15
    done
    if [ "$success" -ne 1 ]; then
        echo "$(date -Is) chunk $i FAILED after $((MAX_RETRIES + 1)) attempts; aborting"
        exit 1
    fi
    INPUTS="$INPUTS --input $OUT_I"
    echo "$(date -Is) END chunk $i"
done

/home/matilda/git/SI/.venv/bin/python -m si.cli lcb-merge $INPUTS --out "$OUT"
echo "$(date -Is) ANCHOR_CHUNKED DONE → $OUT"
