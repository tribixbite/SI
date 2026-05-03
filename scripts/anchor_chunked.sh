#!/bin/bash
# Subprocess-chunked LCB anchor — each chunk runs in its own Python/vLLM
# process to avoid the cumulative cudaErrorUnknown bug on vLLM 0.19.1 + AWQ.
# Usage:
#   anchor_chunked.sh <model_path> <bon> <max_tokens> <chunks> <out_path> [extra anchor args...]
set -euo pipefail

MODEL=${1:?model path}
BON=${2:?bon}
MAX_TOK=${3:?max_completion_tokens}
N_CHUNKS=${4:?n_chunks}
OUT=${5:?out merged json}
shift 5
EXTRA="$@"

WORKDIR=$(dirname "$OUT")/$(basename "$OUT" .json)_chunks
mkdir -p "$WORKDIR"

# LCB v6 has 1054 problems.
N_PROBLEMS=1054
CHUNK=$(( (N_PROBLEMS + N_CHUNKS - 1) / N_CHUNKS ))
echo "chunked anchor: model=$MODEL bon=$BON max_tok=$MAX_TOK n_chunks=$N_CHUNKS chunk_size=$CHUNK"

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
    echo "$(date -Is) START chunk $i (offset=$OFF, limit=$CHUNK)"
    CUDA_VISIBLE_DEVICES=1 /home/matilda/git/SI/.venv/bin/python -m si.cli anchor \
        --benchmark lcb --bon "$BON" --parallel-problems 4 \
        --max-completion-tokens "$MAX_TOK" \
        --problem-offset "$OFF" --problem-limit "$CHUNK" \
        --model "$MODEL" --out "$OUT_I" $EXTRA \
        > "$LOG_I" 2>&1
    INPUTS="$INPUTS --input $OUT_I"
    echo "$(date -Is) END chunk $i"
done

/home/matilda/git/SI/.venv/bin/python -m si.cli lcb-merge $INPUTS --out "$OUT"
echo "$(date -Is) ANCHOR_CHUNKED DONE → $OUT"
