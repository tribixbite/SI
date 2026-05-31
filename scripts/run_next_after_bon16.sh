#!/bin/bash
# Supervisor: wait for the running Qwen3-Coder BoN16 anchor to finish,
# then launch the Qwen3.6-27B Dense BoN1 baseline via SGLang.
#
# Trigger condition: the merged JSON for the BoN16 run lands on disk.
# We poll once a minute. Cheaper than a Python loop with imports.

set -uo pipefail

BON16_OUT=/home/matilda/git/SI/runs/qwen3coder_base_lcb_v6_bon16_chunked.json
NEXT_LOG=/home/matilda/git/SI/runs/qwen36_27b_sglang_bon1.log
NEXT_OUT=/home/matilda/git/SI/runs/qwen36_27b_sglang_lcb_v6_bon1.json

if [ -f "$BON16_OUT" ]; then
    echo "$(date -Is) BoN16 already done; not waiting"
else
    echo "$(date -Is) waiting for $BON16_OUT to land..."
    until [ -f "$BON16_OUT" ]; do sleep 60; done
    echo "$(date -Is) BoN16 done"
    # Give 30s for child Python+vLLM processes to actually exit + GPU mem to release.
    sleep 30
fi

# Verify GPU 1 is free before launching.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 2p)
if [ -z "$USED" ] || [ "$USED" -gt 4000 ]; then
    echo "$(date -Is) GPU 1 still has ${USED}MiB used; bailing — re-launch manually after cleanup"
    nvidia-smi
    exit 1
fi

echo "$(date -Is) launching Qwen3.6-27B SGLang baseline (BoN1, full 1054 problems)"
mkdir -p "$(dirname "$NEXT_OUT")"

# Use the subprocess-chunked pattern from anchor_chunked.sh but invoke our
# SGLang script. Chunked to dodge any equivalent long-run issues SGLang has.
WORKDIR=${NEXT_OUT%.json}_chunks
mkdir -p "$WORKDIR"
N_CHUNKS=8
N_PROBLEMS=1054
CHUNK=$(( (N_PROBLEMS + N_CHUNKS - 1) / N_CHUNKS ))

INPUTS=""
for i in $(seq 0 $((N_CHUNKS - 1))); do
    OFF=$((i * CHUNK))
    OUT_I="$WORKDIR/chunk_$i.json"
    LOG_I="$WORKDIR/chunk_$i.log"
    if [ -f "$OUT_I" ]; then
        INPUTS="$INPUTS --input $OUT_I"
        continue
    fi
    : > "$LOG_I"
    for attempt in 1 2 3; do
        echo "$(date -Is) sglang chunk $i attempt $attempt (off=$OFF lim=$CHUNK)"
        rc=0
        timeout --foreground --kill-after=30s 3600s \
            /home/matilda/git/SI/.venv-sglang/bin/python \
                /home/matilda/git/SI/scripts/lcb_anchor_sglang.py \
                --model /home/matilda/git/SI/cache/qwen3.6-27b-awq-int4 \
                --quantization awq_marlin \
                --bon 1 \
                --parallel-problems 4 \
                --max-completion-tokens 4096 \
                --problem-offset "$OFF" --problem-limit "$CHUNK" \
                --cuda-device 1 \
                --out "$OUT_I" >> "$LOG_I" 2>&1 || rc=$?
        if [ "$rc" -eq 0 ] && [ -f "$OUT_I" ]; then
            echo "$(date -Is) sglang chunk $i OK"
            break
        fi
        echo "$(date -Is) sglang chunk $i attempt $attempt FAILED rc=$rc"
        sleep 30
    done
    [ ! -f "$OUT_I" ] && { echo "$(date -Is) sglang chunk $i abandoned"; exit 1; }
    INPUTS="$INPUTS --input $OUT_I"
done

/home/matilda/git/SI/.venv/bin/python -m si.cli lcb-merge $INPUTS --out "$NEXT_OUT"
echo "$(date -Is) Qwen3.6-27B SGLang baseline DONE → $NEXT_OUT"
