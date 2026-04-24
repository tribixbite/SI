---
name: si-monitor
description: Use this skill when the user asks to check on a running SI Phase 1 training run, asks "how is the run going", references runs/<id>/, or is doing an hourly status check during a multi-hour training job. Produces a concise status (generation, GPU vitals, last anchor score, time-to-next-anchor) and flags any anomalies (stale heartbeat, VRAM collapse, failed subcommand).
---

# si-monitor

## What to do

1. **Find the latest run.** `ls -t /home/matilda/git/SI/runs/ | grep -v '^base_\|^smoke' | head -1` — that's the current run_id.

2. **Read the heartbeat log.** `tail -20 runs/<id>/phase1.log` — each line has: timestamp, gen, step, gpu_used, gpu_free, ram. The last line is the current state.

3. **Check the process is alive.** `pgrep -af phase1_loop` — if empty, the run died. If the last heartbeat is > 15 min old without a completion line, also suspicious.

4. **Check GPU vitals match expectations.**
   - rollout-start → GPU 1 should climb to ~19 GB (vLLM)
   - train-start → GPU 1 should be ~11-13 GB (Unsloth QLoRA + activations)
   - Big drop during a step = died silently

5. **Read the latest anchor.** `ls -t runs/<id>/anchor_gen*.json | head -1` — if present, `pass_at_1` is the score. Compare to base (87.20%) and Phase 1 target (92.20%).

6. **Scan the latest subprocess log for errors.** `tail -40 runs/<id>/rollout_gen*.log` or the train/anchor equivalent. Look for `Traceback`, `OOM`, `CUDA error`, `NaN`.

7. **Report concisely.** Three numbers: current gen, last anchor score, ETA to next anchor. Plus a one-line status (healthy / stalled / failed).

## Output format

```
Run: <run_id>  gen=<g>/<N>  step=<rollout|train|anchor>
GPU1: <used>/<total> GiB    RAM: <used>/<total> MB
Last anchor: gen=<g> pass@1=<x>%  (base=87.20%, target=92.20%)
Status: healthy — last heartbeat <T> ago; next anchor at gen <g+k>
```

If unhealthy, replace the Status line with a diagnosis and the exact log lines supporting it. Keep the whole report under 200 words.

## Do not

- Do not restart the run without explicit user approval. If it's dead, report that and wait.
- Do not edit `src/si/` or `scripts/phase1_loop.sh` in the middle of a live run — they're read once at start and only take effect on next restart.
- Do not read the full per-gen log files; they're multi-megabyte noise from vLLM. Only grep them for failures.
