"""DPO data construction utilities (Mix 2).

Given a list of (passing, failing) sample sets keyed by task, produce DPO
preference pairs in TRL's expected schema:
    {"prompt": chat-messages, "chosen": str, "rejected": str}

Pairing strategy: for each task, take the cartesian product of passing × failing
up to `max_pairs_per_task`. With large pass/fail asymmetry (our case: most tasks
have many failures), this saturates with the passing samples as the bottleneck.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from si.ssd import SSDSample

log = logging.getLogger(__name__)


def build_dpo_pairs(
    passing: list[SSDSample],
    failing: list[SSDSample],
    *,
    max_pairs_per_task: int = 4,
    seed: int = 3407,
) -> list[dict]:
    rng = random.Random(seed)
    by_task_pass: dict[str, list[SSDSample]] = {}
    by_task_fail: dict[str, list[SSDSample]] = {}
    for s in passing:
        by_task_pass.setdefault(s.task_id, []).append(s)
    for s in failing:
        by_task_fail.setdefault(s.task_id, []).append(s)

    pairs: list[dict] = []
    for tid, pass_list in by_task_pass.items():
        fail_list = by_task_fail.get(tid, [])
        if not fail_list:
            continue
        n_pairs = min(max_pairs_per_task, len(pass_list) * len(fail_list))
        rng.shuffle(pass_list)
        rng.shuffle(fail_list)
        seen = set()
        for chosen in pass_list:
            for rejected in fail_list:
                if (id(chosen), id(rejected)) in seen:
                    continue
                seen.add((id(chosen), id(rejected)))
                pairs.append(
                    {
                        "prompt": chosen.prompt_messages,
                        "chosen": chosen.completion_text,
                        "rejected": rejected.completion_text,
                    }
                )
                if len(pairs) % n_pairs == 0:
                    break
            if len([p for p in pairs if p["prompt"] is chosen.prompt_messages]) >= n_pairs:
                break
    log.info(
        "build_dpo_pairs: %d preference pairs from %d tasks (%d pass, %d fail)",
        len(pairs), len(by_task_pass), len(passing), len(failing),
    )
    return pairs


def write_dpo_jsonl(pairs: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


def read_dpo_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
