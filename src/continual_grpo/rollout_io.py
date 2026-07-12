"""Load the original Self-Distillation GRPO rollout caches for reuse.

The original repo saved K sampled completions per GSM8K prompt as text-level
records in ``grpo_rollouts.jsonl``: ``id`` ``train_X#k``, ``group_id``
``train_X``, ``question``, ``rollout`` text, binary ``rollout_correct``, and
the extracted ``gold`` answer. This module reconstructs training groups from
such a file using the exact system/user prompt the rollouts were sampled
under, and keeps the file's verifier rewards so group advantages match the
original run.
"""
from __future__ import annotations

import json
import random

# Copied verbatim from Self-Distillation sdft/text.py so the reused
# trajectories stay in-distribution with how they were generated.
ORIGINAL_GSM8K_SYSTEM = (
    "You are a careful math solver. Show concise step-by-step reasoning, "
    "then end with exactly: Final answer: <number>"
)
ORIGINAL_GSM8K_USER = (
    "Solve the math problem. Show 2-5 concise reasoning steps, then end with "
    "exactly:\nFinal answer: <number>\n\nProblem:\n"
)


def original_gsm8k_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": ORIGINAL_GSM8K_SYSTEM},
        {"role": "user", "content": ORIGINAL_GSM8K_USER + question},
    ]


def _k_index(row_id: str) -> int:
    try:
        return int(str(row_id).rsplit("#", 1)[1])
    except (IndexError, ValueError):
        return 0


def load_external_groups(path: str, seed: int, max_samples: int = 0) -> list[dict]:
    """Group the rollout file by prompt, in sampler order, seeded-shuffled."""
    by_group: dict[str, list[dict]] = {}
    order: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            gid = str(row["group_id"])
            if gid not in by_group:
                by_group[gid] = []
                order.append(gid)
            by_group[gid].append(row)
    groups = []
    for gid in order:
        rows = sorted(by_group[gid], key=lambda r: _k_index(r.get("id", "0#0")))
        groups.append({
            "group_id": gid,
            "question": str(rows[0]["question"]),
            "completions": [str(r["rollout"]) for r in rows],
            "correct": [1.0 if r.get("rollout_correct") else 0.0 for r in rows],
        })
    sizes = {len(g["completions"]) for g in groups}
    if len(sizes) > 1:
        raise ValueError(f"external rollout groups have mixed sizes: {sorted(sizes)}")
    random.Random(seed).shuffle(groups)
    if max_samples and max_samples > 0:
        groups = groups[:max_samples]
    return groups
