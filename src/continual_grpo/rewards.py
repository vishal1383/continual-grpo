"""Task preparation and verifiable rewards shared by every training method."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from datasets import load_dataset


SYSTEM = "Solve step by step, briefly. End with the final answer on its own line: #### <number>"


def _number(text: str) -> Decimal | None:
    """Extract the final numeric answer, preferring text after GSM8K's #### marker."""
    marked = re.findall(r"####\s*([^\n]+)", str(text))
    search = marked[-1] if marked else str(text)
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", search)
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


def correctness_reward(completions, answer, **_):
    """Exact-answer verifier used by GRPO; one reward per sampled completion."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [1.0 if _number(text) is not None and _number(text) == _number(gold) else 0.0
            for text, gold in zip(texts, answer)]


def format_reward(completions, **_):
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [0.1 if re.search(r"####\s*\S+", text) else 0.0 for text in texts]


def prepare_task(spec: dict, seed: int):
    ds = load_dataset(spec["dataset"], spec.get("subset"), split=spec.get("split", "train"))
    limit = int(spec.get("max_samples", 0))
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    prompt_field, answer_field = spec["prompt_field"], spec["answer_field"]
    return ds.map(lambda row: {
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row[prompt_field]}],
        "answer": row[answer_field],
    }, remove_columns=ds.column_names)
