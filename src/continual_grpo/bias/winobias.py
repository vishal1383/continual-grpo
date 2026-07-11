"""WinoBias pro/anti-stereotype likelihood gap.

Scores the full sentence likelihood for the four official WinoBias configs and
reports the mean log-probability gap between pro- and anti-stereotypical
coreference sentences. Copied unchanged from the original run_bias_evals.py.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

from datasets import load_dataset

from .runtime import save_jsonl, select_subset
from .scoring import continuation_logprobs


def _detokenize(tokens) -> str:
    text = " ".join(str(t) for t in tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    return text.strip()


def eval_winobias(model, tok, args, out_dir: str):
    configs = ("type1_pro", "type1_anti", "type2_pro", "type2_anti")
    raw = []
    for config in configs:
        ds = load_dataset("uclanlp/wino_bias", config, split="test")
        ds = select_subset(ds, args.n_eval, args.seed)
        texts = [_detokenize(x["tokens"]) for x in ds]
        scores = continuation_logprobs(model, tok, [""] * len(texts), texts, args.score_batch, args.score_max_length)
        for i, (x, text, score) in enumerate(zip(ds, texts, scores)):
            raw.append(
                dict(
                    id=str(x.get("document_id", f"{config}_{i}")),
                    benchmark="winobias",
                    group=config,
                    stereotype="pro" if config.endswith("_pro") else "anti",
                    type=config.split("_", 1)[0],
                    text=text,
                    score=score,
                )
            )
    save_jsonl(os.path.join(out_dir, "winobias_scores.jsonl"), raw)
    by_group = defaultdict(list)
    for r in raw:
        by_group[r["group"]].append(r["score"])
    means = {group: sum(vals) / len(vals) for group, vals in by_group.items()}
    pro_scores = [r["score"] for r in raw if r["stereotype"] == "pro"]
    anti_scores = [r["score"] for r in raw if r["stereotype"] == "anti"]
    pro_mean = sum(pro_scores) / max(1, len(pro_scores))
    anti_mean = sum(anti_scores) / max(1, len(anti_scores))
    detail = [
        dict(benchmark="winobias", group=group, metric="mean_logprob", value=value, n=len(by_group[group]))
        for group, value in sorted(means.items())
    ]
    return {
        "winobias_pro_mean_logprob": round(pro_mean, 6),
        "winobias_anti_mean_logprob": round(anti_mean, 6),
        "winobias_pro_minus_anti": round(pro_mean - anti_mean, 6),
        "winobias_n": len(raw),
    }, detail
