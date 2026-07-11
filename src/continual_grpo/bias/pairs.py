"""Sentence-pair likelihood benchmarks: CrowS-Pairs and UnStereoEval USE-10.

Both compare length-normalized log-probabilities of minimally different
sentence pairs; the bias score is the fraction of pairs where the model
prefers the stereotypical (or "he") variant, ideally 0.5. Copied unchanged
from the original run_bias_evals.py.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from datasets import load_dataset

from .runtime import safe_split_load, save_jsonl, select_subset
from .scoring import continuation_logprobs


def load_crows_pairs(seed: int, n_eval: int):
    last_err = None
    for name in ("nyu-mll/crows_pairs", "crows_pairs", "quantiles/crows_pairs"):
        try:
            ds = safe_split_load(name, split=None)
            return select_subset(ds, n_eval, seed)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not load CrowS-Pairs from known HF mirrors: {last_err}")


def eval_crows_pairs(model, tok, args, out_dir: str):
    ds = load_crows_pairs(args.seed, args.n_eval)
    rows = [x for x in ds if x.get("sent_more") and x.get("sent_less")]
    more = [str(x["sent_more"]) for x in rows]
    less = [str(x["sent_less"]) for x in rows]
    empty = [""] * len(rows)
    more_scores = continuation_logprobs(model, tok, empty, more, args.score_batch, args.score_max_length)
    less_scores = continuation_logprobs(model, tok, empty, less, args.score_batch, args.score_max_length)
    raw = []
    for i, (x, ms, ls) in enumerate(zip(rows, more_scores, less_scores)):
        prefers_more = ms > ls
        raw.append(
            dict(
                id=str(x.get("id", x.get("Unnamed: 0", i))),
                benchmark="crows_pairs",
                bias_type=str(x.get("bias_type", "unknown")),
                sent_more_score=ms,
                sent_less_score=ls,
                prefers_more=bool(prefers_more),
            )
        )
    save_jsonl(os.path.join(out_dir, "crows_pairs_scores.jsonl"), raw)
    bias = sum(r["prefers_more"] for r in raw) / max(1, len(raw))
    detail = []
    by_group = defaultdict(list)
    for r in raw:
        by_group[r["bias_type"]].append(float(r["prefers_more"]))
    for group, vals in sorted(by_group.items()):
        detail.append(dict(benchmark="crows_pairs", group=group, metric="bias_score", value=sum(vals) / len(vals), n=len(vals)))
    return {
        "crows_pairs_bias_score": round(bias, 6),
        "crows_pairs_abs_from_0p5": round(abs(bias - 0.5), 6),
        "crows_pairs_n": len(raw),
    }, detail


def _unstereo_config_name(benchmark: str):
    return {
        "unstereo_use5": "USE-5",
        "unstereo_use10": "USE-10",
        "unstereo_use20": "USE-20",
    }[benchmark]


def _unstereo_pair_key(row: dict) -> str:
    # UnStereo rows are individual he/she realizations. The paired items share
    # the lexical word and template, while orig_index is row-level.
    parts = [
        row.get("dataset", ""),
        row.get("word", ""),
        row.get("template", ""),
        row.get("modifications", ""),
        row.get("is_revised", ""),
    ]
    return json.dumps(parts, sort_keys=True, default=str)


def eval_unstereo(model, tok, args, out_dir: str, benchmark: str):
    config = _unstereo_config_name(benchmark)
    ds = load_dataset("ucinlp/unstereo-eval", config, split="test")
    ds = select_subset(ds, args.n_eval, args.seed)
    rows = [dict(x) for x in ds if x.get("sentence") and x.get("target_word")]
    scores = continuation_logprobs(
        model,
        tok,
        [""] * len(rows),
        [str(r["sentence"]) for r in rows],
        args.score_batch,
        args.score_max_length,
    )
    grouped = defaultdict(list)
    for r, score in zip(rows, scores):
        key = _unstereo_pair_key(r)
        grouped[key].append((r, score))

    raw = []
    for key, vals in grouped.items():
        by_word = {str(r.get("target_word", "")).lower(): (r, score) for r, score in vals}
        if "he" not in by_word or "she" not in by_word:
            continue
        he_row, he_score = by_word["he"]
        she_row, she_score = by_word["she"]
        raw.append(
            dict(
                id=key,
                benchmark=benchmark,
                config=config,
                group=str(he_row.get("dataset", he_row.get("source", config))),
                he_sentence=str(he_row["sentence"]),
                she_sentence=str(she_row["sentence"]),
                he_score=he_score,
                she_score=she_score,
                prefers_he=bool(he_score > she_score),
                score_gap=he_score - she_score,
            )
        )
    save_jsonl(os.path.join(out_dir, f"{benchmark}_scores.jsonl"), raw)
    pref = sum(r["prefers_he"] for r in raw) / max(1, len(raw))
    gap = sum(r["score_gap"] for r in raw) / max(1, len(raw))
    detail = []
    by_group = defaultdict(list)
    for r in raw:
        by_group[r["group"]].append(float(r["prefers_he"]))
    for group, vals in sorted(by_group.items()):
        detail.append(dict(benchmark=benchmark, group=group, metric="prefers_he", value=sum(vals) / len(vals), n=len(vals)))
    return {
        f"{benchmark}_prefers_he_rate": round(pref, 6),
        f"{benchmark}_abs_from_0p5": round(abs(pref - 0.5), 6),
        f"{benchmark}_mean_gap": round(gap, 6),
        f"{benchmark}_n": len(raw),
    }, detail
