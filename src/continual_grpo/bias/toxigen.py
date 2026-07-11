"""ToxiGen forced-choice toxicity classification.

The model classifies each annotated ToxiGen text as toxic/non-toxic via A/B
generation; accuracy against the human toxicity label, predicted-toxic rate,
and invalid-parse rate are reported overall and per target group. Copied
unchanged from the original run_bias_evals.py.
"""
from __future__ import annotations

import os
from collections import defaultdict

from datasets import load_dataset

from .generation import classify_toxic_texts
from .runtime import save_jsonl, select_subset


def load_toxigen(seed: int, n_eval: int):
    last_err = None
    for split in ("test", "train"):
        try:
            ds = load_dataset("toxigen/toxigen-data", "annotated", split=split)
            return select_subset(ds, n_eval, seed)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not load ToxiGen annotated split: {last_err}")


def eval_toxigen(model, tok, args, out_dir: str):
    ds = load_toxigen(args.seed, args.n_eval)
    rows = [x for x in ds if x.get("text")]
    toxic_preds, valid, classifier_outputs = classify_toxic_texts(model, tok, [str(x["text"]) for x in rows], args)
    raw = []
    for i, (x, pred_toxic, ok, clf_out) in enumerate(zip(rows, toxic_preds, valid, classifier_outputs)):
        gold_toxic = float(x.get("toxicity_human", x.get("toxicity_ai", 1))) >= args.toxigen_toxic_threshold
        raw.append(
            dict(
                id=str(x.get("id", i)),
                benchmark="toxigen",
                target_group=str(x.get("target_group", x.get("group", "unknown"))),
                gold_toxic=bool(gold_toxic),
                classifier_output=clf_out,
                valid_parse=bool(ok),
                pred_toxic=bool(pred_toxic),
                correct=bool(ok and gold_toxic == pred_toxic),
            )
        )
    save_jsonl(os.path.join(out_dir, "toxigen_scores.jsonl"), raw)
    acc = sum(r["correct"] for r in raw) / max(1, len(raw))
    valid_rows = [r for r in raw if r["valid_parse"]]
    toxic_rate = sum(r["pred_toxic"] for r in valid_rows) / max(1, len(valid_rows))
    invalid_rate = sum(not r["valid_parse"] for r in raw) / max(1, len(raw))
    detail = []
    by_group = defaultdict(list)
    pred_by_group = defaultdict(list)
    invalid_by_group = defaultdict(list)
    for r in raw:
        by_group[r["target_group"]].append(float(r["correct"]))
        if r["valid_parse"]:
            pred_by_group[r["target_group"]].append(float(r["pred_toxic"]))
        invalid_by_group[r["target_group"]].append(float(not r["valid_parse"]))
    for group, vals in sorted(by_group.items()):
        detail.append(dict(benchmark="toxigen", group=group, metric="accuracy", value=sum(vals) / len(vals), n=len(vals)))
    for group, vals in sorted(pred_by_group.items()):
        detail.append(dict(benchmark="toxigen", group=group, metric="pred_toxic_rate", value=sum(vals) / len(vals), n=len(vals)))
    for group, vals in sorted(invalid_by_group.items()):
        detail.append(dict(benchmark="toxigen", group=group, metric="invalid_parse_rate", value=sum(vals) / len(vals), n=len(vals)))
    return {
        "toxigen_accuracy": round(acc, 6),
        "toxigen_pred_toxic_rate": round(toxic_rate, 6),
        "toxigen_invalid_rate": round(invalid_rate, 6),
        "toxigen_n": len(raw),
    }, detail
