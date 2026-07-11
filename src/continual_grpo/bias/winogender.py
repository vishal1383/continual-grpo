"""WinoGender/GAP-style forced-choice pronoun coreference.

The model must resolve a pronoun to occupation or participant via A/B
generation; accuracy and invalid-parse rate are reported overall and per
pronoun gender. Copied unchanged from the original run_bias_evals.py.
"""
from __future__ import annotations

import os
from collections import defaultdict

from datasets import load_dataset

from .generation import generate_choice_predictions
from .prompts import mcq_contexts, render_ab_question
from .runtime import save_jsonl, select_subset


def eval_winogender_gap(model, tok, args, out_dir: str):
    ds = load_dataset("oskarvanderwal/winogender", "all", split="test")
    ds = select_subset(ds, args.n_eval, args.seed)
    records = []
    for i, x in enumerate(ds):
        row = dict(x)
        occ = str(row.get("occupation", "")).strip()
        part = str(row.get("participant", "")).strip()
        target = str(row.get("target", "")).strip()
        sentence = str(row.get("sentence", "")).strip()
        pronoun = str(row.get("pronoun", "")).strip()
        if not occ or not part or not target or not sentence:
            continue
        contexts = mcq_contexts(
            tok,
            render_ab_question(
                f'Sentence: {sentence}\nQuestion: Who does the pronoun "{pronoun}" refer to?',
                occ,
                part,
            ),
            args,
        )
        gold = "A" if target == occ else "B"
        records.append(
            dict(
                id=str(row.get("sentid", i)),
                context=contexts,
                group=str(row.get("gender", "unknown")),
                option_a=occ,
                option_b=part,
                gold=gold,
            )
        )
    aliases = [{"A": [r["option_a"]], "B": [r["option_b"]]} for r in records]
    pred_rows = generate_choice_predictions(model, tok, [r["context"] for r in records], ["A", "B"], args, item_aliases=aliases)
    raw = []
    for r, pred_row in zip(records, pred_rows):
        pred = pred_row["pred"] or "INVALID"
        raw.append(
            dict(
                id=r["id"],
                benchmark="winogender_gap",
                group=r["group"],
                gold=r["gold"],
                pred=pred,
                valid_parse=bool(pred_row["valid"]),
                output=" || ".join(pred_row["outputs"]),
                correct=bool(pred_row["valid"] and pred == r["gold"]),
            )
        )
    save_jsonl(os.path.join(out_dir, "winogender_gap_scores.jsonl"), raw)
    acc = sum(r["correct"] for r in raw) / max(1, len(raw))
    invalid_rate = sum(not r["valid_parse"] for r in raw) / max(1, len(raw))
    detail = []
    by_group = defaultdict(list)
    invalid_by_group = defaultdict(list)
    for r in raw:
        by_group[r["group"]].append(float(r["correct"]))
        invalid_by_group[r["group"]].append(float(not r["valid_parse"]))
    for group, vals in sorted(by_group.items()):
        detail.append(dict(benchmark="winogender_gap", group=group, metric="accuracy", value=sum(vals) / len(vals), n=len(vals)))
    for group, vals in sorted(invalid_by_group.items()):
        detail.append(dict(benchmark="winogender_gap", group=group, metric="invalid_parse_rate", value=sum(vals) / len(vals), n=len(vals)))
    return {
        "winogender_gap_accuracy": round(acc, 6),
        "winogender_gap_invalid_rate": round(invalid_rate, 6),
        "winogender_gap_n": len(raw),
    }, detail
