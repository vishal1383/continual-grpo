"""UCCB cultural QA generation probe.

Each cultural knowledge question is answered by short greedy generation and
compared to the gold answer with normalized exact-match and containment;
per-category rates and the max-minus-min category disparity are reported.
Copied unchanged from the original run_bias_evals.py.
"""
from __future__ import annotations

import os
from collections import defaultdict

from .generation import generate_from_contexts, normalize_answer
from .runtime import chat_prompt, safe_split_load, save_jsonl, select_subset


def eval_uccb(model, tok, args, out_dir: str):
    ds = safe_split_load("CraneAILabs/UCCB", split="test")
    ds = select_subset(ds, args.n_eval, args.seed)
    rows = []
    for i, x in enumerate(ds):
        row = dict(x)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        context = chat_prompt(
            tok,
            [
                {"role": "system", "content": "Answer the cultural knowledge question concisely."},
                {"role": "user", "content": f"Question:\n{question}\n\nAnswer:"},
            ],
        )
        rows.append(
            dict(
                id=str(row.get("id", i)),
                category=str(row.get("category", "unknown")),
                question=question,
                answer=answer,
                context=context,
            )
        )
    outputs = generate_from_contexts(model, tok, [r["context"] for r in rows], args, args.qa_max_new)
    raw = []
    for r, output in zip(rows, outputs):
        pred_norm = normalize_answer(output)
        gold_norm = normalize_answer(r["answer"])
        exact = bool(pred_norm and pred_norm == gold_norm)
        contains = bool(pred_norm and gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm))
        raw.append(
            dict(
                id=r["id"],
                benchmark="uccb",
                category=r["category"],
                question=r["question"],
                answer=r["answer"],
                output=output,
                exact_match=exact,
                contains_gold=contains,
            )
        )
    save_jsonl(os.path.join(out_dir, "uccb_scores.jsonl"), raw)

    by_category = defaultdict(list)
    contains_by_category = defaultdict(list)
    for r in raw:
        by_category[r["category"]].append(float(r["exact_match"]))
        contains_by_category[r["category"]].append(float(r["contains_gold"]))
    category_means = {group: sum(vals) / len(vals) for group, vals in by_category.items() if vals}
    contains_means = {group: sum(vals) / len(vals) for group, vals in contains_by_category.items() if vals}
    exact_rate = sum(float(r["exact_match"]) for r in raw) / max(1, len(raw))
    contains_rate = sum(float(r["contains_gold"]) for r in raw) / max(1, len(raw))
    disparity = max(category_means.values()) - min(category_means.values()) if len(category_means) > 1 else 0.0
    detail = []
    for group, value in sorted(category_means.items()):
        detail.append(dict(benchmark="uccb", group=group, metric="exact_match", value=value, n=len(by_category[group])))
    for group, value in sorted(contains_means.items()):
        detail.append(dict(benchmark="uccb", group=group, metric="contains_gold", value=value, n=len(contains_by_category[group])))
    return {
        "uccb_exact_match": round(exact_rate, 6),
        "uccb_contains_gold": round(contains_rate, 6),
        "uccb_exact_category_disparity": round(disparity, 6),
        "uccb_n": len(raw),
    }, detail
