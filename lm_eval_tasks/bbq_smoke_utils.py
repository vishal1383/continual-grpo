"""Stratified BBQ subset so smoke runs produce no NaN metrics.

BBQ reports bias scores per demographic category and per context condition
(ambiguous vs disambiguated).  A plain ``--limit N`` takes the first N rows,
which all belong to one category, so every other bucket aggregates an empty
list and prints NaN.  This task keeps the first ``PER_BUCKET`` documents of
every (category, context_condition) bucket instead: all 22 buckets have data,
every metric is defined, and the whole benchmark stays 22 documents.

Everything except the subsampling is re-exported unchanged from the installed
lm-eval BBQ task, so scores are computed exactly as in the real benchmark.
"""
from __future__ import annotations

import datasets

from lm_eval.tasks.bbq.utils import (  # noqa: F401  (re-exported for the YAML)
    agg_accuracy_amb,
    agg_accuracy_disamb,
    agg_amb_bias_scores,
    agg_disamb_bias_scores,
    doc_to_choice,
    doc_to_target,
    process_results_multiple_choice,
)
from lm_eval.tasks.bbq.utils import process_docs as _process_docs_full

PER_BUCKET = 1


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    counts: dict[tuple[str, str], int] = {}
    keep: list[int] = []
    for index, bucket in enumerate(zip(dataset["category"], dataset["context_condition"])):
        if counts.get(bucket, 0) < PER_BUCKET:
            counts[bucket] = counts.get(bucket, 0) + 1
            keep.append(index)
    return _process_docs_full(dataset.select(keep))
