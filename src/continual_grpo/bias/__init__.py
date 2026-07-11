"""Ten-benchmark bias suite ported verbatim from the original run_bias_evals.py.

Only the glue differs from the original: model loading, chat templating, and
JSONL I/O come from this package instead of the legacy ``sdft`` package, and
checkpoint discovery matches the continual-grpo output layout. Every scoring
rule, prompt template, and metric definition is unchanged.
"""
