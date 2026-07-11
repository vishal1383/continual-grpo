"""Dispatch table for the exact original ten-benchmark bias suite.

Every entry has the uniform signature ``fn(model, tok, args, out_dir)`` and
returns ``(metrics, detail_rows)``: a flat metric dict whose keys are prefixed
with the benchmark name (including a ``<benchmark>_n`` sample count) and a
list of per-group detail rows.
"""
from __future__ import annotations

from .discrim import eval_discrim_eval
from .pairs import eval_crows_pairs, eval_unstereo
from .pat import eval_pat
from .stereoset import eval_stereoset
from .toxigen import eval_toxigen
from .uccb import eval_uccb
from .winobias import eval_winobias
from .winogender import eval_winogender_gap

TEN_BENCHMARKS = [
    "crows_pairs",
    "stereoset_intersentence",
    "stereoset_intrasentence",
    "toxigen",
    "winobias",
    "winogender_gap",
    "pat",
    "unstereo_use10",
    "discrim_eval",
    "uccb",
]


def run_benchmark(benchmark: str, model, tok, args, out_dir: str):
    if benchmark == "crows_pairs":
        return eval_crows_pairs(model, tok, args, out_dir)
    if benchmark == "stereoset_intersentence":
        return eval_stereoset(model, tok, args, "intersentence", out_dir)
    if benchmark == "stereoset_intrasentence":
        return eval_stereoset(model, tok, args, "intrasentence", out_dir)
    if benchmark == "toxigen":
        return eval_toxigen(model, tok, args, out_dir)
    if benchmark == "winobias":
        return eval_winobias(model, tok, args, out_dir)
    if benchmark == "winogender_gap":
        return eval_winogender_gap(model, tok, args, out_dir)
    if benchmark == "pat":
        return eval_pat(model, tok, args, out_dir)
    if benchmark in ("unstereo_use5", "unstereo_use10", "unstereo_use20"):
        return eval_unstereo(model, tok, args, out_dir, benchmark)
    if benchmark == "discrim_eval":
        return eval_discrim_eval(model, tok, args, out_dir)
    if benchmark == "uccb":
        return eval_uccb(model, tok, args, out_dir)
    raise ValueError(f"unknown benchmark: {benchmark}")
