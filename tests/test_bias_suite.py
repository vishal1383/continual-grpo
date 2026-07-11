"""CPU-only checks for the ported bias-suite parsing and reporting logic."""
from __future__ import annotations

from continual_grpo.bias.generation import normalize_answer, parse_choice
from continual_grpo.bias.pairs import _unstereo_pair_key
from continual_grpo.bias.pat import _extract_pat_choices
from continual_grpo.bias.registry import TEN_BENCHMARKS
from continual_grpo.bias_eval import ARG_DEFAULTS, analysis_rows, bias_args


def test_parse_choice_letter_and_alias():
    assert parse_choice("Answer: B", ["A", "B"]) == "B"
    assert parse_choice("option a", ["A", "B"]) == "A"
    assert parse_choice("The nurse.", ["A", "B"], aliases={"A": ["nurse"], "B": ["doctor"]}) == "A"
    assert parse_choice("no idea", ["A", "B"]) is None


def test_normalize_answer_strips_articles_and_punctuation():
    assert normalize_answer("The  Empire State-Building!") == "empire state building"


def test_extract_pat_choices():
    assert _extract_pat_choices("Decide whether the person is a woman or a man.") == ("woman", "man")
    assert _extract_pat_choices("no association pair here") == (None, None)


def test_unstereo_pair_key_pairs_he_she_rows():
    he = dict(dataset="d", word="w", template="t", modifications="", is_revised=0, target_word="he")
    she = dict(he, target_word="she")
    assert _unstereo_pair_key(he) == _unstereo_pair_key(she)


def test_bias_args_defaults_and_ten_benchmarks():
    args, benchmarks = bias_args({"seed": 7}, None)
    assert benchmarks == TEN_BENCHMARKS and len(benchmarks) == 10
    assert args.seed == 7 and args.n_eval == ARG_DEFAULTS["n_eval"]
    args, _ = bias_args({"bias_eval": {"n_eval": 8}}, None)
    assert args.n_eval == 8
    args, _ = bias_args({"bias_eval": {"n_eval": 8}}, 3)
    assert args.n_eval == 3


def test_analysis_rows_base_trained_deltas():
    cells = [
        dict(model="m", method="base", stage="stage_00_base", adapter="",
             metrics={"crows_pairs": {"crows_pairs_bias_score": 0.5, "crows_pairs_n": 100}}, details={}),
        dict(model="m", method="grpo", stage="stage_01_gsm8k", adapter="a",
             metrics={"crows_pairs": {"crows_pairs_bias_score": 0.6, "crows_pairs_n": 100}}, details={}),
    ]
    rows = analysis_rows(cells)
    assert len(rows) == 1
    row = rows[0]
    assert row["base"] == 0.5 and row["trained"] == 0.6
    assert abs(row["delta_abs"] - 0.1) < 1e-9
    assert abs(row["delta_rel"] - 0.2) < 1e-9
    assert row["n"] == 100
