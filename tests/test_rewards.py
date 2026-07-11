from decimal import Decimal

from continual_grpo.rewards import _number, correctness_reward


def test_number_prefers_final_marked_answer():
    assert _number("Work: 6 * 7 = 42\n#### 42") == Decimal("42")


def test_number_accepts_reasoning_without_marker():
    assert _number("Therefore the answer is 1,234.5 apples.") == Decimal("1234.5")


def test_correctness_uses_numeric_answer_not_whole_line():
    completions = [[{"content": "Thus she has 42 apples."}]]
    assert correctness_reward(completions, ["reasoning\n#### 42"]) == [1.0]
