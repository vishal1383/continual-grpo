from decimal import Decimal

from continual_grpo.rewards import (_number, boxed_answer, correctness_reward,
                                    math_correctness_reward)


def test_number_prefers_final_marked_answer():
    assert _number("Work: 6 * 7 = 42\n#### 42") == Decimal("42")


def test_number_accepts_reasoning_without_marker():
    assert _number("Therefore the answer is 1,234.5 apples.") == Decimal("1234.5")


def test_correctness_uses_numeric_answer_not_whole_line():
    completions = [[{"content": "Thus she has 42 apples."}]]
    assert correctness_reward(completions, ["reasoning\n#### 42"]) == [1.0]


def test_boxed_answer_handles_nested_braces():
    assert boxed_answer("So the answer is $\\boxed{\\frac{3}{4}}$.") == "\\frac{3}{4}"
    assert boxed_answer("no boxed content here") is None


def test_math_reward_matches_fraction_and_numeric_forms():
    completions = ["steps...\n#### 3/4", "steps...\n#### 0.75", "steps...\n#### 42"]
    answers = ["\\frac{3}{4}", "\\frac{3}{4}", "42"]
    assert math_correctness_reward(completions, answers) == [1.0, 0.0, 1.0]


def test_math_reward_falls_back_to_boxed_completion():
    completions = ["therefore $\\boxed{x=3}$"]
    assert math_correctness_reward(completions, ["x = 3"]) == [1.0]
