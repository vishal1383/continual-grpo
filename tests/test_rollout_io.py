import json
import os
import tempfile

from continual_grpo.rollout_io import (ORIGINAL_GSM8K_SYSTEM, load_external_groups,
                                       original_gsm8k_messages)


def _write_rollouts(path, groups):
    with open(path, "w", encoding="utf-8") as handle:
        for gid, rollouts in groups:
            for k, (text, correct) in enumerate(rollouts):
                handle.write(json.dumps({
                    "id": f"{gid}#{k}", "group_id": gid, "question": f"q-{gid}",
                    "gold": "16", "rollout": text, "rollout_correct": correct,
                    "rollout_pred": "16",
                }) + "\n")


def test_groups_keep_sampler_order_and_rewards():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rollouts.jsonl")
        _write_rollouts(path, [
            ("train_0", [("a", True), ("b", False)]),
            ("train_1", [("c", False), ("d", False)]),
        ])
        groups = load_external_groups(path, seed=0)
    by_id = {g["group_id"]: g for g in groups}
    assert by_id["train_0"]["completions"] == ["a", "b"]
    assert by_id["train_0"]["correct"] == [1.0, 0.0]
    assert by_id["train_1"]["correct"] == [0.0, 0.0]


def test_shuffle_is_seeded_and_max_samples_caps_groups():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rollouts.jsonl")
        _write_rollouts(path, [(f"train_{i}", [("x", True)]) for i in range(20)])
        first = [g["group_id"] for g in load_external_groups(path, seed=7)]
        second = [g["group_id"] for g in load_external_groups(path, seed=7)]
        assert first == second
        assert len(load_external_groups(path, seed=7, max_samples=3)) == 3


def test_mixed_group_sizes_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rollouts.jsonl")
        _write_rollouts(path, [("train_0", [("a", True)]),
                               ("train_1", [("b", True), ("c", False)])])
        try:
            load_external_groups(path, seed=0)
        except ValueError as error:
            assert "mixed sizes" in str(error)
        else:
            raise AssertionError("expected ValueError for mixed group sizes")


def test_messages_use_verbatim_original_prompt():
    msgs = original_gsm8k_messages("What is 2+2?")
    assert msgs[0]["content"] == ORIGINAL_GSM8K_SYSTEM
    assert msgs[1]["content"].endswith("Problem:\nWhat is 2+2?")
    assert "Final answer: <number>" in msgs[1]["content"]
