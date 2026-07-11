"""Task preparation and verifiable rewards shared by every training method."""
from __future__ import annotations

import re
import json
import subprocess
import sys
import tempfile
import textwrap
from decimal import Decimal, InvalidOperation

from datasets import load_dataset
from datasets import Dataset


SYSTEM = "Solve step by step, briefly. End with the final answer on its own line: #### <number>"
SYSTEM_MATH = "Solve step by step, briefly. End with the final answer on its own line: #### <answer>"


def _number(text: str) -> Decimal | None:
    """Extract the final numeric answer, preferring text after GSM8K's #### marker."""
    marked = re.findall(r"####\s*([^\n]+)", str(text))
    search = marked[-1] if marked else str(text)
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", search)
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


def correctness_reward(completions, answer, **_):
    """Exact-answer verifier used by GRPO; one reward per sampled completion."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [1.0 if _number(text) is not None and _number(text) == _number(gold) else 0.0
            for text, gold in zip(texts, answer)]


def format_reward(completions, **_):
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [0.1 if re.search(r"####\s*\S+", text) else 0.0 for text in texts]


def boxed_answer(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, honoring nested braces."""
    index = str(text).rfind("\\boxed")
    if index == -1:
        return None
    open_brace = str(text).find("{", index)
    if open_brace == -1:
        return None
    depth = 0
    for position in range(open_brace, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:position]
    return None


def _normalize_math(text: str) -> str:
    """Canonicalize a MATH answer: drop spacing/latex decoration, flatten \\frac."""
    s = str(text).strip().strip("$").strip()
    for token in ("\\left", "\\right", "\\,", "\\!", "\\;", " "):
        s = s.replace(token, "")
    while True:
        replaced = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
        if replaced == s:
            break
        s = replaced
    return s.replace("{", "").replace("}", "").rstrip(".")


def _math_equal(pred: str, gold: str) -> bool:
    p, g = _normalize_math(pred), _normalize_math(gold)
    if not p or not g:
        return False
    try:
        return Decimal(p.replace(",", "")) == Decimal(g.replace(",", ""))
    except InvalidOperation:
        return p == g


def math_correctness_reward(completions, answer, **_):
    """MATH verifier: the #### answer (or last \\boxed in the completion) must
    match the gold \\boxed content, numerically when both parse, else as
    normalized text."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    out = []
    for text, gold in zip(texts, answer):
        marked = re.findall(r"####\s*([^\n]+)", text)
        pred = marked[-1].strip() if marked else boxed_answer(text)
        out.append(1.0 if pred is not None and _math_equal(pred, str(gold)) else 0.0)
    return out


def _extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return fenced[-1].strip() if fenced else text.strip()


def code_correct(text: str, serialized_gold: str) -> bool:
    gold = json.loads(serialized_gold)
    code = _extract_code(text)
    entry, prompt = gold["entry_point"], gold["prompt"]
    candidate = code if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", code) else prompt + code
    prelude = textwrap.dedent("""
        import signal
        def _timeout(signum, frame): raise TimeoutError()
        signal.signal(signal.SIGALRM, _timeout); signal.alarm(5)
    """)
    script = prelude + "\n" + candidate + "\n" + gold["test"] + f"\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/candidate.py"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(script)
        try:
            return subprocess.run([sys.executable, path], cwd=directory, timeout=7,
                                  capture_output=True).returncode == 0
        except Exception:
            return False


def code_rewards(completions: list[str], answers: list[str]) -> list[float]:
    return [1.0 if code_correct(text, gold) else 0.0 for text, gold in zip(completions, answers)]


def prepare_task(spec: dict, seed: int):
    if spec.get("name") == "math":
        ds = load_dataset(spec["dataset"], spec.get("subset"), split=spec.get("split", "train"))
        limit = int(spec.get("max_samples", 0))
        if limit:
            ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
        rows = []
        for row in ds:
            gold = boxed_answer(str(row.get("solution", "")))
            if gold is None:
                continue
            rows.append({
                "prompt": [{"role": "system", "content": SYSTEM_MATH},
                           {"role": "user", "content": row["problem"]}],
                "answer": gold,
            })
        return Dataset.from_list(rows)
    if spec.get("name") == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=seed)
        held_out = int(spec.get("held_out", 64))
        ds = ds.select(range(min(held_out, len(ds)), len(ds)))
        limit = int(spec.get("max_samples", 0))
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        rows = []
        for row in ds:
            gold = json.dumps({"test": row["test"], "entry_point": row["entry_point"], "prompt": row["prompt"]})
            rows.append({
                "prompt": [{"role": "system", "content": "Complete the Python function correctly."},
                           {"role": "user", "content": row["prompt"]}],
                "answer": gold,
            })
        return Dataset.from_list(rows)
    ds = load_dataset(spec["dataset"], spec.get("subset"), split=spec.get("split", "train"))
    limit = int(spec.get("max_samples", 0))
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    prompt_field, answer_field = spec["prompt_field"], spec["answer_field"]
    return ds.map(lambda row: {
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row[prompt_field]}],
        "answer": row[answer_field],
    }, remove_columns=ds.column_names)
