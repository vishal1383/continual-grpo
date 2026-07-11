"""Greedy generation and answer parsing for the forced-choice probes.

Items are answered by generating a few tokens per prompt phrasing, parsing an
option letter (with per-item alias phrases as fallback), and majority-voting
across phrasings. Copied unchanged from the original run_bias_evals.py.
"""
from __future__ import annotations

import re
from collections import defaultdict

import torch

from .prompts import TOXICITY_QUESTIONS, toxigen_context
from .runtime import bounded_count


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_choice(text: str, choices: list[str], aliases: dict[str, list[str]] | None = None) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    valid = [c.upper() for c in choices]
    m = re.match(r"^\s*(?:OPTION\s+|CHOICE\s+|ANSWER\s*[:\-]?\s*)?([A-Z])\b", upper)
    if m and m.group(1) in valid:
        return m.group(1)
    for choice in valid:
        if re.search(rf"\b{re.escape(choice)}\b", upper):
            return choice
    if aliases:
        norm_text = normalize_answer(cleaned)
        for choice in valid:
            phrases = aliases.get(choice, []) + aliases.get(choice.lower(), [])
            for phrase in sorted(phrases, key=len, reverse=True):
                norm_phrase = normalize_answer(phrase)
                if norm_phrase and re.search(rf"\b{re.escape(norm_phrase)}\b", norm_text):
                    return choice
    return None


@torch.inference_mode()
def generate_from_contexts(model, tok, contexts: list[str], args, max_new_tokens: int):
    device = next(model.parameters()).device
    model.eval()
    outputs: list[str] = []
    for start in range(0, len(contexts), args.eval_batch):
        batch = contexts[start : start + args.eval_batch]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt,
            add_special_tokens=False,
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for seq in out:
            text = tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for bad in ("\nHuman:", "\nAssistant:", "\nUser:", "<|im_start|>", "<|im_end|>"):
                if bad in text:
                    text = text.split(bad)[0].strip()
            outputs.append(text)
        del enc, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def generate_choice_predictions(
    model,
    tok,
    item_contexts: list[list[str]],
    choices: list[str],
    args,
    item_aliases: list[dict[str, list[str]]] | None = None,
):
    flat = [ctx for contexts in item_contexts for ctx in contexts]
    counts = [len(contexts) for contexts in item_contexts]
    outputs = generate_from_contexts(model, tok, flat, args, args.choice_max_new) if flat else []
    rows = []
    offset = 0
    for item_i, count in enumerate(counts):
        item_outputs = outputs[offset : offset + count]
        aliases = item_aliases[item_i] if item_aliases and item_i < len(item_aliases) else None
        parsed = [parse_choice(text, choices, aliases=aliases) for text in item_outputs]
        votes = defaultdict(int)
        for pred in parsed:
            if pred is not None:
                votes[pred] += 1
        pred = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
        rows.append(
            dict(
                pred=pred,
                valid=pred is not None,
                outputs=item_outputs,
                parsed=parsed,
            )
        )
        offset += count
    return rows


@torch.inference_mode()
def classify_toxic_texts(model, tok, texts: list[str], args):
    n = bounded_count(args.toxicity_prompt_count, len(TOXICITY_QUESTIONS))
    item_contexts = [
        [toxigen_context(tok, text, question) for question in TOXICITY_QUESTIONS[:n]]
        for text in texts
    ]
    aliases = [
        {"A": ["non-toxic", "not toxic", "benign", "safe", "no"], "B": ["toxic", "hateful", "abusive", "unsafe", "yes"]}
        for _ in texts
    ]
    preds = generate_choice_predictions(model, tok, item_contexts, ["A", "B"], args, item_aliases=aliases)
    return [row["pred"] == "B" if row["valid"] else False for row in preds], [row["valid"] for row in preds], [
        " || ".join(row["outputs"]) for row in preds
    ]
