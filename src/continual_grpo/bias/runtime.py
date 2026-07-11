"""Model loading, dataset selection, and file I/O for the bias suite.

These helpers replace the legacy ``sdft`` package imports used by the original
run_bias_evals.py (``chat_prompt``, ``save_jsonl``, ``load_base``,
``load_adapter``) with equivalents that follow this repository's conventions.
Evaluation models load frozen in FP16 with ``device_map='auto'``, matching the
lm-eval utility evaluation path.
"""
from __future__ import annotations

import csv
import gc
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def chat_prompt(tok, msgs: list[dict]) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_subset(ds, n_eval: int, seed: int):
    if n_eval and n_eval > 0:
        ds = ds.shuffle(seed=seed).select(range(min(n_eval, len(ds))))
    return ds


def safe_split_load(name: str, *args, split: str | None = None, **kwargs):
    if split is not None:
        return load_dataset(name, *args, split=split, **kwargs)
    ds = load_dataset(name, *args, **kwargs)
    if hasattr(ds, "keys"):
        for key in ("test", "validation", "train"):
            if key in ds:
                return ds[key]
        return ds[next(iter(ds.keys()))]
    return ds


def bounded_count(value: int, max_count: int) -> int:
    return max(1, min(int(value), max_count))


def load_eval_model(model_name: str, adapter: str | None):
    """Load a frozen FP16 checkpoint, optionally with a saved LoRA adapter."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    if adapter:
        print(f"    loading adapter: {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tok


def unload(model, tok) -> None:
    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
