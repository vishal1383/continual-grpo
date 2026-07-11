"""Answer-conditioned protected axes from held-out knowledge/fairness anchors."""
from __future__ import annotations

import torch
from datasets import load_dataset

from .skill_axes import axes_from_gradient_samples, collect_gradient_sample
from .spectral_update import ProtectedAxes


def _mmlu_examples(spec: dict, seed: int) -> list[tuple[str, str]]:
    ds = load_dataset(spec.get("dataset", "cais/mmlu"), spec.get("subset", "all"),
                      split=spec.get("split", "validation"))
    limit = int(spec.get("max_samples", 64))
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    examples = []
    for row in ds:
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
        examples.append((f"{row['question']}\n\nOptions:\n{options}\n\nAnswer:", "ABCD"[int(row["answer"])]))
    return examples


def _bbq_examples(spec: dict, seed: int) -> list[tuple[str, str]]:
    ds = load_dataset(spec.get("dataset", "oskarvanderwal/bbq"), spec.get("subset", "All"),
                      split=spec.get("split", "test"))
    limit = int(spec.get("max_samples", 64))
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    examples = []
    for row in ds:
        options = "\n".join(f"{letter}. {row[f'ans{i}']}" for i, letter in enumerate("ABC"))
        prompt = f"{row.get('context', '')}\n{row['question']}\n\nOptions:\n{options}\n\nAnswer:"
        examples.append((prompt, "ABC"[int(row["label"])]))
    return examples


def anchor_examples(spec: dict, seed: int) -> list[tuple[str, str]]:
    name = spec.get("name")
    if name == "mmlu":
        return _mmlu_examples(spec, seed)
    if name == "bbq":
        return _bbq_examples(spec, seed)
    ds = load_dataset(spec["dataset"], spec.get("subset"), split=spec.get("split", "train"))
    limit = int(spec.get("max_samples", 32))
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    return [(str(row[spec["prompt_field"]]), str(row[spec["answer_field"]])) for row in ds]


def build_protected_axes(model, tokenizer, cfg: dict) -> dict[str, ProtectedAxes]:
    """Estimate axes using loss only on correct protected answers, not prompts."""
    specs = cfg.get("protected_anchors") or []
    if not specs:
        return {}
    batch_size = int(cfg.get("anchor_batch_size", 2))
    max_length = int(cfg.get("anchor_max_length", 512))
    device = next(model.parameters()).device
    was_training, old_padding = model.training, tokenizer.padding_side
    model.train()
    tokenizer.padding_side = "right"
    samples: dict[str, list[torch.Tensor]] = {}
    try:
        for spec in specs:
            examples = anchor_examples(spec, int(cfg.get("seed", 42)))
            for start in range(0, len(examples), batch_size):
                chunk = examples[start:start + batch_size]
                prompts = [prompt for prompt, _ in chunk]
                texts = [f"{prompt} {answer}" for prompt, answer in chunk]
                batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_length).to(device)
                prompt_tokens = tokenizer(prompts, padding=False, truncation=True,
                                          max_length=max_length)["input_ids"]
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"] == 0] = -100
                for row, ids in enumerate(prompt_tokens):
                    labels[row, :min(len(ids), labels.shape[1])] = -100
                model(**batch, labels=labels).loss.backward()
                collect_gradient_sample(model, samples)
                model.zero_grad(set_to_none=True)
    finally:
        tokenizer.padding_side = old_padding
        model.train(was_training)
    samples = {name: grads for name, grads in samples.items()
               if any(grad.abs().max() > 0 for grad in grads)}
    return axes_from_gradient_samples(samples, int(cfg.get("protected_rank", 2)))
