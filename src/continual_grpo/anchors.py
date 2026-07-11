"""Build protected skill axes from anchor batches (proposal Aim 1, Eqs. 1-2).

Anchor batches are small samples from protected benchmarks (fairness,
knowledge).  Their language-modeling gradients on the trainable parameters
mark the directions that most affect the protected behavior; the SVD of the
mean gradient per parameter gives the low-rank protected basis used by the
skill-isolating spectral transform during training.

Note: with a freshly initialized LoRA adapter the `lora_A` gradients are
exactly zero (because `lora_B` starts at zero), so protection is carried by
the `lora_B` side; all-zero gradient samples are dropped rather than turned
into meaningless axes.
"""
from __future__ import annotations

import torch
from datasets import load_dataset

from .skill_axes import axes_from_gradient_samples, collect_gradient_sample
from .spectral_update import ProtectedAxes


def anchor_texts(spec: dict, seed: int) -> list[str]:
    ds = load_dataset(spec["dataset"], spec.get("subset"), split=spec.get("split", "train"))
    limit = int(spec.get("max_samples", 32))
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    prompt_field, answer_field = spec["prompt_field"], spec["answer_field"]
    return [f"{row[prompt_field]}\n{row[answer_field]}" for row in ds]


def build_protected_axes(model, tokenizer, cfg: dict) -> dict[str, ProtectedAxes]:
    specs = cfg.get("protected_anchors") or []
    if not specs:
        return {}
    batch_size = int(cfg.get("anchor_batch_size", 2))
    max_length = int(cfg.get("anchor_max_length", 512))
    device = next(model.parameters()).device
    was_training = model.training
    model.train()
    samples: dict[str, list[torch.Tensor]] = {}
    for spec in specs:
        texts = anchor_texts(spec, int(cfg.get("seed", 42)))
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                              padding=True, truncation=True, max_length=max_length).to(device)
            labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
            model(**batch, labels=labels).loss.backward()
            collect_gradient_sample(model, samples)
            model.zero_grad(set_to_none=True)
    model.train(was_training)
    samples = {name: grads for name, grads in samples.items()
               if any(grad.abs().max() > 0 for grad in grads)}
    return axes_from_gradient_samples(samples, int(cfg.get("protected_rank", 2)))
