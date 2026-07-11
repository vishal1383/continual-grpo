"""One handwritten GRPO loop shared by every ablation.

The only method differences are additive C-OPSD loss and/or the spectral
gradient transform immediately before optimizer.step().
"""
from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .anchors import build_protected_axes
from .rewards import code_rewards, correctness_reward, format_reward, prepare_task
from .spectral_update import transform_gradients


def _adapter_off(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()


def _token_logps(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def _advantages(rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True, unbiased=False)
    advantages = torch.where(std > 1e-8, (rewards - mean) / std.clamp_min(1e-8), 0.0)
    return advantages, (std.squeeze(1) > 1e-8)


def _divergence_mask(positive: torch.Tensor, negative: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(mask)
    for row in range(positive.shape[0]):
        different = ((positive[row] != negative[row]) & mask[row].bool()).nonzero()
        start = int(different[0]) if different.numel() else 0
        out[row, start:] = mask[row, start:]
    return out


def _opsd_pairs(rewards: torch.Tensor) -> tuple[list[int], list[int]]:
    positives, negatives = [], []
    for group in range(rewards.shape[0]):
        pos = (rewards[group] >= 1.0).nonzero().flatten().tolist()
        neg = (rewards[group] < 1.0).nonzero().flatten().tolist()
        if pos and neg:
            for index in pos:
                positives.append(group * rewards.shape[1] + index)
                negatives.append(group * rewards.shape[1] + neg[0])
    return positives, negatives


def _save_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load_policy(model_name: str, cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    if cfg.get("finetune_mode", "lora") == "lora":
        rank = int(cfg.get("lora_r", 32))
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=int(cfg.get("lora_alpha", rank)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)), bias="none",
            target_modules=cfg["lora_target_modules"], task_type="CAUSAL_LM",
        ))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    return model, tokenizer


def train_cell(model_name: str, task_spec: dict, method: str, cfg: dict, output: Path) -> None:
    final = output / "final_adapter"
    if final.exists():
        print(f"Skipping completed cell: {output}")
        return
    torch.manual_seed(int(cfg.get("seed", 42)))
    random.seed(int(cfg.get("seed", 42)))
    model, tokenizer = load_policy(model_name, cfg)
    rows = prepare_task(task_spec, int(cfg.get("seed", 42)))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.get("learning_rate", 2e-6)), weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    use_opsd = method in {"copsd", "combined"}
    use_spectral = method in {"skill_ortho", "combined"}
    axes = build_protected_axes(model, tokenizer, cfg) if use_spectral else {}
    k = int(cfg.get("num_generations", 4))
    prompt_batch = int(cfg.get("prompt_batch_size", 1))
    clip_eps = float(cfg.get("grpo_clip", 0.2))
    beta = float(cfg.get("kl_beta", 0.04))
    opsd_weight = float(cfg.get("opsd_weight", 0.1))
    margin = float(cfg.get("opsd_margin", 0.2))
    indices = list(range(len(rows)))
    global_step = 0
    for epoch in range(int(cfg.get("epochs", 1))):
        random.shuffle(indices)
        for start in range(0, len(indices), prompt_batch):
            batch = rows.select(indices[start:start + prompt_batch])
            chats = batch["prompt"]
            prompts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in chats]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=int(cfg.get("max_prompt_length", 512))).to(model.device)
            prompt_width = encoded.input_ids.shape[1]
            expanded_ids = encoded.input_ids.repeat_interleave(k, 0)
            expanded_mask = encoded.attention_mask.repeat_interleave(k, 0)
            model.eval()
            with torch.no_grad():
                generated = model.generate(
                    input_ids=expanded_ids, attention_mask=expanded_mask,
                    max_new_tokens=int(cfg.get("max_completion_length", 512)), do_sample=True,
                    temperature=float(cfg.get("rollout_temperature", 0.8)),
                    top_p=float(cfg.get("rollout_top_p", 0.95)), pad_token_id=tokenizer.pad_token_id,
                )
            model.train()
            completion_ids = generated[:, prompt_width:]
            completion_mask = completion_ids.ne(tokenizer.pad_token_id).long()
            attention_mask = torch.cat([expanded_mask, completion_mask], 1)
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            gold = [answer for answer in batch["answer"] for _ in range(k)]
            if task_spec["name"] == "humaneval":
                correct_values = code_rewards(texts, gold)
                format_values = [0.0] * len(texts)
            else:
                correct_values = correctness_reward(texts, gold)
                format_values = format_reward(texts)
            correct = torch.tensor(correct_values, device=model.device).view(-1, k)
            formatting = torch.tensor(format_values, device=model.device).view(-1, k)
            rewards = correct + formatting
            advantages, mixed = _advantages(rewards)

            # Old policy is the policy that produced this rollout.
            with torch.no_grad():
                old_logits = model(input_ids=generated, attention_mask=attention_mask).logits
                old_logps = _token_logps(old_logits[:, prompt_width - 1:-1], completion_ids)
            logits = model(input_ids=generated, attention_mask=attention_mask).logits
            completion_logits = logits[:, prompt_width - 1:-1]
            policy_logps = _token_logps(completion_logits, completion_ids)
            with torch.no_grad(), _adapter_off(model):
                reference_logits = model(input_ids=generated, attention_mask=attention_mask).logits[:, prompt_width - 1:-1]
                reference_logps = _token_logps(reference_logits, completion_ids)

            mask = completion_mask.float()
            lengths = mask.sum(1).clamp_min(1)
            ratio = torch.exp(policy_logps - old_logps)
            flat_adv = advantages.flatten().unsqueeze(1)
            pg = torch.minimum(ratio * flat_adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * flat_adv)
            grpo_loss = -((pg * mask).sum(1) / lengths).mean()
            log_ratio = reference_logps - policy_logps
            per_token_kl = torch.exp(log_ratio) - log_ratio - 1
            kl_loss = ((per_token_kl * mask).sum(1) / lengths).mean()
            opsd_loss = torch.zeros((), device=model.device)
            pairs = 0
            if use_opsd:
                pos, neg = _opsd_pairs(correct)
                pairs = len(pos)
                if pos:
                    pos_t = torch.tensor(pos, device=model.device)
                    neg_t = torch.tensor(neg, device=model.device)
                    divergence = _divergence_mask(completion_ids[pos_t], completion_ids[neg_t], mask[pos_t])
                    ref_probs = F.softmax(reference_logits[pos_t].float(), -1)
                    positive_kl = (F.kl_div(F.log_softmax(completion_logits[pos_t].float(), -1),
                                            ref_probs, reduction="none").sum(-1) * divergence).sum()
                    positive_kl = positive_kl / divergence.sum().clamp_min(1)
                    pos_score = (policy_logps[pos_t] * mask[pos_t]).sum(1) / lengths[pos_t]
                    neg_score = (policy_logps[neg_t] * mask[neg_t]).sum(1) / lengths[neg_t]
                    margin_loss = F.relu(margin - (pos_score - neg_score)).mean()
                    opsd_loss = positive_kl + float(cfg.get("opsd_negative_weight", 0.1)) * margin_loss
            total = grpo_loss + beta * kl_loss + opsd_weight * opsd_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if use_spectral:
                transform_gradients(model, axes, int(cfg.get("spectral_target_rank", 1)))
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            global_step += 1
            log = {
                "step": global_step, "method": method, "total_loss": float(total.detach()),
                "grpo_policy_loss": float(grpo_loss.detach()), "reference_kl_loss": float(kl_loss.detach()),
                "opsd_loss": float(opsd_loss.detach()), "opsd_pairs": pairs,
                "correct_fraction": float(correct.mean()), "mixed_group_fraction": float(mixed.float().mean()),
                "grad_norm": float(grad_norm), "completion_length": float(lengths.float().mean()),
            }
            print(log, flush=True)
            _save_log(output / "train_metrics.jsonl", log)
            del logits, old_logits, reference_logits, completion_logits
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
