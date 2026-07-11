"""One handwritten GRPO loop shared by every ablation.

The only method differences are additive C-OPSD loss and/or the spectral
gradient transform immediately before optimizer.step().

Batching semantics (all read from the config):
  prompt_batch_size           prompts rolled out together (each with
                              num_generations completions)
  per_device_batch_size       completions scored/backpropagated per forward
                              chunk; chunks are whole prompt groups so C-OPSD
                              pairing is never split
  gradient_accumulation_steps prompt batches accumulated per optimizer step

Scoring divides all logits by the rollout temperature, so the trained,
old, and reference log-probabilities describe the same tempered policy the
completions were sampled from. There is exactly one optimizer pass per
rollout, so the old policy equals the sampler and its log-probabilities are
the detached policy log-probabilities (no extra forward); the clipped ratio
becomes active only if multiple updates per rollout are ever added.
"""
from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .anchors import build_protected_axes
from .losses import (clipped_grpo_loss, group_advantages, opsd_loss_for_chunk,
                     reference_kl_loss, token_logps)
from .rewards import (code_rewards, correctness_reward, format_reward,
                      math_correctness_reward, prepare_task)
from .spectral_update import transform_gradients


def _adapter_off(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()


def _save_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _reward_values(task_name: str, texts: list[str], gold: list[str]):
    if task_name == "humaneval":
        return code_rewards(texts, gold), [0.0] * len(texts)
    if task_name == "math":
        return math_correctness_reward(texts, gold), format_reward(texts)
    return correctness_reward(texts, gold), format_reward(texts)


def _apply_step(model, optimizer, axes, use_spectral: bool, cfg: dict,
                gradient_scale: float = 1.0) -> float:
    if gradient_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
    if use_spectral:
        transform_gradients(model, axes, int(cfg.get("spectral_target_rank", 1)))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


def _new_window() -> dict:
    return {"grpo": 0.0, "kl": 0.0, "opsd": 0.0, "total": 0.0, "pairs": 0,
            "correct": 0.0, "mixed": 0.0, "length": 0.0, "seqs": 0, "groups": 0}


def _log_row(step: int, method: str, window: dict, grad_norm: float) -> dict:
    seqs = max(1, window["seqs"])
    groups = max(1, window["groups"])
    return {
        "step": step, "method": method, "total_loss": window["total"] / seqs,
        "grpo_policy_loss": window["grpo"] / seqs, "reference_kl_loss": window["kl"] / seqs,
        "opsd_loss": window["opsd"] / seqs, "opsd_pairs": window["pairs"],
        "correct_fraction": window["correct"] / seqs, "mixed_group_fraction": window["mixed"] / groups,
        "grad_norm": grad_norm, "completion_length": window["length"] / seqs,
    }


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
    per_device = int(cfg.get("per_device_batch_size", k))
    accumulation = int(cfg.get("gradient_accumulation_steps", 1))
    groups_per_chunk = max(1, per_device // k)
    temperature = float(cfg.get("rollout_temperature", 0.8))
    clip_eps = float(cfg.get("grpo_clip", 0.2))
    beta = float(cfg.get("kl_beta", 0.04))
    opsd_weight = float(cfg.get("opsd_weight", 0.1))
    margin = float(cfg.get("opsd_margin", 0.2))
    negative_weight = float(cfg.get("opsd_negative_weight", 0.1))
    opsd_temperature = float(cfg.get("opsd_temperature", 1.0))
    indices = list(range(len(rows)))
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(cfg.get("epochs", 1))):
        random.shuffle(indices)
        window = _new_window()
        pending = 0
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
                    temperature=temperature,
                    top_p=float(cfg.get("rollout_top_p", 0.95)), pad_token_id=tokenizer.pad_token_id,
                )
            model.train()
            completion_ids = generated[:, prompt_width:]
            completion_mask = completion_ids.ne(tokenizer.pad_token_id).long()
            attention_mask = torch.cat([expanded_mask, completion_mask], 1)
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            gold = [answer for answer in batch["answer"] for _ in range(k)]
            correct_values, format_values = _reward_values(task_spec["name"], texts, gold)
            correct = torch.tensor(correct_values, device=model.device).view(-1, k)
            formatting = torch.tensor(format_values, device=model.device).view(-1, k)
            rewards = correct + formatting
            advantages, mixed = group_advantages(rewards)
            mask_all = completion_mask.float()
            lengths_all = mask_all.sum(1).clamp_min(1)
            n_groups = rewards.shape[0]
            batch_seqs = n_groups * k
            for g0 in range(0, n_groups, groups_per_chunk):
                g1 = min(g0 + groups_per_chunk, n_groups)
                sl = slice(g0 * k, g1 * k)
                logits = model(input_ids=generated[sl], attention_mask=attention_mask[sl]).logits
                completion_logits = logits[:, prompt_width - 1:-1] / temperature
                policy_logps = token_logps(completion_logits, completion_ids[sl])
                # Single update per rollout: the sampler is the old policy.
                old_logps = policy_logps.detach()
                with torch.no_grad(), _adapter_off(model):
                    reference_logits = model(
                        input_ids=generated[sl], attention_mask=attention_mask[sl],
                    ).logits[:, prompt_width - 1:-1] / temperature
                    reference_logps = token_logps(reference_logits, completion_ids[sl])
                mask = mask_all[sl]
                lengths = lengths_all[sl]
                flat_adv = advantages[g0:g1].flatten().unsqueeze(1)
                grpo_loss = clipped_grpo_loss(policy_logps, old_logps, flat_adv, mask, lengths, clip_eps)
                kl_loss = reference_kl_loss(policy_logps, reference_logps, mask, lengths)
                opsd_loss = torch.zeros((), device=model.device)
                pairs = 0
                if use_opsd:
                    opsd_loss, pairs = opsd_loss_for_chunk(
                        correct[g0:g1], k, completion_ids[sl], completion_logits,
                        reference_logits, policy_logps, mask,
                        margin, negative_weight, opsd_temperature,
                    )
                total = grpo_loss + beta * kl_loss + opsd_weight * opsd_loss
                chunk_seqs = (g1 - g0) * k
                (total * (chunk_seqs / (batch_seqs * accumulation))).backward()
                window["grpo"] += float(grpo_loss.detach()) * chunk_seqs
                window["kl"] += float(kl_loss.detach()) * chunk_seqs
                window["opsd"] += float(opsd_loss.detach()) * chunk_seqs
                window["total"] += float(total.detach()) * chunk_seqs
                window["pairs"] += pairs
                window["seqs"] += chunk_seqs
                del logits, completion_logits, reference_logits, policy_logps
            window["correct"] += float(correct.sum())
            window["mixed"] += float(mixed.float().sum())
            window["length"] += float(lengths_all.sum())
            window["groups"] += n_groups
            pending += 1
            if pending == accumulation:
                global_step += 1
                grad_norm = _apply_step(model, optimizer, axes, use_spectral, cfg)
                log = _log_row(global_step, method, window, grad_norm)
                print(log, flush=True)
                _save_log(output / "train_metrics.jsonl", log)
                window = _new_window()
                pending = 0
        if pending:
            global_step += 1
            # Losses were divided by the configured accumulation count. Undo
            # that extra division when the epoch ends with a partial window.
            grad_norm = _apply_step(model, optimizer, axes, use_spectral, cfg,
                                    accumulation / pending)
            log = _log_row(global_step, method, window, grad_norm)
            print(log, flush=True)
            _save_log(output / "train_metrics.jsonl", log)
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
