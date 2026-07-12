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

Like the original Self-Distillation implementation, rollouts and sampler
log-probabilities are collected once before optimization.  The fixed buffer is
then shuffled and reused for ``epochs`` passes.  Consequently ``old_logps``
remain fixed while the policy changes, and the clipped ratio is active after
the first update.
"""
from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

from .anchors import build_protected_axes
from .losses import (clipped_grpo_loss, group_advantages, opsd_loss_for_chunk,
                     reference_kl_loss, token_logps)
from .rewards import (code_rewards, correctness_reward, format_reward,
                      math_correctness_reward, prepare_task)
from .rollout_io import (load_external_groups, original_gsm8k_messages,
                         resolve_rollout_source)
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


def _collect_rollout_buffer(model, tokenizer, rows, task_spec: dict, cfg: dict) -> list[dict]:
    """Collect fixed on-policy trajectories and their sampler log-probabilities.

    ``min_signal_groups`` (task-level) stops collection early once that many
    groups carry reward variance — the smoke configs use it so a tiny buffer
    is still guaranteed to exercise nonzero advantages.
    """
    task_name = task_spec["name"]
    k = int(cfg.get("num_generations", 4))
    prompt_batch = int(cfg.get("prompt_batch_size", 1))
    temperature = float(cfg.get("rollout_temperature", 0.8))
    min_signal = int(task_spec.get("min_signal_groups", 0))
    buffer = []
    model.eval()
    for start in range(0, len(rows), prompt_batch):
        if min_signal and sum(item["mixed"] for item in buffer) >= min_signal:
            break
        batch = rows.select(range(start, min(start + prompt_batch, len(rows))))
        prompts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
                   for x in batch["prompt"]]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=int(cfg.get("max_prompt_length", 512))).to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        expanded_ids = encoded.input_ids.repeat_interleave(k, 0)
        expanded_mask = encoded.attention_mask.repeat_interleave(k, 0)
        with torch.no_grad():
            generated = model.generate(
                input_ids=expanded_ids, attention_mask=expanded_mask,
                max_new_tokens=int(cfg.get("max_completion_length", 512)), do_sample=True,
                temperature=temperature, top_p=float(cfg.get("rollout_top_p", 1.0)),
                pad_token_id=tokenizer.pad_token_id,
            )
            completion_ids = generated[:, prompt_width:]
            completion_mask = completion_ids.ne(tokenizer.pad_token_id).long()
            attention_mask = torch.cat([expanded_mask, completion_mask], 1)
            logits = model(input_ids=generated, attention_mask=attention_mask).logits
            old_logps = token_logps(logits[:, prompt_width - 1:-1] / temperature,
                                    completion_ids)
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        gold = [answer for answer in batch["answer"] for _ in range(k)]
        correct_values, format_values = _reward_values(task_name, texts, gold)
        correct = torch.tensor(correct_values).view(-1, k)
        formatting = torch.tensor(format_values).view(-1, k)
        advantages, mixed = group_advantages(correct + formatting)
        for group in range(len(batch)):
            lo, hi = group * k, (group + 1) * k
            buffer.append({
                "generated": generated[lo:hi].cpu(),
                "attention_mask": attention_mask[lo:hi].cpu(),
                "completion_ids": completion_ids[lo:hi].cpu(),
                "completion_mask": completion_mask[lo:hi].cpu(),
                "old_logps": old_logps[lo:hi].cpu(),
                "advantages": advantages[group].cpu(),
                "correct": correct[group].cpu(),
                "mixed": bool(mixed[group]),
                "prompt_width": prompt_width,
            })
        del logits, old_logps, generated
    model.train()
    return buffer


def _external_rollout_buffer(model, tokenizer, task_spec: dict, cfg: dict) -> list[dict]:
    """Rebuild fixed groups from an original Self-Distillation rollout cache.

    Rewards come from the file's verifier, so advantages match the original
    run; only the sampler log-probabilities are rescored (the initial LoRA is
    an identity, so the scoring policy equals the model that sampled them).
    """
    groups = load_external_groups(task_spec["rollout_source"], int(cfg.get("seed", 42)),
                                  int(task_spec.get("max_samples", 0)))
    temperature = float(cfg.get("rollout_temperature", 0.8))
    max_completion = int(cfg.get("max_completion_length", 512))
    buffer = []
    model.eval()
    for group in groups:
        prompt_text = tokenizer.apply_chat_template(
            original_gsm8k_messages(group["question"]), tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                               max_length=int(cfg.get("max_prompt_length", 512)),
                               add_special_tokens=False).input_ids.to(model.device)
        prompt_width = prompt_ids.shape[1]
        completion_lists = [
            tokenizer(text, add_special_tokens=False, truncation=True,
                      max_length=max_completion - 1).input_ids + [tokenizer.eos_token_id]
            for text in group["completions"]
        ]
        k = len(completion_lists)
        width = max(len(ids) for ids in completion_lists)
        completion_ids = torch.full((k, width), tokenizer.pad_token_id,
                                    dtype=torch.long, device=model.device)
        for row, ids in enumerate(completion_lists):
            completion_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)
        completion_mask = completion_ids.ne(tokenizer.pad_token_id).long()
        generated = torch.cat([prompt_ids.expand(k, -1), completion_ids], 1)
        attention_mask = torch.cat(
            [torch.ones((k, prompt_width), dtype=torch.long, device=model.device), completion_mask], 1)
        with torch.no_grad():
            logits = model(input_ids=generated, attention_mask=attention_mask).logits
            old_logps = token_logps(logits[:, prompt_width - 1:-1] / temperature, completion_ids)
        correct = torch.tensor(group["correct"]).view(1, k)
        advantages, mixed = group_advantages(correct)
        buffer.append({
            "generated": generated.cpu(),
            "attention_mask": attention_mask.cpu(),
            "completion_ids": completion_ids.cpu(),
            "completion_mask": completion_mask.cpu(),
            "old_logps": old_logps.cpu(),
            "advantages": advantages[0].cpu(),
            "correct": correct[0].cpu(),
            "mixed": bool(mixed[0]),
            "prompt_width": prompt_width,
        })
        del logits, old_logps, generated
    model.train()
    return buffer


def _load_or_collect_buffer(model, tokenizer, rows, task_spec: dict, cfg: dict,
                            output: Path) -> list[dict]:
    """Build the fixed rollout buffer once per cell and persist it to disk."""
    cache = output / "rollout_buffer.pt"
    if cache.exists():
        buffer = torch.load(cache, map_location="cpu")
        print(f"Reusing persisted rollout buffer: {len(buffer)} groups ({cache})", flush=True)
        return buffer
    if task_spec.get("rollout_source"):
        buffer = _external_rollout_buffer(model, tokenizer, task_spec, cfg)
        source = task_spec["rollout_source"]
    else:
        buffer = _collect_rollout_buffer(model, tokenizer, rows, task_spec, cfg)
        source = "self-collected"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(buffer, cache)
    mixed = sum(item["mixed"] for item in buffer)
    print(f"Rollout buffer: {len(buffer)} groups ({mixed} with signal) from {source}; saved {cache}",
          flush=True)
    return buffer


def _apply_step(model, optimizer, scheduler, axes, use_spectral: bool, cfg: dict,
                gradient_scale: float = 1.0) -> tuple[float, dict[str, float]]:
    if gradient_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
    spectral_stats = {}
    if use_spectral:
        spectral_stats = transform_gradients(model, axes, int(cfg.get("spectral_target_rank", 1)))
    spectral_stats = dict(spectral_stats)
    spectral_stats["lr"] = optimizer.param_groups[0]["lr"]
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm), spectral_stats


def _new_window() -> dict:
    return {"grpo": 0.0, "grpo_abs": 0.0, "kl": 0.0, "opsd": 0.0, "total": 0.0, "pairs": 0,
            "correct": 0.0, "mixed": 0.0, "paired_groups": 0.0, "abs_advantage": 0.0,
            "length": 0.0, "seqs": 0, "groups": 0}


def _log_row(step: int, method: str, window: dict, grad_norm: float,
             spectral_stats: dict[str, float]) -> dict:
    seqs = max(1, window["seqs"])
    groups = max(1, window["groups"])
    row = {
        "step": step, "method": method, "total_loss": window["total"] / seqs,
        "grpo_policy_loss": window["grpo"] / seqs,
        "grpo_seq_abs_loss": window["grpo_abs"] / seqs,
        "reference_kl_loss": window["kl"] / seqs,
        "opsd_loss": window["opsd"] / seqs, "opsd_pairs": window["pairs"],
        "correct_fraction": window["correct"] / seqs, "mixed_group_fraction": window["mixed"] / groups,
        "paired_group_fraction": window["paired_groups"] / groups,
        "mean_abs_advantage": window["abs_advantage"] / seqs,
        "grad_norm": grad_norm, "completion_length": window["length"] / seqs,
    }
    row.update(spectral_stats)
    return row


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
    task_spec = {**task_spec, "rollout_source": resolve_rollout_source(task_spec, model_name)}
    rows = None if task_spec.get("rollout_source") else prepare_task(task_spec, int(cfg.get("seed", 42)))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = cfg.get("optimizer", "adamw")
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=float(cfg.get("learning_rate", 2e-6)),
                                    momentum=float(cfg.get("momentum", 0.9)), weight_decay=0.0)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=float(cfg.get("learning_rate", 2e-6)),
                                      weight_decay=float(cfg.get("weight_decay", 0.0)))
    else:
        raise ValueError("optimizer must be 'sgd' or 'adamw'")
    use_opsd = method in {"copsd", "combined"}
    use_spectral = method in {"skill_ortho", "combined"}
    axes = build_protected_axes(model, tokenizer, cfg) if use_spectral else {}
    k = int(cfg.get("num_generations", 4))
    prompt_batch = int(cfg.get("prompt_batch_size", 1))
    per_device = int(cfg.get("per_device_batch_size", k))
    accumulation = int(cfg.get("gradient_accumulation_steps", 1))
    if per_device < k or per_device % k:
        raise ValueError("per_device_batch_size must be a multiple of num_generations")
    groups_per_chunk = per_device // k
    temperature = float(cfg.get("rollout_temperature", 0.8))
    clip_eps = float(cfg.get("grpo_clip", 0.2))
    beta = float(cfg.get("kl_beta", 0.04))
    opsd_weight = float(cfg.get("opsd_weight", 0.1))
    margin = float(cfg.get("opsd_margin", 0.2))
    negative_weight = float(cfg.get("opsd_negative_weight", 0.1))
    opsd_temperature = float(cfg.get("opsd_temperature", 1.0))
    # Match the original Self-Distillation design: collect trajectories once,
    # before any optimizer update, and keep their sampler log-probabilities.
    # Reset after optional anchor construction so every ablation samples the
    # same trajectories.
    torch.manual_seed(int(cfg.get("seed", 42)))
    random.seed(int(cfg.get("seed", 42)))
    rollout_buffer = _load_or_collect_buffer(model, tokenizer, rows, task_spec, cfg, output)
    indices = list(range(len(rollout_buffer)))
    batches_per_epoch = math.ceil(len(rollout_buffer) / groups_per_chunk)
    total_steps = max(1, math.ceil(batches_per_epoch / accumulation) * int(cfg.get("epochs", 1)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * float(cfg.get("warmup_ratio", 0.1))), total_steps)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(cfg.get("epochs", 1))):
        random.shuffle(indices)
        window = _new_window()
        pending = 0
        for start in range(0, len(indices), groups_per_chunk):
            group_items = [rollout_buffer[i] for i in indices[start:start + groups_per_chunk]]
            # Groups in a collection batch share padded widths; process each
            # independently so buffers collected in different batches need not.
            # Group size comes from each buffer entry: external rollout caches
            # may use a different K than the config's num_generations.
            batch_seqs = sum(item["completion_ids"].shape[0] for item in group_items)
            for item in group_items:
                generated = item["generated"].to(model.device)
                attention_mask = item["attention_mask"].to(model.device)
                completion_ids = item["completion_ids"].to(model.device)
                mask = item["completion_mask"].to(model.device).float()
                lengths = mask.sum(1).clamp_min(1)
                old_logps = item["old_logps"].to(model.device)
                advantages = item["advantages"].to(model.device)
                correct = item["correct"].to(model.device)
                prompt_width = item["prompt_width"]
                logits = model(input_ids=generated, attention_mask=attention_mask).logits
                completion_logits = logits[:, prompt_width - 1:-1] / temperature
                policy_logps = token_logps(completion_logits, completion_ids)
                with torch.no_grad(), _adapter_off(model):
                    reference_logits = model(
                        input_ids=generated, attention_mask=attention_mask,
                    ).logits[:, prompt_width - 1:-1] / temperature
                flat_adv = advantages.unsqueeze(1)
                grpo_loss, grpo_abs = clipped_grpo_loss(policy_logps, old_logps, flat_adv, mask, lengths, clip_eps)
                kl_loss = reference_kl_loss(completion_logits, reference_logits, mask, lengths)
                opsd_loss = torch.zeros((), device=model.device)
                pairs = 0
                if use_opsd:
                    opsd_loss, pairs = opsd_loss_for_chunk(
                        correct.unsqueeze(0), completion_ids.shape[0], completion_ids,
                        completion_logits, reference_logits, policy_logps, mask,
                        margin, negative_weight, opsd_temperature,
                    )
                total = grpo_loss + beta * kl_loss + opsd_weight * opsd_loss
                chunk_seqs = completion_ids.shape[0]
                (total * (chunk_seqs / (batch_seqs * accumulation))).backward()
                window["grpo"] += float(grpo_loss.detach()) * chunk_seqs
                window["grpo_abs"] += grpo_abs * chunk_seqs
                window["kl"] += float(kl_loss.detach()) * chunk_seqs
                window["opsd"] += float(opsd_loss.detach()) * chunk_seqs
                window["total"] += float(total.detach()) * chunk_seqs
                window["pairs"] += pairs
                window["seqs"] += chunk_seqs
                del logits, completion_logits, reference_logits, policy_logps
                window["correct"] += float(correct.sum())
                window["mixed"] += float(item["mixed"])
                window["paired_groups"] += float(correct.ge(1.0).any() and correct.lt(1.0).any())
                window["abs_advantage"] += float(advantages.abs().sum())
                window["length"] += float(lengths.sum())
                window["groups"] += 1
            pending += 1
            if pending == accumulation:
                global_step += 1
                grad_norm, spectral_stats = _apply_step(model, optimizer, scheduler, axes, use_spectral, cfg)
                log = _log_row(global_step, method, window, grad_norm, spectral_stats)
                print(log, flush=True)
                _save_log(output / "train_metrics.jsonl", log)
                window = _new_window()
                pending = 0
        if pending:
            global_step += 1
            # Losses were divided by the configured accumulation count. Undo
            # that extra division when the epoch ends with a partial window.
            grad_norm, spectral_stats = _apply_step(model, optimizer, scheduler, axes, use_spectral, cfg,
                                                    accumulation / pending)
            log = _log_row(global_step, method, window, grad_norm, spectral_stats)
            print(log, flush=True)
            _save_log(output / "train_metrics.jsonl", log)
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
