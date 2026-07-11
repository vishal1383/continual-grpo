from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from .common import dump_json, load_config, model_list, model_slug


SYSTEM = "Solve carefully. Put the final answer after ####."


def _final(text: str) -> str:
    matches = re.findall(r"####\s*([^\n]+)", text)
    return (matches[-1] if matches else text.strip().splitlines()[-1]).replace(",", "").strip()


def correctness_reward(completions, answer, **_):
    """Exact-answer verifier used by GRPO; one reward per sampled completion."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [1.0 if _final(text) == _final(gold) else 0.0 for text, gold in zip(texts, answer)]


def format_reward(completions, **_):
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [0.1 if re.search(r"####\s*\S+", text) else 0.0 for text in texts]


def prepare_task(spec: dict, seed: int):
    ds = load_dataset(spec["dataset"], spec.get("subset"), split=spec.get("split", "train"))
    limit = int(spec.get("max_samples", 0))
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    prompt_field, answer_field = spec["prompt_field"], spec["answer_field"]
    return ds.map(lambda row: {
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row[prompt_field]}],
        "answer": row[answer_field],
    }, remove_columns=ds.column_names)


def run(config_path: str, resume: bool = False) -> None:
    cfg = load_config(config_path)
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "resolved_config.json", cfg)
    history = []
    for base_model in model_list(cfg):
        train_model_stages(cfg, root, base_model, resume, history)


def train_model_stages(cfg: dict, root: Path, base_model: str, resume: bool, history: list) -> None:
    model = base_model
    for index, task in enumerate(cfg["tasks"], start=1):
        stage = root / model_slug(base_model) / f"stage_{index:02d}_{task['name']}"
        final_adapter = stage / "final_adapter"
        if resume and final_adapter.exists():
            model = str(final_adapter)
            continue
        train_data = prepare_task(task, int(cfg.get("seed", 42)))
        args = GRPOConfig(
            output_dir=str(stage), seed=int(cfg.get("seed", 42)),
            beta=float(cfg.get("kl_beta", 0.04)),
            learning_rate=float(cfg.get("learning_rate", 2e-6)),
            num_train_epochs=float(cfg.get("epochs", 1)),
            per_device_train_batch_size=int(cfg.get("per_device_batch_size", 1)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
            num_generations=int(cfg.get("num_generations", 4)),
            max_prompt_length=int(cfg.get("max_prompt_length", 768)),
            max_completion_length=int(cfg.get("max_completion_length", 384)),
            bf16=bool(cfg.get("bf16", True)), fp16=not bool(cfg.get("bf16", True)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1, save_strategy="epoch", report_to="none",
        )
        peft = None
        if cfg.get("use_lora", True):
            peft = LoraConfig(r=int(cfg.get("lora_r", 16)), lora_alpha=int(cfg.get("lora_r", 16)),
                              target_modules="all-linear", task_type="CAUSAL_LM")
        trainer = GRPOTrainer(model=model, reward_funcs=[correctness_reward, format_reward],
                              args=args, train_dataset=train_data, peft_config=peft)
        if cfg.get("gradient_checkpointing", True) and hasattr(trainer.model, "enable_input_require_grads"):
            trainer.model.enable_input_require_grads()
        has_checkpoint = any(stage.glob("checkpoint-*"))
        result = trainer.train(resume_from_checkpoint=True if resume and has_checkpoint else None)
        trainer.save_model(str(final_adapter))
        history.append({"model": base_model, "stage": index, "task": task["name"], "metrics": result.metrics})
        dump_json(root / "train_history.json", history)
        model = str(final_adapter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential GRPO with ordinary reference-policy KL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.resume)


if __name__ == "__main__":
    main()
