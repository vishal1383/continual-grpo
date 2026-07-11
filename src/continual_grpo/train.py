from __future__ import annotations

import argparse
from pathlib import Path

from peft import LoraConfig
from trl import GRPOConfig

from .anchors import build_protected_axes
from .common import dump_json, load_config, method_list, model_list, model_slug
from .opsd_trainer import OPSDTrainer, RecordingReward
from .ortho import SkillOrthoCallback
from .rewards import correctness_reward, format_reward, prepare_task


def run(config_path: str, resume: bool = False) -> None:
    cfg = load_config(config_path)
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "resolved_config.json", cfg)
    history = []
    for method in method_list(cfg):
        for base_model in model_list(cfg):
            train_arm(cfg, root, method, base_model, resume, history)


def train_arm(cfg: dict, root: Path, method: str, base_model: str, resume: bool, history: list) -> None:
    model = base_model
    for index, task in enumerate(cfg["tasks"], start=1):
        stage = root / method / model_slug(base_model) / f"stage_{index:02d}_{task['name']}"
        final_adapter = stage / "final_adapter"
        if resume and final_adapter.exists():
            model = str(final_adapter)
            continue
        print(f"\n=== train {method}: {base_model}: {task['name']} ===", flush=True)
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
            max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
            logging_steps=1, save_strategy="epoch", report_to="none",
        )
        peft = None
        if cfg.get("use_lora", True):
            peft = LoraConfig(r=int(cfg.get("lora_r", 16)), lora_alpha=int(cfg.get("lora_r", 16)),
                              target_modules="all-linear", task_type="CAUSAL_LM")
        trainer = OPSDTrainer(
            model=model, reward_funcs=[RecordingReward(correctness_reward), format_reward],
            args=args, train_dataset=train_data, peft_config=peft,
            method=method,
            opsd_margin=float(cfg.get("opsd_margin", 0.2)),
            opsd_negative_weight=float(cfg.get("opsd_negative_weight", 0.1)),
            opsd_temperature=float(cfg.get("opsd_temperature", 1.0)),
        )
        if cfg.get("gradient_checkpointing", True) and hasattr(trainer.model, "enable_input_require_grads"):
            trainer.model.enable_input_require_grads()
        if method in ("skill_ortho", "combined"):
            axes = build_protected_axes(trainer.model, trainer.processing_class, cfg)
            trainer.add_callback(SkillOrthoCallback(axes, int(cfg.get("spectral_target_rank", 1))))
        has_checkpoint = any(stage.glob("checkpoint-*"))
        result = trainer.train(resume_from_checkpoint=True if resume and has_checkpoint else None)
        trainer.save_model(str(final_adapter))
        history.append({"method": method, "model": base_model, "stage": index,
                        "task": task["name"], "metrics": result.metrics})
        dump_json(root / "train_history.json", history)
        model = str(final_adapter)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continual GRPO with skill-orthogonal and C-OPSD method arms")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.resume)


if __name__ == "__main__":
    main()
