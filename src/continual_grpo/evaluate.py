from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from .common import load_config, method_list, model_list, model_slug


def checkpoint_series(cfg: dict) -> list[tuple[str, str, str | None]]:
    root = Path(cfg["output_dir"])
    points = []
    for model in model_list(cfg):
        slug = model_slug(model)
        points.append((f"base/{slug}/stage_00_base", model, None))
        for method in method_list(cfg):
            for i, task in enumerate(cfg["tasks"], 1):
                adapter = root / method / slug / f"stage_{i:02d}_{task['name']}" / "final_adapter"
                if adapter.exists():
                    points.append((f"{method}/{slug}/stage_{i:02d}_{task['name']}", model, str(adapter)))
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-code-execution", action="store_true",
                        help="Required to execute HumanEval generations; use only in the Docker sandbox.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"]) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tasks = cfg.get("eval_tasks", ["gsm8k", "humaneval", "bbq"])
    limits = cfg.get("eval_limits") or {}
    include_dir = Path(__file__).resolve().parents[2] / "lm_eval_tasks"
    batch_size = str(cfg.get("eval_batch_size", 1))
    max_memory = str(cfg.get("max_memory_per_gpu", "70GB"))
    for label, model, adapter in checkpoint_series(cfg):
        for task in tasks:
            target = out / label / task
            if any(target.glob("**/results*.json")):
                continue
            print(f"\n=== {label}: {task} ===", flush=True)
            model_args = (
                f"pretrained={model},trust_remote_code=True,dtype=float16,"
                f"max_memory_per_gpu={max_memory}"
            )
            if adapter:
                model_args += f",peft={adapter}"
            cmd = ["lm_eval", "--model", "hf", "--model_args", model_args,
                   "--tasks", task, "--batch_size", batch_size, "--output_path", str(target),
                   "--apply_chat_template"]
            if include_dir.is_dir():
                cmd += ["--include_path", str(include_dir)]
            limit = limits.get(task, args.limit)
            if limit is not None:
                cmd += ["--limit", str(limit)]
            if task == "humaneval":
                if not args.allow_code_execution:
                    raise SystemExit("HumanEval executes generated code. Re-run inside Docker with --allow-code-execution.")
                cmd += ["--confirm_run_unsafe_code"]
            env = os.environ.copy()
            if args.allow_code_execution:
                env["HF_ALLOW_CODE_EVAL"] = "1"
            subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
