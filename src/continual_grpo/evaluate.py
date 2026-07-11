from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from .common import load_config


def checkpoint_series(cfg: dict) -> list[tuple[str, str]]:
    root = Path(cfg["output_dir"])
    points = [("stage_00_base", cfg["model"])]
    for i, task in enumerate(cfg["tasks"], 1):
        adapter = root / f"stage_{i:02d}_{task['name']}" / "final_adapter"
        if adapter.exists():
            points.append((f"stage_{i:02d}_{task['name']}", str(adapter)))
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
    tasks = ",".join(cfg.get("eval_tasks", ["gsm8k", "humaneval", "bbq"]))
    for label, model in checkpoint_series(cfg):
        target = out / label
        if (target / "results.json").exists():
            continue
        cmd = ["lm_eval", "--model", "hf", "--model_args", f"pretrained={model},trust_remote_code=True",
               "--tasks", tasks, "--batch_size", "auto", "--output_path", str(target), "--log_samples",
               "--apply_chat_template"]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if "humaneval" in tasks:
            if not args.allow_code_execution:
                raise SystemExit("HumanEval executes generated code. Re-run inside Docker with --allow-code-execution.")
            cmd += ["--confirm_run_unsafe_code"]
        env = os.environ.copy()
        if args.allow_code_execution:
            env["HF_ALLOW_CODE_EVAL"] = "1"
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
