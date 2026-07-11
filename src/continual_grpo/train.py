"""Run the apples-to-apples handwritten GRPO ablation grid."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import dump_json, load_config, method_list, model_list, model_slug
from .custom_grpo import train_cell


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "resolved_config.json", cfg)
    for method in method_list(cfg):
        for model in model_list(cfg):
            for index, task in enumerate(cfg["tasks"], 1):
                output = root / method / model_slug(model) / f"stage_{index:02d}_{task['name']}"
                print(f"\n=== TRAIN {method} | {model} | {task['name']} ===", flush=True)
                train_cell(model, task, method, cfg, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Handwritten LoRA GRPO ablation grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Completed cells are always reused")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
