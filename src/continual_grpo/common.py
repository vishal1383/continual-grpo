from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = {"output_dir", "tasks"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if not cfg.get("models") and not cfg.get("model"):
        raise ValueError("Configuration needs 'model' or a non-empty 'models' list")
    return cfg


def model_list(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("models") or [cfg["model"]])


def model_slug(name: str) -> str:
    return name.replace("/", "__")


def dump_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
