from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output_dir"])
    rows = []
    for file in sorted((root / "eval").glob("**/results*.json")):
        payload = json.loads(file.read_text())
        stage = file.parent.name
        for task, metrics in payload.get("results", {}).items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not metric.endswith("_stderr"):
                    rows.append({"stage": stage, "task": task, "metric": metric, "value": value})
    if not rows:
        raise SystemExit(f"No evaluation JSON found under {root / 'eval'}")
    frame = pd.DataFrame(rows)
    base = frame[frame.stage == "stage_00_base"][["task", "metric", "value"]].rename(columns={"value": "base"})
    frame = frame.merge(base, on=["task", "metric"], how="left")
    frame["delta_from_base"] = frame.value - frame.base
    frame.to_csv(root / "analysis.csv", index=False)
    summary = frame.pivot_table(index=["task", "metric"], columns="stage", values="value")
    (root / "analysis.md").write_text("# Capability and drift report\n\n" + summary.to_markdown() + "\n")
    print(f"Wrote {root / 'analysis.csv'} and {root / 'analysis.md'}")


if __name__ == "__main__":
    main()
