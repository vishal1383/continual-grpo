"""Run the exact original ten-benchmark bias suite over saved checkpoints.

What this does, step by step:

1. Discovers the same checkpoint series as ``continual_grpo.evaluate``: the
   base model plus every ``{method}/{model}/stage_XX_{task}/final_adapter``
   that training saved under ``output_dir``.
2. For each checkpoint, loads it frozen in FP16 (base weights + LoRA adapter)
   and runs each configured benchmark. The default list is the original ten:
   CrowS-Pairs, StereoSet inter/intrasentence, ToxiGen, WinoBias, WinoGender
   gap, P-AT, UnStereoEval USE-10, DiscrimEval, and UCCB.
3. Writes per-item scores to ``output_dir/bias_eval/<cell>/<benchmark>_scores.jsonl``
   and accumulates metrics in ``<cell>/summary.json`` after every benchmark,
   so re-running resumes exactly where it stopped (a failed benchmark is
   retried; a finished one is skipped).
4. After all cells, writes three CSVs under ``output_dir/bias_eval/``:
   ``bias_summary.csv`` (one wide row of metrics per checkpoint),
   ``bias_details.csv`` (per-group values with sample counts), and
   ``bias_analysis.csv`` (per metric: base score, trained score, absolute
   delta, relative delta, and sample count).

Smoke first: ``python3 -m continual_grpo.bias_eval --config configs/smoke.yaml``
caps every benchmark at the config's small ``bias_eval.n_eval`` before any
full run. Options in the config's ``bias_eval`` section mirror the original
run_bias_evals.py argparse defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import traceback
from pathlib import Path
from types import SimpleNamespace

# Must be set before `datasets` is imported (it reads the env at import time):
# script-based suite datasets (CrowS-Pairs mirrors, WinoBias) otherwise trigger
# an interactive trust prompt whose un-cancelled signal.alarm can kill the
# process minutes later, mid model load.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

from .bias.registry import TEN_BENCHMARKS, run_benchmark
from .bias.runtime import load_eval_model, unload, write_csv
from .common import load_config
from .evaluate import checkpoint_series

ARG_DEFAULTS = dict(
    n_eval=0,
    eval_batch=8,
    score_batch=2,
    max_prompt=1024,
    score_max_length=1024,
    mcq_prompt_count=1,
    toxicity_prompt_count=1,
    choice_max_new=4,
    qa_max_new=32,
    toxigen_toxic_threshold=3.0,
)


def bias_args(cfg: dict, n_eval_override: int | None):
    section = {**ARG_DEFAULTS, **(cfg.get("bias_eval") or {})}
    benchmarks = list(section.pop("benchmarks", TEN_BENCHMARKS))
    if n_eval_override is not None:
        section["n_eval"] = n_eval_override
    section.setdefault("seed", cfg.get("seed", 42))
    return SimpleNamespace(**section), benchmarks


def run_cell(label: str, model_name: str, adapter: str | None, benchmarks: list[str], args, out_root: Path) -> dict:
    out_dir = out_root / label
    summary_path = out_dir / "summary.json"
    payload = json.loads(summary_path.read_text()) if summary_path.exists() else {"metrics": {}, "details": {}}
    missing = [b for b in benchmarks if b not in payload["metrics"]]
    if not missing:
        print(f"SKIP complete cell: {label}", flush=True)
        return payload
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n================ {label} ================", flush=True)
    model, tok = load_eval_model(model_name, adapter)
    try:
        for benchmark in missing:
            print(f"\n[{label}] {benchmark}", flush=True)
            try:
                metrics, details = run_benchmark(benchmark, model, tok, args, str(out_dir))
            except Exception as e:
                print(f"    ERROR {benchmark}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue
            finally:
                # A failed dataset-load fallback can leave the trust-prompt
                # alarm armed; a stray SIGALRM would abort a later cell.
                signal.alarm(0)
            payload["metrics"][benchmark] = metrics
            payload["details"][benchmark] = details
            summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        unload(model, tok)
    return payload


def analysis_rows(cells: list[dict]) -> list[dict]:
    base = {}
    for cell in cells:
        if cell["method"] == "base":
            for bench, metrics in cell["metrics"].items():
                base[(cell["model"], bench)] = metrics
    rows = []
    for cell in cells:
        if cell["method"] == "base":
            continue
        for bench, metrics in cell["metrics"].items():
            base_metrics = base.get((cell["model"], bench), {})
            n = metrics.get(f"{bench}_n", "")
            for key, trained in metrics.items():
                if key.endswith("_n") or key.endswith("_skipped"):
                    continue
                base_value = base_metrics.get(key)
                has_base = isinstance(base_value, (int, float))
                delta = trained - base_value if has_base else ""
                rel = delta / abs(base_value) if has_base and base_value else ""
                rows.append(dict(
                    model=cell["model"], method=cell["method"], stage=cell["stage"],
                    benchmark=bench, metric=key, base=base_value if has_base else "",
                    trained=trained, delta_abs=delta, delta_rel=rel, n=n,
                ))
    return rows


def write_reports(cells: list[dict], out_root: Path) -> None:
    summary_rows, detail_rows = [], []
    for cell in cells:
        row = dict(model=cell["model"], method=cell["method"], stage=cell["stage"], adapter=cell["adapter"])
        for metrics in cell["metrics"].values():
            row.update(metrics)
        summary_rows.append(row)
        for details in cell["details"].values():
            for d in details:
                detail_rows.append(dict(model=cell["model"], method=cell["method"], stage=cell["stage"], **d))
    write_csv(str(out_root / "bias_summary.csv"), summary_rows)
    write_csv(str(out_root / "bias_details.csv"), detail_rows)
    write_csv(str(out_root / "bias_analysis.csv"), analysis_rows(cells))
    print(f"\nSaved {out_root / 'bias_summary.csv'}")
    print(f"Saved {out_root / 'bias_details.csv'}")
    print(f"Saved {out_root / 'bias_analysis.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ten-benchmark bias suite over saved checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmarks", default=None, help="comma list overriding the config benchmark list")
    parser.add_argument("--n-eval", type=int, default=None, help="override per-benchmark subset size; 0 = full")
    cli = parser.parse_args()
    cfg = load_config(cli.config)
    args, benchmarks = bias_args(cfg, cli.n_eval)
    if cli.benchmarks:
        benchmarks = [x for x in cli.benchmarks.split(",") if x]
    out_root = Path(cfg["output_dir"]) / "bias_eval"
    cells = []
    for label, model_name, adapter in checkpoint_series(cfg):
        payload = run_cell(label, model_name, adapter, benchmarks, args, out_root)
        method, slug, stage = label.split("/")
        cells.append(dict(model=slug, method=method, stage=stage, adapter=adapter or "",
                          metrics=payload["metrics"], details=payload["details"]))
    write_reports(cells, out_root)


if __name__ == "__main__":
    main()
