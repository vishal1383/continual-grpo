# Continual GRPO

A minimal, self-contained baseline for the NSF proposal's central measurement question: when GRPO improves a target capability, how much do protected capabilities and fairness behavior move? It trains a configurable sequence of tasks with standard GRPO and the ordinary frozen-reference KL penalty, then evaluates every stage on GSM8K, HumanEval, and BBQ.

This repository deliberately excludes archived experiments, old distillation modes, generated outputs, and paper assets. It does **not** claim to implement the proposal's future Skill-Orthogonal/Muon or C-OPSD algorithms. Its task stream and stage-wise evaluation are the clean baseline framework those methods can extend.

## What is measured

- Target reasoning: GSM8K exact-match via `lm-eval`.
- Protected coding: HumanEval pass@1.
- Protected fairness: BBQ accuracy/bias metrics exposed by `lm-eval`.
- Continual drift: each metric's delta from the untouched base model after every training stage.
- Training intervention: GRPO reward optimization with standard reference-policy KL (`kl_beta`).

HumanEval is one protected-capability benchmark in the same evaluation suite. Because its metric executes generated Python, the evaluator requires the explicit `--allow-code-execution` acknowledgement.

## Prerequisites

- Linux, Git, and Docker 27+ with Compose.
- NVIDIA driver and NVIDIA Container Toolkit.
- Hugging Face access/network connectivity for model and dataset downloads.
- GB10: use the vendor CUDA/PyTorch base if it is required by your DGX Spark image; pass it with `--build-arg BASE_IMAGE=...`.

## Docker environment and Python commands

```bash
./run_all.sh smoke
```

After the smoke test succeeds, run the complete GSM8K training and evaluation suite:

```bash
./run_all.sh full
```

The first run builds `continual-grpo:latest`. Later runs reuse that image and the same persistent Compose container; they do not rebuild or use `--rm`. If dependencies change, rebuild manually with `docker compose build experiment`.

To enter the persistent environment after or during a run:

```bash
docker compose exec experiment bash
```

Inside `/workspace`, the phases can also be run separately:

```bash
python3 -m continual_grpo.train --config configs/default.yaml --resume
python3 -m continual_grpo.evaluate --config configs/default.yaml --allow-code-execution
python3 -m continual_grpo.report --config configs/default.yaml
```

The default uses Qwen2.5-0.5B and 256 training examples so it fits broadly. Edit `configs/default.yaml` to select a larger model or full dataset when more compute is available.

## Native installation and individual commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m continual_grpo.train --config configs/default.yaml --resume
python3 -m continual_grpo.evaluate --config configs/default.yaml --allow-code-execution
python3 -m continual_grpo.report --config configs/default.yaml
```

Results are written under the configured `output_dir`: resolved config, stage adapters, raw evaluation JSON/sample logs, `analysis.csv`, and `analysis.md`. Runs are resumable; completed training stages and evaluations are skipped.

## Adding a continual task

Append a task to the YAML `tasks` list with its Hugging Face dataset, split, prompt field, and answer field. The adapter from stage N initializes stage N+1. Add the corresponding `lm-eval` task name to `eval_tasks` to measure retention at every available checkpoint.

For publishable experiments, pin the container digest and Python lockfile, record GPU/driver versions, use full datasets, run multiple seeds, and inspect `lm_eval --tasks list` because task aliases can change across harness releases.
