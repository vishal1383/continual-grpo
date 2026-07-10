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
- NVIDIA driver and NVIDIA Container Toolkit (`docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` must work).
- Hugging Face access/network connectivity for model and dataset downloads.
- GB10: use the vendor CUDA/PyTorch base if it is required by your DGX Spark image; pass it with `--build-arg BASE_IMAGE=...`.

## Docker environment and Python commands

```bash
docker compose build
docker compose run --rm experiment
```

You are now at `/workspace` inside the complete GPU environment. Run each phase explicitly:

```bash
python -m continual_grpo.train --config configs/general.yaml --resume
python -m continual_grpo.evaluate --config configs/general.yaml --allow-code-execution
python -m continual_grpo.report --config configs/general.yaml
```

Or run the same three Python commands through one wrapper:

```bash
scripts/run_all.sh configs/general.yaml
```

Type `exit` to leave the container. The general profile uses Qwen2.5-0.5B and 256 training examples so it can validate the pipeline on a conventional CUDA GPU. Remove `max_samples` limits for a research run.

## GB10 / DGX Spark example

The 7B profile is designed for a single 128 GB unified-memory GB10 system:

```bash
docker compose build --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
docker compose run --rm experiment
python -m continual_grpo.train --config configs/gb10.yaml --resume
python -m continual_grpo.evaluate --config configs/gb10.yaml --allow-code-execution
python -m continual_grpo.report --config configs/gb10.yaml
```

If the host's NVIDIA-provided PyTorch image is named `gb10-rl-saved:latest`, use `docker build --build-arg BASE_IMAGE=gb10-rl-saved:latest -t continual-grpo .` and then:

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=32g \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  continual-grpo
```

Then, inside that container:

```bash
python -m continual_grpo.train --config configs/gb10.yaml --resume
python -m continual_grpo.evaluate --config configs/gb10.yaml --allow-code-execution
python -m continual_grpo.report --config configs/gb10.yaml
```

## Native installation and individual commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m continual_grpo.train --config configs/general.yaml --resume
python -m continual_grpo.evaluate --config configs/general.yaml --allow-code-execution
python -m continual_grpo.report --config configs/general.yaml
```

Results are written under the configured `output_dir`: resolved config, stage adapters, raw evaluation JSON/sample logs, `analysis.csv`, and `analysis.md`. Runs are resumable; completed training stages and evaluations are skipped.

## Adding a continual task

Append a task to the YAML `tasks` list with its Hugging Face dataset, split, prompt field, and answer field. The adapter from stage N initializes stage N+1. Add the corresponding `lm-eval` task name to `eval_tasks` to measure retention at every available checkpoint.

For publishable experiments, pin the container digest and Python lockfile, record GPU/driver versions, use full datasets, run multiple seeds, and inspect `lm_eval --tasks list` because task aliases can change across harness releases.
