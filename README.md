# Continual GRPO

A minimal, self-contained baseline for the NSF proposal's central measurement question: when GRPO improves a target capability, how much do protected capabilities and fairness behavior move? It trains a configurable sequence of tasks with standard GRPO and the ordinary frozen-reference KL penalty, then evaluates every stage on GSM8K, HumanEval, and BBQ.

This repository deliberately excludes archived experiments, old distillation modes, generated outputs, and paper assets. It implements the proposal's method arms as configurable alternatives sharing one pipeline: standard GRPO (the comparison baseline), Skill-Orthogonal gradient projection (Aim 1), Contrastive On-Policy Self-Distillation (Aim 2), and their combination (Aim 3).

## What is measured

- Target reasoning: GSM8K exact-match via `lm-eval`.
- Protected coding: HumanEval pass@1.
- Protected fairness: BBQ accuracy/bias metrics exposed by `lm-eval`.
- Continual drift: each metric's delta from the untouched base model after every training stage.
- Training intervention: one of four method arms per run (see "Method arms" below).

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

The first run builds `continual-grpo:v2`. Later runs reuse that image and the same persistent Compose container; they do not rebuild or use `--rm`. If dependencies change, rebuild manually with `docker compose build experiment`.

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

Results are written under the configured `output_dir` with one subdirectory per base model: stage adapters under `<model_slug>/stage_NN_<task>/`, raw evaluation JSON under `eval/<model_slug>/<stage>/<task>/`, plus the resolved config, `train_history.json`, `analysis.csv`, and `analysis.md`. Runs are resumable; completed training stages and evaluations are skipped.

## Method arms

The `methods` list selects which arms train. Each arm trains its own adapters under `output_dir/<method>/<model_slug>/stage_NN_<task>/` and is evaluated separately under `eval/<method>/...`; the untouched base model is evaluated once per model under `eval/base/`.

- `grpo` — standard GRPO with reference-policy KL (`kl_beta`); the comparison baseline.
- `skill_ortho` — Aim 1. Before training, anchor batches from `protected_anchors` run through the model; the SVD of their mean gradient per parameter defines protected row/column axes (`protected_rank`, built in `anchors.py`/`skill_axes.py`). A callback (`ortho.py`) then transforms every matrix gradient immediately before each optimizer step: project off the protected axes, keep the top `spectral_target_rank` singular directions, and re-project (`spectral_update.py`). With a fresh LoRA adapter the protection acts through the `lora_B` side, since `lora_A` anchor gradients are zero at initialization.
- `copsd` — Aim 2 (`opsd_trainer.py`). Replaces the scalar-reward objective: within each generation group, verified-correct completions are distilled toward the reference distribution on the tokens after the point where they diverge from a verified-wrong completion of the same prompt, plus a bounded margin (`opsd_margin`, weighted by `opsd_negative_weight`) that pushes correct sequence log-likelihood above wrong. Batches whose groups have no correct/wrong pair fall back to the GRPO loss so training never stalls; the `opsd_pairs` metric logs how often the contrastive signal was available.
- `combined` — Aim 3: the C-OPSD objective with the skill-orthogonal projection applied to its updates.

## Model sizes and smoke evaluation

The `models` list in each config runs the entire task stream and evaluation once per base model. Smoke and full cover Qwen2.5-Instruct 0.5B, 1.5B, 3B, and 7B — the sizes whose GRPO LoRA training fits the container's 70 GiB memory limit. A single `model:` key is still accepted (see `configs/default.yaml`).

The smoke config exercises full functionality on minimal data: it trains each model on one example, evaluates GSM8K and HumanEval with a per-task `eval_limits` of 1, and swaps BBQ for the local `bbq_smoke` task (defined in `lm_eval_tasks/` and loaded via `--include_path`). `bbq_smoke` keeps one document per BBQ (category, context) bucket — 22 documents — so every bias metric has data and none degenerates to NaN, which is what happens when a plain `--limit` takes rows from only the first category.

## Adding a continual task

Append a task to the YAML `tasks` list with its Hugging Face dataset, split, prompt field, and answer field. The adapter from stage N initializes stage N+1. Add the corresponding `lm-eval` task name to `eval_tasks` to measure retention at every available checkpoint.

For publishable experiments, pin the container digest and Python lockfile, record GPU/driver versions, use full datasets, run multiple seeds, and inspect `lm_eval --tasks list` because task aliases can change across harness releases.
