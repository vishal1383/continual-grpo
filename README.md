# Continual GRPO

A minimal, self-contained implementation for the NSF proposal's central measurement question. One handwritten LoRA GRPO loop runs every ablation with identical rollout, reward, reference-KL, batching, and optimizer code.

This repository deliberately excludes archived experiments, old distillation modes, generated outputs, and paper assets. It implements the proposal's method arms as configurable alternatives sharing one pipeline: standard GRPO (the comparison baseline), Skill-Orthogonal gradient projection (Aim 1), Contrastive On-Policy Self-Distillation (Aim 2), and their combination (Aim 3).

## What is measured

- Target reasoning: GSM8K and MATH (MATH-lighteval train split; `hendrycks_math` via `lm-eval`).
- Protected coding: HumanEval pass@1 (evaluation only; HumanEval is never trained on).
- Protected knowledge: full MMLU (all 57 subtasks) via `lm-eval`.
- Protected fairness: BBQ accuracy/bias metrics exposed by `lm-eval`, plus the exact original ten-benchmark bias suite (CrowS-Pairs, StereoSet inter/intrasentence, ToxiGen, WinoBias, WinoGender gap, P-AT, UnStereoEval USE-10, DiscrimEval, UCCB) via `continual_grpo.bias_eval`.
- Continual drift: each metric's delta from the untouched base model after every training stage — absolute and relative, with sample counts and per-group bias details.
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

After the smoke test succeeds, run the complete GSM8K/MATH training and retention-evaluation suite:

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
python3 -m continual_grpo.bias_eval --config configs/default.yaml
python3 -m continual_grpo.report --config configs/default.yaml
```

The default uses Qwen2.5-7B with bounded GSM8K/MATH subsets. The full config uses both complete training splits.

## Native installation and individual commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m continual_grpo.train --config configs/default.yaml --resume
python3 -m continual_grpo.evaluate --config configs/default.yaml --allow-code-execution
python3 -m continual_grpo.bias_eval --config configs/default.yaml
python3 -m continual_grpo.report --config configs/default.yaml
```

Results are written under `output_dir/<method>/<model_slug>/stage_NN_<task>/`, with raw utility evaluation under `eval/`, ten-bias results under `bias_eval/`, and aggregate CSV/Markdown reports. Runs are resumable; completed cells and evaluations are skipped.

## Method arms

The `methods` list selects which arms train. Each arm trains its own adapters under `output_dir/<method>/<model_slug>/stage_NN_<task>/` and is evaluated separately under `eval/<method>/...`; the untouched base model is evaluated once per model under `eval/base/`.

- `grpo` — handwritten clipped vanilla GRPO plus reference-policy KL.
- `skill_ortho` — the identical GRPO+KL loss, followed by protected-axis projection, top-k SVD truncation, and final projection before `optimizer.step()`.
- `copsd` — the identical GRPO+KL loss plus positive reference-distribution KL and the bounded correct-versus-wrong likelihood margin.
- `combined` — the additive GRPO+KL+C-OPSD loss followed by the identical spectral gradient transform.

All four paths are explicit in `custom_grpo.py`; training does not use TRL.

All arms use the same AdamW optimizer at 2e-5 with cosine warmup and zero weight decay, matching the original Self-Distillation GRPO runs (SGD+momentum remains selectable via `optimizer: sgd`, but at practical learning rates it moves LoRA weights orders of magnitude less per step). Weight decay stays zero because decoupled decay would move weights along protected directions regardless of the spectral gradient projection; Adam's elementwise preconditioning can still reintroduce some projected components, which is why spectral arms log retained gradient energy and protected overlap before/after projection. Full GSM8K uses top-p 1.0 so the scored policy matches the sampling distribution.

## Model sizes and smoke evaluation

The supplied configs currently use Qwen2.5-7B-Instruct only. Here continual learning means capability-preserving post-training: each GSM8K or MATH cell starts from the same base model, improves one reasoning task, and is checked for drift on HumanEval, full MMLU, full BBQ, and the exact ten-bias suite. It does not mean sequential GSM8K→MATH training. Smoke runs one item from each reasoning task through all four methods. Full uses the complete GSM8K and MATH-lighteval training splits.

The smoke config exercises full functionality on minimal data: it trains each model on one example, evaluates GSM8K and HumanEval with a per-task `eval_limits` of 1, and swaps BBQ for the local `bbq_smoke` task (defined in `lm_eval_tasks/` and loaded via `--include_path`). `bbq_smoke` keeps one document per BBQ (category, context) bucket — 22 documents — so every bias metric has data and none degenerates to NaN, which is what happens when a plain `--limit` takes rows from only the first category.

## Adding a continual task

Append a task to the YAML `tasks` list with its dataset and verifier fields. Each task is an independent post-training cell initialized from the same base model. Add the corresponding `lm-eval` task name to `eval_tasks` to measure retention.

For publishable experiments, pin the container digest and Python lockfile, record GPU/driver versions, use full datasets, run multiple seeds, and inspect `lm_eval --tasks list` because task aliases can change across harness releases.
