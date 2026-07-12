# Running the complete experiment

All training, evaluation, and analysis run inside Docker. The container is persistent: the commands below do not use `--rm`, and the full runner does not stop or delete it afterward.

Training uses one handwritten loop in `src/continual_grpo/custom_grpo.py`, not TRL:

```text
grpo        = clipped vanilla GRPO + reference KL
skill_ortho = the same loss, then protected top-k spectral gradient transform
copsd       = the same GRPO + KL plus additive contrastive OPSD
combined    = GRPO + KL + OPSD, then the same spectral transform
```

Training first collects a fixed on-policy rollout buffer and its sampler
token log-probabilities. The buffer is shuffled and reused for the configured
epochs, following the original Self-Distillation GRPO mechanics. `old_logps`
remain fixed, so policy ratios and clipping become active after the first
optimizer update.

Every cell logs separate policy, KL, OPSD, mixed-group, reward, and gradient metrics in `train_metrics.jsonl` (one row per optimizer step).

All four arms use identical SGD+momentum with zero weight decay. Spectral arms additionally log `spectral_retained_energy`, `protected_overlap_before`, and `protected_overlap_after`. Signal diagnostics include `mean_abs_advantage`, `mixed_group_fraction`, and the correctness-only `paired_group_fraction` used by C-OPSD.

Training tasks are GSM8K and MATH (MATH-lighteval); each is trained as its own independent cell from the same base model. Here continual learning means retaining existing capabilities and behavior during reasoning post-training, not sequential GSM8K→MATH adaptation. HumanEval, full MMLU, full BBQ, and the ten-bias suite are held-out retention axes. Batching config keys are all honored: `prompt_batch_size` prompts roll out together with `num_generations` completions each, `per_device_batch_size` bounds scoring/backpropagation chunks, and `gradient_accumulation_steps` controls optimizer cadence.

## Prerequisites

1. Install Docker Engine with the Compose plugin.
2. Install the NVIDIA driver and NVIDIA Container Toolkit.
3. Confirm GPU container access:

   ```bash
   docker run --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
   ```

4. From this repository, confirm the selected Hugging Face models are accessible. Set `HF_TOKEN` on the host if your model requires authentication.

## Run everything

```bash
./run_all.sh smoke
```

Then run the complete experiment:

```bash
./run_all.sh full
```

The first run builds `continual-grpo:v2`. Every later run reuses that image and the same persistent Compose container without rebuilding or using `--rm`, then runs these four phases inside it:

```bash
python3 -m continual_grpo.train --config configs/default.yaml --resume
python3 -m continual_grpo.evaluate --config configs/default.yaml --allow-code-execution
python3 -m continual_grpo.bias_eval --config configs/default.yaml
python3 -m continual_grpo.report --config configs/default.yaml
```

## Utility evaluation and the ten-benchmark bias suite

`continual_grpo.evaluate` runs the `lm-eval` utility tasks from `eval_tasks`: GSM8K, HumanEval, BBQ, and full MMLU (all 57 subtasks; the smoke config caps every subtask at one example via `eval_limits`).

`continual_grpo.bias_eval` runs the exact original ten-benchmark bias suite over the same checkpoint series (base model plus every saved stage adapter of every method):

```text
crows_pairs               pairwise stereotype-vs-antistereotype likelihood
stereoset_intersentence   official StereoSet split
stereoset_intrasentence   official StereoSet split
toxigen                   toxic/non-toxic forced-choice classification
winobias                  pro/anti-stereotype likelihood gap
winogender_gap            forced-choice pronoun coreference
pat                       full Prompt Association Test
unstereo_use10            full UnStereoEval USE-10 config
discrim_eval              full Anthropic decision-discrimination eval
uccb                      full cultural QA generation probe
```

The scoring logic is ported verbatim from the original `run_bias_evals.py`; only model loading, chat templating, and checkpoint discovery were adapted to this repository. Behavior details:

- Checkpoints load frozen in FP16 with the stage LoRA adapter applied through PEFT.
- Per-item scores go to `output_dir/bias_eval/<method>/<model>/<stage>/<benchmark>_scores.jsonl`; metrics accumulate in that cell's `summary.json` after every benchmark, so an interrupted run resumes without repeating finished benchmarks.
- Final reporting lands in `output_dir/bias_eval/`: `bias_summary.csv` (one wide metric row per checkpoint), `bias_details.csv` (per-group values with sample counts), and `bias_analysis.csv` (per metric: base score, trained score, absolute delta, relative delta, sample count).
- The config's `bias_eval:` section controls the run: `n_eval` (0 = full benchmark; smoke uses 8 shuffled items per benchmark), `eval_batch`, `score_batch`, and the prompt-ensemble counts, mirroring the original script's defaults.
- Quick manual smoke of just this phase: `python3 -m continual_grpo.bias_eval --config configs/smoke.yaml --n-eval 4 --benchmarks crows_pairs,uccb`.

Outputs remain in the host's `outputs/` directory and the container remains available after completion.

Rebuild explicitly only after changing `Dockerfile`, `pyproject.toml`, or another installed dependency:

```bash
docker compose build experiment
```

## Enter the existing container

```bash
docker compose exec experiment bash
```

Inside it, the repository is `/workspace`. You can run or debug any individual Python phase there.

## Container lifecycle

Stop without deleting the container:

```bash
docker compose stop experiment
```

Start it again:

```bash
docker compose start experiment
```

Check status and logs:

```bash
docker compose ps
docker compose logs experiment
```

Only `docker compose down` removes the Compose container. It is intentionally not used by `run_all.sh`.

## Fine-tuning mode

The supplied smoke and full configurations currently run only Qwen2.5-7B-Instruct. They use LoRA over attention Q/K/V/O and FFN gate/up/down projections. The full run uses rank 32; smoke uses rank 16.

Full-parameter fine-tuning is available for larger systems by setting:

```yaml
finetune_mode: full
```

Do not select full-parameter mode for the supplied single-GB10/GB20 workflow; it is not enabled by any default runner configuration.
