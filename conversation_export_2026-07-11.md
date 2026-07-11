# Continual GRPO Project Conversation Export

Date: 2026-07-11  
Timezone: America/New_York

## Initial request

Create a minimal, self-contained repository derived from the existing Self-Distillation project and the NSF continual-learning proposal. Preserve the research philosophy while removing legacy outputs and scripts. The requested scope included GRPO, KL anchoring, reasoning benchmarks, HumanEval, GSM8K, held-out capability evaluation, bias metrics, Docker instructions, GB10 compatibility, a single run command, and a private GitHub repository.

## Repository and Docker decisions

- Created the nested repository `continual-grpo`.
- Added a Python package, Dockerfile, Docker Compose configuration, smoke/full configurations, evaluation, reporting, and instructions.
- Docker is persistent: no `--rm` is used by the runner.
- The image is built only when its expected tag is missing.
- The same Compose container is started or reused.
- Source is bind-mounted at `/workspace`, so source edits do not require rebuilding dependencies.
- Outputs are mounted to the host under `outputs/`.
- Container memory is capped at 70 GiB.
- Evaluation loads models in FP16 with a fixed bounded batch size.
- The configuration records `vllm_gpu_memory_utilization: 0.70`, though the current evaluation path uses Hugging Face rather than vLLM.

Primary commands:

```bash
./run_all.sh smoke
./run_all.sh full
```

## Docker issues and resolutions

1. A legacy check incorrectly required an `nvidia` runtime in `docker info`. GB10 can use CDI without listing that runtime. The check was removed.
2. The initial command used `python`, while Ubuntu exposed `python3`. All commands were changed to `python3`.
3. TRL imported optional vLLM modules even when vLLM was unused. Several version attempts exposed additional optional-dependency conflicts. A clean virtual environment was added to the image.
4. Python modules initially lacked `if __name__ == "__main__": main()` blocks and exited without doing work. Entry points were fixed.
5. The GB10 PyTorch native Triton path failed compiling a CUDA helper. `TORCH_DISABLE_NATIVE_JIT=1` was added to use ordinary CUDA operations.
6. HumanEval required both `--confirm_run_unsafe_code` and `HF_ALLOW_CODE_EVAL=1`. Both are now supplied inside Docker.
7. Automatic evaluation batching caused memory pressure. Evaluation now uses fixed batches, FP16, separate task processes, and bounded memory.
8. LoRA adapters were initially passed to `lm-eval` as standalone models. Evaluation now loads the original Qwen model with the adapter supplied through the `peft` argument.

## Agreed experiment scope

Models were initially planned as Qwen2.5 0.5B, 3B, and 7B. Later, because smaller-model binary-reward behavior was poor, the supplied configurations were restricted to:

```text
Qwen/Qwen2.5-7B-Instruct
```

Training tasks:

- GSM8K
- HumanEval, using 64 held-out problems and the remaining 100 as the training partition

Methods:

1. Vanilla GRPO plus reference KL
2. Skill-isolated spectral GRPO
3. GRPO plus contrastive on-policy self-distillation (C-OPSD)
4. Combined GRPO plus C-OPSD followed by spectral gradient isolation

Held-out evaluation requirements:

- Full MMLU, not MMLU-STEM
- Full BBQ
- GSM8K and HumanEval cross-task retention

Exact original ten-benchmark bias suite:

1. CrowS-Pairs
2. StereoSet intersentence
3. StereoSet intrasentence
4. ToxiGen
5. WinoBias
6. WinoGender gap
7. PAT
8. UnStereoEval USE-10
9. DiscrimEval
10. UCCB

Required reporting includes base score, trained score, absolute delta, relative delta, sample count, and per-group bias details.

## Vanilla GRPO discussion

TRL was initially used for vanilla GRPO. Logs often showed a scalar loss of zero. The correct explanation established in the conversation was:

For normalized group advantages,

```text
sum_i A_i = 0
```

At the on-policy point the importance ratio equals one, so the scalar surrogate value can be exactly zero:

```text
J(theta) = mean_i A_i = 0
```

while its gradient can remain nonzero:

```text
grad J(theta) = mean_i A_i grad log pi_theta(y_i)
```

Therefore `loss: 0.0` with nonzero reward variation and a finite nonzero gradient can be valid. When every rollout in a prompt group receives the same reward, the normalized advantages and policy gradient are genuinely zero.

The 7B logs showed many all-correct groups. That means vanilla binary-reward GRPO receives little signal on easy GSM8K prompts. Possible future remedies discussed were harder prompts, more generations, increased sampling diversity, curriculum filtering for mixed-outcome prompts, or denser rewards.

## Decision to remove TRL from training

The user correctly noted that using TRL only for vanilla GRPO and separate implementations for spectral/C-OPSD would not be an apples-to-apples ablation.

Training was therefore replaced with one handwritten LoRA GRPO loop in:

```text
src/continual_grpo/custom_grpo.py
```

Every method now shares:

- Model and LoRA initialization
- Rollout generation
- Old-policy sampled-token log probabilities
- Binary verification
- Group-relative advantages
- Clipped GRPO ratio objective
- Reference-policy KL
- Optimizer
- Gradient clipping
- Seeds and task splits

Only two switches change:

- Add C-OPSD loss or not
- Apply spectral gradient isolation or not

## Handwritten loss

The shared objective is:

```text
total_loss = grpo_policy_loss
           + kl_beta * reference_kl_loss
           + opsd_weight * opsd_loss
```

The C-OPSD term is zero for methods that do not use it.

Each step records:

```text
total_loss
grpo_policy_loss
reference_kl_loss
opsd_loss
opsd_pairs
correct_fraction
mixed_group_fraction
grad_norm
completion_length
```

Metrics are written to each cell's `train_metrics.jsonl`.

## Spectral skill-isolating update

This is not standard Muon. Protected row and column axes are estimated from anchor-batch gradients. For a matrix gradient `G`:

```text
G_safe = Q_L G Q_R
G_safe = U Sigma V^T
G_topk = U[:, :k] Sigma[:k] V[:k, :]
G_final = Q_L G_topk Q_R
```

Only the top-k singular target directions are retained. All remaining singular components are discarded. The final projection prevents the spectral reconstruction from reintroducing protected components.

The gradient transformation occurs explicitly after `backward()` and before clipping and `optimizer.step()`.

Relevant files:

```text
src/continual_grpo/spectral_update.py
src/continual_grpo/skill_axes.py
src/continual_grpo/anchors.py
```

## Contrastive OPSD

Correct and incorrect completions are paired only when they come from the same prompt rollout group.

The positive term distills the reference token distribution on correct trajectories after the correct and incorrect traces diverge.

The negative term uses length-normalized sequence log-likelihood:

```text
l_theta(y) = sum_t log pi_theta(y_t | x, y_<t) / |y|
```

and a bounded margin:

```text
L_negative = max(0, margin - (l_theta(y_positive) - l_theta(y_negative)))
```

If no correct/incorrect pair exists in a group, that group contributes no C-OPSD pair. Vanilla GRPO and reference KL remain part of the objective.

## LoRA and full fine-tuning

The full research configuration uses rank-32 LoRA. Smoke uses rank 16.

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Thus both attention and FFN projections are adapted.

Full-parameter fine-tuning is available through:

```yaml
finetune_mode: full
```

but no supplied GB10/GB20-oriented configuration selects it.

## HumanEval training

The official 164 HumanEval problems are shuffled deterministically. Sixty-four are held out for evaluation and the remaining 100 form the training pool. Generated code is verified in a temporary directory with a timeout.

## Validation status

- GRPO advantage, C-OPSD pairing, spectral rank, and protected-orthogonality invariants passed inside Docker.
- A real handwritten 7B vanilla-GRPO GSM8K smoke cell completed and saved an adapter.
- Its single smoke group was all incorrect, so it correctly logged zero mixed-group fraction and zero gradient.
- The full four-method smoke command is `./run_all.sh smoke`.
- The full research command is `./run_all.sh full`.

## Current limitations at export time

- Utility evaluation currently runs GSM8K, HumanEval, and BBQ through `lm-eval`.
- Full MMLU and the exact original ten-bias evaluator remain the major unfinished integration item.
- Results must not be presented as the final requested study until that evaluation integration is complete.

## Post-export implementation addendum

Later on 2026-07-11, the specification was finalized as independent GSM8K and MATH post-training cells. “Continual learning” means preserving held-out capabilities and bias behavior during each reasoning update, not sequential GSM8K→MATH training. Full MMLU, full BBQ, HumanEval retention, and the exact original ten-benchmark bias suite were integrated. Protected spectral axes were subsequently corrected to use answer-conditioned MMLU and BBQ gradients rather than unsupervised BBQ prompt text.

## GitHub commands

```bash
cd ~/Desktop/Self-Distillation/continual-grpo
gh auth login -h github.com
gh auth status
gh repo create continual-grpo --private --source=. --remote=origin --push
```

Later updates:

```bash
git add .
git commit -m "Describe changes"
git push origin main
```

The untracked `NSF_proposal_continual_learning.pdf` should be intentionally included or added to `.gitignore` before using `git add .`.
