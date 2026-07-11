# Running the complete experiment

All training, evaluation, and analysis run inside Docker. The container is persistent: the commands below do not use `--rm`, and the full runner does not stop or delete it afterward.

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

The first run builds `continual-grpo:v2`. Every later run reuses that image and the same persistent Compose container without rebuilding or using `--rm`, then runs these three phases inside it:

```bash
python3 -m continual_grpo.train --config configs/default.yaml --resume
python3 -m continual_grpo.evaluate --config configs/default.yaml --allow-code-execution
python3 -m continual_grpo.report --config configs/default.yaml
```

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
