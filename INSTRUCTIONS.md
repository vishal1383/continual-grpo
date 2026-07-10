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

General/smaller GPU profile:

```bash
./run_all.sh configs/general.yaml
```

GB10/DGX Spark profile:

```bash
./run_all.sh configs/gb10.yaml
```

The script checks Docker, Compose, NVIDIA Container Toolkit, the selected config, and GPU visibility. It builds the image with all Python prerequisites only when `continual-grpo:latest` does not exist, starts or reuses the persistent Compose service, and runs these three phases inside it:

```bash
python -m continual_grpo.train --config CONFIG --resume
python -m continual_grpo.evaluate --config CONFIG --allow-code-execution
python -m continual_grpo.report --config CONFIG
```

Outputs remain in the host's `outputs/` directory and the container remains available after completion.

Normal repeated runs reuse the existing image and container. Rebuild explicitly only after changing `Dockerfile`, `pyproject.toml`, or another installed dependency:

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
