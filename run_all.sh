#!/usr/bin/env bash
# One host-side command: check prerequisites, build once, start/reuse Docker,
# then train, evaluate, and report. The container is kept after the run.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-configs/general.yaml}"
IMAGE="continual-grpo:latest"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || die "Docker is not installed or is not on PATH."
docker info >/dev/null 2>&1 || die "Docker daemon is not available to this user."
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is not installed."
[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"

echo "Checking NVIDIA GPU access from Docker..."
docker info --format '{{json .Runtimes}}' | grep -q nvidia \
  || die "NVIDIA Container Toolkit is not configured (the nvidia runtime is missing)."

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found; building it with all Python prerequisites..."
  docker compose build experiment
else
  echo "Reusing existing image: $IMAGE"
  echo "To rebuild after dependency changes, run: docker compose build experiment"
fi

echo "Starting or reusing the persistent experiment container..."
docker compose up -d --no-build experiment

echo "Verifying the GPU inside the container..."
docker compose exec experiment nvidia-smi >/dev/null

echo "Training: $CONFIG"
docker compose exec experiment python -m continual_grpo.train --config "$CONFIG" --resume

echo "Evaluating GSM8K, HumanEval, and bias benchmarks..."
docker compose exec experiment python -m continual_grpo.evaluate --config "$CONFIG" --allow-code-execution

echo "Building the analysis report..."
docker compose exec experiment python -m continual_grpo.report --config "$CONFIG"

echo "Run complete. Container is still running."
echo "Enter it with: docker compose exec experiment bash"
echo "Stop it with:  docker compose stop experiment"
