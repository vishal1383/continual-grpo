#!/usr/bin/env bash
# Host-side complete runner. Keeps the Docker container after the run.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-configs/general.yaml}"

docker compose up -d --build experiment
docker compose exec experiment python -m continual_grpo.train --config "$CONFIG" --resume
docker compose exec experiment python -m continual_grpo.evaluate --config "$CONFIG" --allow-code-execution
docker compose exec experiment python -m continual_grpo.report --config "$CONFIG"

echo "Run complete. Container is still running."
echo "Enter it with: docker compose exec experiment bash"
echo "Stop it with:  docker compose stop experiment"
