#!/usr/bin/env bash
# Build once, start/reuse Docker, then train, evaluate, and report.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="configs/default.yaml"

docker compose build experiment
docker compose up -d --no-build experiment
docker compose exec experiment python3 -m continual_grpo.train --config "$CONFIG" --resume
docker compose exec experiment python3 -m continual_grpo.evaluate --config "$CONFIG" --allow-code-execution
docker compose exec experiment python3 -m continual_grpo.report --config "$CONFIG"

echo "Run complete. Container is still running."
echo "Enter it with: docker compose exec experiment bash"
echo "Stop it with:  docker compose stop experiment"
