#!/usr/bin/env bash
# Build once, start/reuse Docker, then train, evaluate, and report.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-smoke}"
IMAGE="continual-grpo:v2"
case "$MODE" in
  smoke)
    CONFIG="configs/smoke.yaml"
    EVAL_LIMIT=(--limit 1)
    ;;
  full)
    CONFIG="configs/full.yaml"
    EVAL_LIMIT=()
    ;;
  *)
    echo "Usage: $0 smoke|full" >&2
    exit 2
    ;;
esac

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker compose build experiment
fi
docker compose up -d --no-build experiment
docker compose exec experiment python3 -m continual_grpo.train --config "$CONFIG" --resume
docker compose exec experiment python3 -m continual_grpo.evaluate --config "$CONFIG" --allow-code-execution "${EVAL_LIMIT[@]}"
docker compose exec experiment python3 -m continual_grpo.report --config "$CONFIG"

echo "Run complete. Container is still running."
echo "Enter it with: docker compose exec experiment bash"
echo "Stop it with:  docker compose stop experiment"
