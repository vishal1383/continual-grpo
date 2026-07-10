#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="configs/default.yaml"
python -m continual_grpo.train --config "$CONFIG" --resume
python -m continual_grpo.evaluate --config "$CONFIG" --allow-code-execution
python -m continual_grpo.report --config "$CONFIG"
