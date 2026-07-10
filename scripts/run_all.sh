#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="configs/default.yaml"
python3 -m continual_grpo.train --config "$CONFIG" --resume
python3 -m continual_grpo.evaluate --config "$CONFIG" --allow-code-execution
python3 -m continual_grpo.report --config "$CONFIG"
