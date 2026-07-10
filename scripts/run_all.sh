#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/general.yaml}"
continual-grpo --config "$CONFIG" --resume
continual-eval --config "$CONFIG" --allow-code-execution
continual-report --config "$CONFIG"
