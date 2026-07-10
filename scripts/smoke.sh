#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q src
continual-grpo --help >/dev/null
continual-eval --help >/dev/null
continual-report --help >/dev/null
echo "Static smoke checks passed."
