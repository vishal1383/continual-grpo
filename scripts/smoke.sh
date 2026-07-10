#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q src
python -m continual_grpo.train --help >/dev/null
python -m continual_grpo.evaluate --help >/dev/null
python -m continual_grpo.report --help >/dev/null
echo "Static smoke checks passed."
