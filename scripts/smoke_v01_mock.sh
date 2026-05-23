#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-runs/smoke_v01_mock}"
STEPS="${STEPS:-2}"

python -m visual_rl.cli smoke-imports
python -m visual_rl.cli smoke-mock --output-dir "${OUTPUT_DIR}" --steps "${STEPS}"

