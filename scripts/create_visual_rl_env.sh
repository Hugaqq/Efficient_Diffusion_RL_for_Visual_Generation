#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-visual-rl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to create ${ENV_NAME}" >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e ".[dev]"

echo "Created ${ENV_NAME}. Activate with: conda activate ${ENV_NAME}"

