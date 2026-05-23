#!/usr/bin/env bash
set -euo pipefail

# Conservative wrapper around the legacy World-R1 launcher.
# Pick idle GPUs explicitly, for example:
#   CUDA_VISIBLE_DEVICES=6,7 MODEL_PATH=/models/Wan2.1-T2V-1.3B-Diffusers bash scripts/run_v01_world_r1_baseline.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_R1_DIR="${WORLD_R1_DIR:-${REPO_DIR}/reference_code/World-R1-main}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Wan2.1 diffusers checkpoint path}"
TRAIN_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${TRAIN_VISIBLE_DEVICES:-6,7}}"
NUM_PROCESSES="${NUM_PROCESSES:-$(python - <<PY
devices = "${TRAIN_VISIBLE_DEVICES}".split(",")
print(len([d for d in devices if d.strip()]))
PY
)}"

export MODEL_PATH
export TRAIN_VISIBLE_DEVICES
export NUM_PROCESSES
export TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-2}"
export TRAIN_EVAL_NUM_STEPS="${TRAIN_EVAL_NUM_STEPS:-2}"
export TRAIN_NUM_BATCHES_PER_EPOCH="${TRAIN_NUM_BATCHES_PER_EPOCH:-1}"
export TRAIN_TEST_BATCH_SIZE="${TRAIN_TEST_BATCH_SIZE:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export TRAIN_NUM_IMAGE_PER_PROMPT="${TRAIN_NUM_IMAGE_PER_PROMPT:-1}"
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-256}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-448}"
export TRAIN_FRAMES="${TRAIN_FRAMES:-17}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/runs/world_r1_v01}"

cd "${WORLD_R1_DIR}"
bash scripts/run_training.sh
