#!/usr/bin/env bash
set -euo pipefail

# Conservative wrapper around the legacy World-R1 launcher.
# Pick idle GPUs explicitly, for example:
#   CUDA_VISIBLE_DEVICES=6,7 MODEL_PATH=/models/Wan2.1-T2V-1.3B-Diffusers bash scripts/run_world_r1_reference_baseline.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_CODE_ROOT="${REFERENCE_CODE_ROOT:-${VISUAL_RL_REFERENCE_CODE_ROOT:-${REPO_DIR}/../code_base/reference_code}}"
WORLD_R1_DIR="${WORLD_R1_DIR:-${REFERENCE_CODE_ROOT}/World-R1-main}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Wan2.1 diffusers checkpoint path}"
TRAIN_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${TRAIN_VISIBLE_DEVICES:-6,7}}"
if [[ ! -d "${WORLD_R1_DIR}" ]]; then
  echo "World-R1 reference repo not found: ${WORLD_R1_DIR}" >&2
  echo "Set WORLD_R1_DIR or VISUAL_RL_REFERENCE_CODE_ROOT to the directory that contains World-R1-main." >&2
  exit 1
fi

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  NUM_PROCESSES=0
  IFS=',' read -ra DEVICE_ITEMS <<< "${TRAIN_VISIBLE_DEVICES}"
  for device in "${DEVICE_ITEMS[@]}"; do
    if [[ -n "${device//[[:space:]]/}" ]]; then
      NUM_PROCESSES=$((NUM_PROCESSES + 1))
    fi
  done
fi
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "TRAIN_VISIBLE_DEVICES must contain at least one GPU index." >&2
  exit 1
fi

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
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/runs/world_r1_reference}"

cd "${WORLD_R1_DIR}"
bash scripts/run_training.sh
