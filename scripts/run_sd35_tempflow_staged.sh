#!/usr/bin/env bash
set -euo pipefail

ROOT="${SD35_TEMPFLOW_ROOT:-/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke}"
RUN_ROOT="${SD35_TEMPFLOW_RUN_ROOT:-/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded}"
MODEL_PATH="${FLOW_GRPO_SD3_MODEL:-/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium}"
ENV_NAME="${SD35_TEMPFLOW_ENV:-visual-rl-sd35}"
STAGES="${SD35_TEMPFLOW_STAGES:-5 20 200}"
RESUME_LORA="${SD35_TEMPFLOW_RESUME_LORA:-0}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/stage_markers"

source /home/v-qiaoqifan/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

latest_lora() {
  find "${RUN_ROOT}/checkpoints" -type d -path "*/lora" 2>/dev/null \
    | sort -V \
    | tail -n 1
}

for epochs in ${STAGES}; do
  marker="${RUN_ROOT}/stage_markers/stage_${epochs}.done"
  if [[ -f "${marker}" ]]; then
    echo "[stage ${epochs}] already complete; skipping"
    continue
  fi

  lora_path=""
  if [[ "${RESUME_LORA}" == "1" ]]; then
    lora_path="$(latest_lora || true)"
  fi
  save_freq=20
  if [[ "${epochs}" -le 5 ]]; then
    save_freq=2
  elif [[ "${epochs}" -le 20 ]]; then
    save_freq=5
  fi

  args=(
    scripts/train_sd3_tempflow.py
    --config=config/grpo.py:tempflow_sd3_server_smoke
    --config.num_epochs="${epochs}"
    --config.run_name="sd35_tempflow_guarded_${epochs}epoch"
    --config.logdir="${RUN_ROOT}/accelerate_logs"
    --config.save_dir="${RUN_ROOT}"
    --config.save_freq="${save_freq}"
    --config.eval_freq=0
    --config.num_checkpoint_limit=20
  )
  if [[ -n "${lora_path}" ]]; then
    args+=(--config.train.lora_path="${lora_path}")
    echo "[stage ${epochs}] resuming LoRA from ${lora_path}"
  else
    echo "[stage ${epochs}] starting from base model"
  fi

  export FLOW_GRPO_SD3_MODEL="${MODEL_PATH}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export TOKENIZERS_PARALLELISM=false

  echo "[stage ${epochs}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[stage ${epochs}] command: python ${args[*]}"
  python "${args[@]}"
  date -Is > "${marker}"
  echo "[stage ${epochs}] complete"
done
