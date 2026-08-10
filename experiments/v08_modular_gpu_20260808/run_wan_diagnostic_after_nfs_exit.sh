#!/usr/bin/env bash
set -u

readonly STALLED_PID=1645127
readonly BASE=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808
readonly SOURCE=/dev/shm/v-qiaoqifan/visualrl-v08-training-source-20260808-1730
readonly LOG="$BASE/logs/flow-wan-gpu3-attempt3-tmpfs-diagnostics.log"
readonly EXITCODE="$BASE/logs/flow-wan-gpu3-attempt3-tmpfs-diagnostics.exitcode"

while kill -0 "$STALLED_PID" 2>/dev/null; do
  sleep 60
done

gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
  --id=3 | tr -d ' ')
if [[ ! "$gpu_used_mib" =~ ^[0-9]+$ ]] || (( gpu_used_mib > 4096 )); then
  printf '%s\n' "SKIPPED: GPU 3 used memory is ${gpu_used_mib:-unknown} MiB" > "$LOG"
  printf '%s\n' 125 > "$EXITCODE"
  exit 125
fi

cd "$SOURCE" || exit 126
env \
  PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES=3 \
  /home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35/bin/python \
  -m visual_rl.train \
  experiments/v08_modular_gpu_20260808/configs/flow_grpo_wan_gpu3_attempt3_diagnostics.yaml \
  > "$LOG" 2>&1
status=$?
printf '%s\n' "$status" > "$EXITCODE"
exit "$status"
