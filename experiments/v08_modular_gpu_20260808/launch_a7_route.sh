#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: launch_a7_route.sh GPU_INDEX ROUTE CONFIG_PATH" >&2
  exit 64
fi

gpu_index=$1
route=$2
config_path=$3

if [[ ! $gpu_index =~ ^[0-9]+$ ]]; then
  echo "GPU_INDEX must be a non-negative integer" >&2
  exit 64
fi
if [[ ! $route =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  echo "ROUTE must use lowercase dash-separated tokens" >&2
  exit 64
fi
if [[ $config_path != /* || ! -f $config_path ]]; then
  echo "CONFIG_PATH must name an existing absolute file" >&2
  exit 66
fi

source_root=/dev/shm/v-qiaoqifan/visualrl-v08-candidate-56507f6e-source
trainer_python=/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35/bin/python
evidence_root=/dev/shm/v-qiaoqifan/visualrl-v08-a7-final-56507f-6f1533ef
sampler=$source_root/experiments/v08_modular_gpu_20260808/sample_gpu_memory.sh

if [[ ! -x $trainer_python || ! -x $sampler ]]; then
  echo "frozen trainer interpreter or memory sampler is unavailable" >&2
  exit 69
fi

log_dir=$evidence_root/logs
run_dir=$evidence_root/runs/$route
stdout_log=$log_dir/$route.log
memory_log=$log_dir/$route.gpu-memory.csv
pid_file=$log_dir/$route.pid
exitcode_file=$log_dir/$route.exitcode

mkdir -p "$log_dir"
for path in "$run_dir" "$stdout_log" "$memory_log" "$pid_file" "$exitcode_file"; do
  if [[ -e $path ]]; then
    echo "refusing to overwrite existing A7 artifact: $path" >&2
    exit 73
  fi
done

cd "$source_root"
env PYTHONPATH=. CUDA_VISIBLE_DEVICES="$gpu_index" \
  "$trainer_python" -m visual_rl.train "$config_path" \
  >"$stdout_log" 2>&1 &
trainer_pid=$!
printf '%s\n' "$trainer_pid" >"$pid_file"

"$sampler" "$gpu_index" "$trainer_pid" "$memory_log" &
sampler_pid=$!

set +e
wait "$trainer_pid"
trainer_status=$?
wait "$sampler_pid"
sampler_status=$?
set -e

printf '%s\n' "$trainer_status" >"$exitcode_file"
if [[ $sampler_status -ne 0 ]]; then
  echo "GPU memory sampler failed with exit code $sampler_status" >&2
  exit "$sampler_status"
fi
exit "$trainer_status"
