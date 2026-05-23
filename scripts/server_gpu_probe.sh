#!/usr/bin/env bash
set -euo pipefail

SERVER="${1:-v-qiaoqifan@10.130.140.73}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${SERVER}" \
  'hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; echo "--- processes ---"; nvidia-smi pmon -c 1 || true'

