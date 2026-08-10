#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  printf '%s\n' "usage: sample_gpu_memory.sh GPU_INDEX TARGET_PID OUTPUT_CSV" >&2
  exit 64
fi

readonly GPU_INDEX=$1
readonly TARGET_PID=$2
readonly OUTPUT_CSV=$3
readonly SAMPLE_INTERVAL_SECONDS=15

if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "GPU_INDEX must be a non-negative integer" >&2
  exit 64
fi
if [[ ! "$TARGET_PID" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' "TARGET_PID must be a positive integer" >&2
  exit 64
fi
if [[ "$OUTPUT_CSV" != /* ]]; then
  printf '%s\n' "OUTPUT_CSV must be an absolute path" >&2
  exit 64
fi
if [[ ! -d "${OUTPUT_CSV%/*}" ]]; then
  printf '%s\n' "OUTPUT_CSV parent directory does not exist" >&2
  exit 66
fi

printf '%s\n' \
  "sample_epoch_s,target_pid,target_alive,gpu_index,gpu_uuid,memory_used_mib,memory_total_mib,utilization_gpu_percent" \
  > "$OUTPUT_CSV"

sample() {
  local target_alive=$1
  local gpu_row
  gpu_row=$(nvidia-smi \
    --id="$GPU_INDEX" \
    --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits)
  printf '%s,%s,%s,%s\n' \
    "$(date +%s)" "$TARGET_PID" "$target_alive" "$gpu_row" \
    >> "$OUTPUT_CSV"
}

while kill -0 "$TARGET_PID" 2>/dev/null; do
  sample 1
  sleep "$SAMPLE_INTERVAL_SECONDS"
done

# Capture one final driver state after the training wrapper exits.  This row is
# not attributed to a live training process; TARGET_PID remains recorded so the
# evidence consumer can distinguish it through the explicit target_alive field.
sample 0
