#!/usr/bin/env bash
set -euo pipefail

# One-shot, event-driven A7 completion watcher.  It waits on known supervisor
# PIDs, never reads live training logs, and exits permanently after acceptance
# plus archival or at the first terminal failure.
evidence_root=/dev/shm/v-qiaoqifan/visualrl-v08-a7-final-56507f-6f1533ef
config_root=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/configs/a7-final-56507f-6f1533ef
python_bin=/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35/bin/python
capture=$evidence_root/tools/capture_a7_launch_receipt.py
finalize=$evidence_root/tools/finalize_a7_acceptance.sh
archive=$evidence_root/tools/archive_a7_acceptance.sh
automation_root=$evidence_root/automation

flow_sd3_supervisor=1688705
tempflow_sd3_supervisor=1690267
flow_wan_supervisor=1693030
flash_wan_supervisor=1694549
world_core_supervisor=1696016
world_release_supervisor=1696017

for path in "$python_bin" "$capture" "$finalize" "$archive"; do
  [[ -x "$path" || ( -f "$path" && $path == *.py ) ]] || {
    echo "missing completion dependency: $path" >&2
    exit 66
  }
done
command -v flock >/dev/null || { echo "flock is unavailable" >&2; exit 69; }
command -v tail >/dev/null || { echo "tail is unavailable" >&2; exit 69; }
mkdir -p "$automation_root"
exec 9>"$automation_root/terminal-watcher.lock"
flock -n 9 || { echo "another terminal watcher holds the lock" >&2; exit 73; }
printf '%s\n' "$$" >"$automation_root/terminal-watcher.owner-pid"

wait_for_exit() {
  local pid=$1
  local label=$2
  echo "$label: waiting for supervisor $pid"
  if kill -0 "$pid" 2>/dev/null; then
    tail --pid="$pid" -f /dev/null
  fi
  echo "$label: supervisor $pid exited"
}

finalize_terminal_rows() {
  local status
  set +e
  "$finalize"
  status=$?
  set -e
  if [[ $status -ne 0 && $status -ne 75 ]]; then
    echo "terminal acceptance failed with status $status" >&2
    exit "$status"
  fi
}

capture_after_upstream() {
  local upstream_pid=$1
  local upstream_label=$2
  local route=$3
  local gpu_index=$4
  local config=$5
  local pid_file=$evidence_root/logs/$route.pid

  wait_for_exit "$upstream_pid" "$upstream_label"
  for _attempt in $(seq 1 120); do
    if [[ -f "$pid_file" ]]; then
      "$python_bin" "$capture" "$route" "$gpu_index" "$config"
      echo "$route: live launch receipt captured"
      return 0
    fi
    sleep 1
  done
  echo "$route: trainer PID did not appear after upstream completion" >&2
  return 75
}

capture_after_upstream \
  "$flow_sd3_supervisor" \
  flow-grpo-sd3 \
  world-r1-core-wan \
  3 \
  "$config_root/world_r1_core_wan.yaml" &
core_capture_pid=$!

capture_after_upstream \
  "$tempflow_sd3_supervisor" \
  tempflow-sd3 \
  world-r1-release-surrogate-wan \
  6 \
  "$config_root/world_r1_release_surrogate_wan.yaml" &
release_capture_pid=$!

for specification in \
  "$flow_sd3_supervisor:flow-grpo-sd3" \
  "$tempflow_sd3_supervisor:tempflow-sd3" \
  "$flow_wan_supervisor:flow-grpo-wan" \
  "$flash_wan_supervisor:flash-wan" \
  "$world_core_supervisor:world-r1-core-wan" \
  "$world_release_supervisor:world-r1-release-surrogate-wan"; do
  pid=${specification%%:*}
  label=${specification#*:}
  wait_for_exit "$pid" "$label"
  finalize_terminal_rows
done

wait "$core_capture_pid"
wait "$release_capture_pid"
finalize_terminal_rows
[[ -f "$evidence_root/acceptance/matrix.json" ]] || {
  echo "all supervisors exited without a matrix acceptance receipt" >&2
  exit 1
}
"$archive"
echo "A7 six-route acceptance and archive completed"
