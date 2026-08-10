#!/usr/bin/env bash
set -euo pipefail

# One-shot recovery for the final World-R1 release-surrogate route after its
# first frozen run was externally terminated with status 143.  The script
# waits for the already-running core route and the original terminal watcher,
# preserves the failed release attempt on tmpfs, starts one fresh run on the
# GPU released by the accepted core route, and archives the failed attempt to
# NFS in parallel with that run.  Slow NFS durability must not leave a healthy
# GPU idle.  The script never reads live training logs and never overwrites an
# existing artifact.
evidence_root=/dev/shm/v-qiaoqifan/visualrl-v08-a7-final-56507f-6f1533ef
config_root=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/configs/a7-final-56507f-6f1533ef
failure_archive_parent=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/failed_attempts/a7-final-56507f-6f1533ef
python_bin=/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35/bin/python
launcher=$evidence_root/tools/launch_a7_route.sh
capture=$evidence_root/tools/capture_a7_launch_receipt.py
finalize=$evidence_root/tools/finalize_a7_acceptance.sh
archive=$evidence_root/tools/archive_a7_acceptance.sh
automation_root=$evidence_root/automation

route=world-r1-release-surrogate-wan
retry_gpu=3
config=$config_root/world_r1_release_surrogate_wan.yaml
core_supervisor=1696016
original_watcher=1704242
failure_label=world-r1-release-surrogate-wan-sigterm-143-20260809
tmpfs_failure=$evidence_root/failed-attempts/$failure_label
nfs_failure=$failure_archive_parent/$failure_label

for path in "$python_bin" "$launcher" "$capture" "$finalize" "$archive" "$config"; do
  [[ -x "$path" || ( -f "$path" && ( "$path" == *.py || "$path" == *.yaml ) ) ]] || {
    echo "missing recovery dependency: $path" >&2
    exit 66
  }
done
command -v flock >/dev/null || { echo "flock is unavailable" >&2; exit 69; }
command -v tail >/dev/null || { echo "tail is unavailable" >&2; exit 69; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 69; }

mkdir -p "$automation_root"
exec 9>"$automation_root/world-release-recovery.lock"
flock -n 9 || { echo "another World-R1 release recovery holds the lock" >&2; exit 73; }
printf '%s\n' "$$" >"$automation_root/world-release-recovery.owner-pid"

wait_for_exit() {
  local pid=$1
  local label=$2
  echo "$label: waiting for PID $pid"
  if kill -0 "$pid" 2>/dev/null; then
    tail --pid="$pid" -f /dev/null
  fi
  echo "$label: PID $pid exited"
}

wait_for_exit "$core_supervisor" world-r1-core-wan
wait_for_exit "$original_watcher" original-terminal-watcher

[[ -f "$evidence_root/acceptance/world-r1-core-wan.json" ]] || {
  echo "core route did not produce an acceptance receipt; refusing retry" >&2
  exit 1
}
[[ -f "$evidence_root/runs/world-r1-core-wan/SUCCESS" ]] || {
  echo "core route did not publish SUCCESS; refusing retry" >&2
  exit 1
}
[[ -f "$evidence_root/logs/$route.exitcode" ]] || {
  echo "the terminated release route has no exit-code evidence" >&2
  exit 66
}
[[ $(tr -d '[:space:]' <"$evidence_root/logs/$route.exitcode") == 143 ]] || {
  echo "the release route no longer has the expected status 143" >&2
  exit 65
}
[[ ! -e "$evidence_root/runs/$route/SUCCESS" ]] || {
  echo "the terminated release route unexpectedly contains SUCCESS" >&2
  exit 65
}
[[ ! -e "$tmpfs_failure" && ! -e "$nfs_failure" ]] || {
  echo "refusing to overwrite an existing failed-attempt archive" >&2
  exit 73
}

mkdir -p \
  "$tmpfs_failure/logs" \
  "$tmpfs_failure/runs" \
  "$tmpfs_failure/launch-receipts" \
  "$tmpfs_failure/acceptance" \
  "$tmpfs_failure/automation"
mv "$evidence_root/runs/$route" "$tmpfs_failure/runs/$route"
for path in "$evidence_root/logs/$route".*; do
  [[ -e "$path" ]] && mv "$path" "$tmpfs_failure/logs/"
done
mv "$evidence_root/launch-receipts/$route.json" "$tmpfs_failure/launch-receipts/"
for path in "$evidence_root/acceptance/.$route".*; do
  [[ -e "$path" ]] && mv "$path" "$tmpfs_failure/acceptance/"
done
for path in \
  "$automation_root/watcher.log" \
  "$automation_root/watcher.exitcode"; do
  [[ -e "$path" ]] && cp -a "$path" "$tmpfs_failure/automation/"
done
(
  cd "$tmpfs_failure"
  find acceptance automation launch-receipts logs runs -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)

archive_failed_attempt() {
  local nfs_staging
  mkdir -p "$failure_archive_parent"
  nfs_staging=$(mktemp -d "$failure_archive_parent/.$failure_label.staging.XXXXXX")
  cp -a "$tmpfs_failure/." "$nfs_staging/"
  (
    cd "$nfs_staging"
    sha256sum -c SHA256SUMS >/dev/null
  )
  sync -f "$nfs_staging"
  mv "$nfs_staging" "$nfs_failure"
  (
    cd "$nfs_failure"
    sha256sum -c SHA256SUMS >/dev/null
  )
}

archive_failed_attempt &
failure_archive_pid=$!
printf '%s\n' "$failure_archive_pid" >"$automation_root/world-release-failure-archive.pid"

baseline_mib=$(nvidia-smi --id="$retry_gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ ! $baseline_mib =~ ^[0-9]+$ || $baseline_mib -gt 2048 ]]; then
  echo "GPU $retry_gpu baseline is not safe for retry: ${baseline_mib:-unknown} MiB" >&2
  exit 75
fi

"$launcher" "$retry_gpu" "$route" "$config" \
  >"$automation_root/world-release-retry-launcher.log" 2>&1 &
retry_supervisor=$!
printf '%s\n' "$retry_supervisor" >"$automation_root/world-release-retry-supervisor.pid"

pid_file=$evidence_root/logs/$route.pid
for _attempt in $(seq 1 120); do
  if [[ -f "$pid_file" ]]; then
    "$python_bin" "$capture" "$route" "$retry_gpu" "$config" \
      >"$automation_root/world-release-retry-launch-receipt.json"
    break
  fi
  if ! kill -0 "$retry_supervisor" 2>/dev/null; then
    echo "release retry launcher exited before publishing a trainer PID" >&2
    wait "$retry_supervisor"
    exit 1
  fi
  sleep 1
done
[[ -f "$evidence_root/launch-receipts/$route.json" ]] || {
  echo "release retry did not produce a launch receipt" >&2
  exit 75
}

set +e
wait "$retry_supervisor"
retry_status=$?
wait "$failure_archive_pid"
failure_archive_status=$?
set -e
if [[ $failure_archive_status -ne 0 ]]; then
  echo "failed-attempt NFS archive failed with status $failure_archive_status" >&2
  exit "$failure_archive_status"
fi
if [[ $retry_status -ne 0 ]]; then
  echo "release retry failed with status $retry_status" >&2
  exit "$retry_status"
fi

"$finalize"
[[ -f "$evidence_root/acceptance/matrix.json" ]] || {
  echo "release retry completed without six-route matrix acceptance" >&2
  exit 1
}
"$archive"
echo "World-R1 release retry accepted and A7 archive completed"
