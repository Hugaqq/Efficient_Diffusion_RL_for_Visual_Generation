#!/usr/bin/env bash
set -euo pipefail

# One-shot handoff for the deployed A7 recovery whose pre-retry NFS sync is
# stuck in folio_wait_bit_common.  This script terminates only the recovery
# shell, never the sync worker, trainer, or reward services.  It preserves the
# rejected route in checksummed tmpfs, launches the same frozen route on the
# released GPU, and finishes the already-created NFS staging after training.
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
old_recovery_wrapper=2270592
old_recovery_shell=2270594
old_sync_worker=2582884
failure_label=world-r1-release-surrogate-wan-sigterm-143-20260809
tmpfs_failure=$evidence_root/failed-attempts/$failure_label
nfs_failure=$failure_archive_parent/$failure_label
nfs_staging=$failure_archive_parent/.$failure_label.staging.snsQM4

for path in "$python_bin" "$launcher" "$capture" "$finalize" "$archive" "$config"; do
  [[ -x "$path" || ( -f "$path" && ( "$path" == *.py || "$path" == *.yaml ) ) ]] || {
    echo "missing handoff dependency: $path" >&2
    exit 66
  }
done
for command_name in flock nvidia-smi sha256sum tail; do
  command -v "$command_name" >/dev/null || {
    echo "$command_name is unavailable" >&2
    exit 69
  }
done

mkdir -p "$automation_root"
exec 8>"$automation_root/world-release-handoff-v2.lock"
flock -n 8 || { echo "another World-R1 handoff holds the lock" >&2; exit 73; }

cmdline_contains() {
  local pid=$1
  local expected=$2
  [[ -r /proc/$pid/cmdline ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -F -- "$expected" >/dev/null
}

[[ -f "$evidence_root/acceptance/world-r1-core-wan.json" ]] || {
  echo "core acceptance is missing" >&2
  exit 66
}
[[ -f "$evidence_root/runs/world-r1-core-wan/SUCCESS" ]] || {
  echo "core SUCCESS is missing" >&2
  exit 66
}
[[ -f "$evidence_root/logs/$route.exitcode" ]] || {
  echo "rejected release exit code is missing" >&2
  exit 66
}
[[ $(tr -d '[:space:]' <"$evidence_root/logs/$route.exitcode") == 143 ]] || {
  echo "rejected release no longer has exit status 143" >&2
  exit 65
}
[[ ! -e "$evidence_root/runs/$route/SUCCESS" ]] || {
  echo "rejected release unexpectedly has SUCCESS" >&2
  exit 65
}
[[ ! -e "$tmpfs_failure" && ! -e "$nfs_failure" ]] || {
  echo "a committed failed-attempt archive already exists; refusing handoff" >&2
  exit 73
}
[[ -d "$nfs_staging" && -f "$nfs_staging/SHA256SUMS" ]] || {
  echo "expected NFS staging is missing" >&2
  exit 66
}
[[ ! -e "$automation_root/world-release-retry-supervisor.pid" ]] || {
  echo "the deployed recovery already published a retry supervisor; handoff is unnecessary" >&2
  exit 75
}
cmdline_contains "$old_sync_worker" "sync -f $nfs_staging" || {
  echo "old NFS sync worker identity does not match" >&2
  exit 65
}

# Freeze the old shell before checking the launch sentinel again.  This closes
# the race where the NFS syscall returns between validation and handoff.  A
# previous safe handoff attempt may already have terminated the shell while its
# inherited sync worker remains in D state; that state is also safe to resume.
if kill -0 "$old_recovery_shell" 2>/dev/null; then
  cmdline_contains "$old_recovery_shell" recover_a7_world_release_after_core.sh || {
    echo "old recovery shell identity does not match" >&2
    exit 65
  }
  kill -STOP "$old_recovery_shell"
  if [[ -e "$automation_root/world-release-retry-supervisor.pid" ]]; then
    kill -CONT "$old_recovery_shell"
    echo "the deployed recovery began launching the retry; handoff aborted safely" >&2
    exit 75
  fi
fi

{
  date -Is
  echo "old_recovery_wrapper=$old_recovery_wrapper"
  echo "old_recovery_shell=$old_recovery_shell"
  echo "old_sync_worker=$old_sync_worker"
  echo "nfs_staging=$nfs_staging"
  sha256sum "$nfs_staging/SHA256SUMS"
} >"$automation_root/world-release-handoff-v2.preflight.txt"

# The shell is experiment orchestration, not a trainer.  SIGKILL is deliberate:
# a stopped shell cannot advance and later move the fresh run into the rejected
# namespace when its orphaned sync finally returns.
if kill -0 "$old_recovery_shell" 2>/dev/null; then
  kill -KILL "$old_recovery_shell"
fi
for _attempt in $(seq 1 30); do
  kill -0 "$old_recovery_wrapper" 2>/dev/null || break
  sleep 1
done
if kill -0 "$old_recovery_wrapper" 2>/dev/null; then
  kill -TERM "$old_recovery_wrapper"
fi

# The orphaned sync worker inherited the original recovery lock descriptor, so
# that lock can remain held until NFS returns even though no recovery shell can
# advance.  The v2 handoff lock above is the live orchestration lock; requiring
# the old shell and wrapper to be gone is the actual no-race condition.
if kill -0 "$old_recovery_shell" 2>/dev/null || kill -0 "$old_recovery_wrapper" 2>/dev/null; then
  echo "old recovery orchestration is still live" >&2
  exit 75
fi

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
  "$automation_root/watcher.exitcode" \
  "$automation_root/world-release-recovery.log" \
  "$automation_root/world-release-recovery.exitcode"; do
  [[ -e "$path" ]] && cp -a "$path" "$tmpfs_failure/automation/"
done
(
  cd "$tmpfs_failure"
  find acceptance automation launch-receipts logs runs -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)

baseline_mib=$(nvidia-smi --id="$retry_gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ ! $baseline_mib =~ ^[0-9]+$ || $baseline_mib -gt 2048 ]]; then
  echo "GPU $retry_gpu baseline is not safe for retry: ${baseline_mib:-unknown} MiB" >&2
  exit 75
fi

"$launcher" "$retry_gpu" "$route" "$config" \
  >"$automation_root/world-release-handoff-v2-launcher.log" 2>&1 &
retry_supervisor=$!
printf '%s\n' "$retry_supervisor" >"$automation_root/world-release-handoff-v2-supervisor.pid"

pid_file=$evidence_root/logs/$route.pid
for _attempt in $(seq 1 120); do
  if [[ -f "$pid_file" ]]; then
    "$python_bin" "$capture" "$route" "$retry_gpu" "$config" \
      >"$automation_root/world-release-handoff-v2-launch-receipt.json"
    break
  fi
  if ! kill -0 "$retry_supervisor" 2>/dev/null; then
    echo "handoff launcher exited before publishing a trainer PID" >&2
    wait "$retry_supervisor"
    exit 1
  fi
  sleep 1
done
[[ -f "$evidence_root/launch-receipts/$route.json" ]] || {
  echo "handoff retry did not produce a launch receipt" >&2
  exit 75
}

set +e
wait "$retry_supervisor"
retry_status=$?
set -e
if [[ $retry_status -ne 0 ]]; then
  echo "handoff retry failed with status $retry_status" >&2
  exit "$retry_status"
fi

# The NFS durability barrier is intentionally joined only after GPU work.  If
# the original worker has already exited this is a no-op.
if cmdline_contains "$old_sync_worker" "sync -f $nfs_staging"; then
  tail --pid="$old_sync_worker" -f /dev/null
fi
if [[ ! -e "$nfs_failure" ]]; then
  [[ -d "$nfs_staging" ]] || { echo "NFS failed-attempt staging disappeared" >&2; exit 66; }
  mv "$nfs_staging" "$nfs_failure"
fi
(
  cd "$nfs_failure"
  sha256sum -c SHA256SUMS >/dev/null
)

"$finalize"
[[ -f "$evidence_root/acceptance/matrix.json" ]] || {
  echo "handoff retry completed without six-route matrix acceptance" >&2
  exit 1
}
"$archive"
echo "World-R1 release handoff accepted and A7 archive completed"
