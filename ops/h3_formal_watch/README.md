# H3 formal watch — Python ≥3.10, standard library only

Infrastructure only: **not deployed or authorized for a real experiment**. Do not
use paused runs as test targets. No training math, flags, HPS service, monitor
budgets or scientific semantics are changed. Linux/systemd and external CLIs still
need separately authorized acceptance. All development tests/drills are offline.

## Watchdog contract

* First tick on **every start/restart** arms read-only: validate absolute paths and
  recipe bindings, resolved lifecycle output, existing artifacts, ready HPS with
  expected full identity, monitor health, readable metrics and process ownership.
  Failure goes to NEEDS_HUMAN **without any training signal**. Provision only after
  initial metrics exist; starting too early deliberately fails arming.
* Every elapsed deadline uses the GPU host's monotonic clock and persisted boot ID.
  Reboot/schema changes require reconciliation; never reuse old monotonic values.
  UTC fields (`now`, `heartbeat_at`, `last_progress_at`) are reporting only.
  Restart does not reset update/stop/recovery budgets. Monitor time is sampled
  **after** the atomic health.json read to avoid a negative-age publication race.
* HPS onload/idle-unload normally returns 503/unready. Consecutive bad health ticks
  must persist for `hps_grace_seconds=600`; a healthy tick resets this interval.
  Unreachable/read failures share that grace. Verified identity changes after
  arming are immediately fatal. HTTP uses `ProxyHandler({})`, never ambient proxies.
* Monitor unhealthy/stale and low disk require `bad_ticks=3` consecutive bad ticks.
  Runtime ownership read failures also debounce for `bad_ticks` without any action;
  initial/restart arming still fails closed immediately. Per-PID ENOENT/ESRCH races
  are skipped; permission/other IO failures remain unknown.
  Read/parse failures and unavailable inode capacity are UNKNOWN, never inferred
  no-progress or low-disk. Persistent unknown checks latch NEEDS_HUMAN without
  signalling. `inode_floor=0` explicitly disables the inode requirement.
* Progress is completed metrics `step+1`, not PID existence or training reward.
  Content/step identify new metrics; filesystem/wall-clock mtime is not a timer.
  Old failed-attempt metrics cannot masquerade as resumed progress.
* `monitor_stop`, `disk_low`, HPS failure after grace request a safe-boundary pause.
  Deadline = `last_progress_mono + max_update_seconds + checkpoint_allowance_seconds`
  (defaults 3h + 30min). New metrics extend the deadline; a newly published matching
  step checkpoint permits escalation after the TERM allowance. A stalled run
  (`no_progress`) gets TERM **immediately**, not an unreachable boundary request.
  TERM wait defaults 60s, then KILL, then 60s before NEEDS_HUMAN if still populated.
* Pause requires the operator-provisioned dangling symlink below. Missing/wrong
  link on a boundary fault means NEEDS_HUMAN, not a surprise hard stop. A request
  that exists without a watchdog STOPPING ownership record is an **operator pause**.
  It is never removed or auto-resumed, including when it uses the same symlink.
* Resume allowlist: explicit NCCL watchdog timeout, actual card/process-monitor
  unhealthy/stale exit log, or watchdog monitor/disk boundary stop with a new
  successful checkpoint and H3SafeBoundaryPause/log or matching TERM exit receipt.
  Generic death, unknown exits, HPS failure/identity change, graph/Adam/gradient
  audits and integrity errors require humans. Permanent evidence wins over
  transient patterns. Unrecognized exception/error records veto recovery, while
  normal launcher JSON (including identity metadata) is not failure evidence.
  Log matching is conservative and bounded to the attempt's
  last 256 KiB; missing evidence never authorizes retry.
* Resume requires healthy prerequisites, latest complete checkpoint metadata and
  **full exact-tree/shard hashing**. No rollback past a corrupt publication. Hashing
  runs in a daemon worker with no signalling/launch authority; the main loop keeps
  reporting RECOVERING/verifying plus byte/file progress and honest heartbeats.
  Default verification deadline 2h, persisted once per recovery incident across
  restarts; a completed result on the deadline tick is accepted. Prerequisite
  recovery window 30min. Exact-tree exclusions are root metadata paths only.
  HPS identity verification uses the producer's ASCII-escaped canonical JSON.
* At most 2 attempted auto-resumes per boot-bound rolling 24h; no same-checkpoint
  retry without a new high-water step. Only `launch.resume_from` changes. All
  artifacts/config paths must remain absolute and exist immediately before launch;
  original recipe identity, argv, full env and cwd are preserved. Launcher remains
  the authoritative semantic/runtime admission gate. Ambiguous launches are never
  replayed. Old watchdog-owned pause targets are archived, not deleted.
* NEEDS_HUMAN keeps doing read-only checks and archiving observations, never signals
  or launches, and remains sticky until explicit acknowledgement. DONE additionally
  requires matching SUCCESS/run/checkpoint data, empty ownership and exit-code-0
  receipt. An adopted process without a receipt cannot establish DONE.
* `status.json` retains only 32 incident summaries, bounded checks and 128 resume
  summaries (above the maximum allowed rolling limit of 100). Full incident data
  goes to fsynced `incidents.jsonl`; events to `events.jsonl`. At 1 MiB these files
  rotate by rename into uniquely named immutable segments; **nothing is deleted**.
  Monitor archive disk usage. Old observer outages may miss summaries outside this
  window: archive review, not the small status snapshot, is the complete history.

## Ownership: systemd first, conservative adoption fallback

New/relaunched training always runs in a unique transient **user service**, named
`h3-training-UNIQUE.service`, with `KillMode=control-group`, `Restart=no`,
`RemainAfterExit=yes`. The receipt records boot ID and systemd InvocationID. Never
reuse the unit name. HPS, monitor, watchdog and unrelated jobs must be outside it.
Liveness means **any process in its cgroup**, including elastic ranks with their
own sessions. TERM uses `systemctl --user kill --kill-whom=all --signal=SIGTERM`;
KILL escalates the whole unit and requests its stop. Training survives watcher
service restart/disable because it belongs to a different unit.

For operator-started training, set `training_unit` in config or record it once:

```sh
python3 -m ops.h3_formal_watch.process adopt-unit h3-training-UNIQUE.service /ABS/watch/state
# Only if NOT already in a suitable training unit:
python3 -m ops.h3_formal_watch.process record TORCHRUN_PID /ABS/watch/state
```

Fallback tracks PPID descendants plus PID/starttime/boot identity, retaining known
orphan ranks and signalling rank groups **before** the agent. Agent must have its
own session. Unowned group peers/reused identities fail closed. This is inherently
weaker than cgroups: a child that forks and reparents entirely between snapshots
cannot be discovered; PID-group signalling still has a small kernel TOCTOU window.
Do not claim equivalent containment. Prefer a new approved unit-owned launch; do
not migrate or restart a healthy operator run merely to simplify bookkeeping.

For a separately approved fresh launch, write private `state/launch-initial.json`:

```json
{"launch_id":"initial-UNIQUE","unit":"h3-training-initial-UNIQUE.service","argv":["/ABS/venv/bin/torchrun","EXACT_APPROVED_ARGUMENTS"],"cwd":"/ABS/source","env":{"EXACT_CAPTURED_ENV":"VALUE"}}
```

```sh
python3 -m ops.h3_formal_watch.process supervise /ABS/watch/state launch-initial.json
# It returns after systemd submission/receipt, not after training completion.
# Copy process-initial-UNIQUE.json to initial-process.json ONCE, or adopt training_unit.
```

The in-unit waiter saves the actual exit receipt. No systemd automatic training
restart is permitted. An uncatchable waiter KILL may lose the exit receipt; this
is unknown, not permission to fabricate one or replay the launch claim.

## GPU configuration

JSON serialization of the approved original training recipe is required (JSON is
valid YAML). Convert YAML-relative artifact paths to absolute **from the original
recipe's directory before provisioning**, never relative to watchdog state_dir.
All five artifact keys are required: model, text_cache, train, eval, hps_manifest.
No YAML parser or model import exists in ops. Config/recipe/cwd/output paths must
be absolute; output must equal its resolved path, just as lifecycle compilation.
Keep state private (0700), non-symlink, distinct from output/run, with no untrusted
writers. `launcher.env` is the complete environment, not overrides.

```json
{
  "state_dir": "/ABS/watch/state",
  "run_dir": "/ABS/run",
  "output_dir": "/ABS/run/output",
  "monitor_dir": "/ABS/run/monitor",
  "training_unit": "h3-training-initial-UNIQUE.service",
  "training_log": "/ABS/run/training.log",
  "hps_healthz_url": "http://127.0.0.1:18823/healthz",
  "hps_identity": "REPLACE_WITH_FULL_APPROVED_64_HEX_SHA256",
  "pause_via_symlink": true,
  "thresholds": {
    "interval_seconds": 60, "monitor_max_age": 30, "bad_ticks": 3,
    "hps_grace_seconds": 600, "max_update_seconds": 10800,
    "checkpoint_allowance_seconds": 1800, "max_auto_resumes": 2,
    "disk_floor_bytes": 21474836480, "inode_floor": 20000
  },
  "launcher": {
    "recipe_json": "/ABS/original-recipe.json",
    "cwd": "/ABS/approved-source",
    "argv": ["/ABS/venv/bin/torchrun", "--nproc_per_node=8", "/ABS/approved-source/scripts/launch_h3_formal.py", "{resume_config}", "EXACT_OTHER_APPROVED_FLAGS"],
    "env": {"EXACT_CAPTURED_ENV": "VALUE"}
  }
}
```

Alternatively supply `process` receipt or `initial-process.json` instead of
`training_unit`. Other thresholds: `term_wait_seconds=60`, `kill_wait_seconds=60`,
`recovery_wait_seconds=1800`, `launch_wait_seconds=60`, `verification_wait_seconds=7200`.
Old `stop_wait_seconds` is rejected; migrate config/state explicitly, not in place.
A ~41-minute resumed initialization consumes the 3h progress budget: choose the
approved update budget with initialization/export durations in mind.

Operator provisioning, never overwrite an existing PAUSE_REQUESTED:

```sh
ln -s /ABS/watch/state/pause-request /ABS/run/output/PAUSE_REQUESTED
python3 -m ops.h3_formal_watch.watchdog /ABS/dry-config.json --dry-run --once
```

Dry-run must use a separate state_dir/identity and never signals or submits units.
After investigating NEEDS_HUMAN and reconciling real processes/checkpoints/logs:

```sh
python3 -m ops.h3_formal_watch.watchdog /ABS/config.json --acknowledge 'Reviewed incident; safe to rearm'
```

This writes a token-bound acknowledgement consumed by the running watchdog on its
next tick. It does not clear retry limits, remove pause requests, bypass a reboot,
change ownership, or force a resume. Active pause requests must be resolved by the
operator first; a failed rearming gate returns to NEEDS_HUMAN with a new token.

## Observer / notifications

VPS observer is independent, read-only on the GPU, and cannot control training.
Optional pi diagnosis has no tools/MCP/extensions/context, no persistent session,
and no authority in the deterministic watchdog. Sending logs to a model requires
separately approved disclosure. Feishu receives **alerts only**, as a direct message
from the app bot to **one configured human user**. No group or chat-ID target is
supported, including multi-human chats. Full pi diagnoses and daily summaries are
local-only; they are never sent to Feishu.

```json
{
  "state_dir": "/ABS/observer/state",
  "report_dir": "/ABS/observer/reports",
  "progress_log_path": "/ABS/progress/10.22-progress.md",
  "ssh_host": "ubtroot@APPROVED_ALIAS",
  "ssh_identity_file": "/ABS/read-only-key",
  "remote_python": "/usr/bin/python3",
  "remote_status": "/ABS/watch/state/status.json",
  "remote_events": "/ABS/watch/state/events.jsonl",
  "remote_logs": ["/ABS/run/training.log"],
  "include_training_log": true,
  "expected_config_id": "REPLACE_WITH_APPROVED_STATUS_CONFIG_ID",
  "timezone": "Asia/Shanghai", "daily_time": "09:00",
  "heartbeat_max_age_seconds": 600, "unreachable_seconds": 1800,
  "ssh_attempts": 3, "ssh_timeout_seconds": 30,
  "notify_retry_seconds": 300, "alert_repeat_seconds": 3600,
  "analysis_timeout_seconds": 180, "max_analyses_per_tick": 4,
  "pi_binary": "/ABS/pi",
  "notifier": {"binary":"/ABS/lark-cli","alert_user_open_id":"ou_REPLACE","identity":"bot","timeout_seconds":30}
}
```

SSH uses verified host keys, BatchMode, bounded retries/backoff and only the bundled
read-only snapshot program. Prefer an operator-enforced forced-command read-only
key. Snapshot bounds: status 1 MiB; events 64 KiB; four 16 KiB log tails including
the current supervised attempt. Missing logs are explicit unknowns. No cached
snapshot is evidence of fresh reachability. Heartbeat age is judged ONLY on a
fresh fetch, using the GPU snapshot's remote monotonic clock when boot IDs match,
otherwise that same remote host's wall time — never Mac/VPS time against cached
status. Cached DONE cannot suppress the independent 30-minute SSH-unreachable
rule. Observer cooldown/unreachability durations use its own boot-bound monotonic
clock. Daily calendar scheduling alone intentionally uses wall time.

`notifier.alert_user_open_id` must be a single `ou_` user open_id, not a list,
chat ID, email, whitespace/comma-separated targets or CLI syntax. `identity` must
be `bot`. The notifier accepts only the four documented keys; **any** extra key
(including `chat_id`, `group_id`, `user_id`, or target lists) is rejected, even if
empty. Legacy top-level chat/group keys are also rejected. Every observer start
validates before any fetch/analysis/send and records refusals locally in
`state_dir/notifier-errors.jsonl`; delivery rechecks the policy. The only send argv
is `lark-cli im +messages-send --as bot --user-id <open_id> --text ...
--idempotency-key ...` (flags checked against local CLI help). No chat lookup,
group fallback, or sender-identity fallback exists.

`report_dir` and `progress_log_path` are required absolute local paths, private
and non-symlink, with trusted writers only. Keep the progress file distinct from
observer state and report files. Each day's summary and optional pi diagnosis go
to `report_dir/DAILY.md`, one dated section per day; each section is capped at
8,000 Unicode characters. A short dated line pointing to DAILY.md is appended to
the progress log. Non-daily pi diagnoses go to `report_dir/INCIDENTS.md` with the
same section cap. Alert DMs contain only a bounded event/status summary and the
local INCIDENTS.md path; the initial alert may precede its diagnosis file write.
The follow-up DM announces the saved diagnosis, without copying model output.
Reports use whole-file temp+fsync+rename, locks and durable markers to avoid
partial publication or duplicate sections/progress lines after a crash. History
is retained without rotation; monitor local disk usage. Dry-run also writes these
local artifacts, so use separate dry-run state, reports and progress paths.

Daily summaries and their diagnoses never enter the alert outbox; legacy queued
daily messages are retained but suppressed, never sent. Legacy incident-diagnosis
outbox text is replaced by a short local-file pointer before retry. Report-write
failures are recorded and retried without repeating pi or blocking unrelated alert
outbox draining. Existing configurations
with `chat_id` must be explicitly reconciled with the new user-only configuration
and config-bound state before deployment; do not delete evidence or reset pending
alerts to bypass the identity check.

Alert text/attempts form a durable outbox. Undelivered messages drain even
when the triggering condition has resolved; failed sends exit nonzero, with a
5-minute service retry/cooldown and stable idempotency key. Non-object CLI replies
are failures. Text starting with `@`/`-` is prefixed to prevent file/option parsing;
UTF-8 argv is never shell-interpolated. An overlapping timer's busy lock exits 0
with an explicit log line, without constructing/running the observer. Exactly-once
network delivery is not guaranteed beyond Feishu's upstream deduplication window.
Daily and incident pi sessions are claimed once; ambiguous model execution is not
automatically replayed. Independent VPS monitoring is still required: a dead VPS
cannot report its own outage.

Preview without any SSH/pi/lark subprocess (use separate dry state):

```sh
python3 -m ops.h3_formal_watch.observer /ABS/dry-observer.json --dry-run
python3 -m ops.h3_formal_watch.observer /ABS/dry-observer.json --dry-run --snapshot /ABS/offline-snapshot.json
```

## Install, stop and revoke (authorized hosts only)

Do not deploy from this worktree as part of the code-fix task. After Linux/CLI
acceptance, render `systemd/*.service` placeholders `@REPO@`, `@PYTHON@`, `@CONFIG@`
with absolute paths without whitespace or `%`, into `~/.config/systemd/user/`.
Install only the GPU watchdog on GPU, observer service/timer on VPS. Confirm
`loginctl show-user "$USER" -p Linger`; enabling linger may require separate admin
permission. Never change linger for unrelated jobs.

Example rendering on the **authorized host**, after replacing the paths:

```sh
export REPO=/ABS/watch-checkout PYTHON=/ABS/python3 CONFIG=/ABS/config.json
export UNIT=h3-formal-watchdog  # on VPS: h3-formal-observer
python3 - <<'PY'
import os
from pathlib import Path
root = Path.home() / '.config/systemd/user'
root.mkdir(parents=True, exist_ok=True)
unit = os.environ['UNIT']
assert unit in ('h3-formal-watchdog', 'h3-formal-observer')
source = Path(os.environ['REPO']) / 'ops/h3_formal_watch/systemd'
text = (source / (unit + '.service')).read_text()
for key in ('REPO', 'PYTHON', 'CONFIG'):
    value = os.environ[key]
    assert Path(value).is_absolute() and not any(c.isspace() or c == '%' for c in value)
    text = text.replace('@' + key + '@', value)
(root / (unit + '.service')).write_text(text)
if unit == 'h3-formal-observer':
    (root / (unit + '.timer')).write_text((source / (unit + '.timer')).read_text())
PY
systemctl --user daemon-reload
# GPU, after arming evidence and ownership are ready:
systemctl --user enable --now h3-formal-watchdog.service
# VPS, after recipient/delivery and CLI acceptance:
systemctl --user enable --now h3-formal-observer.timer
systemctl --user start h3-formal-observer.service
journalctl --user -u h3-formal-watchdog.service -n 50 --no-pager
```

Revoke automation before manual process intervention:

```sh
systemctl --user disable --now h3-formal-watchdog.service  # does NOT stop training
systemctl --user disable --now h3-formal-observer.timer
systemctl --user stop h3-formal-observer.service
# Only with separate authorization to stop training:
systemctl --user kill --kill-whom=all --signal=SIGTERM h3-training-UNIQUE.service
```

Preserve state, archives, recipes and receipts. A pending pause is not undone by
stopping the watchdog. For permanent revocation, archive the installed unit files,
then mask their names (mask cannot replace a regular installed user-unit file).
Revoke observer SSH/Feishu permissions separately. Never delete status, reuse a
launch claim, or reset history to bypass gates. Restart only after reconciliation.

## Offline acceptance

```sh
PYTHONPATH=. python -m pytest -q tests/test_h3_formal_watch.py
python ops/h3_formal_watch/drill.py
```

Drill defaults to a unique OS temporary directory, retains artifacts, and accepts
no real config. Nine scenarios: monitor stale, no progress, identity change, HPS
503 burst without stop, audit error, operator pause, fake orphan-rank PPID tree,
one transient resume then repeated-checkpoint refusal, matching success. Only
local sleeps are launched and cleaned up; HTTP and systemd submission are mocked.
These are infrastructure tests, not GPU learning/performance or systemd acceptance.
