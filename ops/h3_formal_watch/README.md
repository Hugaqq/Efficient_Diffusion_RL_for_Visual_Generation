# H3 formal-run watch (stdlib only, Python >= 3.10)

This installs **infrastructure**, not authorization to start/resume a real experiment.
No deployment or real-run acceptance has been performed. Test the installed Linux
systemd/SSH/pi/lark versions and obtain run authorization separately. Never use the
old paused experiments as test targets. No changes to training source, flags, HPS,
monitor budgets or scientific semantics are made here.

## Contract and limitations

* GPU watchdog: one exclusive `watchdog.lock`, atomic `status.json`, fsynced
  append-only `events.jsonl`. 60-second checks; timings/config persist over restarts.
  `NEEDS_HUMAN` is sticky and continues heartbeating without actions; `DONE` exits.
* Progress: `output_dir/h3-last-metrics.json.step + 1` (rank0, after optimizer
  completion); checkpoint `complete.json.step`/`progress.next_optimizer_step` are
  durable fallback. `metrics.jsonl` is terminal-only. Journals are diagnostics, not
  optimizer evidence. Checkpoint cadence is 1, every 5, terminal, or requested pause.
  A restart starts its own progress timer; old metrics cannot fake new progress.
* Monitor age uses the **same host's monotonic clock**, `<30s`; `healthy=false`,
  missing/invalid health and stale health stop training. `degraded=true` alone only
  logs. Budget fields are retained, not modified; formal launcher checks capacity.
* HPS `/healthz`: `status=ready`, exact full `service_identity`; when present,
  identity facts are rehashed. Unavailable/unready HPS is fatal, not auto-resumable.
  Disk floor defaults to 20 GiB and 20,000 free inodes; unknown inode count fails
  closed. These emergency floors do NOT replace launcher's full-run disk preflight.
* Training must be an **exclusive session/process group containing only torchrun
  and its ranks**. Never include HPS, card monitor, watchdog or unrelated jobs.
  Linux identity is boot ID + PID start ticks + PGID/SID, not command names. Every
  signal rechecks identity. PID reuse/unknown members/reboot require a human.
  Escaped/new sessions cannot be discovered by group tracking: torchrun must not
  daemonize ranks outside this group. Mac `ps` support is for the drill only.
* Only watchdog state files are written. To opt into lifecycle's graceful pause,
  the **operator**, before starting the run, provisions a dangling symlink:
  `output/PAUSE_REQUESTED -> state/pause-request`. Watchdog only creates the target.
  Do not replace an existing request. Without this exact link it uses SIGTERM.
  A pause waits at most 900s (default); then TERM, 60s, KILL of the verified training
  group only; after another 60s an unkillable group needs a human. A 75–90 minute
  update often cannot reach its safe boundary in 15 minutes: choose a larger
  `stop_wait_seconds` if preserving the in-flight update outweighs fault exposure.
* Auto-resume: only process death, NCCL timeout, monitor-induced stop; known
  identity/OOM/integrity errors are permanent. All monitor/HPS/disk checks must
  recover first (default within 30min). No fresh starts. Require latest published
  checkpoint, consistent hashed metadata and **full shard hashes/exact tree**;
  no silent fallback from a corrupt newest published checkpoint. Launcher still
  performs authoritative contract/run/code/scorer checks. It alone writes training
  artifacts. At most 2 attempted resumes / rolling 24h, and no reuse of the same
  checkpoint without a new global progress high-water mark. Pending spawn is
  journaled before execution; ambiguous launch after a crash is never replayed.
  A watchdog-owned pause target is **renamed to an archive**, never deleted.
  An operator's independent pause request prevents resume.
* Full checkpoint verification can take minutes and temporarily delay heartbeat;
  observer dead-man threshold must exceed measured worst-case verification time.
  Configure clocks with NTP; VPS heartbeat age uses UTC epoch seconds. Host reboot
  and configuration changes require explicit manual reconciliation, not a reset.
* `SUCCESS` alone is insufficient. DONE needs matching run/checkpoint metadata,
  all training group members exited, and a matching supervisor exit-code-0 receipt.
  An adopted PID has no knowable exit status; missing receipt becomes NEEDS_HUMAN.

## GPU configuration and initial process identity

Keep code, run/output and state paths absolute and separate. State is private to
ubtroot, non-symlink, mode 0700. Never grant untrusted writers access to it. Example
`/home/ubtroot/h3-watch/config.json` (replace ALL placeholders; full HPS hash, not
`542df399...`). The initial `process` may be omitted when `initial-process.json`
exists in state_dir.

```json
{
  "run_dir": "/ABS/formal-run",
  "output_dir": "/ABS/formal-run/output",
  "state_dir": "/home/ubtroot/h3-watch/state",
  "monitor_dir": "/ABS/formal-run/monitor",
  "hps_healthz_url": "http://127.0.0.1:18823/healthz",
  "hps_identity": "REPLACE_64_HEX_CHARACTERS",
  "pause_via_symlink": true,
  "training_log": "/ABS/formal-run/initial-training.log",
  "thresholds": {
    "max_update_seconds": 10800, "max_auto_resumes": 2,
    "stop_wait_seconds": 900, "disk_floor_bytes": 21474836480, "inode_floor": 20000
  },
  "launcher": {
    "argv": ["/ABS/venv/bin/torchrun", "--nproc_per_node=8", "scripts/launch_h3_formal.py", "--config", "{resume_config}", "--expected-sha", "APPROVED_SHA", "--monitor-dir", "/ABS/formal-run/monitor", "REPLACE_WITH_EXACT_APPROVED_FLAGS"],
    "env": {"REPLACE_WITH_EXACT_LAUNCH_ENV": "VALUE"},
    "cwd": "/ABS/approved-training-checkout",
    "recipe_json": "/ABS/approved-original-recipe.json"
  }
}
```

The formal launcher has **no `--resume-from` flag**: it reads `launch.resume_from`
from its recipe. Supply an operator-verified JSON serialization of the original
recipe (JSON is valid YAML), with absolute artifact/output paths. The watchdog
writes a private immutable copy changing **only `launch.resume_from`**, passes its
path in the single `{resume_config}` argv slot, and preserves every other flag,
env value and cwd. Do not insert shell strings or unapproved launcher-upgrade
switches. `launcher.env` must be the **complete captured launch environment** (not
just overrides); the supervisor does not inherit watchdog-specific Python paths
or other settings into training. No YAML dependency is needed by ops.

For already-running training, record its isolated torchrun PGID leader once:

```sh
cd /ABS/watch-checkout
PYTHONDONTWRITEBYTECODE=1 python3 -m ops.h3_formal_watch.process record TORCHRUN_PID /home/ubtroot/h3-watch/state
```

For new **separately approved** training, use the durable supervisor rather than
adoption to retain exit status. Operator writes `state/launch-initial.json` with
`{"launch_id":"initial-UNIQUE","argv":[...exact fresh argv...],"cwd":"/ABS/training-checkout","env":{...}}`,
then runs the following in an independently managed persistent session/service:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m ops.h3_formal_watch.process supervise /home/ubtroot/h3-watch/state launch-initial.json
```

It creates a new isolated session, logs stdout/stderr under state_dir and records
`process-initial-UNIQUE.json` and `exit-initial-UNIQUE.json`. Before enabling the
watchdog, copy the process receipt to `initial-process.json` (or put its JSON in
config's `process`). Never fabricate an exit receipt or overwrite a launch claim.
Auto-resumes use the same supervisor detached from the watchdog. Configure any
external supervisor with **no automatic training retry**. Never run two controllers.

To provision pause (operator action; not performed by watchdog):

```sh
# Only before the output directory is admitted/created, plan this link; once the
# launcher has created output, install it without overwriting any existing path.
ln -s /home/ubtroot/h3-watch/state/pause-request /ABS/formal-run/output/PAUSE_REQUESTED
```

Dry-run uses a **separate state_dir**, recorded process identity and the same checks.
`python3 -m ops.h3_formal_watch.watchdog /ABS/dry-config.json --dry-run --once`
prints intended actions and writes only dry state; never sends signals or launches.
Do not reuse dry status for a live service.

## Install user units (GPU ubtroot; observer VPS qvanium)

Do NOT run these commands from this development worktree against a real host.
On each authorized host, logged in as the correct user, set the paths and render
only its unit. The following are exact shell commands after editing the variables:

```sh
REPO=/ABS/watch-checkout
PYTHON=/ABS/python3
CONFIG=/ABS/watch-config.json
UNIT=h3-formal-watchdog  # GPU; on VPS use h3-formal-observer
loginctl show-user "$USER" -p Linger
# If Linger=no, request/administer permission first (often needs sudo):
sudo loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger  # must be yes for unattended logout/reboot
mkdir -p "$HOME/.config/systemd/user"
export REPO PYTHON CONFIG UNIT
python3 - <<'PY'
import os
from pathlib import Path
repo, unit = os.environ['REPO'], os.environ['UNIT']
text = (Path(repo) / 'ops/h3_formal_watch/systemd' / (unit + '.service')).read_text()
for key in ('REPO', 'PYTHON', 'CONFIG'):
    text = text.replace('@' + key + '@', os.environ[key])
# Unit path placeholders must not contain whitespace or systemd percent specifiers.
assert not any(any(c.isspace() or c == '%' for c in os.environ[k]) for k in ('REPO', 'PYTHON', 'CONFIG'))
(Path.home() / '.config/systemd/user' / (unit + '.service')).write_text(text)
PY
systemctl --user daemon-reload
# GPU only:
systemctl --user enable --now h3-formal-watchdog.service
systemctl --user status h3-formal-watchdog.service
journalctl --user -u h3-formal-watchdog.service -n 50 --no-pager
# VPS only, after configuring/testing observer and notifier:
cp "$REPO/ops/h3_formal_watch/systemd/h3-formal-observer.timer" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now h3-formal-observer.timer
systemctl --user start h3-formal-observer.service
systemctl --user list-timers h3-formal-observer.timer
```

Stopping/revoking (on the respective host):

```sh
systemctl --user disable --now h3-formal-watchdog.service   # does NOT stop training
systemctl --user disable --now h3-formal-observer.timer
systemctl --user stop h3-formal-observer.service
systemctl --user mask h3-formal-watchdog.service            # prevent accidental restart
systemctl --user mask h3-formal-observer.service h3-formal-observer.timer
```

Preserve all state/evidence. Revoke the observer's read-only SSH key and Feishu
permissions separately using the operator's normal credential process. Do not
`disable-linger` if other user services depend on it. Stopping a watchdog while a
pause/stop is in progress does not undo the already issued request; reconcile
training/checkpoints/receipts before any manual restart. Unmask/re-enable only
following that reconciliation; never delete status to bypass retry limits.

## Offline drill and tests

```sh
PYTHONPATH=. python3 -m ops.h3_formal_watch.drill --parent tmp/h3-watch-drills
PYTHONPATH=. python3 -m pytest -q tests/test_h3_formal_watch.py
```

The drill creates a unique FAKE directory, synthetic markers, `/bin/sleep` groups
and a fake Python sleep resume command. HTTP is stubbed; it cannot accept a real
run config. It exercises monitor stale, no progress, changed HPS identity, transient
death + one resume + repeated failure, and matching success. Every owned dummy is
reaped/killed; evidence is retained, no real artifacts removed. Unit tests add pause
escalation, PID reuse, limits, corruption, write confinement, and observer cases.
