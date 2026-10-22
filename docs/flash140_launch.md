# Flash140: Git deployment, bounded real validation, and formal continuation

**Local implementation handoff, not GPU validation.** A is
`a9a3b245ecdb3f95594fc6807fae4cbe3304033c`. B is the second commit containing
this document and `scripts/launch_flash140.py`; use the full B SHA returned by
the coding handoff, not a package version or an unexamined future main tip.
The simplified full384/two192 GPU path and formal140 have NOT run locally.

## Deployment (main session, authorized server terminal only)

Do not edit server source or overwrite either existing worktree. Preserve the
old service source path while its process may still import it. Main owns any
necessary reward-service source-path reconciliation; this launcher never
restarts a worker, changes its device, installs packages, or touches other jobs.

Set `AGREED_SHA` to the exact B SHA from the final coding result, then:

```bash
: "${AGREED_SHA:?export the exact full B commit SHA from the coding handoff}"
BASE=/home/ubuntu/qifan/VisualRL
SOURCE=/home/ubuntu/qifan/visualrl-flash140-20261022/simplified-source
git -C "$BASE" status --short --branch
git -C "$BASE" fetch origin main
git -C "$BASE" cat-file -e "$AGREED_SHA^{commit}"
# SOURCE must not exist. No reset, clean, force push, or patch uploads.
git -C "$BASE" worktree add --detach "$SOURCE" "$AGREED_SHA"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$AGREED_SHA"
test -z "$(git status --porcelain --untracked-files=normal)"
git log -2 --oneline
PY=/home/ubuntu/qifan/envs/visual-rl/bin/python
OVERLAY=/home/ubuntu/qifan/visualrl-flash140-20261022/dependency-overlay
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE:$OVERLAY"
```

Use the approved GPU index after checking ownership; never stop other processes
to satisfy admission. Do not upgrade shared environments. The launcher requires
the training venv and checks actual imports, not just installed metadata:
Torch2.7.1+cu128, Diffusers0.33.1, overlay ftfy6.3.1/wcwidth0.2.14. It records
paths/versions for transformers, peft, accelerate, NumPy, YAML, PIL, requests,
imageio and imageio-ffmpeg as well. `visual_rl` and collection.py must resolve to
the clean pulled source. Overlay copies of visual_rl are rejected. Model loading
is already `local_files_only=True`; offline environment flags are also set.

## Fixed recipe and admission

`experiments/flash_wan_capacity_20261022/approved140-runtime.yaml` keeps:

- 140 optimizer commits,70 fixed-policy384 collections, two nominal192 updates;
  full384 batch std, group8, chunk16, forward4/replay4/decode1, KL0/PPO clip.001.
- Seed42,20 train steps,CFG4.5,480x832x81,LoRA16/32,BF16, stage text encoding,
  gradient checkpointing; no model/precision/objective relaxation for validation.
- EMA every8 commits;300 prompts x50 deterministic ODE at0/36/76/116, all81
  HPS frames/top24 aggregation; no additional evaluation at140.
- Wan `/mnt/data/qifan/models/Wan2.1-T2V-1.3B-Diffusers` and author
  `/home/ubuntu/qifan/Flash-GRPO/dataset/video/{train,test}.txt` (19700/300 rows,
  exact manifest SHA checks). Full model contents are still hashed by controller
  preflight; the launcher additionally checks local component configs/weights.
- Existing private worker `http://127.0.0.1:18820` and
  `/mnt/data/qifan/runs/flash140-20261022/hpsv3-service.json`, identity
  `4c831491cb7b413839ad07459f777bd674cbd59aadbc7d7d3e2ce25b11c993e9`.
  Only its real typed `/healthz` and `/score` protocol is used; this is not the
  unrelated author's different HTTP interface. Artifact/live identity must agree.

Admission requires exactly one visible CUDA GPU, at least **68.03GiB free GPU**
(52.03GiB measured batch4 reservation +16GiB margin) and **88GiB available host**
(64GiB provisional workload budget +24GiB margin). 52.03 is a prior probe, NOT
an end-to-end measurement;64 is conservative budgeting, NOT observed host use.
Validation/preflight require64GiB disk free; formal `run` reserves1TiB free at
admission because all checkpoints are retained. There is no automatic deletion
or pruning. Use actual `checkpoint_disk_bytes` from the first odd/even checkpoints
to budget the remaining70 odd pools plus even/model states, chunks, temp writes
and at least64GiB disk margin. If that exceeds available space, use a larger
output filesystem before formal launch;1TiB is not a proven upper bound.

A15s heartbeat and synchronous post-checkpoint sample enforce >=16GiB free GPU,
>=24GiB host available and >=64GiB disk free during execution. Torch's cumulative
peak reservation is also checked against initial admitted free GPU minus16GiB,
so a missed transient Torch reservation cannot be hidden by later offload.
Minima are sampled device-wide free memory, not a proof of continuous headroom
from other processes; Torch peak excludes non-Torch allocations. Host peak RSS
is per launcher, not system-wide. Native CUDA/HTTP calls can delay a safety
signal until Python regains control. A breach aborts ONLY this launcher, keeps
the last durable checkpoint, and reports failure; no unsafe in-process retry.

## Executable real validation

Choose a new output outside source. `preflight` is read-only apart from a parent
directory/global lock record; it does not create the experiment output or load
model weights. This is admission, not a GPU smoke pass.

```bash
OUT=/mnt/data/qifan/runs/flash140-20261022/simplified-validation
"$PY" -u scripts/launch_flash140.py preflight --expected-sha "$AGREED_SHA" --output "$OUT"
# Baseline300x50 evaluation runs BEFORE any training; this can take many hours.
# First full384 rollout/reward/advantage then one192 update, durable odd safe point.
"$PY" -u scripts/launch_flash140.py validate --expected-sha "$AGREED_SHA" --output "$OUT" --until 1
# Read only OWN trusted checkpoints; deserialization is not safe for untrusted files.
"$PY" -u scripts/launch_flash140.py inspect --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$OUT/checkpoints/step-1"
# New process must reuse the pending pool: no new collection/score/evaluation.
"$PY" -u scripts/launch_flash140.py validate --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$OUT/checkpoints/step-1" --until 2
"$PY" -u scripts/launch_flash140.py inspect --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$OUT/checkpoints/step-2" --compare "$OUT/checkpoints/step-1"
```

Alternatively one fresh `validate --until 2` performs the complete384->192+192
path continuously. Validation always renders the SAME140 training semantics and
LR endpoint; it never sets max_steps2 or disables baseline evaluation. The
wrapper observes the existing checkpoint sink and pauses only after its atomic
non-terminal receipt. The controller unwinds/tears down resources; the launcher
records `paused_not_success` and exits0, NOT training SUCCESS or GPU approval.
No checkpoint ownership, optimizer, reward or training loop is replaced.

For the real interval8 EMA/new-collection check, after accepting step2 results:

```bash
"$PY" -u scripts/launch_flash140.py validate --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$OUT/checkpoints/step-2" --until 8
"$PY" -u scripts/launch_flash140.py inspect --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$OUT/checkpoints/step-8" --compare "$OUT/checkpoints/step-2"
```

This executes three additional full384 collections, NOT a toy EMA simulation.
Inspection loads one full checkpoint/pool at a time on CPU and returns only
scalar summaries/hashes; ensure host budget, and run it after pausing. It verifies
complete atomic checkpoints, finite CPU trainables/EMA and schema2, reporting
pending rows/cursor, evaluation cursor, persisted metrics and parameter hashes.
It is a read-only aid, not an automatic assertion of learning or full-method
correctness. Main must accept these actual GPU-only checks:

1. Baseline300 receipts are real50-step ODE/all81-frame HPS, with finite rewards.
   Every training chunk uses actual81-frame scoring; no simulated client in CLI.
2. Step0 diagnostic passes using normal current forward; both updates have
   finite nonzero gradients. Step2 trainable hash differs from step1; real
   second ratio may differ from1 and clipfrac may be nonzero, legally.
3. Odd step1 has pending384/cursor1; resumed step2 has no pending/cursor0, no new
   collection/HPS score events. Sampler only advanced48 prompts for this pool.
   CPU tests prove exact state continuation with fake leaves; real GPU precision
   and restored optimizer/LR/RNG/EMA still need observation on this environment.
4. EMA hash stays unchanged through step2 and changes at8; parameter swap,
   checkpoint restoration and subsequent generation behave correctly.
5. Inspect heartbeat minima, exact Torch peak, host RSS, disk sizes and any
   service logs, including merge/checkpoint peaks. Confirm required headroom,
   not just successful process exit. Evaluation/collection interruption recovery
   uses receipts/chunks and physical attempt journals; do not delete them.

## Formal140 and recovery

After real acceptance and disk planning, either continue the validation trajectory
with its unchanged production semantics, or use a NEW output to restart from seed42.
No auto promotion and no silent background launch is performed. Example continuing
from the above step8 (use step2 instead if that was the last accepted point):

```bash
# Keep console log outside source, and choose a NEW console filename per attempt.
LOG=/mnt/data/qifan/runs/flash140-20261022/formal140-console-$(date +%Y%m%d-%H%M%S).log
nohup "$PY" -u scripts/launch_flash140.py run --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$OUT/checkpoints/step-8" >"$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!
printf 'pid=%s log=%s\n' "$LAUNCH_PID" "$LOG"
tail -f "$LOG"
```

A fresh formal run instead uses `run --output /mnt/data/qifan/runs/flash140-20261022/approved140-simplified`
without `--resume`; it reruns baseline evaluation and all140 updates. Only the
real terminal140 finalizer writes SUCCESS. Formal run is not considered started
by this handoff or by CPU tests.

The persistent global flock at
`/mnt/data/qifan/runs/flash140-20261022/.flash140-launch.lock` excludes concurrent
cooperating launches across different outputs/checkouts. Never unlink it or
kill a PID just because stale owner JSON exists. It cannot detect an unrelated
manual trainer that bypasses this entry; main must verify process/GPU ownership
before launching. `launcher-pid.json` and per-attempt `owner.json` record PID,
UID, source, commit, output, mode and start time; compare `/proc/PID/cmdline` and
process start time before signalling ONLY your process. Read-only inspection
also holds this lock and must not overlap another launcher.

Each `OUT/launcher/<time_ns>/` has `owner.json` (including exact import paths and
admission), rendered `launch.yaml`, `events.jsonl`, `resources.json`, durable
`checkpoint-N.json` (metrics/disk bytes) and `result.json`. Live stdout also shows
JSON events: heartbeat, evaluation video/receipt, collection chunk/generated/
durable/reused, HPS video start/complete, stage/commit and checkpoint. Media and
prompts are not retained in these logs. A long baseline evaluation is not a hung
training update; its events explicitly state that training has not started yet.

Fresh mode rejects ANY existing output. For a failure BEFORE the first complete
checkpoint, explicitly rerun the same mode/output with `--retry-incomplete` to
reuse evaluation receipts and durable collection chunks. After a complete
checkpoint, explicitly use `--resume OUT/checkpoints/step-N` selecting the latest
final checkpoint. Atomic orphan `step-N.tmp-*` directories are ignored and left
untouched, never treated as checkpoints. Damaged final checkpoints fail closed.
Unknown output ownership, changed Git/config identity, old schema1 pools,
finished outputs and attempts to overwrite newer checkpoint history are rejected.
Do not hot-change config, drop pending pools, reset counters or remove cache
identity journals to get past a failure. Use old code for historical checkpoints.

CPU evidence is exactly the existing seven-test file, run in the real Mac
candidate before each commit/push. It now also covers the checkpoint-observer
pause and inspection, launcher config rendering, admission boundaries, peak
safety, real flock exclusion, output ownership/recovery and progress events,
using explicit CPU fake leaves. None of this is evidence of a server GPU pass.
