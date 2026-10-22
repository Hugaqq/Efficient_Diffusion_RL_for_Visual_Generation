# User-approved SMALL40 — Git deploy, preflight, then background RUN40

This supersedes the immediate140/validate plan. The user already authorized this
new experiment. **Do not run140 validation or pause at1/2/8.** The main session
reports it verified and stopped owned PID704136; do not signal it again or stop,
move, restart or patch any reward service. Preserve all old outputs/worktrees.
This document is a local implementation handoff, **not GPU execution evidence**.
Use the full tested commit SHA from the coding result, not a package version.

## Fixed experiment

`experiments/flash_wan_capacity_20261022/small40-runtime.yaml` is explicitly
`training.flash_preset: small40`. The default `approved140-runtime.yaml` and
`workload-140.json` remain unchanged and separately selectable. No new training
scheduler, optimizer, reward implementation or service owner was added.

- Wan1.3B,480x832,81 decoded frames,15fps;20 train steps,CFG4.5,group8,
  BF16,LoRA16/32,lr1e-4,PPO clip.001,seed42,stage GPU text encoding,
  gradient checkpointing,forward4/replay4/decode1,KL0.
- Four prompts x8 =32 fixed-policy training videos, streamed in two16 chunks.
  Normalize rewards over **all32** before two16-row optimizer commits. No new
  generation/scoring between those commits.20 collections =40 commits =640
  logical training videos. Checkpoints every2 commits, no artificial pauses.
- EMA still updates every8 commits with the existing warmup decay. Evaluation
  uses EMA at **0,20,40**,16 heldout prompts x**20 ODE steps**,48 eval videos.
  The terminal40 evaluation runs explicitly in the existing terminal-checkpoint
  path before capturing terminal state and before SUCCESS. A failure cannot
  silently skip it or produce a successful terminal checkpoint.
- `data/flash_small40/evaluation-selection.json` freezes zero-based author test
  indices, exact prompts, four categories (four each), author SHA hashes and
  explicit seeds42000..42015. `evaluation-16.txt` is the same ordered selection.
  Selection was made without viewing SMALL40 results or using reward values.
  Full author train19700/test300 hashes are still checked, selected indices must
  match the300 source, and training overlap is rejected (trimmed exact strings).
  No external data download. The same per-prompt noise seed is reset at each
  round, independent of the step-specific sample/RPC identity.
- Every round saves16 H264 MP4s,15fps,exact81 decoded frames (Wan's21 latent
  frames are NOT the saved video length). File layout:
  `OUT/evaluations/step-{0,20,40}/video-NNN-seed-SEED.mp4`, accompanied by
  `video-NNN.json` with sample ID,prompt,seed,filename,SHA256,fps,finite HPS
  and all81 frame scores/top24 indices. `summary.json` contains means, ordered
  per-prompt records and paired differences against earlier rounds. These are
  descriptive measurements, **not a quality-improvement claim**.
- MP4 is closed/fsynced/atomically renamed before its atomic JSON score receipt.
  An interrupted physical generation/RPC is journaled separately and may retry;
  logical IDs remain stable. A receipt with a missing/corrupt MP4 is rejected,
  not silently counted. SUCCESS finalization also checks all48 MP4 hashes and
  finite receipts, even if earlier evaluations are marked completed. Evaluation
  restores EMA-swapped weights and all global RNG on success/failure and releases
  media after save/score. No generated training videos are retained by this change.

The full model/data/service/overlay paths remain the previous handoff's paths.
Private HPS18820 and its code/artifact identity are unchanged:
`4c831491cb7b413839ad07459f777bd674cbd59aadbc7d7d3e2ce25b11c993e9`.
The production scorer remains real HPS on all81 frames then top24 aggregation.

## Deploy exact Git commit (sole authorized deployment role)

Use the authorized `capacity_visualrl` browser bridge/terminal, not SSH. Only the
single deployment role operates it. Main must not concurrently send commands.
The deployment task's old140/validate mandate is superseded by this document.

```bash
: "${AGREED_SHA:?export the full tested SMALL40 SHA from the coding result}"
BASE=/home/ubuntu/qifan/VisualRL
SOURCE=/home/ubuntu/qifan/visualrl-flash140-20261022/small40-source
OUT=/mnt/data/qifan/runs/flash-small40-20261022/run-001
PY=/home/ubuntu/qifan/envs/visual-rl/bin/python
OVERLAY=/home/ubuntu/qifan/visualrl-flash140-20261022/dependency-overlay
hostname
nvidia-smi
ps -eo pid,lstart,args | grep -E 'launch_flash140|704136|hpsv3' | grep -v grep || true
git -C "$BASE" status --short --branch
git -C "$BASE" fetch origin main
git -C "$BASE" cat-file -e "$AGREED_SHA^{commit}"
# These must both be NEW. If either exists, inspect and preserve it; do NOT reset,
# clean, overwrite, reuse old140 output, or choose an untested hotfix.
test ! -e "$SOURCE" && test ! -e "$OUT"
git -C "$BASE" worktree add --detach "$SOURCE" "$AGREED_SHA"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$AGREED_SHA"
test -z "$(git status --porcelain --untracked-files=normal)"
export PYTHONPATH="$SOURCE:$OVERLAY"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "$OUT")"
```

Execute subsequent commands only if all prior checks passed. If the source
already exists, the deployment role must first verify clean exact SHA/import
paths and its ownership rather than running `worktree add` or overwriting it.
If output already exists, do not reuse it: report the collision for a new fresh
output decision. This launcher rejects SMALL40 resume/retry/validate; it cannot
accidentally resume the stopped140 pool. Framework CPU receipt recovery tests
are separate from this explicitly fresh launch policy.

## One preflight, then directly RUN40

```bash
PREFLIGHT=/mnt/data/qifan/runs/flash-small40-20261022/preflight-$(date +%Y%m%d-%H%M%S).log
"$PY" -u scripts/launch_flash140.py preflight --preset small40 \
  --expected-sha "$AGREED_SHA" --output "$OUT" >"$PREFLIGHT" 2>&1
PREFLIGHT_RC=$?
printf 'preflight_rc=%s log=%s\n' "$PREFLIGHT_RC" "$PREFLIGHT"
tail -n 50 "$PREFLIGHT"
# STOP on nonzero RC; do not launch after failed preflight.
test "$PREFLIGHT_RC" -eq 0
```

After actual preflight RC0, the user has already authorized background launch;
no additional approval or384/300 validation is required:

```bash
LOG=/mnt/data/qifan/runs/flash-small40-20261022/RUN40-console-$(date +%Y%m%d-%H%M%S).log
nohup "$PY" -u scripts/launch_flash140.py run --preset small40 \
  --expected-sha "$AGREED_SHA" --output "$OUT" >"$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!
printf 'RUN40 pid=%s log=%s output=%s sha=%s\n' "$LAUNCH_PID" "$LOG" "$OUT" "$AGREED_SHA"
sleep 20
ps -p "$LAUNCH_PID" -o pid,lstart,etime,args
readlink -f "/proc/$LAUNCH_PID/cwd"
tr '\0' ' ' < "/proc/$LAUNCH_PID/cmdline"; printf '\n'
cat "$OUT/launcher-pid.json"
tail -n 80 "$LOG"
nvidia-smi
git -C "$SOURCE" status --short --branch
```

Do not treat `test`/shell exit status as a scheduler: the deployment role must
observe actual success before sending the next command. The `run` invocation
rechecks normal ownership/resource/HPS invariants; no redundant bounded training
phase is inserted. Confirm process stays active through model loading and logs
`experiment_plan` with optimizer_target40,collection_rows32,rows_per_commit16,
training_videos640,evaluation_steps[0,20,40],evaluation_prompts16,
evaluation_sampling_steps20,save_evaluation_videos=true. Then observe actual
`evaluation_start` committed_steps0,prompts16,sampling_steps20 and continuing15s
resource heartbeats. Record PID/cmdline/cwd,clean exact Git SHA,actual imported
`visual_rl`/collection paths,log,attempt path and live resource evidence in
`outputs/flash140-deploy/SMALL40_REPORT.md`. Return the live process, do not wait
for40 completion and do not stop/restart it for a progress inquiry. A running
baseline means experiment started, **not** completed baseline or optimizer work.

## Existing safety and evidence boundaries

The global flock is still
`/mnt/data/qifan/runs/flash140-20261022/.flash140-launch.lock`, shared by both
presets/worktrees/outputs. Never unlink its inode or kill another owner. Shared
GPU admission remains **68.03GiB free** = prior batch4 reservation52.03 +16 margin;
host88GiB = provisional workload64 +24 margin. The model's batch4 memory cost
has NOT shrunk just because the collection is32 rather than384. These budgets
are not newly measured peaks. During execution the existing monitor enforces
>=16GiB GPU free,>=24GiB available host,>=64GiB disk and Torch reservation peak
within the admitted GPU budget. It signals only this owned launcher on breach.
No new heavyweight admission layer, service change, or model-quality relaxation.
SMALL40 admission reserves64GiB disk (all20 even/model checkpoints retained;
no pending odd384 pools); original140 formal run retains its1TiB floor. Observe
actual disk/checkpoint/media growth and preserve safety margin, never delete
other runs to make room.

Each `OUT/launcher/<time_ns>/` contains owner/import/admission evidence, rendered
launch.yaml,events.jsonl,resources.json,non-terminal checkpoint metrics/bytes
and eventual result.json. Owner filenames and the logger keep historical
`flash140` names for compatibility; the explicit `preset` and rendered config
establish SMALL40 identity. result.json is absent while running; training success
requires the real terminal40 checkpoint, all three evaluation rounds and SUCCESS.
Resources are sampled, not a proof of every transient allocation or other job's
headroom. No GPU speed,precision,resource peak,MP4 backend or HPS quality result
is established by local tests.

## Local CPU gate and code paths

Run only the original seven test functions (extended subcases), not a substitute
full suite or parameterized expansion:

```bash
cd outputs/flash140-coding
/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q
```

The SMALL40 integration uses actual task-local author `Flash-GRPO/dataset/video`
data hashes, real typed dispatch/controller/PPO/checkpoints and40 commits with
explicit fake tiny model ports/scorer/MP4 encoder. It proves terminal40 failure
cannot write SUCCESS, checkpoint38 recovery reuses3 terminal receipts and scores
only13 remaining, same seeds,81-frame contracts,48 saved-video associations,
paired means,missing-video rejection,EMA/RNG restoration and media release. The
original384/192,300x50,cache,schema2,step0/PPO and exact-resume tests remain.
It is CPU fake-leaf evidence, **not** successful real ffmpeg/GPU/HTTP execution.

Implementation: `visual_rl/algorithms/trainer/flash_presets.py`, typed
`composition/config/specs.py`, existing `runtime/algorithm_binding.py`,
`trainer/collection.py`, `trainer/evaluation.py`, existing terminal
`runtime/checkpoint_binding.py` and `scripts/launch_flash140.py`.
All source repairs must again be Mac edit -> same7 tests -> commit/push -> server
Git fetch exact SHA. No server patching or package installation by this launcher.
