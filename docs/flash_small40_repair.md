# SMALL40: audited launcher-only repair; resume original run-002 step2

This mandate supersedes the fresh-only policy in `flash_small40_launch.md`.
**Do not reset, create a new output, warm-start LoRA only, edit checkpoints or
restart baseline.** Target remains40. This is Mac code/CPU handoff, not a report
of real GPU recovery. Sole deployment role uses `capacity_visualrl`; coding
never accesses the server. Preserve all other processes/worktrees/services.

## Narrow repair and compatibility proof

Historical source/owner: `00cee786df783f36cb254a9a666a48a5952dc0a3`.
Original output: `/mnt/data/qifan/runs/flash-small40-20261022/run-002`.
Explicit checkpoint: `OUT/checkpoints/step-2`. User reports complete step2,
16 completed baseline videos, first32 training rewards and two commits, then
HTTP503 after the next16 generated videos. There was no observed CUDA OOM.
Allocator caching is a hypothesis, not a proven leak/root cause.

- `scripts/flash_small40_repair.py` uses the existing default-controller
  `reward_resource_factory` seam. The default factory still acquires/checks the
  real HPS client and returns the same binding facts. A newly constructed owned
  adapter (HPS subtype required by evaluation) delegates scoring/transport,
  validation, activation and close to that client. No live prepared module or
  deep function is patched. No replacement rewards, HTTP retries or service
  commands. All81 frames/top24 remain genuine.
- At the synchronous reward leaf boundary, rollout has returned and exited its
  execution context; reward media is detached CPU data. One `empty_cache()` per
  generated collection chunk (not per training step/slot/frame), or per newly
  generated eval video, releases **only unused trainer allocator cache**. No GC,
  parameter offload, gradient mutation, owner/context change or loop rewrite.
  Synchronize and log `reward_cuda_handoff_before` and
  `reward_cuda_handoff_after` with `phase`, `allocated_bytes`, `reserved_bytes`,
  `card_free_bytes`, `card_total_bytes`. No fake reclamation guarantee: live
  allocations stay live; other processes can change card free independently.
- `visual_rl/`, **all** `experiments/` and `data/` Git objects remain byte-for-byte
  unchanged from the historical base. Thus the existing actual code artifact
  (root `visual_rl`) and config identity can remain equal. The new launcher
  integration is explicitly audited, not disguised as old launcher code.
- `--upgrade-launcher-from 00cee...` is bounded to the exact original output and
  its latest step2, exact old owner SHA/config hash, base ancestry, allowlisted
  launcher/helper/test/doc/AGENTS paths, equal runtime/config/data tree objects,
  and actual config file blob/hash equality. Arbitrary source/schema/config
  changes are not accepted. Main additionally checks full clean HEAD and actual
  imported paths; no package-version inference.
- Preflight authorization is read-only. RUN under the existing global flock
  verifies latest path + latest.json + complete checkpoint tree/schema2,
  finite CPU model/EMA, even2..38 cursor/pool/evaluation state; then atomically
  publishes ONE `flash140-owner.json` containing the new identity plus full
  `launcher_upgrade` proof. Old checkpoint files are never edited. Audit denotes
  launcher authorization, **not** completed GPU/full-contract validation.
  The unchanged production preflight/restore still requires the actual recipe,
  prepared and full checkpoint contracts (model/optimizer/LR, RNG/sampler,
  EMA/cursors, reward/execution/data identities). No IDs are substituted, no
  strictness disabled. A genuine incompatibility fails normally, not fresh.
  An interrupted launch may leave the audited owner but cannot alter step2.
  Reusing that audit requires recomputing the same proof at the same new SHA.
- Later explicit latest resumes of the same tested owner are supported; no
  SMALL40 retry-without-checkpoint or bounded validation. Original140 behavior
  remains unchanged. For original run-002, recovery requires explicit
  `--gpu-memory-policy oom`: no16GiB launcher GPU floor/peak abort is restored.
  Host/disk/invalid-telemetry checks and real CUDA OOM still fail normally.
  Independent HPS admission40GiB/runtime16GiB guards are NOT changed.

Baseline cursor `[0]` in step2 skips the already completed16 evaluations, using
existing receipts/videos. Sampling restarts the next logical collection at
step2, not0. Failed unscored16 media were not a durable chunk: regeneration may
repeat that logical chunk; `physical-attempts/collection-2/rows-0.json` counts
physical attempts separately. Durable scored chunks, if any, reuse the original
weight/recipe/row identity and saved generator state. No extra optimizer steps.
Specs stay480x832x81, train20/CFG4.5/BF16/group8/LoRA16/32/lr1e-4/clip.001/
seed42/forward4/replay4/decode1,32->16+16/full32 std,40/640,EMA every8,
eval0/20/40 fixed16 seeds42000..42015 at20 ODE steps,48 MP4/finite receipts
required before SUCCESS. Original140 config/manifest are untouched.

## Server commands (only deployment role; exact tested SHA from coding result)

Use the authorized bridge, not SSH. First verify no owned run-002 training PID
is alive and private HPS18820 is ready; no killing/moving services. Do not act
on historical PID718759 without checking actual ownership/liveness. If there is
another active training owner, report the conflict; do not launch a duplicate.

```bash
set -euo pipefail
: "${AGREED_SHA:?export the full tested repair SHA from coding handoff}"
OLD=00cee786df783f36cb254a9a666a48a5952dc0a3
BASE=/home/ubuntu/qifan/VisualRL
SOURCE=/home/ubuntu/qifan/visualrl-flash140-20261022/small40-repair-source
OUT=/mnt/data/qifan/runs/flash-small40-20261022/run-002
CKPT="$OUT/checkpoints/step-2"
PY=/home/ubuntu/qifan/envs/visual-rl/bin/python
OVERLAY=/home/ubuntu/qifan/visualrl-flash140-20261022/dependency-overlay
hostname
ps -eo pid,lstart,args | grep -E 'launch_flash140|718759|hpsv3' | grep -v grep || true
nvidia-smi
curl --fail --silent http://127.0.0.1:18820/healthz; printf '\n'
git -C "$BASE" status --short --branch
git -C "$BASE" fetch origin main
git -C "$BASE" cat-file -e "$AGREED_SHA^{commit}"
test ! -e "$SOURCE"
test -d "$CKPT"
git -C "$BASE" worktree add --detach "$SOURCE" "$AGREED_SHA"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$AGREED_SHA"
test -z "$(git status --porcelain --untracked-files=normal)"
# Independently display the runtime/config/data equality used by the CLI proof.
git diff --exit-code "$OLD" "$AGREED_SHA" -- visual_rl experiments data
git diff --name-only "$OLD" "$AGREED_SHA"
export PYTHONPATH="$SOURCE:$OVERLAY" CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
"$PY" -c 'import visual_rl; from visual_rl.algorithms.trainer import collection; print(visual_rl.__file__); print(collection.__file__)'
PREFLIGHT="$(dirname "$OUT")/repair-step2-preflight-$(date +%Y%m%d-%H%M%S).log"
"$PY" -u scripts/launch_flash140.py preflight --preset small40 \
  --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$CKPT" \
  --gpu-memory-policy oom --upgrade-launcher-from "$OLD" >"$PREFLIGHT" 2>&1
printf 'preflight_rc=0 log=%s\n' "$PREFLIGHT"
tail -n 60 "$PREFLIGHT"
```

If source already exists, preserve it: verify exact clean SHA/import paths
rather than overwrite/add blindly. Any nonzero preflight stops the command
sequence; preserve logs and report. No fallback output, checkpoint edit or
server hotfix. After actual RC0, launch directly (no1/2/8 validation pauses):

```bash
LOG="$(dirname "$OUT")/RUN40-repair-step2-console-$(date +%Y%m%d-%H%M%S).log"
nohup "$PY" -u scripts/launch_flash140.py run --preset small40 \
  --expected-sha "$AGREED_SHA" --output "$OUT" --resume "$CKPT" \
  --gpu-memory-policy oom --upgrade-launcher-from "$OLD" >"$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!
printf 'pid=%s log=%s output=%s sha=%s\n' "$LAUNCH_PID" "$LOG" "$OUT" "$AGREED_SHA"
sleep 20
ps -p "$LAUNCH_PID" -o pid,lstart,etime,args
readlink -f "/proc/$LAUNCH_PID/cwd"
tr '\0' ' ' < "/proc/$LAUNCH_PID/cmdline"; printf '\n'
cat "$OUT/flash140-owner.json"
cat "$OUT/launcher-pid.json"
tail -n 100 "$LOG"
git status --short --branch
nvidia-smi
```

## Actual acceptance and failure boundary

Return live PID/cmdline/cwd, exact Git/import identity, console/attempt paths,
owner audit, rendered resume path + target40 + oom and sustained15s heartbeats.
`resume_checkpoint_validated` reports committed2/evaluations[0], but is only
read-only checkpoint inspection; prove actual restore by subsequent
`training_stage_start` / `collection_start` at optimizer_step2 (not step0), with no new
`evaluation_start` at0 and unchanged baseline receipts. Observe the first
post-generation before/after cache records and **actual**
`hpsv3_video_complete` finite reward, not merely a video_start or HTTP health.
Generation can take~25min; leave it running while observing. Record real
allocated/reserved/cardfree values; don't infer release from CPU fake counters.
If503/OOM/contract failure persists, preserve exact error/telemetry and return
blocked; no automatic restart, fresh run, fake reward or safety bypass.

Report `outputs/flash140-deploy/SMALL40_REPAIR_REPORT.md` and evidence, leave
active process in background after the first genuine score/cache handoff.
Do not cancel/pause on progress inquiries. No claim of completed40 or quality
from this recovery milestone. Only the main session updates the external daily
Action1 log; coding/deployment scoped permissions do not include that file.

## Mac gate and verification boundaries

Same seven top-level CPU functions, extended subcases, not a full-suite substitute:

```bash
cd outputs/flash140-coding
/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q
```

Coverage retains all prior384/PPO/EMA/schema/cache/300x50/SMALL40 tests. New
subcases test managed-context exit -> no-grad cache handoff -> original scorer,
actual adapter ownership/close, unchanged checkpoint states versus no adapter,
ordinary reward failure propagation, oom low-free policy, strict explicit
latest resume and code/recipe mismatch rejection, bounded launcher proof and
atomic audit roundtrip/config/runtime/ancestry rejection. Real CPU controller
SMALL40 pauses at durable2 ONLY as a test fault, resumes without baseline
re-score or receipt changes, then matches continuous40 model/optimizer/LR/RNG/
sampler/EMA state exactly; existing terminal40 failure/38 receipt retry remains.
Heavy model/scorer/MP4/CUDA telemetry leaves are fake in CPU tests. No real GPU
cache release, HTTP503 remedy, ffmpeg or server compatibility success is claimed
before deployment evidence. Commit/push follows this exact seven-test gate.
