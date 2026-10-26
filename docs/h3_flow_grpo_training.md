# H3 direct PyTorch joint Flow-GRPO diagnostic

**Fresh-only, one then five real updates; NOT normal-resolution training capacity.**
The original successful inference source/results, Wan/SMALL40/RL5, reward service,
and review snapshot are unchanged. No vLLM/SGLang/Comfy, weight-content inventories,
mandatory conversion receipts, quantization, approximate cache or external trainer.

## Current GPU policy — user override after the first real attempt

**No GPU reserve on H3 training.** The training CLI, initial checks, event/15s
heartbeat checks and every encoder/DiT/decoder stage use `gpu_memory_policy='oom'`.
No16GiB free floor, no32GiB stage GPU admission, no max-reserved limit. Telemetry
is real and retained; native CUDA OOM still fails with full evidence. Host-stage
component+shard+24GiB admission, host24GiB/disk64GiB floors, locks and services
remain protected. Wan/SMALL40 and H3 inference defaults are NOT changed.
Owner/environment/SUCCESS/FAILED explicitly record `gpu_memory_policy='oom'`,
`gpu_reserve_enabled=false`, `gpu_reserved_limit_gib=null`.

Historical4cd18b8 diagnostic-one-001 loaded real encoder/DiT/PEFT but stopped at
GPUfree15.963806GiB<16GiB during first rollout, zero updates. This was NOT CUDA OOM
or backward acceptance. Preserve that output; this authorized policy change uses
new `diagnostic-one-oom-001` and exactly the same diagnostic config.

## Read this call chain first

```
python -m visual_rl.train_h3
  -> existing shared launch flock / Git+import checks / resource monitor
  -> H3TrainingAdapter.encode (frozen native Qwen hidden_states[50])
  -> prepare_policy (native DiT + PEFT -> H3PreparedRoot -> AdamW/LambdaLR)
  -> H3GRPODiagnostic (the H3-specific algorithm, not another service scheduler)
      group2, same prompt, distinct seeds, B1 serial forwards:
        prepare_latents/layout -> root.__call__ -> paired Euler/Flow-SDE
        save selected current/action/conditions/layout/clocks/masks/old density
      park SAME root/Parameter objects on CPU; decode both AV branches
      export/full decode original MP4+float WAV; explicit H3->HPS adapter; real RPC
      normalize true group rewards (degenerate group FAILS, no update)
      same root -> fixed-action replay -> existing VisualRL clipped_surrogate
      accumulate two B1 losses / 2 -> backward inside root residency
      finite/nonzero LoRA gradients -> clip -> real AdamW.step -> LR -> zero_grad
      finite actual LoRA change -> atomic step-N/training.pt -> durable counter
  -> close model/reward; final resource check -> SUCCESS.json
```

Source map: `models/h3_training.py` owns model/trainable lifetime and identity;
`algorithms/dynamics/h3_joint.py` owns paired sampling/evaluation math;
`algorithms/trainer/h3.py` owns explicit joint records, group/reward/PPO/checkpoint.
`models/h3_components.py` adds an explicit stage-policy seam; inference retains
its original reserve default, while only the training subclass selects 'oom'. No
packed AV is disguised as Wan's single-tensor `ModelInput`/trajectory contract.

## Exact prospective diagnostic (fixed BEFORE the first server attempt)

`configs/h3/training-diagnostic.json`:
- width96 x height64, 124 frames, 24fps; native stereo32kHz, no reference/prefix.
  This is an extremely small **diagnostic**, not a normal sample/quality claim.
  Spatial32 multiples follow native patch/layout. Fixed upstream modular tests
  exercise32x32/124; real pretrained96x64 encode/decode still needs GPU validation.
- 40 sigma grid points including terminal zero =39 joint predictions per sample.
  video shift12, audio shift3; both full branches always retained.
- One group of2 same-prompt samples per update, seeds42000/42001; next group's
  seeds42002/42003 etc. Distinct initial and transition generators. Batch1 only.
- Only zero-based transition **20** is stochastic/trainable, noise0.7. Other38
  transitions are native deterministic Euler; they have **no recorded density**.
  Index0 (sigma1) and last (next sigma0) explicitly cannot enter RL masks.
- LoRA rank4/alpha8/dropout0 on all native attention `to_q/to_k/to_v/to_out.0`
  (joint blocks and token-refiner); encoder/VAEs/base frozen. FP32 LoRA storage,
  original BF16 base with native FP32 patch/time/output projections retained.
- LR1e-4, AdamW decay0, constant LR, clip range0.001, max gradient norm1;
  one update per FRESH group (no multi-epoch/pending-pool reuse). KL0, no reference
  model or EMA claim. Five updates mean five newly collected groups.
- Non-reentrant native gradient checkpointing enabled for DiT and token refiner;
  exact token-local FF chunks1024; no inference group/CPU offload hooks on DiT.
  DiT fully resides on GPU for collection and replay/backward/optimizer. Park the
  entire root on CPU only after all collection graphs are gone, for AV decoding;
  PyTorch2.10's documented `swap_module_params_on_conversion` is enabled only
  around `.to(device)` and restored in finally; optimizer Parameter identities
  are checked after every move. No swap is allowed with a live backward graph. Small Adam
  state can stay on GPU while parameters park; next optimization runs on GPU.
- **No FP16 or scaler**: native BF16 compute/FP32 optimized storage with ordinary
  PyTorch accumulation, one optimizer call after both graphs; no double scaling.
  Current/collection both eval mode; grad enabled only for replay, never train().

One fresh `--updates 1` must pass first. Only then, with resources safe, one fresh
`--updates 5` at identical settings is allowed. This is NOT resume: the five-update
run starts its own initial LoRA and explicitly collects five groups. No CLI knob
silently reduces dimensions, audio, group, timesteps, weights or base model after
a failure. Long training and normal1344x768 capacity are not implemented/claimed.

## Probability definition (review before using the metrics)

From fixed Flow-Factory3dd5ca4d `scheduler/minimax_h3.py:490-659`, Flow-SDE only:
for each modality, `s=sigma`, `dt=next_sigma-s < 0`, native data-ward velocity `v`,

```
g = noise * sqrt(s / (1 - where(s==1, sigma_grid[1], s)))
mu = x*(1 + g**2/(2*s)*dt) - v*(1 + g**2*(1-s)/(2*s))*dt
std_effective = g * sqrt(-dt)
action = mu + std_effective * epsilon
scalar_logp = -(action.detach()-mu)**2/(2*std_effective**2)
              - log(std_effective) - 0.5*log(2*pi)
Lv = mean(scalar_logp_video[generated_mask]); La = mean(audio[generated_mask])
L = (Nv*Lv + Na*La) / (Nv+Na)
```

This is FF's **scalar-count normalized surrogate**, not unreduced joint log density
and not a proven optimal weighting. No0.5/0.5 substitution. Mean/std/action are
FP32. Each modality has its OWN sigma/next_sigma/timestep. Stochastic interior
steps only: `0<next_sigma<sigma<1`, positive finite noise/std. Endpoints/ODE have
no fake logprob0. T2VA packed layout must exactly partition text/video/audio;
no condition/padding AV rows are accepted. Per-scalar bool masks/counts are saved;
all AV scalars are generated/scored in this T2VA baseline; text never contributes.
For this diagnostic: video37x2x3 packed rows x96 =21312 scalars; stereo audio
414 rows x32 =13248 scalars. Actual runtime counts, not these constants, weight L.

Collection/replay call the SAME helper. Replay passes saved action and cannot
sample RNG. Save detached oldLv/La/counts/aggregate, original B1 forward partition,
state, action, prompt/layout, clocks, sigma_max/noise/masks and generator-after.
Initial generator seed is request.seed; SDE generator seed is request.seed+1000000. Per-modality and
aggregate matching tolerance1e-4 is fixed, not relaxed to conceal failure.
Each fresh group is collected and updated exactly once, so parity is appropriate
for every such group. Legitimate later nonunit PPO ratios are not generally an
error; reusing this routine for multiple policy epochs is NOT supported. The
canonical PPO implementation is unchanged and retains exp(new-old)/clipping.

Source/provenance references are code, not weight integrity checks:
- Diffusers `aaf57d03a3edfbd9de457d2872417a92f149a91d`, H3 model forward
  `torch.is_grad_enabled() and gradient_checkpointing` in both block loops;
  ModelMixin non-reentrant default checkpoint implementation.
- PEFT0.18.1 LoraConfig + Diffusers PeftAdapterMixin.add_adapter ->
  inject_adapter_in_model; PEFT Linear.forward casts to FP32 LoRA and back to
  original result dtype. Small fetched sources in integration/reference.
- Flow-Factory `_common.py:403-497` weights per-component means by generated
  scalar counts; scheduler formula adapted with Apache2 attribution in source.
  No entire adapter/trainer copied. Existing `docs/h3_pytorch_sources.md` applies.

## Honest video-only reward definition (original service UNCHANGED)

`H3VideoReward` uses existing `Hpsv3VideoClient`/JPEG transport and `/healthz`.
Preserve service identity
`4c831491cb7b413839ad07459f777bd674cbd59aadbc7d7d3e2ce25b11c993e9` and strict
81x480x832/top24 contract. We do NOT send124x64x96 or claim all124-frame scoring.

1. On ORIGINAL CPU uint8 RGB FHWC, choose81 indices:
   `torch.linspace(0,F-1,81,dtype=float64).round().long()` (round ties-to-even).
2. PIL RGB BICUBIC resize each entire chosen frame to832x480; aspect ratio is
   intentionally warped, not cropped. No alteration of original video/audio.
3. Existing JPEG encoder transmits all81 adapted frames; original scorer returns
   finite top24-of81 aggregation. Store original/adapted shapes, source indices,
   interpolation, scorer frame indices/scores/selection and sample identity.

This is a defined diagnostic video reward, probably poor at tiny resolution.
It drives the JOINT surrogate, but provides **no audio-specific quality reward**.
Equal scores/nonfinite rewards cause honest failure before optimizer.step.
No reward perturbations, dummy images, fallback score, service restarts or /device.
Original MP4/WAV and complete AV-decode/bytes/SHA receipts live in each group.

## Independent server deployment (Experimenter only)

Use the exact tested SHA supplied in `coder-handoff.md` as `AGREED_SHA`; do not
use package version as identity. Before any GPU command freshly inspect existing
inference three success/results, active jobs, safety and HPS original ready
identity. Do not rerun Wan5/SMALL40/accepted inference or stop unrelated services.

```bash
export AGREED_SHA=REPLACE_WITH_FULL_TESTED_SHA
SOURCE=/home/ubuntu/qifan/visualrl-h3-training-20261026/source
# Existing TRAINING worktree only, after prior owned PIDs have exited and clean
# status is verified. Do NOT switch the successful inference checkout.
git -C "$SOURCE" status --short --branch
git -C "$SOURCE" fetch origin h3-pytorch-training
git -C "$SOURCE" switch --detach "$AGREED_SHA"
git -C "$SOURCE" rev-parse HEAD
git -C "$SOURCE" status --short --branch
cd "$SOURCE"
```

The independent training environment already passed installation/pip check and
PEFT0.18.1/CUDA import tests in the first attempt. **Reuse it unmodified: no
reinstall, model download or reconversion.** Verify actual source/module paths:

```bash
mkdir -p /mnt/data/qifan/runs/visualrl-h3-training-20261026
PY=/mnt/data/qifan/envs/visualrl-h3-train/bin/python
"$PY" -m pip check
"$PY" -m pip freeze > /mnt/data/qifan/runs/visualrl-h3-training-20261026/pip-freeze-no-gpu-reserve.txt
export PYTHONPATH="$SOURCE" CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$PY" -c 'import torch,peft,visual_rl; from visual_rl.models.h3_components import audit_runtime; print(visual_rl.__file__,peft.__file__,audit_runtime()); print(torch.__version__)'
RAW=/mnt/data/qifan/models/h3-native/hub-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
DERIVED=/mnt/data/qifan/models/h3-torch/FL2VA-diffusers-aaf57d03
OUT=/mnt/data/qifan/runs/visualrl-h3-training-20261026/diagnostic-one-oom-001
mkdir -p "$(dirname "$OUT")"
export H3_RC="$OUT.rc"
# A new log/RC wrapper; once only, record $!, child PID/cwd/argv and ownership.
nohup bash -c '"$@"; rc=$?; printf "%s\n" "$rc" > "${H3_RC}"; exit "$rc"' h3 \
  "$PY" -u -m visual_rl.train_h3 --expected-sha "$AGREED_SHA" \
  --model-root "$RAW" --converted-root "$DERIVED" \
  --model-revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --output "$OUT" --updates 1 >"$OUT.console.log" 2>&1 < /dev/null &
```

Do not literally execute placeholders. Record the actual background PID from
`$!` immediately; a launched process is not training acceptance. Reuse existing raw/derived
assets: **no download, conversion or content integrity scan**. Actual loading
key/shape errors are propagated. Prior real environment/model preparation is
not backward/update proof; if API fails preserve exact error and return it.
No server source/package monkeypatch. No automatic reinstallation on retry.

Only after one real update acceptance and fresh resource checks, use a distinct
`diagnostic-five-oom-001` OUT/RC/log and exactly the same command with `--updates 5`.
Do not pass `--resume` (unsupported), alter config, or claim the first run's
checkpoint resumed. On bridge failure read existing PID/log first, never replay
nohup. The global flock prevents coexistence with other owned Flash/H3 launches.

## Evidence / safety / limits

- Full DiT GPU residency/model preparation was observed in the historical first
  attempt; full rollout/backward/optimizer capacity remains **unverified**. Host
  admission uses largest live component + loading shard +24GiB, not sum134GiB.
  Training now has **no GPU reserve or max-reserved rejection**, per user override;
  real telemetry and OOM propagation remain. Host24GiB/disk64GiB checks run at15s
  and explicit stage boundaries; sampled checks do not prove transient minima.
  No automatic precision/model/geometry change or host/disk guard removal.
- Initial stage admission rejection is **not CUDA OOM**. Native OOM or host safety
  failure must immediately return config/phase/resource snapshots/traceback/RC
  and notify main/user. Keep the failed output, do not shrink and retry. No HPS
  outage is required/authorized by this implementation; reward must remain real.
- `owner.json`, `environment.json`, `events.jsonl`, `resources.json` track source,
  modules, phase, true joint steps, optimizer attempts and capacities. `FAILED.json`
  includes full traceback, resources, last phase, durable count, diagnostic config
  and capacity-notification flag. Do not treat CLI launch/CUDA allocation as RL.
- Per group: `trajectory.pt` is a trusted torch diagnostic record (not external
  untrusted input); `rewards.json` has real scorer/AV receipts; `update.json` has
  each modality/aggregate old/current/count/ratio/parity, gradients, actual LoRA
  before/after hash AND nonzero delta L2/max. Hashes concern small LoRA update
  evidence/output media/source code only, never full base weights.
- `step-N/training.pt`: atomic file+directory publication with fsync, real LoRA,
  AdamW/LR, Python/NumPy/Torch/CUDA RNG, step/config/source/revision/audit.
  No pending group; **resume is explicitly unsupported**. No exact-restart claim,
  base-weight export, checkpoint editing, cross-code conversion or strict=False.
  Failure after optimizer attempt is fatal; in-memory retry is forbidden. Durable
  counters advance after checkpoint publication, not merely after backward.
- Accept1 only if real pretrained joint rollouts, complete AV decode, matching
  per-modality+aggregate fixed-action density, distinct finite real rewards,
  nonzero finite selected LoRA gradients, actual finite nonzero Parameter delta,
  durable checkpoint1, RC0/SUCCESS all exist. Then inspect five identical-setting
  real commits/checkpoints1..5; no normal1344x768 training capacity/quality claim.
- CPU tests use explicit fake heavy leaves and real small-tensor math/autograd;
  they are NOT real model/PEFT CUDA/offload/reward/gradient-capacity evidence.
