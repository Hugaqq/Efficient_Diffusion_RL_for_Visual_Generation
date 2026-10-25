# H3 PyTorch — executable handoff (supersedes earlier engine instructions)

**No H3 GPU success claimed.** The tested Mac candidate provides component code
and CPU evidence; Experimenter must validate dependencies, raw→derived conversion,
normal generation, AV decode, resource peaks and service restoration.
Never restart the already completed Wan5 or SMALL40.

## Computation boundary

- `models/h3.py`: raw official inventory, request, joint CPU AV result validation.
- `models/h3_components.py`: known HF Qwen3-VL encoder, native Diffusers DiT and
  video/audio VAEs. Public ports are `encode`, `prepare_latents`, `predict`, `decode`.
- `algorithms/rollout/h3.py:sample_h3`: explicit joint video/audio sampling loop.
  One forward per step returns both velocities; independently shifted clocks,
  positive data-ward velocity, FP32 transition state. No pipeline invocation,
  engine registry, HTTP, subprocess generator, optimizer, LoRA or RL probabilities.
- `infer_h3.py`: Git/import identity, offline gates, shared flock,15s resource
  heartbeat, fresh outputs, overall SUCCESS only after close and final checks.
- Native checkpoint is CFG-distilled: one conditional pass, no negative branch.
  No added Turbo/distillation/quantization/cache approximation.

Current supported task is **text-only T2VA**, B=1, on official FL2VA weights.
Image/reference requests fail explicitly; they are not silently converted to text.
H3 RL is not implemented. Wan training and its numerical/restore paths are unchanged.

### Schedule is explicit, not the old engine's RES schedule

Use native Diffusers H3 baseline **Euler eta=0**,40 sigma grid points INCLUDING
terminal zero, hence **39 joint forwards**. `sigma=s*b/(1+(s-1)*b)` with
`b=linspace(1,0,40)`, shifts video12/audio3. Model time `t=1-sigma`.
`x0=x+(1-t)*v`, then `x_next=(sigma_next/sigma)*x+(1-sigma_next/sigma)*x0`.
The sigma→time→sigma float32 round trip in x0 follows the pinned implementation.
There is no extra appended zero or forward at sigma0. This is not20 RES steps.
Normal geometry remains1344×768×124 at24fps; audio40Hz latent clock gives207
latents/channel →165600 samples/channel at32kHz. Original prompts/seeds42009,
42012,42013 remain byte-identical. `--diagnostic` uses384×224×107 and is NOT
normal acceptance. Do not publish diagnostic output as the requested result.

## Fixed source and original assets

See [provenance](h3_pytorch_sources.md). Diffusers fixed commit
`aaf57d03a3edfbd9de457d2872417a92f149a91d`, Transformers5.19.0.
Actual installed H3 source hashes and live imports are checked, not just versions.
Runtime refuses installed/imported vllm, vllm_omni, sglang or Comfy packages.
No checkpoint-local Python is imported (`trust_remote_code=False`). No model
Python-review digest option is needed: native torch classes replace custom wrappers.
Original Python files remain inventoried and preserved but inert.

Raw root (already downloaded; **do not redownload134GiB**):

```bash
export MODEL=/mnt/data/qifan/models/h3-native/hub-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export REV=42ed227ee7df40d41602854ae760620d6eb651fe
```

It contains `FL2VA/{transformer,text_encoder,video_vae,audio_vae,tokenizer,processor}`.
Video VAE source is `video_vae/source/model.safetensors`, selected by its config,
not a guessed flat file.81 original files were verified in the preceding task;
that verification is asset evidence, NOT PyTorch generation evidence.
Quantized/pruned/AWQ/NVFP4 files remain unsupported.

Raw DiT/VAE keys differ from Diffusers. Preparation uses the **reviewed SHA-checked
upstream streaming converter**, not `from_pretrained(raw DiT)` and not strict=False.
It deinterleaves per-head QKV before splitting q/k/v, swaps gate/value FF halves,
preserves FP32 patch/time/output islands, drops only the documented recomputed
RoPE buffer and unused VAE mask token, verifies full key plans, and writes separate
derived components. Audio VAE keeps its exact state tree. Original raw files stay
unchanged. Tokenizer and Qwen weights continue loading from the original snapshot.

## Server Git and isolated environment

Only after Mac tests+push, set the **exact SHA from coder-handoff.md**. Preserve
all old worktrees/results/services. Check unknown GPU processes and ownership first.
Do not reuse cancelled inference PIDs or environments. No server source patches.

```bash
export AGREED_SHA='<40-character tested SHA from coder-handoff.md>'
export SOURCE=/home/ubuntu/qifan/visualrl-readable-h3-20261024/source
# Only after checking this task checkout is clean and has no active old process:
git -C "$SOURCE" fetch origin h3-readable-integration
git -C "$SOURCE" switch --detach "$AGREED_SHA"
git -C "$SOURCE" rev-parse HEAD
git -C "$SOURCE" status --short --branch
```

Prepare a NEW environment, without the old Wan overlay or SG LD_LIBRARY_PATH.
Do not install the project's broad training extras or upgrade a shared venv.
The following dependency pins were source-reviewed, **not installed/tested on Mac**;
record real resolver/pip/CUDA outcomes, return incompatibilities rather than patching.

```bash
python3.12 -m venv /mnt/data/qifan/envs/visualrl-h3-torch
export PY=/mnt/data/qifan/envs/visualrl-h3-torch/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r "$SOURCE/configs/h3/requirements-torch.txt"
"$PY" -m pip check
mkdir -p /mnt/data/qifan/runs/visualrl-h3-torch-20261025
"$PY" -m pip freeze > /mnt/data/qifan/runs/visualrl-h3-torch-20261025/freeze.txt
export PYTHONPATH="$SOURCE" CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$SOURCE"
"$PY" -c 'import torch, visual_rl; from visual_rl.models.h3_components import audit_runtime; print(visual_rl.__file__); print(audit_runtime()); print(torch.cuda.get_device_name()); print((torch.ones(2,device="cuda")+1).cpu())'
```

Prepare the exact converter source (plain component library, not an engine):

```bash
export DIFF=/mnt/data/qifan/src/diffusers-h3-aaf57d03
# If absent, create once. If already present, inspect HEAD/dirty; do not overwrite.
git clone https://github.com/huggingface/diffusers.git "$DIFF"
git -C "$DIFF" checkout --detach aaf57d03a3edfbd9de457d2872417a92f149a91d
git -C "$DIFF" status --short --branch
export DERIVED=/mnt/data/qifan/models/h3-torch/FL2VA-diffusers-aaf57d03
```

Run conversion as one owned background command with log/RC, not on an interactive
foreground shell that requires polling input. It may take minutes; check existing
PID/output before any repeat. `*.partial` is preserved on failure, never reused.

```bash
mkdir -p /mnt/data/qifan/models/h3-torch
export RUNS=/mnt/data/qifan/runs/visualrl-h3-torch-20261025
LOG="$RUNS/convert.log"
nohup bash -c '"$PY" -u -m scripts.prepare_h3_torch_assets --model-root "$MODEL" --model-revision "$REV" --diffusers-source "$DIFF" --output "$DERIVED"; rc=$?; printf "%s\n" "$rc" > "$RUNS/convert.rc"; exit "$rc"' >"$LOG" 2>&1 < /dev/null &
printf 'WRAPPER_PID=%s LOG=%s\n' "$!" "$LOG"
# Record wrapper/child cmdline and log immediately. Confirm convert.rc=0, final
# conversion.json and byte hashes before moving to inference. No success by PID alone.
```

The converter itself logs key counts and fails on unexpected/missing keys/dtypes.
At failure return code/problem to Coder. No installed-package monkeypatches.
`conversion.json` atomically binds raw inventory, converter SHA and every derived
file SHA. No conversion receipt means the derived directory cannot be loaded.

## Stage/resource model

Only **one** CPU model is live at a time: Qwen encoder → release → joint DiT →
release → video VAE → release → audio VAE → release. Per request, unused components
are deleted; no134GiB all-components loader. Qwen uses Accelerate CPU offload on
its actual called `.model` root, reads hidden_states[50] without the LM head.
DiT uses synchronous one-block group offload (no stream-prefetch/pinned full-copy),
FF token chunks1024, no approximation to full attention. PyTorch flash/efficient
SDPA only; no dangerous quadratic math-attention fallback. VAEs decode sequentially,
video uses its native tiling and per-channel/ImageNet denormalization, audio stays
stereo (two mono decoder batch items), never averaged.

Phase admission derives from actual files: live phase weight bytes + largest phase
shard +24GiB host reserve. Initial admission checks the maximum of these per-phase
requirements, not a sum. Conversion needs48GiB host available and raw-size+64GiB
disk. Inference admission GPU free>=32GiB; runtime global floors remain GPU16GiB,
host24GiB,disk64GiB. These are conservative engineering budgets, **not measured
full H3 peaks**.15s global sampling can miss brief minima. Stage logs and Torch peak
reservation/RSS are recorded. If resources fail, preserve failure; do not weaken
guards or assume HPS memory is available. Existing HPS restoration and temporary
release permissions remain governed by TASK/TRANSITION and main ownership.

## Actual one, then three

```bash
export RUNS=/mnt/data/qifan/runs/visualrl-h3-torch-20261025
"$PY" -u -m visual_rl.infer_h3 configs/h3/one-scene.json --expected-sha "$AGREED_SHA" --model-root "$MODEL" --model-revision "$REV" --converted-root "$DERIVED" --output "$RUNS/one-001" --preflight
# Only RC0 plus fresh safety/resource checks permits the following ONCE:
nohup bash -c '"$PY" -u -m visual_rl.infer_h3 configs/h3/one-scene.json --expected-sha "$AGREED_SHA" --model-root "$MODEL" --model-revision "$REV" --converted-root "$DERIVED" --output "$RUNS/one-001"; rc=$?; printf "%s\n" "$rc" > "$RUNS/one-001.rc"; exit "$rc"' >"$RUNS/one-001.log" 2>&1 < /dev/null &
printf 'WRAPPER_PID=%s LOG=%s\n' "$!" "$RUNS/one-001.log"
```

Inspect owned PID/log and15s heartbeats; wait bounded multi-minute intervals, no
injected foreground commands or automatic retry. `h3_phase_start/end` surrounds
loads/compute; `h3_joint_step` records both sigmas and39 joint forwards. Confirm
`one-001/SUCCESS.json`, launch result, actual MP4 and float WAV, full decode and
finite nonzero stereo/audio lengths and hashes. Only then fresh safety check and
one new run with `configs/h3/three-scenes.json`, output `three-001`, separate log:

```bash
# Execute only AFTER accepted one-001 and fresh safety/resource checks.
nohup bash -c '"$PY" -u -m visual_rl.infer_h3 configs/h3/three-scenes.json --expected-sha "$AGREED_SHA" --model-root "$MODEL" --model-revision "$REV" --converted-root "$DERIVED" --output "$RUNS/three-001"; rc=$?; printf "%s\n" "$rc" > "$RUNS/three-001.rc"; exit "$rc"' >"$RUNS/three-001.log" 2>&1 < /dev/null &
printf 'WRAPPER_PID=%s LOG=%s\n' "$!" "$RUNS/three-001.log"
```

No80-prompt batch. Every sample has receipt/MP4/float WAV; all3 must pass.

Failure returns to Coder with exact tested SHA, actual imports/versions, traceback,
phase/resource data and asset/conversion receipt. Preserve evidence; no hotfix.
Main downloads only the newly verified files, compares remote bytes/SHA locally and
fully decodes them before reporting completion. HPS must be original-identity ready
on completion/failure. Main's task evidence records restored ready/identity/81-frame
contract at11:30:52 UTC on2026-10-25 (PID1001593); this is not a fresh Coder HTTP
check. Recheck the existing service, do not duplicate-start restoration. No Feishu
success update before real normal AV + Mac verification succeeds.
