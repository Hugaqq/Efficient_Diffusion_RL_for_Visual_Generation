# H3 PyTorch inference — current handoff

**Weight integrity is not an RL-infrastructure responsibility.** Preflight and
initialization read only layout/config/index metadata and stat weight sizes for
resource planning. They do not open/hash weight contents, build asset inventories,
require conversion receipts, or offer fast/cached/full verification modes.
Existing `conversion.json` files are ignored, not upgraded or regenerated.
Declared model revision/path is provenance supplied by the caller, not a claim
that local bytes were authenticated. Actual model loading retains key/shape/dtype
errors; unsupported quantized/pruned layouts still fail.

Source Git/import identity, installed executable component-source hashes, offline
loading, resource gates, output AV decode/SHA and download receipts remain required.
No checkpoint-local Python is imported (`trust_remote_code=False`). See
[h3_pytorch_sources.md](h3_pytorch_sources.md) for fixed source/license attribution.

## Computation (unchanged)

- `models/h3.py`: lightweight raw layout/config, request and joint AV contracts.
- `models/h3_components.py`: staged Qwen3-VL encode, packed latent preparation,
  joint Diffusers DiT prediction, video/audio VAE decode. No whole pipeline call.
- `algorithms/rollout/h3.py`: VisualRL owns the explicit two-clock Euler loop.
- `infer_h3.py`: Git/import/offline checks, shared flock, 15s resource heartbeat,
  fresh output and SUCCESS only after native close and final checks.

T2VA only, B=1, official CFG-distilled FL2VA: one conditional pass, no negative
branch, optimizer, LoRA, reference policy, reward or H3 RL. No vLLM/SGLang/Comfy,
HTTP backend, cache approximation, quantization or Turbo switch.

Euler eta=0 uses **40 sigma points including terminal zero = 39 joint forwards**.
For b=linspace(1,0,40), sigma=s*b/(1+(s-1)*b), video shift12/audio shift3,
t=1-sigma; x0=x+(1-t)*v, x_next=(sigma_next/sigma)*x+
(1-sigma_next/sigma)*x0, including the pinned float32 round trip.
Normal output is1344×768×124 at24fps, stereo32kHz,207 audio latents/channel →
165600 samples/channel. Prompts/seeds42009/42012/42013 remain unchanged.
`--diagnostic` means384×224×107 and is not normal acceptance.

## Preserve the old live run; deploy into a new worktree

Old `one-001` wrapper1007707/Python1007710 belongs to7ac7dba; actual log starts
2026-10-25 13:15:51UTC. Check its current state/log/RC first. Do not cancel,
restart, duplicate, overwrite output or update its source while live. Old results
retain their old SHA. No new competing GPU task. Do not rerun Wan5/SMALL40.

Use the exact tested SHA from outer
`outputs/h3-remove-weight-integrity/coder-handoff.md`:

```bash
export AGREED_SHA='<tested full SHA>'
export OLD_SOURCE=/home/ubuntu/qifan/visualrl-readable-h3-20261024/source
export SOURCE=/home/ubuntu/qifan/visualrl-readable-h3-20261024/no-weight-integrity-source
# Confirm SOURCE is new; preserve all dirty changes in existing worktrees.
git -C "$OLD_SOURCE" fetch origin h3-readable-integration
git -C "$OLD_SOURCE" worktree add --detach "$SOURCE" "$AGREED_SHA"
git -C "$SOURCE" rev-parse HEAD
git -C "$SOURCE" status --short --branch
export PY=/mnt/data/qifan/envs/visualrl-h3-torch/bin/python
export MODEL=/mnt/data/qifan/models/h3-native/hub-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
export REV=42ed227ee7df40d41602854ae760620d6eb651fe
export DERIVED=/mnt/data/qifan/models/h3-torch/FL2VA-diffusers-aaf57d03
export RUNS=/mnt/data/qifan/runs/visualrl-h3-torch-20261025
export PYTHONPATH="$SOURCE" CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$SOURCE"
"$PY" -c 'import visual_rl; from visual_rl.models import h3; from visual_rl.models.h3_components import audit_runtime; print(visual_rl.__file__, h3.__file__); print(audit_runtime())'
```

**Reuse existing environment, raw files and converted components. Do not install,
redownload or reconvert for this change.** Prior conversion completed RC0. The
existing derived folder works without any receipt. Source checks still require
Diffusers aaf57d03a3edfbd9de457d2872417a92f149a91d and pinned component sources;
Transformers5.19.0, Torch2.10.0+cu128 were recorded by Experimenter. The raw video
VAE uses its config-owned `video_vae/source/model.safetensors` path.

For a genuinely new asset preparation, `scripts.prepare_h3_torch_assets` remains
an offline streaming converter, executing only the pinned clean Git/SHA-checked
upstream conversion script. It preserves full key-plan/dtype conversion checks,
writes a separate `.partial` directory and atomically publishes on success;
failure leaves `FAILED.json` and originals untouched. It does not scan raw or
converted contents for checksums or create a mandatory asset receipt. An existing
output is refused, not reconverted. Do not run it on the already prepared assets.

## Resources (unchanged)

One model phase at a time: encoder → release → DiT → release → video VAE → release
→ audio VAE → release. Qwen hidden_states[50] uses Accelerate CPU offload on the
called `.model`; DiT has synchronous one-block group offload, FF chunks1024,
FP32 input/time/output islands and BF16 remaining parameters, flash/efficient
SDPA only. Sequential native VAE tiling/denormalization preserves stereo.

Admission is live component weight bytes + largest shard +24GiB host reserve,
not total asset size. Initial GPU free>=32GiB; runtime floors GPU16GiB,
host24GiB,disk64GiB. File sizes are metadata, not integrity evidence. Each phase
rechecks resources. These are engineering budgets, not measured peak guarantees;
15s sampling can miss brief minima. Preserve guards and unrelated services.
HPS1001593 was previously restored ready with the original identity/81-frame
contract; recheck current health, do not duplicate-start or assume its RAM is free.

## New-SHA preflight, then normal one → three

Use a fresh output (example `one-no-integrity-001`); do not overwrite `one-001`.
Preflight is metadata-only, not a model/GPU generation test. Record elapsed time
and actual I/O, not only its exit code; do not run manual weight hashes instead.

```bash
"$PY" -u -m visual_rl.infer_h3 configs/h3/one-scene.json --expected-sha "$AGREED_SHA" --model-root "$MODEL" --model-revision "$REV" --converted-root "$DERIVED" --output "$RUNS/one-no-integrity-001" --preflight
# ONLY after old GPU work ends naturally, preflight RC0, fresh safety/resource checks:
nohup bash -c '"$PY" -u -m visual_rl.infer_h3 configs/h3/one-scene.json --expected-sha "$AGREED_SHA" --model-root "$MODEL" --model-revision "$REV" --converted-root "$DERIVED" --output "$RUNS/one-no-integrity-001"; rc=$?; printf "%s\n" "$rc" > "$RUNS/one-no-integrity-001.rc"; exit "$rc"' >"$RUNS/one-no-integrity-001.log" 2>&1 < /dev/null &
printf 'WRAPPER_PID=%s\n' "$!"
# ONLY after accepted one, fresh resources and no competing GPU work:
nohup bash -c '"$PY" -u -m visual_rl.infer_h3 configs/h3/three-scenes.json --expected-sha "$AGREED_SHA" --model-root "$MODEL" --model-revision "$REV" --converted-root "$DERIVED" --output "$RUNS/three-no-integrity-001"; rc=$?; printf "%s\n" "$rc" > "$RUNS/three-no-integrity-001.rc"; exit "$rc"' >"$RUNS/three-no-integrity-001.log" 2>&1 < /dev/null &
printf 'WRAPPER_PID=%s\n' "$!"
```

Record actual imports/SHA,39 joint forwards, phase/peak logs, RC0/SUCCESS, every
MP4/float WAV, full dual-track decode, finite nonzero stereo, metadata/lengths and
output hashes. Only output media is hashed. Main downloads those exact new files,
checks bytes/SHA and full decode on Mac. Keep HPS original-identity ready on final
handoff. Genuine model errors return to Coder → CPU gates → Git; no server hotfix.
No GPU success is claimed by these CPU changes. Feishu publishes only final
successful adaptation/new samples after normal AV and Mac verification.
