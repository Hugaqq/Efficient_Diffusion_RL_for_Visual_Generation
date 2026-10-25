# H3 PyTorch implementation provenance and review boundary

## Fixed upstream references

1. HuggingFace Diffusers **aaf57d03a3edfbd9de457d2872417a92f149a91d**,
   Apache-2.0, Copyright MiniMax/HuggingFace teams.
   <https://github.com/huggingface/diffusers/tree/aaf57d03a3edfbd9de457d2872417a92f149a91d>
   - `scripts/convert_minimax_h3_to_diffusers.py`, SHA256
     `86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c`:
     reviewed key plans, per-head QKV deinterleave, FF swaps, FP32 islands,
     VAE metadata/normalization, streamed shard writes, completeness validation.
     Invoked from an exact clean upstream Git checkout by preparation only.
   - `models/transformers/transformer_minimax_h3.py`: real joint PyTorch denoiser,
     FP32 input/time/output islands and BF16 attention/FF/AdaLN weights.
   - `models/autoencoders/autoencoder_kl_minimax_h3{,_audio}.py`: native torch VAEs.
   - `modular_pipelines/minimax_h3/{encoders,before_denoise,decoders}.py`:
     raw Qwen hidden-state50 presentation (no template/special tokens), packed
     layout/FP64 coordinates, native noise draw order, 24fps/40Hz, channel-major
     stereo packing and exact inverse patchification/denormalization.
   - `schedulers/scheduling_minimax_h3.py`: shifted grid INCLUDING terminal0,
     positive data-ward Euler velocity and independent12/3 clocks. VisualRL's
     explicit numerical step is tested **bitwise** against extracted reference
     methods for2/20/40 grid points and distinct video/audio shapes.
   - `models/modeling_utils.py`, `hooks/group_offloading.py`: reviewed load dtype
     preservation and synchronous block-level offload (no pinned duplicate).
2. Transformers **v5.19.0**, Apache-2.0:
   <https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py>
   Reviewed `.model` input/mm-token-type behavior, language model outputs and
   hidden-state50 versus final normalized state. Known Qwen class only; no remote
   checkpoint code. Pin5.19 avoids old4.x Hub<1 dependency conflict with the fixed
   Diffusers requirement `huggingface-hub>=1.32,<3`, `safetensors>=0.8`.
3. Flow-Factory **3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1**, Apache-2.0,
   Copyright2026 Jayce-Ping:
   <https://github.com/X-GenGroup/Flow-Factory/tree/3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1/src/flow_factory/models/minimax_h3>
   Reviewed workflow/denoise/blocks/dependency and token-local FF chunking.
   VisualRL retains the narrow joint prediction boundary, not its BaseAdapter,
   trainers, replay/scheduler registry or full modular pipeline. Small FF chunk
   wrapper is adapted from `_ChunkedFeedForward`, retaining modules/parameter
   objects, with remainder chunks and no attention approximation.
4. Original official assets MiniMaxAI/MiniMax-H3 **42ed227ee7df40d41602854ae760620d6eb651fe**.
   Read official FL2VA configs (raw Qwen3-VL, DiT, video wrapper/source, audio
   wrapper/metadata). Video wrapper uses `source/model.safetensors`. No new
   weights downloaded on Mac. Server's previous81-file verification is separate
   historical evidence, not inferred from this source review. Current RL infra
   does not verify weight contents or require this historical inventory; revision
   and path are declared provenance. Output media and executable source hashes
   are distinct and retained.

## What is reused versus implemented

- Runtime imports fixed Diffusers/Transformers **model components** and two pure
  packed-layout/time helper APIs. It does **not** instantiate/call any pipeline,
  modular denoise block loop, serving executor or engine.
- VisualRL implements the model adapter lifecycle/encode/joint-predict/decode,
  explicit two-schedule sampling loop, validation, media export and receipt path.
- Tests retain an attributed **inert extraction of numerical reference methods**
  and conversion mappings (`tests/h3_pinned_reference.py`), not a full framework.
  Extraction preserves math; no downloaded script was executed by the fetcher.
- License text retained at `docs/licenses/h3-diffusers-Apache-2.0.txt`.
- Source text and hashes were fetched by `scripts/inspect_h3_torch_references.py`
  into the outer task `outputs/h3-pytorch-integration/reference/`. That script is
  a bounded fixed-source audit fetcher, not an installer/runtime or dispatcher.

## Evidence limits

CPU tests execute exact small-tensor schedule/step/reference geometry, real
VisualRL loop, component ownership with explicit fake heavy torch classes,
executable-source identity refusals, metadata-only layout checks without weight
reads/mandatory conversion receipts, and real FFmpeg encode/full decode. They do
not prove the isolated GPU dependency environment, full raw conversion, mixed
precision GPU kernels, memory peak or normal H3 output. Those require Experimenter
on the tested pushed SHA. Never use a preflight/diagnostic/old clip as that proof.
