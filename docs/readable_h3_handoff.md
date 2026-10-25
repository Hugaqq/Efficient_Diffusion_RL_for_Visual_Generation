# Readable model/algorithm boundary — current handoff

Latest mandate: **direct PyTorch H3**, no serving engines. See
[h3_native_inference.md](h3_native_inference.md) for current commands and
[h3_pytorch_sources.md](h3_pytorch_sources.md) for fixed source/license evidence.
Earlier engine setup instructions are superseded and preserved only in Git.

## Model versus algorithm

```text
Wan training (unchanged by this H3 change):
  existing controller → BaseTrainer → encode / prepare_latents / predict
  → DynamicsSession sample/ODE → decode/reward → advantage/credit
  → current predict / fixed-action Dynamics evaluate → PPO/backward/commit
  → existing checkpoint/EMA/pending-pool/RNG owners

H3 inference:
  H3InferenceAdapter.load(raw+separate derived components)
  → VisualRL sample_h3:
      model.encode (Qwen hidden-state50) → release encoder
      model.prepare_latents (native packed video/audio geometry)
      with model.denoising():
          for video_sigma,audio_sigma in independent native12/3 grids:
              velocity = model.predict(video,audio,prompt,packed_layout,times)
              video,audio = joint_euler_step(...)
      release denoiser
      model.decode: video VAE → release → audio VAE → release
  → actual AV export / full decode / byte SHA / receipts
  → close and final resource/import checks → overall SUCCESS
```

No H3 optimizer, reward acquisition, training recipe, reference policy or logprob.
No whole pipeline delegation or scheduler hidden in model.predict. B=1 text-only
T2VA; unsupported conditioning fails explicitly. No claim of H3 RL support.

## Preserved completed work

- Earlier boundary refactor removed duplicate preprocess/initialize_latents and
  model transition facade; algorithms call their existing DynamicsSession.
- Preserve Wan prepared root, CFG, BF16 model/FP32 trajectory, replay partitions,
  old/current PPO, finite/nonzero-gradient checks, step0-only initial diagnostic,
  full-pool std, sampling-weight cache fingerprint, schema2 pending pool/cursor,
  CPU EMA and exact restore/RNG. No H3 edits touch these owners/configs.
- Wan RL5 on **36ca4d6344fded2083711b1e347e178dee25c3ff** was independently
  verified5/5, SUCCESS, step5 pool4/cursor1 and12 real81-frame HPS calls by the
  server role (2026-10-25). **Do not rerun Wan5/SMALL40.** This historical GPU
  evidence is not evidence for H3 or execution of a later SHA.
- CPU original7 gate stays separate from H3 and readable专项. Candidate import
  checks and exact assertions remain. Counts/timings/precise new pushed SHA are
  in outer `outputs/h3-pytorch-integration/coder-handoff.md` (avoids self-SHA).

## Deployment boundary

Mac source edits → original7 + H3/readable tests → commit/push → server Git precise
SHA, clean/import checks → isolated dependency check/conversion → normal1→3 AV.
Never hotfix server source/packages, edit old checkpoints, discard pending pools,
use cancelled engine environments or overwrite old outputs. Preserve services;
main owns HPS restoration; latest task evidence records ready/identity match at11:30:52 UTC. Resource/service actions require current
identity checks and TASK authorization, not stale PIDs or inferred availability.

Current H3 implementation is a tested CPU candidate, **not a successful GPU model
run**. Final Mac download must match remote byte SHA and fully decode before main
publishes today's final success-only Feishu result. Local failure history remains
truthful and complete; no premature external success entry.
