# Plan Evaluation After Adding GenRL

The revised direction is sound: `GenRL-main` should become the training-runtime
reference, while `visual_rl` remains the integration and abstraction layer.

## What Changed

Previous trunk choice:

- World-R1 as the main trainer.

Updated trunk choice:

- GenRL as the production training runtime reference.
- World-R1 as the video/world specialization layer: 3D rewards, general reward,
  camera-aware latent initialization, dynamic prompts, and world-generation eval.

## Why This Is Better

GenRL has a cleaner trainer split and already handles several pieces that are
expensive to retrofit from scratch:

- typed config
- BaseTrainer lifecycle
- FSDP-aware checkpoint/resume pattern
- epoch-tagged prompt sampler
- same-latent and deterministic prompt seeds
- SDE window controls
- raw vs weighted rewards
- per-reward advantages
- reward-model CPU/GPU offload contract

## v0.2 Scope

Implemented now:

- typed config in `visual_rl.configs.schema`
- GenRL-style `BaseTrainer`
- epoch-aware sampler utility
- per-prompt and per-reward advantage utilities
- RewardRouter v2 with raw/weighted reward separation
- media-aware reward cache keys
- mock trainer kept as the smoke-test path
- World-R1 plan generation remains local/dry-run only

Deferred:

- real FSDP execution
- real Wan checkpoint loading
- real reward server calls
- server SSH/GPU probing
- Flash and TempFlow algorithm implementation
- Inferix online rollout

## Risk Notes

- Do not import all legacy `flow_grpo` packages in the same Python process.
- Keep GenRL/World-R1/Flash/TempFlow code lazy-loaded through adapters.
- Do not silently return zero reward for unknown reward names.
- Do not run server experiments until connectivity and idle GPU availability are confirmed.

## Inferix / BlockVid Note

For future work, the main thing to learn from `Inferix-main` is its BlockVid /
semi-autoregressive block-diffusion design: diffusion blocks, segment-level
long-video scheduling, KV cache management, `PER_BLOCK` streaming decode, and
`NO_DECODE` latent-only paths.

Treat BlockVid as the relevant systems idea. Do not treat Inferix as a
replacement for the GenRL-style training runtime unless it later exposes a clean
`sample_with_logprob` / `recompute_logprob` policy contract.
