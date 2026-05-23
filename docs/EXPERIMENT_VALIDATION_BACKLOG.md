# Experiment Validation Backlog

These are validation gaps discovered during the v0.3 TempFlow smoke review.
They are intentionally deferred so feature integration can continue without
pretending the tiny smoke tests prove full training correctness.

## Near-Term Tiny Checks

- Run `tiny_diffusion` for 20-50 steps and assert reward improves against a
  fixed-seed baseline. TempFlow completed on 2026-05-23.
- Assert trainable parameters change after GRPO, Flash-GRPO, and TempFlow-GRPO
  updates. TempFlow completed on 2026-05-23.
- Add deterministic golden tests for rollout expansion, selected timesteps,
  branch IDs, advantage masks, and cache filenames.
- Compare full-trajectory GRPO, Flash single-step GRPO, and TempFlow branching
  on the same prompt/reward set.

## Infra Checks

- Verify reward cache hit/miss behavior with media hash and reward version.
  Numpy/PIL/tensor content hashing has unit coverage; end-to-end reward-cache
  replay is still pending.
- Verify resume from checkpoint restores model state and keeps metrics/logging
  append behavior sane.
- Add failure-path tests for reward timeout, invalid mask, and strict unknown
  reward names.
- Add config validation for incompatible rollout/algorithm pairs.

## Server Checks

- Repeat tiny smoke on the server with CPU-only execution after each major
  feature addition. TempFlow and Flash completed.
- Add a one-GPU memory probe for small image models, pinned to an explicitly
  idle GPU only. TempFlow two-GPU isolated probe completed on GPU0/GPU1.
- Record GPU ID, visible devices, VRAM before/after, package versions, and
  commit hash for every server run.

## Real-Model Checks

- Add SD1.5 LoRA adapter contract tests before SD3/FLUX/QwenImage. Deferred
  adapter registration tests are complete; loaded-model numeric tests are still
  pending.
- Validate real model `sample()` and `recompute_log_probs()` numerics on a tiny
  batch before any reward optimization run. SD3.5 reference script path passed
  a minimal TempFlow smoke on 2026-05-23; `visual_rl` SD3 adapter path is
  implemented but not yet server-validated.
- Compare Flash selected-step loss against full-trajectory loss on a controlled
  toy scheduler.
- Validate TempFlow branch reward alignment on SD1.5 before moving to SD3. The
  SD3.5 reference script now has a smoke pass, but adapter-level parity is still
  pending.
- Validate FLUX and QwenImage adapters with low-resolution smoke batches after
  SD3.5 adapter parity is confirmed.

## Video/Inferix Checks

- Keep Wan/World-R1 runs as smoke-only until small image RL curves are stable.
- Validate World-R1 reward server clients independently before online training.
- For Inferix, validate BlockVid preview/profiling/no-decode paths first; do
  not use it for online RL until logprob/recompute contracts exist.
