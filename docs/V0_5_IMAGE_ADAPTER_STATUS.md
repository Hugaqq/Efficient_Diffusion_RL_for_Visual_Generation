# v0.5 Image Adapter Status

This checkpoint moves the project from tiny-only image RL toward real Diffusers
image model contracts.

Implemented:

- Pytest collection scoped to project tests, so `reference_code/` no longer
  breaks plain `pytest`.
- Device/dtype-safe GRPO, Flash-GRPO, and TempFlow-GRPO loss inputs.
- Reward normalization now feeds advantage computation.
- Content-aware reward media hashing for tensors, numpy arrays, PIL images,
  lists, tuples, and dicts.
- Strict `RolloutBatch` validation for adapter probes and image trainers.
- `SD15LoRAAdapter` using Diffusers SD1.5, LoRA trainable UNet parameters, and a
  DDIM-style surrogate transition logprob.
- `ImageRLTrainer`, which enables strict rollout validation by default.
- `SD3TempFlowAdapter`, wrapping TempFlow-GRPO's verified SD3 patched
  `pipeline_with_logprob` and `sde_step_with_logprob` path.
- `FluxTempFlowAdapter` and `QwenImageTempFlowAdapter` entries using the
  TempFlow reference patched pipeline contracts.
- Presets for SD1.5, SD3, FLUX, and QwenImage adapter probes/smokes.

Validated locally:

- `compileall`: passed.
- `ruff`: passed.
- `pytest`: 21 passed.
- `smoke-imports`: passed.
- `adapter-probe`: passed for SD1.5, SD3.5, FLUX, and QwenImage in
  deferred-load mode.
- Existing mock, TempFlow tiny, and Flash tiny smoke commands still pass.

Important limitations:

- SD1.5 logprob is currently a surrogate DDIM transition for infra validation,
  not a paper-parity probability formula.
- SD3.5 adapter wraps the same reference path previously validated on the
  server, but the new `visual_rl` adapter itself has not yet run on the server.
- FLUX and QwenImage adapters are implemented from the reference contracts but
  still need low-resolution GPU smoke tests.
- No real-model reward-improvement curve has been established through
  `VisualRLTrainer` yet.
