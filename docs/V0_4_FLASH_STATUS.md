# v0.4 Flash-GRPO Status

This checkpoint adds the first working Flash-GRPO-style path on the cheap
`tiny_diffusion` adapter.

## Implemented

- `single_step` rollout engine.
- Prompt-wise expansion into same-prompt sample groups.
- `iso_temporal`, `cycle`, `first`, `last`, `middle`, and seeded random
  selected timestep strategies.
- Timestep range clamping through `rollout.timestep_range`.
- Selected timestep metadata on every sample.
- Flash-GRPO algorithm plugin with PPO-style ratio clipping.
- Scheduler-style temporal rectification weights.
- CLI smoke command: `visual-rl flash-smoke`.

## Validation

```bash
conda run -n visual-rl python -m pytest -q tests
conda run -n visual-rl python -m ruff check visual_rl tests
conda run -n visual-rl python -m visual_rl.cli flash-smoke --output-dir runs/flash_tiny_smoke --steps 2
```

Server:

```bash
CUDA_VISIBLE_DEVICES="" python -m visual_rl.cli flash-smoke --output-dir runs/server_flash_tiny_smoke --steps 2
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_flash_grpo.py tests/test_import.py
```

The server smoke was run in the isolated `visual-rl-tempflow` conda environment
with CPU-only execution.

## Still Pending

- Reward-trend validation over 20-50 tiny steps.
- Speed/reward comparison against full-trajectory GRPO and TempFlow branching.
- SD1.5 LoRA single-step rollout.
- Wan/Flash parity checks against the reference Flash-GRPO implementation.
- Real scheduler-specific rectification formulas beyond the current tiny
  scheduler proxy.
