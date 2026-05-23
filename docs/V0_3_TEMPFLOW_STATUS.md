# v0.3 TempFlow Status

This checkpoint adds the first working TempFlow-style path without requiring
large image or video checkpoints.

## Implemented

- `tiny_diffusion` image adapter with a trainable color bias.
- `prompt_color` reward client for cheap image RL smoke tests.
- Branching rollout expansion with main trajectory plus branch samples.
- `RolloutBatch.branch_ids` population and branch metadata.
- TempFlow-GRPO algorithm plugin with branch/timestep credit assignment.
- Optional noise-aware weighting through `algorithm.noise_weighting`.
- Lazy SD3, FLUX, and QwenImage adapter bridges for the TempFlow reference path.
- CLI smoke command: `visual-rl tempflow-smoke`.

## Validation

Local:

```bash
conda run -n visual-rl python -m pytest -q tests
conda run -n visual-rl python -m ruff check visual_rl tests
conda run -n visual-rl python -m visual_rl.cli tempflow-smoke --output-dir runs/tempflow_tiny_smoke --steps 2
```

Server:

```bash
CUDA_VISIBLE_DEVICES="" python -m visual_rl.cli tempflow-smoke --output-dir runs/server_tempflow_tiny_smoke --steps 2
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_tempflow_branching.py tests/test_import.py
```

The server smoke was run in an isolated `visual-rl-tempflow` conda environment
with CPU-only PyTorch, so it did not allocate shared GPUs.

## Still Pending

- Real SD1.5 adapter and LoRA path.
- Real SD3/FLUX/QwenImage rollout and logprob contracts.
- Sequential branch execution for large models under 32 GB VRAM.
- Flash-GRPO single-step plugin.
- Inferix BlockVid eval backend.
