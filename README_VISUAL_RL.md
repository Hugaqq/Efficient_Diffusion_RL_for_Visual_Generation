# Enfficient Diffusion RL infra for Visual Generaion

`visual_rl` is the implementation package for a unified diffusion RL training
infra targeting image, video, and world-generation experiments.

The project integrates ideas from World-R1, Flash-GRPO, TempFlow-GRPO, Inferix
/ BlockVid, and GenRL. The goal is not to copy these repositories into one
environment, but to extract common contracts for rollouts, rewards, algorithms,
training, caching, and evaluation.

## Current Progress

- `v0.1`: core package skeleton, `RolloutBatch`, `RewardRouter`, mock trainer,
  CLI, cache, and smoke tests.
- `v0.2`: typed config, trainer lifecycle, advantage utilities, reward
  raw/weighted logging, and safer local validation.
- `v0.3`: TempFlow tiny branching path with prompt-color reward and
  branch/timestep credit assignment.
- `v0.4`: Flash-GRPO tiny single-step path with iso-temporal grouping and
  scheduler-style rectification.

Current smoke tests validate infra wiring on cheap workloads. Real SD3, FLUX,
QwenImage, Wan, and World-R1 training are still pending.

## Quick Smoke

```bash
conda activate visual-rl
pip install -e ".[dev]"
visual-rl smoke-imports
visual-rl smoke-mock --output-dir runs/smoke --steps 2
visual-rl tempflow-smoke --output-dir runs/tempflow_tiny_smoke --steps 2
visual-rl flash-smoke --output-dir runs/flash_tiny_smoke --steps 2
python -m pytest -q tests
python -m ruff check visual_rl tests
```

Use `CUDA_VISIBLE_DEVICES=""` for CPU-only server smoke tests on shared
machines.

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) and
[`docs/EXPERIMENT_VALIDATION_BACKLOG.md`](docs/EXPERIMENT_VALIDATION_BACKLOG.md)
for the roadmap and known validation gaps.
