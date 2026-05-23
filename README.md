# Enfficient Diffusion RL infra for Visual Generaion

This repository builds a unified reinforcement-learning infrastructure for
diffusion-based visual generation. The goal is to make image, video, and
world-generation RL experiments share the same training contracts instead of
living as isolated research scripts.

The implementation package is still named `visual_rl`. It provides common
interfaces for rollout batches, reward routing, algorithm plugins, checkpointing,
logging, and cheap smoke tests. The legacy research projects remain reference
material and are not vendored into the Python package.

## What This Project Is For

- Train diffusion models with GRPO-style objectives for visual generation.
- Compare full-trajectory, Flash single-step, and TempFlow branching rollouts
  under one trainer interface.
- Route local or remote visual rewards through a shared `RewardRouter`.
- Cache rollouts and reward results so failed or expensive runs can be resumed
  or replayed.
- Start with tiny and small image models, then move toward SD3/FLUX/QwenImage,
  Wan/World-R1 video, and Inferix/BlockVid eval.

## Integrated Directions

- **World-R1**: video/world-generation specialization, including 3D rewards,
  general rewards, camera-aware latents, and future Wan/CogVideoX paths.
- **Flash-GRPO**: low-cost selected-timestep training with iso-temporal prompt
  grouping and temporal rectification.
- **TempFlow-GRPO**: image diffusion RL with branching rollouts and
  timestep-level credit assignment.
- **Inferix / BlockVid**: future eval, preview, profiling, long-video serving,
  and latent-only/no-decode workflows.
- **GenRL**: engineering reference for typed config, trainer lifecycle,
  sampling, rewards, checkpointing, and distributed training patterns.

## Current Progress

- **v0.1**: created the first `visual_rl` skeleton with `RolloutBatch`,
  `RewardRouter`, mock training, rollout cache, CLI, and local smoke tests.
- **v0.2**: added typed config, `BaseTrainer`, per-prompt/per-reward advantages,
  raw/weighted reward logging, epoch-aware sampling, and safer validation.
- **v0.3**: added TempFlow tiny branching support with `tiny_diffusion`,
  prompt-color reward, branch metadata, branch/timestep credit assignment, and
  lazy SD3/FLUX/QwenImage bridges.
- **v0.4**: added Flash-GRPO tiny single-step support with selected timestep
  rollout, iso-temporal grouping, scheduler-style rectification weights, and
  CLI/tests.

The current implementation proves the infra plumbing on cheap toy workloads.
It does **not** yet prove real SD3/FLUX/QwenImage/Wan training quality or
paper-level algorithm parity.

## Validation

Safe local checks:

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

Latest verified state:

- Local tests: `15 passed`
- Local ruff: passed
- Local TempFlow tiny smoke: passed
- Local Flash tiny smoke: passed
- Server CPU-only TempFlow tiny smoke: passed
- Server CPU-only Flash tiny smoke: passed
- Server two-GPU TempFlow correctness probe: passed

Server smoke tests were run with `CUDA_VISIBLE_DEVICES=""`, so they did not
consume shared GPUs.
The two-GPU TempFlow correctness probe used GPU0 and GPU1 only, while GPU2-7
were already occupied by other jobs.

## Near-Term Tasks

- Run the validation backlog: reward trends, parameter updates, deterministic
  golden tests, cache/resume behavior, and failure-path tests.
- Extend tiny Flash/TempFlow paths to SD1.5 LoRA.
- Implement real SD3/FLUX/QwenImage adapters after SD1.5 is stable.
- Add World-R1 reward/camera probes before any real Wan RL training.
- Build Inferix/BlockVid as an eval and profiling backend, not as the online RL
  trainer until clean logprob contracts exist.

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the detailed roadmap and
[`docs/EXPERIMENT_VALIDATION_BACKLOG.md`](docs/EXPERIMENT_VALIDATION_BACKLOG.md)
for known validation gaps.
