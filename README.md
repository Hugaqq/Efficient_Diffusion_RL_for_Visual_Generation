# Visual RL Infra for Diffusion Generation

This repository is a local integration workspace for building a unified diffusion RL training infrastructure for visual generation.

The current direction is:

> `visual_rl` provides the unified abstraction layer, `GenRL-main` provides the production-style training runtime reference, and the four research systems below are integrated as specialization modules.

The initial target is not to merge every codebase mechanically. The goal is to extract stable contracts, trainer/runtime patterns, reward interfaces, rollout formats, and evaluation backends into one maintainable infra.

## Current Status

- `v0.1`: created the first `visual_rl` skeleton with `RolloutBatch`, `RewardRouter`, mock rollout/training, cache, CLI, and local smoke tests.
- `v0.2`: upgraded toward a GenRL-style runtime with typed config, `BaseTrainer`, reward raw/weighted scoring, per-prompt/per-reward advantage computation, epoch-aware sampler, and safer local validation.

Server-side experiments are intentionally not part of the current committed state. The repo is set up for local import/smoke validation first.

## Project Roles

| Component | Role in This Repo | What We Reuse | What We Avoid |
| --- | --- | --- | --- |
| `visual_rl/` | Our integration layer | Stable dataclasses, config, trainer API, reward router, rollout cache, algorithm plugins | One-off script coupling |
| `GenRL-main/` | Training runtime reference | Typed config, FSDP/checkpoint/resume design, Wan training loop structure, sampler, reward offload, advantage logic | Treating it as a black-box dependency |
| `World-R1-main/` | World/video specialization | 3D reward, general reward, camera-aware latent initialization, camera trajectory metadata, dynamic prompt phase | Making World-R1 the global trainer trunk |
| `Flash-GRPO-main/` | Low-cost video RL plugin | Single-step selected rollout, iso-temporal grouping, temporal gradient rectification | Duplicating trainer/runtime code |
| `TempFlow-GRPO-main/` | Image flow RL plugin | Branching rollout, timestep credit assignment, noise-aware weighting for SD3/FLUX/QwenImage | Polluting Wan-specific trainer logic |
| `Inferix-main/` | BlockVid/inference/eval backend | Block-diffusion serving, semi-autoregressive block scheduling, KV cache ideas, streaming preview, profiling, `NO_DECODE` latent paths | Using it as the online RL trainer until logprob/recompute contracts exist |

## Four Works Taxonomy

### 1. World-R1: 3D-Constrained Video/World RL

World-R1 is the best source for world-generation specialization.

Key ideas to keep:

- 3D-aware rewards for geometry consistency.
- General visual reward paired with 3D reward to avoid quality collapse.
- Camera-aware latent initialization from camera trajectory metadata.
- Dynamic-scene prompt phase for motion diversity.

Integration target:

- `visual_rl.model_adapters.wan`
- `visual_rl.rewards.world_r1_rewards`
- future `visual_rl.rollout.full_trajectory`
- future world-generation eval presets

World-R1 should become a specialization layer on top of a stronger runtime, not the root trainer architecture.

### 2. Flash-GRPO: Efficient Single-Step Video RL

Flash-GRPO is the low-cost GRPO route for video diffusion.

Key ideas to keep:

- Selected single-step rollout instead of full trajectory rollout.
- Prompt-wise same-timestep grouping to remove timestep-confounded variance.
- Temporal gradient rectification for scheduler/timestep-dependent scale.
- Wan video RL ablation path with much lower sampling cost.

Integration target:

- `visual_rl.rollout.single_step`
- `visual_rl.algorithms.flash_grpo`
- shared `visual_rl.trainer.BaseTrainer`
- shared `visual_rl.rewards.RewardRouter`

Flash should be an algorithm plugin, not a copied trainer.

### 3. TempFlow-GRPO: Branching Credit Assignment for Image Flow Models

TempFlow-GRPO is the image/flow RL branch of the infra.

Key ideas to keep:

- Trajectory branching at selected timesteps.
- Branch-level process rewards without a separate intermediate reward model.
- Noise-aware weighting to focus learning on high-impact timesteps.
- SD3, FLUX, and QwenImage adapters.

Integration target:

- `visual_rl.rollout.branching`
- `visual_rl.algorithms.tempflow_grpo`
- `visual_rl.model_adapters.sd3`
- `visual_rl.model_adapters.flux`
- `visual_rl.model_adapters.qwenimage`

TempFlow should not leak image-specific branching assumptions into the Wan trainer.

### 4. Inferix / BlockVid: Semi-AR Block-Diffusion Inference for Eval and Serving

Inferix is now treated primarily as the source for BlockVid-style inference architecture.

The important part is its semi-autoregressive block-diffusion design:

- Diffusion block: model-level generation unit, such as 3 frames per block in Self-Forcing.
- Segment: framework-level long-video generation unit composed of multiple blocks.
- KV cache management across generated blocks.
- Decode timing modes:
  - `AFTER_ALL`: decode after all diffusion blocks are produced.
  - `PER_BLOCK`: decode each generated block immediately for streaming preview.
  - `NO_DECODE`: keep latent-only outputs for profiling or integration.
- Long-video generation through block/segment scheduling and overlap.
- Profiling for diffusion time, VAE time, memory, and block-level latency.

Integration target:

- `visual_rl.eval.inferix_backend`
- preview generation after checkpoint export
- long-video eval
- profiling-guided cost analysis
- future rollout acceleration only after a clean `sample_with_logprob` / `recompute_logprob` contract exists

Important policy for future work:

> For Inferix, prioritize learning and adapting BlockVid / semi-autoregressive block-diffusion ideas. Do not treat Inferix as the main RL training runtime.

## Target Architecture

```text
visual_rl
  core/
    types.py
    registry.py

  configs/
    schema.py
    presets/

  trainer/
    base.py
    trainer.py
    wan_trainer.py
    image_flow_trainer.py

  rollout/
    full_trajectory.py
    single_step.py
    branching.py
    cache.py

  algorithms/
    grpo.py
    flow_grpo.py
    flash_grpo.py
    tempflow_grpo.py
    longcat.py

  rewards/
    router.py
    local_models.py
    remote_clients.py
    cache.py

  data/
    prompts.py
    samplers.py

  eval/
    inferix_backend.py
```

## Local Validation

The current safe local checks are:

```bash
conda run -n visual-rl visual-rl smoke-imports
conda run -n visual-rl python -m visual_rl.cli smoke-mock --output-dir runs/smoke_v02_mock --steps 2
conda run -n visual-rl python -m pytest -q tests
conda run -n visual-rl python -m ruff check visual_rl tests
```

No server access is required for these checks.

## Near-Term Roadmap

1. Stabilize `visual_rl` v0.2 as the local abstraction/runtime layer.
2. Add a GenRL-style `WanTrainer` behind `visual_rl.trainer`.
3. Move reward composition fully into `RewardRouter` v2.
4. Add Flash-GRPO as `single_step` rollout plus algorithm plugin.
5. Add TempFlow as `branching` rollout plus image adapter plugins.
6. Use Inferix primarily for BlockVid-style eval, streaming preview, profiling, and latent-only `NO_DECODE` flows.
