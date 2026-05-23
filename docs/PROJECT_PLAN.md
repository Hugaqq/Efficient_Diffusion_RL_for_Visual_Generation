# VisualRL Project Plan

This is the canonical plan for the current project direction. It supersedes the
older GenRL-centered planning note and uses the latest principle:

- `GenRL-main` is only an engineering reference.
- `World-R1-main`, `Flash-GRPO-main`, `TempFlow-GRPO-main`, and `Inferix-main`
  are the four projects this repository must integrate.
- Experiments should start with small image/tiny diffusion models, then move to
  larger image models, and only later to Wan/World-R1 video training.

## Current Position

`visual_rl` is the integration infra. It should not become a copy of any single
reference project. The current package already has:

- typed config dataclasses inspired by GenRL
- shared `RolloutBatch` and `RewardBatch` contracts
- lightweight `BaseTrainer`
- mock trainer smoke path
- `RewardRouter` v2 with raw/weighted rewards, valid mask, media-aware cache
- per-prompt and per-reward advantage utilities
- epoch-aware repeat sampler
- World-R1 dry-run launch plan
- Wan runtime plan shell
- tiny diffusion image adapter
- prompt-color image reward
- TempFlow branching rollout and branch/timestep credit assignment
- lazy TempFlow SD3/FLUX/QwenImage adapter bridges

Current validation status:

```bash
conda run -n visual-rl python -m pytest -q tests
conda run -n visual-rl python -m ruff check visual_rl tests
conda run -n visual-rl python -m visual_rl.cli smoke-imports
conda run -n visual-rl python -m visual_rl.cli smoke-mock --output-dir runs/smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli tempflow-smoke --output-dir runs/tempflow_tiny_smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli wan-plan --output-dir runs/wan_runtime_plan
```

Local tests, ruff, local TempFlow tiny smoke, and CPU-only server TempFlow tiny
smoke pass. Real GPU training, real Wan checkpoint loading, reward server calls,
and Inferix eval are not wired yet.

## Reference Code Policy

`reference_code/` is local-only upstream source material and should remain
optional at import time:

```text
reference_code/
  World-R1-main/
  Flash-GRPO-main/
  TempFlow-GRPO-main/
  Inferix-main/
  GenRL-main/
```

The four integration targets are:

| Project | Integration Role |
| --- | --- |
| `World-R1-main` | video/world specialization: Wan/CogVideoX path, camera-aware latents, 3D reward, general reward, dynamic prompts |
| `Flash-GRPO-main` | low-cost single-step GRPO: iso-temporal grouping, timestep sampling, temporal rectification |
| `TempFlow-GRPO-main` | image/flow RL: branching rollout, timestep credit assignment, SD3/FLUX/QwenImage paths |
| `Inferix-main` | eval/preview/profiling/serving backend: BlockVid, long-video eval, latent-only and chunked decode ideas |

`GenRL-main` is a reference only. Borrow its typed config, trainer lifecycle,
sampler, advantage, reward offload, and checkpoint ideas, but do not make
`genrl` the runtime dependency or the architectural trunk.

The legacy projects must stay isolated because several share the `flow_grpo`
package name and have incompatible dependency assumptions. Heavy imports should
be lazy, subprocess-based, or behind adapters.

## Hardware Constraint

The expected training budget is 1-2 RTX 5090 GPUs with 32 GB VRAM each.

This changes the plan:

- Do not start with full Wan video RL.
- Do not target Wan 14B, full finetune, 480x832x81 frames, or online
  multi-heavy-reward training.
- Prefer LoRA, small resolution, sequential group rollout, single-step/SDE-window
  updates, cheap rewards, reward cache, and small image models first.

## Target Architecture

```text
visual_rl/
  core/
    types.py
    registry.py
    contracts.py

  configs/
    schema.py
    presets/
      tiny_diffusion_rl.yaml
      sd15_lora_rl.yaml
      flash_tiny_rl.yaml
      tempflow_tiny_branching.yaml
      tempflow_sd3_plan.yaml
      world_r1_wan_plan.yaml
      inferix_eval_plan.yaml

  model_adapters/
    tiny_diffusion.py
    sd15.py
    sd3.py
    flux.py
    qwenimage.py
    wan.py
    cogvideox.py

  rollout/
    full_trajectory.py
    single_step.py
    branching.py
    cache.py

  algorithms/
    grpo.py
    flash_grpo.py
    tempflow_grpo.py
    world_r1_grpo.py

  rewards/
    router.py
    image_rewards.py
    world_r1_rewards.py
    remote_clients.py
    cache.py

  integrations/
    world_r1/
      launcher.py
      rewards.py
      camera.py

    flash_grpo/
      single_step.py
      timestep_sampler.py
      rectification.py

    tempflow_grpo/
      branching.py
      sd3_bridge.py
      flux_bridge.py
      qwenimage_bridge.py

    inferix/
      eval_backend.py
      preview.py
      profiling.py

  trainer/
    base.py
    mock_trainer.py
    image_trainer.py
    video_trainer.py
    eval_runner.py
```

The current `WanTrainer` should remain a planning shell until small-model and
image-model training loops are stable.

## Code Plan

### Phase A: Tiny Diffusion RL

Status: first implementation complete for the TempFlow smoke path.

Implement a tiny, cheap training loop that proves the RL circuit works.

Files to add:

```text
visual_rl/model_adapters/tiny_diffusion.py
visual_rl/rewards/image_rewards.py
visual_rl/configs/presets/tiny_diffusion_rl.yaml
tests/test_tiny_diffusion_rl.py
```

Requirements:

- CPU or single-GPU runnable
- 32x32 or 64x64 image output
- prompt-color reward such as red/green/blue matching
- real `sample()` and `recompute_log_probs()` contract
- GRPO update changes trainable parameters
- reward trend can improve in a short run
- rollout cache and checkpoint path are exercised

This is the lowest-cost end-to-end testbed for the whole infra.

### Phase B: Flash-GRPO Abstraction

Implement low-cost single-step training before touching Wan.

Files to complete:

```text
visual_rl/rollout/single_step.py
visual_rl/algorithms/flash_grpo.py
visual_rl/integrations/flash_grpo/timestep_sampler.py
visual_rl/integrations/flash_grpo/rectification.py
```

Start on `tiny_diffusion`, then move to SD1.5. Required features:

- selected timestep rollout
- iso-temporal grouping
- scheduler-aware timestep weights
- temporal gradient rectification
- sequential group rollout for 32 GB cards

### Phase C: TempFlow Branching Abstraction

Status: first tiny-diffusion implementation complete.

Implement branching/process-reward mechanics on small models first.

Files to complete:

```text
visual_rl/rollout/branching.py
visual_rl/algorithms/tempflow_grpo.py
visual_rl/integrations/tempflow_grpo/branching.py
```

Start with tiny diffusion, then SD1.5, then SD3. Required features:

- main trajectory plus branch samples: implemented for tiny diffusion
- branch IDs in `RolloutBatch`: implemented
- branch/timestep-level reward assignment: implemented in `tempflow_grpo`
- branch-level advantage computation: implemented through shared GRPO advantages
- sequential branch execution for low VRAM

### Phase D: Real Small Image Model

Add Stable Diffusion 1.5 LoRA RL before SD3/FLUX/QwenImage.

Files to add:

```text
visual_rl/model_adapters/sd15.py
visual_rl/trainer/image_trainer.py
visual_rl/configs/presets/sd15_lora_rl.yaml
tests/test_sd15_adapter_contract.py
```

Suggested experiment config:

- Stable Diffusion 1.5
- 256x256
- LoRA rank 4/8/16
- batch size 1
- group size 4 via sequential rollout
- 10-20 denoise steps
- single-step or small SDE window
- cheap image reward first

### Phase E: TempFlow Model Expansion

After SD1.5 is stable, integrate models from the TempFlow path:

1. SD3
2. FLUX
3. QwenImage

Each adapter must implement:

```python
parameters()
sample()
recompute_log_probs()
save_pretrained()
```

Each model needs a tiny config, adapter contract test, and at least one
low-resolution smoke path.

### Phase F: World-R1 Integration

World-R1 is important but should not be the first real training target.

Integrate in this order:

1. reward server clients
2. general reward probe
3. 3D reward probe
4. camera metadata parsing
5. camera-aware latent init
6. dynamic prompt/world-generation hooks
7. low-VRAM Wan smoke
8. CogVideoX path

Suggested files:

```text
visual_rl/integrations/world_r1/rewards.py
visual_rl/integrations/world_r1/camera.py
visual_rl/rewards/world_r1_rewards.py
visual_rl/configs/presets/world_r1_reward_probe.yaml
```

Wan should start as a smoke target only:

- Wan2.1 1.3B
- LoRA only
- 240x416
- 17 frames
- 8-12 denoise steps
- `sde_window_size=1`
- batch size 1
- group size 2 via sequential rollout
- mock or cheap reward first

### Phase G: Inferix Eval Integration

Inferix should not enter online RL until it exposes a clean policy contract:

```python
sample_with_logprob()
recompute_logprob()
```

Short-term use:

- checkpoint preview
- long-video eval
- profiling
- VAE chunking
- latent-only/no-decode eval
- BlockVid/semi-autoregressive scheduling ideas

Files to complete:

```text
visual_rl/integrations/inferix/eval_backend.py
visual_rl/integrations/inferix/profiling.py
visual_rl/eval/inferix_backend.py
```

## Experiment Plan

### Experiment 0: Tiny Diffusion GRPO

Purpose: prove the base infra.

- model: tiny diffusion adapter
- resolution: 32x32 or 64x64
- reward: prompt color match
- group size: 4 or 8
- algorithm: GRPO
- expected result: reward rises, loss/clip/approx KL are logged, checkpoint saves

### Experiment 1: Tiny Flash-GRPO

Purpose: prove low-cost single-step optimization.

- selected timestep
- iso-temporal grouping
- compare speed/reward curve against full-trajectory GRPO

### Experiment 2: Tiny TempFlow Branching

Purpose: prove branch credit assignment.

- main path plus branch samples: implemented
- reward assigned to branch timestep: implemented
- compare full GRPO, single-step GRPO, and branching GRPO

### Experiment 3: SD1.5 LoRA RL

Purpose: first real image diffusion RL curve.

- 256x256
- LoRA rank 8
- group size 4 sequential
- cheap reward
- 1x 5090 runnable

### Experiment 4: SD3 TempFlow-Style RL

Purpose: begin real TempFlow integration.

- low resolution
- branching rollout
- small group
- reward cache
- no large multi-reward setup initially

### Experiment 5: Wan Low-VRAM Smoke

Purpose: verify video path only, not final quality.

- Wan2.1 1.3B
- 240x416
- 17 frames
- LoRA
- group size 2 sequential
- mock or cheap reward
- `sde_window_size=1`

### Experiment 6: World-R1 Reward and Camera

Purpose: integrate World-R1-specific features after video smoke works.

- reward server probe
- camera metadata probe
- camera-aware latent init
- small video RL run only after probes pass

### Experiment 7: Inferix Eval

Purpose: reduce video eval cost.

- checkpoint preview
- long-video eval
- profiling
- no online training

## Stability and Cost Controls

Add these utilities before running expensive experiments:

- `visual-rl validate-config`
- `visual-rl rollout-probe`
- `visual-rl reward-probe`
- `visual-rl adapter-probe`
- reward cache enabled by default
- rollout cache for debug runs
- media hash and reward version in cache key
- strict reward-name validation
- no silent zero reward fallback
- sequential group rollout for low VRAM
- no-decode/latent-only rollout probe
- frame subsampling for video rewards
- heavy reward as eval first, online later

## Dependency and Environment Notes

Local smoke environment:

```text
conda env: visual-rl
Python: 3.10
numpy: 1.26
pytest/ruff for local validation
```

Server target:

- Python 3.10
- CUDA PyTorch matching the 5090 driver
- `numpy==1.26.4`
- isolated environments for legacy projects when needed
- no shared install of overlapping `flow_grpo` packages

## Current Non-Wan Support

Non-Wan support now has one complete tiny path plus lazy bridges:

| Model | Status |
| --- | --- |
| Tiny diffusion | implemented for GRPO/TempFlow tiny smoke |
| SD1.5 | planned after tiny diffusion |
| SD3 | lazy TempFlow bridge, real adapter pending |
| FLUX | lazy TempFlow bridge, real adapter pending |
| QwenImage | lazy TempFlow bridge, real adapter pending |
| CogVideoX | placeholder, World-R1 path later |
| Inferix | eval placeholder |

## Priority Summary

The new priority order is:

```text
1. TinyDiffusion RL
2. TempFlow branching on tiny/small image models
3. Flash-GRPO single-step on tiny/small image models
4. SD1.5 LoRA RL
5. SD3/FLUX/QwenImage integration
6. Wan low-VRAM smoke
7. World-R1 reward/camera/video extensions
8. Inferix eval/preview/profiling
```

The core project identity is:

```text
VisualRL is an integration infra for World-R1, Flash-GRPO, TempFlow-GRPO,
and Inferix. GenRL is a reference for good engineering patterns only.
```
