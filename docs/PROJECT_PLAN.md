# VisualRL Project Plan

This is the canonical plan for the current project direction. It supersedes the
older GenRL-centered planning note and uses the latest principle:

- `GenRL-main` is only an engineering reference.
- `World-R1-main`, `Flash-GRPO-main`, `TempFlow-GRPO-main`, and `Inferix-main`
  are the four projects this repository must integrate.
- Experiments should start with small image/tiny diffusion models, then move to
  larger image models, and only later to Wan/World-R1 video training.

## Current Position

Current milestone: `visual_rl` v0.5.0.

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
- TempFlow branching rollout and branch/timestep credit assignment on tiny diffusion
- Flash-GRPO single-step rollout and selected-timestep loss on tiny diffusion
- SD1.5 LoRA adapter and image trainer entry point
- SD3.5 TempFlow reference adapter
- FLUX/QwenImage TempFlow adapter entries
- config validation for incompatible rollout/algorithm pairs
- CLI smoke commands for mock, TempFlow tiny, Flash tiny, image training,
  adapter probes, World-R1 plan, and Wan plan

Current implementation status:

| Area | Status |
| --- | --- |
| Core contracts/config/trainer shell | implemented for local smoke |
| TinyDiffusion adapter | implemented |
| Prompt-color reward | implemented |
| Full-trajectory GRPO | implemented for mock/tiny paths |
| TempFlow branching | implemented for tiny diffusion |
| Flash-GRPO single-step | implemented for tiny diffusion |
| SD1.5 | LoRA adapter and numeric-smoke CLI implemented; real GPU numeric probe deferred because no checkpoint path is available |
| SD3 | TempFlow reference adapter implemented; adapter-level server parity is now the next active real-model target |
| FLUX/QwenImage | TempFlow adapter entries implemented; low-resolution smoke pending |
| World-R1 | dry-run launcher and reward client stubs; real integration pending |
| Wan | runtime plan shell only; real checkpoint/logprob training pending |
| Inferix | eval placeholder only |

Current validation status:

```bash
conda run -n visual-rl python -m pytest -q
conda run -n visual-rl python -m ruff check visual_rl tests
conda run -n visual-rl python -m visual_rl.cli smoke-imports
conda run -n visual-rl python -m visual_rl.cli adapter-probe --adapter sd15_lora
conda run -n visual-rl python -m visual_rl.cli sd15-numeric-smoke --help
conda run -n visual-rl python -m visual_rl.cli adapter-probe --adapter sd3_tempflow
conda run -n visual-rl python -m visual_rl.cli smoke-mock --output-dir runs/smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli tempflow-smoke --output-dir runs/tempflow_tiny_smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli flash-smoke --output-dir runs/flash_tiny_smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli wan-plan --output-dir runs/wan_runtime_plan
```

Latest local validation:

- `compileall`: passes for `visual_rl` and `tests`.
- `pytest`: 32 tests pass.
- `ruff`: passes.
- `smoke-imports`: registers `grpo`, `flash_grpo`, `tempflow_grpo`,
  `mock_wan`, `tiny_diffusion`, SD1.5, SD3, FLUX, QwenImage,
  `world_r1_wan_legacy`, `mock`, `prompt_color`, and `remote_pickle`.
- `adapter-probe`: passes for SD1.5, SD3.5, FLUX, and QwenImage in
  deferred-load mode.
- `tempflow-smoke`: passes locally on tiny diffusion.
- `flash-smoke`: passes locally on tiny diffusion.
- `wan-plan`: still reports empty `model.model_path` and mock rewards by default.
- 2026-05-24 local tiny validation: deterministic golden tests now cover
  rollout cache filenames/metadata, Flash selected timesteps, parent prompt
  indices, TempFlow branch IDs/timestep metadata, advantage masks, and
  advantage expansion.
- 2026-05-24 local CPU experiments:
  `flash-smoke --output-dir /tmp/visualrl_eval_flash --steps 1` passed with
  `flash_active_timestep_frac: 1.0` and `flash_selected_timestep_mean: 0.0`;
  `tempflow-smoke --output-dir /tmp/visualrl_eval_tempflow --steps 1` passed
  with `tempflow_active_timestep_frac: 0.25`.
- 2026-05-24 local advantage-mask validation:
  `smoke-mock --output-dir /tmp/visualrl_eval_mock_adv --steps 1` passed with
  `loss: 0.0`, `approx_kl: 0.0`, and `group_size: 2.0`;
  `flash-smoke --output-dir /tmp/visualrl_eval_flash_adv --steps 1` passed
  with `flash_active_timestep_frac: 1.0`,
  `flash_rectification_weight_mean: 1.0`, and
  `flash_selected_timestep_mean: 0.0`;
  `tempflow-smoke --output-dir /tmp/visualrl_eval_tempflow_adv --steps 1`
  passed with `tempflow_active_timestep_frac: 0.25` and
  `tempflow_noise_weight_mean: 1.0`.
- 2026-05-24 local tiny algorithm comparison validation:
  `tests/test_tiny_algorithm_comparison.py` passed and compares
  full-trajectory GRPO, Flash single-step GRPO, and TempFlow branch-timestep
  credit assignment on the same prompts, rewards, timesteps, and logprob
  fixture; `smoke-mock --output-dir /tmp/visualrl_eval_compare_mock --steps 1`,
  `flash-smoke --output-dir /tmp/visualrl_eval_compare_flash --steps 1`, and
  `tempflow-smoke --output-dir /tmp/visualrl_eval_compare_tempflow --steps 1`
  passed locally.
- 2026-05-24 config validation:
  `tests/test_config_v02.py` passed with 3 tests. Positive config load passed
  for `flash_tiny_single_step.yaml` with `GOOD flash_grpo/single_step`.
  A temporary invalid config with `algorithm.name: flash_grpo` and
  `sample.name: full_trajectory` was rejected with
  `ValueError: Incompatible config: algorithm.name='flash_grpo' requires
  sample.name in {'single_step'}, got sample.name='full_trajectory'.`
- 2026-05-24 SD1.5 numeric-smoke CLI validation:
  `tests/test_real_image_adapters.py` passed with 3 tests, including the
  mocked `sd15-numeric-smoke` CLI contract and explicit model-path propagation.
  `sd15-numeric-smoke --help` passed and exposes the required `--model-path`.
  `adapter-probe --adapter sd15_lora` returned deferred-load status
  `{"adapter": "sd15_lora", "loaded": false, "model_path": ""}`.
- 2026-05-24 remote SD1.5 real numeric-smoke attempt:
  `ssh v-qiaoqifan@10.130.140.73 'hostname; nvidia-smi; nvidia-smi pmon -c 1'`
  reported host `node01`; GPU 1 was idle at 3 MiB, 0% utilization, and no
  `pmon` process, while GPUs 0 and 2-7 were busy. Read-only checkpoint search
  over `$HOME`, `/data`, `/mnt`, `/share`, `/workspace`, and `/home` found
  SD3.5 and Wan Diffusers paths only. No usable SD1.5 Diffusers checkpoint was
  found, so the real GPU numeric smoke is deferred and no longer gates SD3.5.

Server validation completed so far:

- CPU-only server TempFlow tiny smoke.
- CPU-only server Flash tiny smoke.
- Two-GPU TempFlow tiny correctness probe on 5090 GPUs.
- Real SD3.5-medium TempFlow reference-script smoke on one 5090 GPU.

Real Wan checkpoint loading, World-R1 reward server calls, real SD3/FLUX/
QwenImage reward-improvement runs through `VisualRLTrainer`, and Inferix eval
are not validated yet. SD1.5 has a local numeric-smoke CLI contract, but the
real model run is deferred because no valid SD1.5 Diffusers checkpoint path is
available. SD1.5 is no longer a gate for SD3.5 adapter parity work.

Known validation gaps are tracked in
[`docs/EXPERIMENT_VALIDATION_BACKLOG.md`](EXPERIMENT_VALIDATION_BACKLOG.md).
The latest TempFlow GPU validation is recorded in
[`docs/TEMPFLOW_GPU_VALIDATION_2026_05_23.md`](TEMPFLOW_GPU_VALIDATION_2026_05_23.md).
The latest SD3.5 TempFlow reference validation is recorded in
[`docs/SD35_TEMPFLOW_VALIDATION_2026_05_23.md`](SD35_TEMPFLOW_VALIDATION_2026_05_23.md).

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
image-model adapter loops are stable.

## Code Plan

### Phase A: Tiny Diffusion RL

Status: complete for the current local smoke target.

Implemented files:

```text
visual_rl/model_adapters/tiny_diffusion.py
visual_rl/rewards/image_rewards.py
visual_rl/configs/presets/tempflow_tiny_branching.yaml
visual_rl/configs/presets/flash_tiny_single_step.yaml
tests/test_tempflow_branching.py
tests/test_flash_grpo.py
tests/test_rollout_cache.py
tests/test_advantage_masks.py
tests/test_tiny_algorithm_comparison.py
```

Implemented behavior:

- CPU or single-GPU runnable
- tiny RGB image output
- prompt-color reward
- real `sample()` and `recompute_log_probs()` contract
- GRPO-style update changes trainable parameters in validation probes
- TempFlow reward trend improved in two 100-step 5090 correctness probes
- rollout cache and checkpoint path are exercised
- deterministic golden tests cover cache filenames/metadata, selected
  timesteps, branch IDs, parent prompt indices, advantage masks, and advantage
  expansion across full-trajectory GRPO, Flash single-step, and TempFlow
  branching
- deterministic tiny comparison covers full-trajectory GRPO, Flash
  single-step GRPO, and TempFlow branch-timestep credit assignment on the same
  prompt/reward/logprob/timestep fixture

This is the lowest-cost end-to-end testbed for the whole infra.

### Phase B: Flash-GRPO Abstraction

Status: complete for the tiny-diffusion smoke target.

Implement low-cost single-step training before touching Wan.

Implemented files:

```text
visual_rl/rollout/single_step.py
visual_rl/algorithms/flash_grpo.py
visual_rl/integrations/flash_grpo/timestep_sampler.py
visual_rl/integrations/flash_grpo/rectification.py
```

Implemented features:

- selected timestep rollout: implemented for tiny diffusion
- iso-temporal grouping: implemented
- scheduler-aware timestep weights: implemented as a tiny scheduler formula
- temporal gradient rectification: implemented in `flash_grpo`
- prompt-wise sample expansion for low-cost groups

Still pending before real-model use:

- 20-50 step reward-trend validation for Flash tiny.
- Scheduler-specific rectification parity against Flash-GRPO reference code.
- SD3/FLUX/QwenImage single-step paths after adapter parity. SD1.5
  single-step LoRA can be revisited only if a checkpoint path becomes
  available.

### Phase C: TempFlow Branching Abstraction

Status: complete for the tiny-diffusion smoke target.

Implement branching/process-reward mechanics on small models first.

Implemented files:

```text
visual_rl/rollout/branching.py
visual_rl/algorithms/tempflow_grpo.py
visual_rl/integrations/tempflow_grpo/branching.py
```

Implemented features:

- main trajectory plus branch samples: implemented for tiny diffusion
- branch IDs in `RolloutBatch`: implemented
- branch/timestep-level reward assignment: implemented in `tempflow_grpo`
- branch-level advantage computation: implemented through shared GRPO advantages
- 100-step reward-improvement probes on two 5090 GPUs: completed

Still pending before real-model use:

- sequential branch execution for large models under 32 GB VRAM
- adapter-level parity with the SD3.5 reference script
- SD3 real `sample()` and `recompute_log_probs()` contracts through
  `visual_rl`

### Phase D: SD1.5 Small Image Model

Add Stable Diffusion 1.5 LoRA RL as an optional small-model path. This phase is
implemented at the CLI/adapter-contract level, but the real GPU smoke is
deferred and must not block SD3/FLUX/QwenImage integration.

Implemented files:

```text
visual_rl/model_adapters/sd15.py
visual_rl/trainer/image_trainer.py
visual_rl/configs/presets/sd15_lora_rl.yaml
visual_rl/cli.py
tests/test_real_image_adapters.py
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

Status: code-ready for a minimal SD1.5 LoRA numeric smoke, but real-model GPU
validation is skipped for now because the server does not currently have a
usable SD1.5 Diffusers checkpoint path.

The `sd15-numeric-smoke` CLI requires an explicit `--model-path`, constructs
the `sd15_lora` adapter, calls `sample()`, runs strict rollout validation,
calls `recompute_log_probs()`, checks finite media/logprob tensors, checks
logprob agreement, and reports JSON metrics.

Optional command once a checkpoint exists:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n visual-rl python -m visual_rl.cli sd15-numeric-smoke --model-path /path/to/sd15-diffusers --resolution 128 --num-steps 1 --prompt "a red square"
```

### Phase E: TempFlow Model Expansion

With SD1.5 deferred, integrate models from the TempFlow path next:

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

Important update: the legacy TempFlow SD3.5-medium reference script already
passes a minimal server smoke on one 5090 GPU. The next step is not to prove the
reference script again; it is to wrap that working behavior behind the
`visual_rl` adapter contract and run it through `VisualRLTrainer`.

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
- resolution: tiny RGB output
- reward: prompt color match
- group size: 4 or 8
- algorithm: GRPO
- expected result: reward rises, loss/clip/approx KL are logged, checkpoint saves
- status: base tiny adapter path is implemented; deterministic comparison
  against Flash and TempFlow is covered; long reward-trend comparison across
  GRPO/Flash/TempFlow is still pending

### Experiment 1: Tiny Flash-GRPO

Purpose: prove low-cost single-step optimization.

- selected timestep: implemented
- iso-temporal grouping: implemented
- tiny smoke: passes
- next: 20-50 step reward-trend validation
- next: compare speed/reward curve against full-trajectory GRPO

### Experiment 2: Tiny TempFlow Branching

Purpose: prove branch credit assignment.

- main path plus branch samples: implemented
- reward assigned to branch timestep: implemented
- two-GPU 5090 correctness probe: passed
- deterministic comparison with full GRPO and Flash single-step GRPO: passed

### Experiment 3: SD1.5 LoRA RL

Purpose: first real image diffusion RL curve.

- 256x256
- LoRA rank 8
- group size 4 sequential
- cheap reward
- 1x 5090 runnable
- status: LoRA adapter, image trainer entry point, and `sd15-numeric-smoke`
  CLI are implemented; local CLI contract test passes; real GPU numeric smoke
  is deferred because no SD1.5 Diffusers checkpoint path is available. This
  experiment is no longer gating SD3.5 adapter parity.

### Experiment 4: SD3 TempFlow-Style RL

Purpose: begin real TempFlow integration.

- low resolution
- branching rollout
- small group
- reward cache
- no large multi-reward setup initially
- status: reference TempFlow SD3.5 smoke passed; `visual_rl` adapter parity is
  the next active real-model validation target

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

Already available or partially available:

- reward cache with media hash and reward version
- rollout cache for smoke/debug runs
- strict `fail_policy: raise` path for configured rewards
- config validation for known rollout/algorithm compatibility
- tiny smoke commands for Flash and TempFlow

Still missing as CLI tools:

- `visual-rl validate-config`
- `visual-rl rollout-probe`
- `visual-rl reward-probe`
- `visual-rl adapter-probe`

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
| Tiny diffusion | implemented for GRPO/TempFlow/Flash tiny smoke |
| SD1.5 | LoRA adapter, image trainer path, and numeric-smoke CLI implemented; GPU numeric probe deferred until a checkpoint path exists |
| SD3 | TempFlow reference adapter implemented; SD3.5 reference script smoke passed; adapter parity is next |
| FLUX | TempFlow adapter implemented; low-resolution smoke pending |
| QwenImage | TempFlow adapter implemented; low-resolution smoke pending |
| CogVideoX | placeholder, World-R1 path later |
| Inferix | eval placeholder |

## Priority Summary

The current priority order is:

```text
1. Run SD3.5 TempFlow adapter parity smoke through `VisualRLTrainer`
2. Validate FLUX and QwenImage low-resolution adapter smoke paths
3. Optionally return to SD1.5 only if a valid Diffusers checkpoint path appears
4. Add World-R1 reward/camera probes
5. Run Wan low-VRAM smoke only after image model contracts are stable
6. Add Inferix eval/preview/profiling backend
```

The core project identity is:

```text
VisualRL is an integration infra for World-R1, Flash-GRPO, TempFlow-GRPO,
and Inferix. GenRL is a reference for good engineering patterns only.
```

## Static Review Notes - 2026-05-23

Scope: local logic and syntax review only. No server was started and no runtime
training service was contacted.

Checks run:

```bash
conda run -n visual-rl python -m compileall -q visual_rl tests
conda run -n visual-rl python -m ruff check visual_rl tests
conda run -n visual-rl python -m pytest -q tests
conda run -n visual-rl python -m visual_rl.cli smoke-imports
conda run -n visual-rl python -m visual_rl.cli smoke-mock --output-dir /tmp/visual_rl_review_smoke_mock --steps 1
conda run -n visual-rl python -m visual_rl.cli tempflow-smoke --output-dir /tmp/visual_rl_review_tempflow --steps 1
conda run -n visual-rl python -m visual_rl.cli flash-smoke --output-dir /tmp/visual_rl_review_flash --steps 1
conda run -n visual-rl python -m visual_rl.cli wan-plan --output-dir /tmp/visual_rl_review_wan_plan
```

Results:

- Syntax compile: passed.
- Ruff: passed.
- Project tests under `tests/`: 15 passed.
- Local CLI smoke commands: passed.
- Direct `conda run -n visual-rl python -m pytest -q` currently fails because
  pytest also collects `reference_code/Inferix-main/tests`, which imports the
  uninstalled `inferix` package.

Findings to fix before the next real-model phase:

1. Pytest discovery leaks into vendored reference code.
   `pyproject.toml` has no pytest `testpaths` or `norecursedirs`, so running
   plain `pytest` collects tests under `reference_code/`. This breaks local
   validation on `reference_code/Inferix-main/tests/unit/test_profiling.py`.
   Add pytest config that limits collection to `tests/` or ignores
   `reference_code/`.

2. Full-trajectory `GRPOAlgorithm` is not device-safe for GPU adapters.
   `visual_rl/algorithms/grpo.py` uses CPU advantages from
   `AdvantageComputer` directly with `new_log_probs`. Flash and TempFlow move
   rewards to `new_log_probs.device`, but GRPO does not. A CUDA image/video
   adapter can hit a CPU/CUDA tensor mismatch. At loss entry, coerce
   advantages and `batch.old_log_probs` to `new_log_probs.device` and dtype.

3. Reward normalization is computed but not used for training.
   `RewardRouter.score()` returns `normalized_total`, but
   `VisualRLTrainer.train()` passes `rewards.weighted_total` into
   `AdvantageComputer`. As a result, `rewards.normalize: per_batch` currently
   affects the returned `RewardBatch` only, not the optimization signal. Decide
   whether trainer advantages should use `normalized_total`, or remove/rename
   the config to avoid a false control knob.

4. Reward cache media hashing collides for non-tensor media.
   `stable_hash_media()` hashes tensor bytes, but for numpy arrays, PIL images,
   lists, or decoded frame containers it falls back to type and shape only.
   Same-shape different images therefore share a reward cache key. This is
   safe for the current tensor-based tiny path, but unsafe for SD/FLUX/QwenImage
   adapters if they return non-tensor media. Add content hashing for numpy/PIL
   media or disable caching for unknown media types.

5. `weight_advantages=True` reports misleading prompt stats.
   In `AdvantageComputer.compute()`, weighted-advantage mode updates
   `reward_trackers`, but the emitted `group_size` and `trained_prompt_num`
   metrics are read from `total_tracker`, which stays empty. The advantages are
   shaped, but the metrics report `0.0`. Aggregate stats from reward trackers
   or update `total_tracker` consistently.

6. `RolloutBatch.validate_lightweight()` only checks prompt/metadata length.
   It does not verify that media, latents, timesteps, logprobs, KL, and branch
   IDs have compatible batch dimensions. This lets adapter mistakes surface
   later inside loss computation. Add a stricter optional validation path for
   `visual-rl rollout-probe`, `adapter-probe`, and smoke tests before wiring
   real image adapters.

Fix status after v0.5 implementation:

- Plain `pytest` is scoped to `tests/` and ignores `reference_code/`.
- GRPO, Flash-GRPO, and TempFlow-GRPO move old logprobs/KL to the new-logprob
  device and dtype before loss math.
- `VisualRLTrainer` now feeds `rewards.normalized_total` into advantage
  computation, so `rewards.normalize` affects the optimization signal.
- Reward media hashing includes tensor dtype/shape/content, numpy content, PIL
  content, lists, tuples, and dicts.
- Weighted-advantage prompt metrics are aggregated from the per-reward trackers.
- `RolloutBatch.validate_lightweight(strict=True)` checks media, latent,
  timestep, logprob, KL, and branch batch dimensions.
- Adapter probes pass for SD1.5, SD3.5, FLUX, and QwenImage in deferred-load
  mode. SD1.5 also has a local `sd15-numeric-smoke` CLI contract test, but the
  real GPU numeric probe is deferred until a valid SD1.5 Diffusers checkpoint
  path is available. SD3.5 adapter parity is now the active next step.
