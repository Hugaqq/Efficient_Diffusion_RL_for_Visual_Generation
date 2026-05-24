# VisualRL Project Plan

This is the canonical plan for the current project direction. It supersedes the
older GenRL-centered planning note and uses the latest principle:

- `GenRL-main` is only an engineering reference.
- `World-R1-main`, `Flash-GRPO-main`, `TempFlow-GRPO-main`, and `Inferix-main`
  are the four projects this repository must integrate.
- Tiny diffusion is now a completed regression gate, not the mainline. The
  current mainline is a complete small-scale SD3.5 image RL loop before moving
  to FLUX/QwenImage, World-R1/Wan, and Inferix.

## Current Goal

Build a real, inspectable SD3.5 mini end-to-end pipeline:

```text
SD3.5 checkpoint
  -> VisualRL SD3 adapter sample
  -> save preview PNGs
  -> reward router
  -> TempFlow/GRPO update
  -> LoRA checkpoint
  -> before/after preview
  -> metrics and blocker documentation
```

Success for the next phase means the project can generate real SD3.5 preview
images through VisualRL, run a bounded low-cost SD3.5 trainer step without
NaN/OOM, save LoRA/checkpoint artifacts, and report reward/logprob/KL metrics.
Tiny probes remain mandatory regression checks after shared infra edits, but no
new tiny-only feature work should be selected unless it directly protects the
real SD3.5 path.

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
- SD3 numeric-smoke CLI contract with explicit `--repo-root`
- shared TempFlow image numeric-smoke CLI contract for SD3, FLUX, and QwenImage
- `image-preview` CLI for registry-backed adapter PNG previews and metadata
- SD3 adapter local compatibility coverage for TempFlow reference pipeline
  kwargs, return shape, no-KL sample returns, and dtype helper behavior
- SD3 `recompute_log_probs()` CPU fake transformer/SDE dtype coverage for
  guidance off/on
- FLUX/QwenImage TempFlow adapter entries
- config validation for incompatible rollout/algorithm pairs
- `validate-config` CLI for lightweight preset/config validation
- `rollout-probe` CLI for lightweight adapter/rollout contract validation
- `reward-probe` CLI for lightweight reward routing/config validation
- `checkpoint-inventory` CLI for local/remote Diffusers checkpoint discovery
- `world-r1-reward-server-probe` CLI for bounded World-R1 endpoint payload
  checks with synthetic image/video media
- `tiny-loss-probe` CLI for deterministic loss-descent infra validation
- CLI smoke commands for mock, TempFlow tiny, Flash tiny, image training,
  config validation, adapter probes, World-R1 plan, and Wan plan

Current implementation status:

| Area | Status |
| --- | --- |
| Core contracts/config/trainer shell | implemented for local smoke |
| TinyDiffusion adapter | implemented; regression gate only |
| Prompt-color reward | implemented |
| Full-trajectory GRPO | implemented for mock/tiny paths |
| TempFlow branching | implemented for tiny diffusion |
| Flash-GRPO single-step | implemented for tiny diffusion |
| SD1.5 | LoRA adapter and numeric-smoke CLI implemented; real GPU numeric probe deferred because no checkpoint path is available |
| SD3 | current mainline; TempFlow reference adapter, numeric-smoke CLI contract with `--repo-root`, local reference-pipeline compatibility coverage, CPU fake recompute dtype coverage, real direct Python SD3.5 checkpoint/CUDA sample/recompute parity, `image-preview` PNG artifact smoke, remote `tempflow-image-numeric-smoke --adapter sd3_tempflow`, staged remote 1-step/5-step bounded trainer runs, one-step resume smoke, and guarded 20-step trend smoke passed with checkpoint, metrics, parameter-delta, and before/after PNG artifacts; meaningful convergence remains unproven |
| FLUX/QwenImage | TempFlow adapter entries and shared low-resolution numeric-smoke CLI/mocked contracts implemented; real smoke blocked by missing checkpoint paths |
| World-R1 | dry-run launcher, endpoint URL validation, mock reward probe, and bounded reward-server probe CLI implemented; real reward-server success and real Wan sample/logprob validation pending |
| Wan | runtime plan shell with model-path/reward-server readiness checks; real checkpoint loading, sample/logprob path, and reward-backed training pending |
| Inferix | dry-run eval/preview/profiling plan backend implemented with execution disabled; real checkpoint execution and logprob/recompute contracts pending |

Current validation status:

```bash
conda run -n visual-rl python -m pytest -q
conda run -n visual-rl python -m ruff check visual_rl tests
conda run -n visual-rl python -m visual_rl.cli smoke-imports
conda run -n visual-rl python -m visual_rl.cli adapter-probe --adapter sd15_lora
conda run -n visual-rl python -m visual_rl.cli sd15-numeric-smoke --help
conda run -n visual-rl python -m visual_rl.cli sd3-numeric-smoke --help
conda run -n visual-rl python -m visual_rl.cli tempflow-image-numeric-smoke --help
conda run -n visual-rl python -m visual_rl.cli sd3-bounded-trainer-smoke --help
conda run -n visual-rl python -m visual_rl.cli adapter-probe --adapter sd3_tempflow
conda run -n visual-rl python -m visual_rl.cli validate-config visual_rl/configs/presets/*.yaml
conda run -n visual-rl python -m visual_rl.cli rollout-probe visual_rl/configs/presets/world_r1_wan_v02_mock.yaml --seed 123
conda run -n visual-rl python -m visual_rl.cli rollout-probe visual_rl/configs/presets/flash_tiny_single_step.yaml --batch-size 1 --num-steps 4 --seed 77
conda run -n visual-rl python -m visual_rl.cli rollout-probe visual_rl/configs/presets/tempflow_tiny_branching.yaml --batch-size 1 --num-steps 4 --seed 88
conda run -n visual-rl python -m visual_rl.cli reward-probe world_r1_wan_v02_mock --seed 123
conda run -n visual-rl python -m visual_rl.cli reward-probe flash_tiny_single_step --batch-size 2 --seed 77
conda run -n visual-rl python -m visual_rl.cli checkpoint-inventory --help
conda run -n visual-rl python -m visual_rl.cli world-r1-reward-server-probe --help
conda run -n visual-rl python -m visual_rl.cli smoke-mock --output-dir runs/smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli tempflow-smoke --output-dir runs/tempflow_tiny_smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli flash-smoke --output-dir runs/flash_tiny_smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli tiny-loss-probe --output-dir runs/tiny_loss_probe --steps 100 --learning-rate 0.1
conda run -n visual-rl python -m visual_rl.cli wan-plan --output-dir runs/wan_runtime_plan
```

Latest local validation:

- `compileall`: passes for `visual_rl` and `tests`.
- `pytest`: 95 tests pass in the latest 2026-05-24 local pass.
- `ruff`: passes.
- `smoke-imports`: registers `grpo`, `flash_grpo`, `tempflow_grpo`,
  `mock_wan`, `tiny_diffusion`, SD1.5, SD3, FLUX, QwenImage,
  `world_r1_wan_legacy`, `mock`, `prompt_color`, and `remote_pickle`.
- `checkpoint-inventory`: help output passes; local tests cover Diffusers
  adapter classification, missing required adapters, and malformed entries.
- `world-r1-reward-server-probe`: help output passes; local tests cover
  `reward_general` image payloads, `reward_3d` video payloads, timeout/retry
  failures, invalid response shape, URL validation, and urllib fallback when
  `requests` is absent. A loopback probe against `127.0.0.1:9` returns
  structured `Connection refused`; no real reward server has passed yet.
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
- 2026-05-24 local tiny loss-descent validation:
  `tiny-loss-probe --output-dir /tmp/visualrl_tiny_loss_probe --steps 100`
  passed with `loss_start: 0.0510348`, `loss_end: 6.127e-7`,
  `bias_error_start: 0.565685`, `bias_error_end: 0.0331612`, and
  `grpo_policy_loss_start: -0.726546`,
  `grpo_policy_loss_end: -0.998916`.
- 2026-05-24 remote GPU tiny loss-descent validation on
  `v-qiaoqifan@10.130.140.73`, `CUDA_VISIBLE_DEVICES=2`:
  `tiny-loss-probe --device cuda --steps 100` passed with
  `loss_start: 0.0511422`, `loss_end: 6.099e-7`,
  `bias_error_start: 0.565685`, and `bias_error_end: 0.0336335`.
- 2026-05-24 remote staged SD3.5 full-loop validation on `node01` GPU1:
  `remote-sd3-cli-smoke --execute` now runs `image-preview`,
  `tempflow-image-numeric-smoke`, and `sd3-bounded-trainer-smoke` in order with
  an `nvidia-smi pmon -i "$GPU" -c 1` idle guard and artifact checks. A
  1-step stage at
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_full_loop_20260524_1215`
  passed with preview `media_shape: [1, 3, 256, 256]`,
  `latents_shape: [1, 2, 16, 32, 32]`, numeric
  `max_abs_logprob_delta: 0.0`, bounded `latest.step: 1`,
  `checkpoint_000001`, `reward_mean: 0.5154668688774109`,
  `parameter_delta_abs_max: 9.99999883788405e-06`, and before/after preview
  rewards `0.5285151600837708` -> `0.5295270681381226`. A 5-step stage at
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_full_loop_20260524_1215_5step`
  passed with bounded `latest.step: 5`, `checkpoint_000005`, 5 metrics rows,
  final `reward_mean: 0.5767983794212341`,
  `parameter_delta_abs_max: 5.019741365686059e-05`,
  `parameter_delta_l2: 0.0421081454686555`, and before/after preview rewards
  `0.5285151600837708` -> `0.5278942584991455`. Main preview, bounded before,
  and bounded after PNGs were verified as 256x256 RGB. GPU1 was `3MiB`,
  `0%`, and `pmon` showed no process before runs; the final script check also
  returned GPU1 to `3MiB` with no process. This validates bounded plumbing,
  checkpoint/metrics writing, PNG artifacts, and parameter updates. Later
  resume and 20-step trend smokes also passed; these runs do not prove
  meaningful convergence or paper-scale training.
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
  `visual-rl validate-config` is implemented and validated as a lightweight
  JSON CLI. It accepts one or more config paths, calls `load_config`, validates
  known registry names for model adapters, algorithms, and reward clients,
  checks nonempty reward weights, and checks reward weight/client alias
  consistency. It does not load models, construct trainers, execute rewards,
  or write output directories. `tests/test_config_v02.py` passed with 12 tests,
  `validate-config visual_rl/configs/presets/*.yaml` reported all 10 presets
  valid, `validate-config --help` passed, full `tests/` now passes with 46
  tests, and Ruff passed for `visual_rl tests`. Evaluator coverage found and
  fixed a missing reward-client alias check.
- 2026-05-24 rollout-probe CLI validation:
  `visual-rl rollout-probe` is implemented and validated as a lightweight JSON
  CLI. It loads config, validates registry/config names, constructs only the
  selected adapter and rollout engine, batches prompts, calls
  `rollout.sample()`, validates the returned `RolloutBatch` strictly by
  default, and emits JSON shapes/metadata. It does not construct
  `VisualRLTrainer`, optimizer, rewards, checkpoints, cache writes, or output
  directories. Local probes passed for `world_r1_wan_v02_mock.yaml`,
  `flash_tiny_single_step.yaml`, and `tempflow_tiny_branching.yaml`; evaluator
  confirmed `runs/world_r1_wan_v02_mock`, `runs/flash_tiny_single_step`, and
  `runs/tempflow_tiny_branching` were not created. Missing config paths and the
  SD3 real adapter preset without local dependencies/checkpoint return
  non-zero structured JSON without a Python traceback. This does not prove real
  SD3.5 CUDA parity.
- 2026-05-24 reward-probe CLI validation:
  `visual-rl reward-probe` is implemented and validated as a lightweight JSON
  CLI. It loads config, reuses config/registry/reward alias validation, takes a
  small `PromptDataset` batch, generates deterministic synthetic RGB media,
  runs `RewardRouter(config.rewards, cache_dir=None)`, and emits a JSON reward
  summary. It does not construct `VisualRLTrainer`, load model adapters,
  construct rollout engines, run optimizers, write output directories,
  checkpoints, or reward cache entries. Evaluator checks passed for mock
  World-R1/Wan rewards and `prompt_color` rewards; invalid reward alias config
  returns non-zero structured JSON without a Python traceback. Preset coverage
  passed locally on 9/10 shipped presets; `world_r1_wan_v01_server.yaml`
  returns the expected structured dataset failure because it is a server/legacy
  plan preset without local prompts. This validates reward routing/config with
  synthetic media, not real adapter output shape, CUDA, or model correctness.
- 2026-05-24 reward-probe synthetic media sizing validation:
  synthetic media sizing now resolves dimensions in priority order from
  `model.extra.height`/`model.extra.width`, `model.extra.resolution`,
  `model.extra.image_size`, top-level model `height`/`width`/`resolution`/
  `image_size`, then `media_shape`, and finally defaults to 16x16. The JSON
  output includes `media_height` and `media_width`. Invalid dimensions return a
  structured JSON failure without a Python traceback. Evaluator checks passed
  for `sd3_tempflow_adapter.yaml`, `flux_tempflow_adapter.yaml`,
  `qwenimage_tempflow_adapter.yaml`, and `sd15_lora_rl.yaml` with 256x256
  synthetic media; `tests/test_config_v02.py` passed with 16 tests, full
  `tests/` passed with 46 tests, Ruff passed for `visual_rl tests`, and
  `compileall` passed for `visual_rl tests`. This remains a synthetic media
  reward routing/config probe only; it does not validate real adapter output,
  CUDA execution, or model correctness.
- 2026-05-24 SD1.5 numeric-smoke CLI validation:
  `tests/test_real_image_adapters.py` includes the mocked
  `sd15-numeric-smoke` CLI contract and explicit model-path propagation.
  `sd15-numeric-smoke --help` passed and exposes the required `--model-path`.
  `adapter-probe --adapter sd15_lora` returned deferred-load status
  `{"adapter": "sd15_lora", "loaded": false, "model_path": ""}`.
- 2026-05-24 SD3 numeric-smoke CLI validation:
  `tests/test_real_image_adapters.py` passed with 4 tests, including mocked
  `sd15-numeric-smoke` and `sd3-numeric-smoke` CLI contracts. The SD3 CLI
  requires explicit `--model-path`, constructs `sd3_tempflow`, calls
  `sample()`, runs strict rollout validation, calls `recompute_log_probs()`,
  checks finite tensors, shape equality, and logprob agreement, and reports
  JSON metrics. `sd3-numeric-smoke --help`, full `tests/`, Ruff, and
  `py_compile` all passed locally. This entry was contract coverage only; later
  direct Python SD3.5 checkpoint/CUDA adapter parity and the staged remote CLI
  path both passed.
- 2026-05-24 SD3 adapter local compatibility hardening:
  evaluator validation passed `tests/test_real_image_adapters.py` with 8
  tests, `tests/test_base_trainer.py` with 1 test, full `tests/` with 52
  tests, Ruff, and `compileall`. The SD3 adapter now has local coverage that
  `_call_pipeline_with_logprob()` filters unsupported kwargs, passes
  `return_dict=False` when the reference pipeline supports `return_dict`,
  accepts the TempFlow reference pipeline 3-tuple no-KL return from `sample()`,
  zero-fills `kl` to match `old_log_probs`, and tests `_transformer_dtype()`
  at helper level. Evaluator comparison against `reference_code` found the
  real TempFlow SD3 pipeline signature includes `return_dict` and `kl_reward`,
  so these helper tests track the real reference surface. This was not a GPU
  experiment; it has since been superseded by direct Python SD3.5 CUDA adapter
  parity, staged remote new-CLI validation, and staged 1-step/5-step bounded
  trainer validation.
- 2026-05-24 SD3 recompute dtype local validation:
  evaluator validation passed `tests/test_real_image_adapters.py` with 9
  tests, `tests/test_base_trainer.py` with 1 test, full `tests/` with 55
  tests, Ruff, and `compileall`. CPU fake transformer/SDE coverage now checks
  `SD3TempFlowAdapter.recompute_log_probs()` for `guidance_scale == 1.0`
  direct path without batch doubling and `guidance_scale == 3.0` CFG path with
  batch doubling. In both paths hidden states are cast to transformer dtype
  `torch.float16`, the fake SDE step sees float-cast `noise_pred`, `sample`,
  and `prev_sample`, and embeddings come from `RolloutBatch.model_tensors`
  without invoking text encoders, diffusers, checkpoints, or GPU. This remains
  a local fake test only; direct Python SD3.5 checkpoint/CUDA adapter parity
  and staged remote new-CLI validation later passed. The direct branch check is
  sufficient for the current `> 1.0` guidance split.
- 2026-05-24 BaseTrainer local config compatibility:
  `setup_optimizer()` accepts string-like numeric config values and builds a
  valid AdamW optimizer; evaluator validation passed
  `tests/test_base_trainer.py`.
- 2026-05-24 TempFlow image numeric-smoke CLI validation:
  code worker added `sd3-numeric-smoke --repo-root` and the shared
  `tempflow-image-numeric-smoke --adapter {sd3_tempflow,flux_tempflow,qwenimage_tempflow}`
  path. The shared CLI supports explicit model path, repo root, prompt,
  resolution, denoise steps, guidance, seed, device, dtype, LoRA rank/alpha,
  logprob tolerance, and LoRA disablement; SD3 keeps `--max-sequence-length`.
  Local tests cover SD3 repo-root propagation plus FLUX/QwenImage mocked
  contracts and help output.
- 2026-05-24 image-preview CLI and artifact validation:
  `image-preview` uses the existing `MODEL_ADAPTERS` registry, constructs the
  selected adapter, calls `adapter.sample()`, runs strict
  `RolloutBatch.validate_strict()`, and writes `preview_000.png` plus
  `metadata.json`. Local validation passed with `tiny_diffusion` and
  `media_shape: [1, 3, 4, 4]`; remote SD3.5 validation passed from a staged copy
  on `node01` GPU1 with `sd3_tempflow`, producing a 256x256 PNG and
  `latents_shape: [1, 2, 16, 32, 32]`. Metadata is sufficient for artifact path
  and shape proof, but does not yet record dtype, LoRA rank/alpha, or trainable
  parameter count.
- 2026-05-24 remote staged SD3 CLI validation:
  current `visual_rl` and `pyproject.toml` were staged with `rsync` into
  `/home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/framecode/`;
  this was not a full remote git checkout update. The server env
  `visual-rl` was missing with `EnvironmentLocationNotFound`, while
  `visual-rl-sd35` worked with PyTorch `2.11.0+cu130` and CUDA available.
  On `CUDA_VISIBLE_DEVICES=1`, `tempflow-image-numeric-smoke --adapter
  sd3_tempflow` passed with `valid: true`, `max_abs_logprob_delta: 0.0`,
  finite media/logprobs, and `trainable_parameters: 4694016`. GPU1 returned to
  3 MiB/0% and `pmon` showed no process after the run.
- 2026-05-24 bounded SD3.5 trainer smoke validation:
  code worker added `sd3-bounded-trainer-smoke`, which runs the real
  `ImageRLTrainer`/`VisualRLTrainer` path, disables rollout cache by default,
  saves `summary.json`, and reports metrics/checkpoint/parameter-delta evidence.
  Local checks passed: `conda run -n visual-rl python -m compileall -q visual_rl
  tests`, `conda run -n visual-rl python -m ruff check visual_rl tests` with
  `All checks passed!`, `conda run -n visual-rl python -m pytest -q` with 67
  tests, and `conda run -n visual-rl python -m visual_rl.cli
  sd3-bounded-trainer-smoke --help`.
- 2026-05-24 remote bounded SD3.5 trainer smoke:
  from
  `/home/v-qiaoqifan/visual_rl_experiments/sd3_bounded_trainer_eval_20260524_115914/framecode`,
  `CUDA_VISIBLE_DEVICES=1` and env `visual-rl-sd35` ran
  `sd3-bounded-trainer-smoke --adapter sd3_tempflow` for 1 trainer step with
  SD3.5-medium, resolution 256, `num_steps=2`, `guidance_scale=4.5`,
  bfloat16, LoRA rank 8/alpha 16, and `--disable-rollout-cache`. Result:
  `valid: true`, `latest.step: 1`, `checkpoint_000001` contained `README.md`,
  `adapter_config.json`, and `adapter_model.safetensors`,
  `trainable_parameters: 4694016`, `reward_mean: 0.5165802240371704`,
  `reward_std: 0.011329561471939087`, `old_logprob_mean:
  1.2957947254180908`, `new_logprob_mean: 1.2957947254180908`,
  `logprob_delta_abs_max: 0.0`, `rollout_kl_mean: 0.0`,
  `tempflow_active_timestep_frac: 0.5`, `parameter_delta_abs_max:
  9.999987014452927e-06`, `parameter_delta_l2: 0.015154718097586433`,
  `parameter_delta_nonzero_count: 2334720`, and `rollout_cache_disabled: true`.
  Artifact check passed with `artifact_status=ok`, `metrics_lines=1`, and no
  `rollouts/` directory. GPU1 returned to `3MiB / 32607MiB`, `0%`, with no
  `pmon` process. This proves bounded trainer/checkpoint/parameter update
  plumbing, not reward improvement or post-update policy divergence.
- 2026-05-24 remote SD1.5 real numeric-smoke attempt:
  `ssh v-qiaoqifan@10.130.140.73 'hostname; nvidia-smi; nvidia-smi pmon -c 1'`
  reported host `node01`; GPU 1 was idle at 3 MiB, 0% utilization, and no
  `pmon` process, while GPUs 0 and 2-7 were busy. Read-only checkpoint search
  over `$HOME`, `/data`, `/mnt`, `/share`, `/workspace`, and `/home` found
  SD3.5 and Wan Diffusers paths only. No usable SD1.5 Diffusers checkpoint was
  found, so the real GPU numeric smoke is deferred and no longer gates SD3.5.
- 2026-05-24 remote SD3 facts:
  server `v-qiaoqifan@10.130.140.73` is `node01`. Before the initial probe,
  GPUs 1-7 were idle at 3 MiB/0% and GPU0 had about 3221 MiB allocated. Remote
  checkpoint discovery found SD3.5 and Wan Diffusers paths only, with no FLUX or
  QwenImage checkpoint. Later staged-copy validation used GPU1, the
  `visual-rl-sd35` env, and did not overwrite the old non-git
  `/home/v-qiaoqifan/visual_rl_experiments/framecode` tree.
- 2026-05-24 previous bounded SD3 `VisualRLTrainer` smoke:
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_sd35_parity_20260524_101348/smoke`
  had `latest.json` at step 1 and metrics with `loss: 0.0`,
  `reward_mean: 0.942878246307373`, `reward_std: 0.05712178349494934`, and
  `tempflow_active_timestep_frac: 0.5`. The first two logs captured earlier
  `return_dict` kwarg and float/bfloat16 mismatch failures that local fixes
  addressed. Treat this as a one-step smoke only, not reward improvement.

Server validation completed so far:

- CPU-only server TempFlow tiny smoke.
- CPU-only server Flash tiny smoke.
- Two-GPU TempFlow tiny correctness probe on 5090 GPUs.
- Real SD3.5-medium TempFlow reference-script smoke on one 5090 GPU.
- Real direct Python SD3.5 `SD3TempFlowAdapter` sample/recompute parity on
  `node01` GPU1 with explicit `repo_root`; `valid: true`,
  `max_abs_logprob_delta: 0.0`, media `[1, 3, 256, 256]`, latents
  `[1, 2, 16, 32, 32]`, 4,694,016 trainable LoRA parameters, CUDA bfloat16.
- Real remote SD3.5 `image-preview --adapter sd3_tempflow` from a staged copy on
  `node01` GPU1; `valid: true`, media `[1, 3, 256, 256]`, latents
  `[1, 2, 16, 32, 32]`, and PNG artifact verified as 256x256 RGB.
- Real remote SD3.5 `tempflow-image-numeric-smoke --adapter sd3_tempflow` from
  the same staged copy on `node01` GPU1; `max_abs_logprob_delta: 0.0`,
  finite media/logprobs, and 4,694,016 trainable parameters.
- Real remote SD3.5 `remote-sd3-cli-smoke --execute` staged full loop on
  `node01` GPU1; the runner executed `image-preview`,
  `tempflow-image-numeric-smoke`, and `sd3-bounded-trainer-smoke` in order for
  1-step and 5-step bounded runs. Both runs produced valid 256x256 preview PNGs,
  finite numeric-smoke logprobs, LoRA checkpoints, metrics JSONL, and nonzero
  parameter deltas. The 5-step before/after preview reward was
  `0.5285151600837708` -> `0.5278942584991455`, so the reward trend is
  recorded, not improved.

Real Wan checkpoint loading, World-R1 reward server calls, real SD3/FLUX/
QwenImage reward-improvement runs through `VisualRLTrainer`, and Inferix eval
are not validated yet. SD1.5 and SD3 both have local numeric-smoke CLI
contracts. SD1.5 real-model validation is deferred because no valid SD1.5
Diffusers checkpoint path is available. SD3.5 now has direct Python
checkpoint/CUDA adapter sample/recompute parity, staged remote `image-preview`
PNG artifact validation, staged remote `tempflow-image-numeric-smoke
--adapter sd3_tempflow` parity, and staged remote 1-step/5-step bounded trainer
artifact validation. SD1.5 is no longer a gate for SD3.5 adapter parity work.

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

Still pending before broader real-model use:

- sequential branch execution for large models under 32 GB VRAM
- repeated/extended SD3.5 trend evidence before claiming convergence

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
passes a minimal server smoke on one 5090 GPU, and the local
`sd3-numeric-smoke` CLI contract plus SD3 reference-pipeline compatibility
helpers are covered by mocked/local tests. CPU fake transformer/SDE coverage
now locally validates `recompute_log_probs()` dtype casting for guidance
off/on. A direct Python `SD3TempFlowAdapter` run against the real SD3.5
checkpoint on `node01` GPU1 passed strict validation and recompute parity with
`max_abs_logprob_delta: 0.0`. The staged remote `image-preview --adapter
sd3_tempflow` and `tempflow-image-numeric-smoke --adapter sd3_tempflow` CLIs
also passed from a fresh experiment directory without overwriting shared remote
code. Staged `remote-sd3-cli-smoke --execute` then passed 1-step and 5-step
bounded trainer runs, a one-step resume smoke, and a guarded 20-step trend
smoke with LoRA checkpoints, metrics JSONL, nonzero parameter deltas, and
before/after PNG artifacts. This proves smoke-scale plumbing and short trend
behavior, not meaningful convergence or long training.
FLUX and QwenImage should remain at mocked-contract status until real
checkpoint paths are found.

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

Files to validate next:

```text
visual_rl/eval/inferix_backend.py
visual_rl/integrations/inferix/eval_backend.py
visual_rl/integrations/inferix/profiling.py
```

Current boundary: VisualRL can build dry-run checkpoint preview, profiling, and
long-video eval plans for the vendored Inferix scripts. Execution intentionally
raises until real checkpoint execution, no-decode/profiling behavior, and
logprob/recompute contracts are validated outside the online RL loop.

## Experiment Plan

### Experiment 0: Tiny Diffusion Regression Gate

Purpose: keep a cheap regression gate for shared infra. This is no longer the
mainline experiment.

- model: tiny diffusion adapter
- resolution: tiny RGB output
- reward: prompt color match
- group size: 4 or 8
- algorithm: GRPO
- expected result: reward rises, loss/clip/approx KL are logged, checkpoint saves
- status: base tiny adapter path is implemented; deterministic comparison
  against Flash and TempFlow is covered; `tiny-loss-probe` passed on local CPU
  and remote CUDA. Do not add new tiny-only tasks unless a shared infra change
  breaks this gate.

### Experiment 1: Tiny Flash-GRPO Regression Gate

Purpose: keep Flash-GRPO single-step mechanics covered on the cheapest path.
This is not the current mainline.

- selected timestep: implemented
- iso-temporal grouping: implemented
- tiny smoke: passes
- next: only rerun when Flash/shared rollout code changes

### Experiment 2: Tiny TempFlow Branching Regression Gate

Purpose: keep TempFlow branch credit assignment covered on the cheapest path.
This is not the current mainline.

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

Purpose: deliver the first complete real-model VisualRL loop.

- SD3.5-medium checkpoint
- low resolution, initially 256
- batch size 1, group size 2 or sequential groups
- branch count/window kept at the smallest viable setting
- cheap reward first; no large multi-reward setup initially
- save preview PNGs before and after training
- no rollout cache for long runs unless explicitly debugging
- status: reference TempFlow SD3.5 smoke passed; `visual_rl` direct Python
  adapter sample/recompute parity passed against the real SD3.5 checkpoint on
  CUDA. The local `sd3-numeric-smoke --repo-root` contract, shared
  `tempflow-image-numeric-smoke` contract, SD3 reference-pipeline compatibility
  helpers, CPU fake transformer/SDE `recompute_log_probs()` dtype coverage,
  registry-backed `image-preview`, staged remote SD3.5 PNG preview, and staged
  remote SD3.5 CLI numeric parity are implemented and tested. Staged
  `remote-sd3-cli-smoke --execute` passed 1-step and 5-step bounded trainer
  runs with checkpoints, metrics, parameter-delta evidence, and before/after
  PNG artifacts; the 5-step before/after preview reward was
  `0.5285151600837708` -> `0.5278942584991455`, so the reward trend is
  recorded, not improved.

Immediate SD3.5 sequence:

```text
1. DONE: Run the new SD3 CLI smoke remotely from a staged copy on an idle GPU.
2. DONE: Add `image-preview` CLI and save real PNG outputs from the SD3 adapter.
3. DONE: Add/run bounded SD3.5 trainer smoke: 1-5 steps, fixed prompt, cheap
   reward, strict rollout validation, no rollout cache, explicit output dir.
4. DONE: Save LoRA/checkpoint and after-preview from the same smoke path.
5. DONE: Compare before/after previews as artifacts, and record reward trend,
   finite logprob/KL, and parameter delta.
6. DONE: Add resume-from-checkpoint validation for the same bounded trainer
   path.
7. DONE: Run a guarded 20-step trend smoke after resume/artifact behavior
   stayed stable; treat it as short trend evidence, not convergence.
```

### Experiment 5: Wan Low-VRAM Smoke

Purpose: verify video path only, not final quality.

- Wan2.1 1.3B
- 240x416
- 17 frames
- LoRA
- group size 2 sequential
- mock or cheap reward
- `sde_window_size=1`
- status: planning shell only. A Wan Diffusers checkpoint path was found on the
  server, and `wan-plan` can now surface model-path and reward-server readiness
  metadata, but VisualRL has not loaded the checkpoint through a real
  Wan/World-R1 adapter and has not validated sample/logprob tensors or
  reward-backed training.
- next command once GPU/runtime is available:
  `conda run -n visual-rl python -m visual_rl.cli wan-plan --config visual_rl/configs/presets/wan_runtime_v02_plan.yaml --model-path /mnt/data/v-yingqi/models/Wan2.1-T2V-1.3B-Diffusers --output-dir runs/wan_runtime_plan_real_path`

### Experiment 6: World-R1 Reward and Camera

Purpose: integrate World-R1-specific features after video smoke works.

- reward server probe
- camera metadata probe
- camera-aware latent init
- small video RL run only after probes pass
- current local command:
  `conda run -n visual-rl python -m visual_rl.cli world-r1-plan --model-path /mnt/data/v-yingqi/models/Wan2.1-T2V-1.3B-Diffusers --repo-dir reference_code/World-R1-main --gpus 1 --output-root runs/world_r1_wan_plan`
- current mock reward command:
  `conda run -n visual-rl python -m visual_rl.cli reward-probe visual_rl/configs/presets/world_r1_wan_v02_mock.yaml --seed 123`
- current endpoint probe command shape:
  `conda run -n visual-rl python -m visual_rl.cli world-r1-reward-server-probe --reward reward_general --url http://HOST:PORT/reward --timeout 5 --retries 0 --batch-size 1 --height 64 --width 64`
- missing: a successful real World-R1 reward-server endpoint call. The new
  endpoint probe validates payload shape, URL handling, timeout/retry behavior,
  and structured failures, but synthetic `reward-probe` and loopback failure
  checks do not validate a live server.

### Experiment 7: Inferix Eval

Purpose: reduce video eval cost.

- checkpoint preview
- long-video eval
- profiling
- no online training
- status: dry-run plan backend implemented. Execution remains disabled in
  VisualRL and `online_rl_ready` remains false.
- next validation: instantiate `InferixEvalBackend` against a real checkpoint
  path to record the preview/profile command payload, then execute the generated
  command outside VisualRL only after confirming the checkpoint and vendored
  Inferix environment. Do not wire it into online RL until `sample_with_logprob`
  and `recompute_logprob` contracts exist.

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
- `validate-config` CLI for lightweight config/preset validation
- `rollout-probe` CLI for lightweight adapter/rollout validation
- `reward-probe` CLI for lightweight reward routing/config validation
- config validation for known rollout/algorithm compatibility
- reward weight/client alias mismatch validation
- tiny smoke commands for Flash and TempFlow
- `image-preview` for saving real generated PNGs and metadata from VisualRL
  adapters

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

Non-Wan support now has a complete tiny regression path and an SD3.5 mainline
with staged 1-step/5-step bounded trainer artifact validation:

| Model | Status |
| --- | --- |
| Tiny diffusion | implemented for GRPO/TempFlow/Flash tiny smoke; regression gate only |
| SD1.5 | LoRA adapter, image trainer path, and numeric-smoke CLI implemented; GPU numeric probe deferred until a checkpoint path exists |
| SD3 | current mainline; TempFlow reference adapter, numeric-smoke CLI with `--repo-root`, local reference-pipeline compatibility coverage, CPU fake recompute dtype coverage, SD3.5 reference script smoke, direct Python real checkpoint/CUDA adapter parity, staged remote `image-preview`, staged remote CLI numeric parity, staged 1-step/5-step bounded trainer artifact validation, one-step resume validation, and a guarded 20-step trend smoke passed; meaningful convergence and long training remain unproven |
| FLUX | TempFlow adapter and mocked shared numeric-smoke CLI contract implemented; real low-resolution smoke blocked by missing checkpoint path |
| QwenImage | TempFlow adapter and mocked shared numeric-smoke CLI contract implemented; real low-resolution smoke blocked by missing checkpoint path |
| CogVideoX | placeholder, World-R1 path later |
| Inferix | dry-run eval/preview/profiling plan backend; execution and training contracts pending |

## Priority Summary

The current priority order is:

```text
1. Keep SD3.5 trend experiments guarded and staged; the current 20-step smoke
   is positive plumbing/trend evidence, not convergence.
2. Repeat or extend SD3.5 bounded runs only when GPU idle checks pass, with
   fixed-prompt before/after PNG comparison, reward trend review, finite
   logprob/KL evidence, and LoRA/checkpoint parameter delta.
3. Keep tiny probes as regression gates only; do not add new tiny-only features.
4. Keep FLUX/QwenImage real smokes blocked until checkpoint paths are found.
5. Validate World-R1 reward server/camera probes before online video training.
6. Run Wan low-VRAM smoke only after the image-model loop is stable.
7. Validate Inferix dry-run preview/profiling plans against a real checkpoint,
   then keep execution outside online RL until logprob/recompute contracts exist.
```

Latest SD3.5 evidence from 2026-05-24:

- Resume smoke stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_resume_eval_20260524_171836`
  passed with `resume_loaded: true`, `resume_base_step: 1`,
  `effective_total_step: 2`, finite reward/logprob/KL/clip/TempFlow metrics,
  nonzero parameter delta, checkpoint files, and 256x256 RGB before/after PNGs.
- 20-step trend stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_20step_trend_20260524_1725`
  passed with `latest.step: 20`, 20 metrics rows, checkpoint files, final
  `reward_mean: 0.5748612880706787`, metric reward movement
  `0.5154668688774109 -> 0.5748612880706787`, before/after preview reward
  `0.5285151600837708 -> 0.5303504467010498`, finite KL/clip/logprob fields,
  and nonzero parameter delta (`abs_max: 0.00017927215958479792`, `l2:
  0.09039463499422448`, `nonzero_count: 4669060`). This is short-run trend
  evidence, not proof of meaningful convergence.

The core project identity is:

```text
VisualRL is an integration infra for World-R1, Flash-GRPO, TempFlow-GRPO,
and Inferix. GenRL is a reference for good engineering patterns only.
```

## Engineering Checkpoint - 2026-05-24

Scope: dirty-worktree review and local validation before starting the next
guarded SD3.5 trend experiment.

Results:

- `docs/ENGINEERING_CHECKPOINT_2026_05_24.md` records the checkpoint scope,
  validation commands, issue scan, and open risk register.
- `wan-checkpoint-probe` no longer lets parent directory names contaminate Wan
  manifest classification.
- `checkpoint-inventory` uses the same bounded classification rule.
- Local validation passed: `compileall`, `ruff check visual_rl tests`, and
  `pytest tests` with 99 tests.

The next SD3.5 move should be a staged 50-step guarded run with fixed prompts,
before/after PNG grids, reward/logprob/KL/clipfrac review, checkpoint files,
and parameter-delta evidence collected in one experiment report. Do not treat
the existing 20-step trend as convergence evidence.

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

Historical results from that review pass:

- Syntax compile: passed.
- Ruff: passed.
- Project tests under `tests/`: 15 passed.
- Local CLI smoke commands: passed.
- Direct `conda run -n visual-rl python -m pytest -q` failed in that older
  pass because pytest collected `reference_code/Inferix-main/tests`, which
  imported the uninstalled `inferix` package. This is fixed below by the
  current pytest scoping.

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
  path is available. SD3.5 has local CPU fake `recompute_log_probs()` dtype
  coverage, direct Python real checkpoint/CUDA adapter parity, staged remote
  `image-preview`, staged remote `tempflow-image-numeric-smoke` parity, staged
  1-step/5-step bounded trainer artifact validation, one-step resume
  validation, and a guarded 20-step trend smoke. The active SD3.5 constraint is
  to keep any further trend/convergence work guarded and staged.
