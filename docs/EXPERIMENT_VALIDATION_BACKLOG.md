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
  branch IDs, advantage masks, and cache filenames. Completed locally on
  2026-05-24 for cache filenames/metadata, Flash selected timesteps, parent
  prompt indices, TempFlow branch IDs/timestep metadata, full-trajectory GRPO
  reward expansion, Flash single-step advantage expansion and rectification
  row masking, and TempFlow branch credit-assignment masks.
- Compare full-trajectory GRPO, Flash single-step GRPO, and TempFlow branching
  on the same prompt/reward/logprob/timestep fixture. Completed locally on
  2026-05-24.

## Infra Checks

- Verify reward cache hit/miss behavior with media hash and reward version.
  Numpy/PIL/tensor content hashing has unit coverage; end-to-end reward-cache
  replay is still pending.
- Verify resume from checkpoint restores model state and keeps metrics/logging
  append behavior sane.
- Add failure-path tests for reward timeout, invalid mask, and strict unknown
  reward names.
- Add config validation for incompatible rollout/algorithm pairs. Completed
  locally on 2026-05-24 for the known contracts `grpo`/`full_trajectory`,
  `flash_grpo`/`single_step`, and `tempflow_grpo`/`branching`.

## Server Checks

- Repeat tiny smoke on the server with CPU-only execution after each major
  feature addition. TempFlow and Flash completed.
- Add a one-GPU memory probe for small image models, pinned to an explicitly
  idle GPU only. TempFlow two-GPU isolated probe completed on GPU0/GPU1.
- SD3.5-medium TempFlow guarded 5/20/200-epoch staged run completed on one idle
  GPU1 on 2026-05-24.
- Record GPU ID, visible devices, VRAM before/after, package versions, and
  commit hash for every server run.

## Latest Local Tiny Validation

- 2026-05-24: `conda run -n visual-rl python -m compileall -q visual_rl tests`
  passed.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q` passed with
  32 tests.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli flash-smoke
  --output-dir /tmp/visualrl_eval_flash --steps 1` passed with
  `flash_active_timestep_frac: 1.0` and `flash_selected_timestep_mean: 0.0`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli tempflow-smoke
  --output-dir /tmp/visualrl_eval_tempflow --steps 1` passed with
  `tempflow_active_timestep_frac: 0.25`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli smoke-mock
  --output-dir /tmp/visualrl_eval_mock_adv --steps 1` passed with
  `loss: 0.0`, `approx_kl: 0.0`, and `group_size: 2.0`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli flash-smoke
  --output-dir /tmp/visualrl_eval_flash_adv --steps 1` passed with
  `flash_active_timestep_frac: 1.0`,
  `flash_rectification_weight_mean: 1.0`, and
  `flash_selected_timestep_mean: 0.0`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli tempflow-smoke
  --output-dir /tmp/visualrl_eval_tempflow_adv --steps 1` passed with
  `tempflow_active_timestep_frac: 0.25` and
  `tempflow_noise_weight_mean: 1.0`.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_tiny_algorithm_comparison.py` passed with 1 test. The fixture
  compares full-trajectory GRPO, Flash single-step GRPO, and TempFlow
  branch-timestep credit assignment on the same prompts, rewards, timesteps,
  and zero logprob tensors.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli smoke-mock
  --output-dir /tmp/visualrl_eval_compare_mock --steps 1` passed with
  `loss: 0.0`, `approx_kl: 0.0`, and `group_size: 2.0`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli flash-smoke
  --output-dir /tmp/visualrl_eval_compare_flash --steps 1` passed with
  `flash_active_timestep_frac: 1.0`,
  `flash_rectification_weight_mean: 1.0`, and
  `flash_selected_timestep_mean: 0.0`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli tempflow-smoke
  --output-dir /tmp/visualrl_eval_compare_tempflow --steps 1` passed with
  `tempflow_active_timestep_frac: 0.25` and
  `tempflow_noise_weight_mean: 1.0`.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_config_v02.py` passed with 3 tests covering shipped preset loads
  and invalid known rollout/algorithm pairs.
- 2026-05-24: positive config load passed for
  `visual_rl/configs/presets/flash_tiny_single_step.yaml` with
  `GOOD flash_grpo/single_step`.
- 2026-05-24: a temporary invalid config with
  `algorithm.name: flash_grpo` and `sample.name: full_trajectory` was rejected
  with `ValueError: Incompatible config: algorithm.name='flash_grpo' requires
  sample.name in {'single_step'}, got sample.name='full_trajectory'.`
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 3 tests, including the mocked
  `sd15-numeric-smoke` CLI contract and explicit model-path propagation.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  sd15-numeric-smoke --help` passed and exposes the required `--model-path`
  plus smoke options.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli adapter-probe
  --adapter sd15_lora` returned deferred-load status
  `{"adapter": "sd15_lora", "loaded": false, "model_path": ""}`.

## Latest Remote SD1.5 Validation Attempt and Skip Decision

- 2026-05-24: `ssh v-qiaoqifan@10.130.140.73 'hostname; nvidia-smi;
  nvidia-smi pmon -c 1'` reported host `node01`; GPU 1 was idle at 3 MiB,
  0% utilization, and no `pmon` process. GPUs 0 and 2-7 were busy with active
  Python processes.
- 2026-05-24: read-only checkpoint search over `$HOME`, `/data`, `/mnt`,
  `/share`, `/workspace`, and `/home` found SD3.5 and Wan Diffusers paths only:
  `/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium/model_index.json`,
  `/home/v-qiaoqifan/TempFlow-GRPO/models/stable-diffusion-3.5-medium/model_index.json`,
  and `/mnt/data/v-yingqi/models/Wan2.1-T2V-1.3B-Diffusers/model_index.json`.
  No usable SD1.5 Diffusers checkpoint was found.
- Real SD1.5 GPU numeric smoke remains pending and blocked by missing SD1.5
  Diffusers checkpoint/model path. No model download was attempted.
- 2026-05-24 decision: skip SD1.5 as a gating phase. Keep the local
  `sd15-numeric-smoke` CLI and tests, but continue the main integration path
  with SD3.5 adapter parity because a validated SD3.5 checkpoint is available.
- Optional command to run once a valid checkpoint exists:
  `CUDA_VISIBLE_DEVICES=1 conda run -n visual-rl python -m visual_rl.cli
  sd15-numeric-smoke --model-path /path/to/sd15-diffusers --resolution 128
  --num-steps 1 --prompt "a red square"`.

## Real-Model Checks

- SD1.5 LoRA adapter contract tests are complete enough for a deferred path:
  deferred adapter registration tests and the mocked `sd15-numeric-smoke` CLI
  contract pass. Loaded-model numeric tests are still pending and blocked by
  missing SD1.5 checkpoint path, but they are no longer gating SD3/FLUX/
  QwenImage work.
- Validate real model `sample()` and `recompute_log_probs()` numerics on a tiny
  batch before any reward optimization run. SD3.5 reference script path passed
  a minimal TempFlow smoke on 2026-05-23; `visual_rl` SD3 adapter path is
  implemented but not yet server-validated. This is now the next active
  real-model validation task.
- Compare Flash selected-step loss against full-trajectory loss on a controlled
  toy scheduler.
- Validate TempFlow branch reward alignment through the SD3.5 `visual_rl`
  adapter path. The SD3.5 reference script now has a smoke pass, but
  adapter-level parity is still pending.
- Validate FLUX and QwenImage adapters with low-resolution smoke batches after
  SD3.5 adapter parity is confirmed.

## Video/Inferix Checks

- Keep Wan/World-R1 runs as smoke-only until small image RL curves are stable.
- Validate World-R1 reward server clients independently before online training.
- For Inferix, validate BlockVid preview/profiling/no-decode paths first; do
  not use it for online RL until logprob/recompute contracts exist.
