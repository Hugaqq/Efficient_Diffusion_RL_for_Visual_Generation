# Experiment Validation Backlog

These are validation gaps discovered while moving VisualRL from tiny correctness
checks to a real SD3.5 mini end-to-end loop. Tiny checks are now regression
gates only; new work should prioritize real-model preview, smoke, bounded
training, and artifact comparison.

## Tiny Regression Gates

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
- Add a deterministic loss-descent gate that does not depend on noisy online
  RL reward trends. Completed on 2026-05-24 with `tiny-loss-probe`: local CPU
  loss fell from `0.0510348` to `6.127e-7`, and remote CUDA
  `v-qiaoqifan@10.130.140.73` GPU2 loss fell from `0.0511422` to `6.099e-7`.
  Details and command are in `docs/LOSS_DESCENT_VALIDATION_PLAN.md`.
- Do not add new tiny-only feature work unless it protects a real-model bug or
  a shared infra regression.

## SD3.5 Complete Mini-Loop Gates

- Add `image-preview` CLI that saves PNG previews and metadata from a VisualRL
  adapter. Completed on 2026-05-24: the CLI uses the existing `MODEL_ADAPTERS`
  registry, calls `adapter.sample()`, runs strict `RolloutBatch` validation, and
  writes `preview_000.png` plus `metadata.json`.
- Run `image-preview` locally with `tiny_diffusion` or mocked adapter coverage
  and remotely with `sd3_tempflow` on one explicitly idle 5090 GPU. Completed on
  2026-05-24: local `tiny_diffusion` produced `media_shape: [1, 3, 4, 4]` and a
  4x4 RGB PNG; staged remote `sd3_tempflow` on `node01` GPU1 produced
  `media_shape: [1, 3, 256, 256]`, `latents_shape: [1, 2, 16, 32, 32]`, and a
  verified 256x256 RGB PNG.
- Run the new `tempflow-image-numeric-smoke --adapter sd3_tempflow` remotely
  from a staged current-code copy, without overwriting shared remote code.
  Completed on 2026-05-24 from
  `/home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/framecode`;
  result included `max_abs_logprob_delta: 0.0`, finite media/logprobs, and
  `trainable_parameters: 4694016`.
- Run a bounded SD3.5 `VisualRLTrainer` smoke: 1-5 steps first, no rollout
  cache, cheap reward, strict rollout validation, and explicit output dir.
  Closed on 2026-05-24 by staged `remote-sd3-cli-smoke --execute` runs on
  `node01` GPU1. The staged runner executed `image-preview`,
  `tempflow-image-numeric-smoke`, and `sd3-bounded-trainer-smoke` in order.
  The 1-step stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_full_loop_20260524_1215`
  passed with bounded `latest.step: 1`; the 5-step stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_full_loop_20260524_1215_5step`
  passed with bounded `latest.step: 5`.
- Validate bounded trainer artifacts: `latest.json`, metrics JSON/JSONL, LoRA
  checkpoint directory/files, and adapter `save_pretrained()` output. Closed on
  2026-05-24 for the staged 1-step and 5-step SD3.5 runs: `checkpoint_000001`
  and `checkpoint_000005` contained `README.md`, `adapter_config.json`, and
  `adapter_model.safetensors`; the 5-step run wrote 5 metrics rows.
- Compare fixed-prompt before/after preview PNGs from the same SD3.5 smoke path.
  Closed for artifact existence/format on 2026-05-24: main preview, bounded
  before, and bounded after PNGs were all verified as
  `PNG image data, 256 x 256, 8-bit/color RGB, non-interlaced`. The reward
  trend is recorded, not improved: 1-step before/after preview reward was
  `0.5285151600837708` -> `0.5295270681381226`; 5-step was
  `0.5285151600837708` -> `0.5278942584991455`.
- Record reward/logprob/KL/parameter-delta evidence for the bounded SD3.5 smoke:
  reward mean/std, finite old and recomputed logprobs, finite/expected KL,
  trainable parameter count, and at least one parameter delta after optimizer
  step. Closed for smoke-scale evidence on 2026-05-24. The staged 1-step run
  reported numeric `max_abs_logprob_delta: 0.0`,
  `trainable_parameters: 4694016`, final `reward_mean:
  0.5154668688774109`, `old_logprob_mean/new_logprob_mean:
  1.2926936149597168`, `rollout_kl_mean: 0.0`, `clipfrac: 0.0`,
  `parameter_delta_abs_max: 9.99999883788405e-06`,
  `parameter_delta_l2: 0.015129816539832258`, and
  `parameter_delta_nonzero_count: 2334720`. The staged 5-step run reported
  numeric `max_abs_logprob_delta: 0.0`, `trainable_parameters: 4694016`, final
  `reward_mean: 0.5767983794212341`, `reward_std: 0.06672149896621704`,
  `old_logprob_mean/new_logprob_mean: 1.3020650148391724`,
  `rollout_kl_mean: 0.0`, `clipfrac: 0.0`,
  `parameter_delta_abs_max: 5.019741365686059e-05`,
  `parameter_delta_l2: 0.0421081454686555`, and
  `parameter_delta_nonzero_count: 4667335`.
- Run 20-100 SD3.5 steps and compare before/after preview PNGs, reward trend,
  finite logprob/KL, and LoRA/checkpoint parameter delta. Closed at 20-step
  smoke scale on 2026-05-24 by staged `remote-sd3-cli-smoke --execute` on
  `node01` GPU1 with `--bounded-steps 20 --skip-resume-validation
  --allow-long-run`. Stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_20step_trend_20260524_1725`
  returned `ok: true`; `latest.step: 20`, `metrics_line_count: 20`,
  `checkpoint_000020` contained `README.md`, `adapter_config.json`, and
  `adapter_model.safetensors`; final metrics included `reward_mean:
  0.5748612880706787`, `reward_std: 0.029813051223754883`,
  `old_logprob_mean/new_logprob_mean: 1.2945268154144287`,
  `rollout_kl_mean: 0.0`, `clipfrac: 0.0`,
  `tempflow_active_timestep_frac: 0.5`, and
  `tempflow_noise_weight_mean: 0.9999999403953552`. Reward metrics moved from
  `0.5154668688774109` to `0.5748612880706787` with max
  `0.7232511043548584`; before/after preview reward was
  `0.5285151600837708` -> `0.5303504467010498`. Parameter delta was nonzero:
  `parameter_delta_abs_max: 0.00017927215958479792`,
  `parameter_delta_l2: 0.09039463499422448`, and
  `parameter_delta_nonzero_count: 4669060`. Before/after PNGs were verified as
  256x256 RGB. This is a useful short trend, not proof of meaningful
  convergence.

## Infra Checks

- Freeze an engineering checkpoint before adding new features. Completed
  locally on 2026-05-24 and recorded in
  `docs/ENGINEERING_CHECKPOINT_2026_05_24.md`: dirty-worktree scan completed,
  Wan/checkpoint-inventory parent-path classification bug fixed, `compileall`
  passed, `ruff check visual_rl tests` passed, and `pytest tests` passed with
  99 tests.
- Verify reward cache hit/miss behavior with media hash and reward version.
  Completed locally on 2026-05-24: `tests/test_reward_router.py` covers
  end-to-end cache replay without re-calling the reward client, media-content
  cache misses for same-shape tensors with different contents, and no cache
  write when an invalid reward result is handled through `fail_policy:
  invalid`.
- Verify resume from checkpoint restores model state and keeps metrics/logging
  behavior sane. Closed at one-step smoke scale on 2026-05-24 by staged
  `remote-sd3-cli-smoke --execute` on `node01` GPU1. Stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_resume_eval_20260524_171836`
  returned `ok: true`; the resume run reported `resume_loaded: true`,
  `resume_base_step: 1`, `effective_total_step: 2`, `metrics_line_count: 1`,
  checkpoint files `README.md`, `adapter_config.json`, and
  `adapter_model.safetensors`, finite reward/logprob/KL/clip/TempFlow metrics,
  and nonzero parameter delta (`abs_max: 9.999996109399945e-06`, `l2:
  0.01731845204397767`, `nonzero_count: 4668478`). Before/after resume preview
  PNGs were verified as 256x256 RGB; final GPU1 state was `3 MiB`, `0%`, with
  no `pmon` process.
- Add failure-path tests for reward timeout, invalid mask, and strict unknown
  reward names. Completed locally on 2026-05-24: bad reward result shapes now
  either mark the batch invalid or raise depending on `fail_policy`; unknown
  reward client names raise a registry error; and `remote_pickle` timeout/retry
  behavior has a network-free mocked test through `RewardRouter`.
- Add config validation for incompatible rollout/algorithm pairs. Completed
  locally on 2026-05-24 for the known contracts `grpo`/`full_trajectory`,
  `flash_grpo`/`single_step`, and `tempflow_grpo`/`branching`.
- Add `visual-rl validate-config` CLI coverage for shipped presets, registry
  names, nonempty reward weights, and reward weight/client alias consistency.
  Completed locally on 2026-05-24, including evaluator-found missing reward
  client alias mismatch coverage.
- Add `visual-rl rollout-probe` CLI coverage for lightweight adapter/rollout
  construction, strict `RolloutBatch` validation, JSON shape/metadata output,
  and no trainer/reward/checkpoint/output-dir side effects. Completed locally
  on 2026-05-24 for mock World-R1/Wan, Flash tiny single-step, and TempFlow
  tiny branching presets.
- Add `visual-rl reward-probe` CLI coverage for configured reward clients,
  cache behavior, strict reward-name validation, and structured failure output.
  Completed locally on 2026-05-24 for mock World-R1/Wan rewards,
  `prompt_color`, invalid reward alias structured failures, no reward cache
  writes, and no output-directory side effects.
- Strengthen `visual-rl reward-probe` synthetic media sizing so it honors
  model config resolution fields where present. Completed and evaluator
  validated on 2026-05-24: dimensions are resolved from
  `model.extra.height`/`model.extra.width`, `model.extra.resolution`,
  `model.extra.image_size`, top-level model `height`/`width`/`resolution`/
  `image_size`, `media_shape`, then a 16x16 default; JSON output includes
  `media_height` and `media_width`; invalid dimensions return structured JSON
  failure without a Python traceback.
- Add BaseTrainer optimizer config compatibility coverage for string-like
  numeric values. Completed locally on 2026-05-24: `setup_optimizer()`
  accepts string-like numeric config values and builds a valid AdamW optimizer.
- Add SD3 adapter local compatibility coverage against the TempFlow reference
  pipeline surface. Completed locally on 2026-05-24 for unsupported kwarg
  filtering, `return_dict=False` propagation when supported, 3-tuple no-KL
  `sample()` returns with zero-filled `kl`, and `_transformer_dtype()` helper
  behavior. This is not real SD3.5 checkpoint/CUDA adapter parity.
- Add CPU fake transformer coverage for SD3 `recompute_log_probs()` dtype
  casting with guidance on/off. Completed locally on 2026-05-24: direct
  `guidance_scale == 1.0` path casts hidden states to transformer dtype
  `torch.float16` without batch doubling; CFG `guidance_scale == 3.0` path
  casts hidden states to `torch.float16` with batch doubling; fake SDE step
  asserts `noise_pred.float()`, `sample.float()`, and `prev_sample.float()`.
  Embeddings come from `RolloutBatch.model_tensors`, so the test does not
  invoke text encoders, diffusers, checkpoints, or GPU. This is local fake
  coverage only, not real SD3.5 checkpoint/CUDA adapter parity.
- Add shared real-image TempFlow numeric-smoke CLI contract for SD3, FLUX, and
  QwenImage. Completed locally on 2026-05-24: `sd3-numeric-smoke` accepts
  `--repo-root` and records `extra["repo_root"]`; the shared
  `tempflow-image-numeric-smoke --adapter {sd3_tempflow,flux_tempflow,qwenimage_tempflow}`
  CLI runs adapter sample, strict validation, recompute logprobs, shape/finite/
  allclose checks, and trainable-parameter JSON. Local mocked contracts cover
  SD3 repo-root propagation, FLUX, QwenImage, and help output.
- Add `checkpoint-inventory` for bounded checkpoint discovery before real
  smokes. Completed locally on 2026-05-24: the CLI scans `model_index.json`
  files, classifies SD1.5/SD3/FLUX/QwenImage/Wan Diffusers paths, emits
  structured JSON records/errors, supports `--require-adapter`, and does not
  load checkpoints or write output directories.
- Add `world-r1-reward-server-probe` for bounded endpoint payload validation.
  Completed locally on 2026-05-24: the CLI builds synthetic image payloads for
  `reward_general` and video payloads for `reward_3d`, calls the existing
  pickle-over-HTTP client, validates reward shape/finiteness, records no
  trainer/checkpoint/output-dir side effects, and returns structured failures
  for bad URLs, timeout/retry exhaustion, invalid output shapes, and connection
  errors. `RemotePickleRewardClient` now falls back to stdlib `urllib` when
  optional `requests` is absent.

## Server Checks

- Repeat tiny smoke on the server with CPU-only execution after each major
  feature addition. TempFlow and Flash completed.
- Add a one-GPU memory probe for small image models, pinned to an explicitly
  idle GPU only. TempFlow two-GPU isolated probe completed on GPU0/GPU1.
- SD3.5-medium TempFlow guarded 5/20/200-epoch staged run completed on one idle
  GPU1 on 2026-05-24.
- Record GPU ID, visible devices, VRAM before/after, package versions, and
  commit hash for every server run.
- Run `tiny-loss-probe` on server CUDA after changes that touch adapter
  logprob, optimizer, metrics, or checkpoint plumbing. Completed on
  2026-05-24 on GPU2 with output directory
  `/home/v-qiaoqifan/visual_rl_experiments/tiny_loss_probe_20260524`.
- Record remote checkpoint availability before real-image smokes. Completed on
  2026-05-24 on `v-qiaoqifan@10.130.140.73` (`node01`): available Diffusers
  `model_index.json` paths were
  `/home/v-qiaoqifan/TempFlow-GRPO/models/stable-diffusion-3.5-medium/model_index.json`,
  `/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium/model_index.json`,
  and `/mnt/data/v-yingqi/models/Wan2.1-T2V-1.3B-Diffusers/model_index.json`.
  No FLUX or QwenImage checkpoint path was found, so real FLUX/QwenImage
  low-resolution smoke remains blocked.
- Run real SD3.5 TempFlow adapter sample/recompute parity on CUDA. Completed
  by direct Python on 2026-05-24 on `node01` GPU1 with explicit
  `repo_root=/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke`,
  `model_path=/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium`,
  resolution 256, `num_steps=2`, `guidance_scale=4.5`, dtype bfloat16,
  LoRA rank 8/alpha 16, `max_sequence_length=128`, and device CUDA. Result:
  `valid: true`, `max_abs_logprob_delta: 0.0`, media `[1, 3, 256, 256]`,
  latents/next latents `[1, 2, 16, 32, 32]`, timesteps/logprobs `[1, 2]`,
  382 trainable tensors, 4,694,016 trainable parameters, CUDA bfloat16. GPU1
  readout was 3 MiB/0% before and 3 MiB/0% after. This proves adapter numeric
  parity for direct Python, not the new remote CLI.
- Run real Wan/World-R1 loading smoke using the discovered Wan checkpoint.
  Pending. First bounded command is:
  `conda run -n visual-rl python -m visual_rl.cli wan-plan --config visual_rl/configs/presets/wan_runtime_v02_plan.yaml --model-path /mnt/data/v-yingqi/models/Wan2.1-T2V-1.3B-Diffusers --output-dir runs/wan_runtime_plan_real_path`.
  This is still a plan/readiness check only; a later task must load the real
  checkpoint through the Wan/World-R1 adapter path and validate sample/logprob
  tensor shapes before training.
- Validate real World-R1 reward-server calls. Pending. Current
  `reward-probe world_r1_wan_v02_mock` uses synthetic media and mock rewards; it
  does not contact the real server. `world-r1-reward-server-probe` now validates
  the real client payload path locally with mocked servers and structured
  connection-failure output, including a loopback `127.0.0.1:9` failure that
  returned `Connection refused` after running outside the network sandbox. A
  successful live `reward_general`/`reward_3d` endpoint call is still required
  before claiming World-R1 reward integration.
- Run the new remote CLI smoke after non-invasive code sync. Completed on
  2026-05-24 by staging only `visual_rl` and `pyproject.toml` into
  `/home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/framecode`.
  This validated the current CLI without claiming the remote
  `/home/v-qiaoqifan/visual_rl_experiments/framecode` tree was updated.
  Remote env note: `/home/v-qiaoqifan/miniconda3/bin/conda run -n visual-rl`
  returned `EnvironmentLocationNotFound`; use
  `/home/v-qiaoqifan/miniconda3/bin/conda run -n visual-rl-sd35` for SD3.5
  CUDA smokes.
- Run `image-preview` from the staged remote copy on real SD3.5. Completed on
  2026-05-24 on `node01` GPU1 with `CUDA_VISIBLE_DEVICES=1`,
  `model_path=/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium`,
  `repo_root=/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke`,
  resolution 256, `num_steps=2`, `guidance_scale=4.5`, and seed 101. Result:
  `valid: true`, `media_shape: [1, 3, 256, 256]`,
  `latents_shape: [1, 2, 16, 32, 32]`, and verified PNG
  `/home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/preview/preview_000.png`
  as `PNG image data, 256 x 256, 8-bit/color RGB, non-interlaced`.
- Run the staged remote SD3.5 numeric CLI. Completed on 2026-05-24 on `node01`
  GPU1 with `CUDA_VISIBLE_DEVICES=1`, env `visual-rl-sd35`, dtype bfloat16,
  LoRA rank 8/alpha 16, and `max_sequence_length=128`. Result:
  `max_abs_logprob_delta: 0.0`, `media_finite: true`,
  `old_log_probs_finite: true`, `recomputed_log_probs_finite: true`, and
  `trainable_parameters: 4694016`.
- Verify final GPU state after staged remote SD3.5 preview/numeric smokes.
  Completed on 2026-05-24: GPU1 returned to `3, 0` and `pmon` showed no
  process.
- Run the staged remote SD3.5 full CLI loop with artifact checks. Completed on
  2026-05-24 on `node01` GPU1. Before runs, GPU1 was `3MiB / 32607MiB`, `0%`,
  and `nvidia-smi pmon -i 1 -c 1` showed no process. The command used
  `remote-sd3-cli-smoke --execute --gpu 1`, env `visual-rl-sd35`, SD3.5-medium
  at `/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium`,
  `repo_root=/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke`,
  prompt `a small red cube on a white table`, resolution 256, `num_steps=2`,
  `guidance_scale=4.5`, seed 101, dtype bfloat16, LoRA rank 8/alpha 16, and
  `max_sequence_length=128`. A 1-step stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_full_loop_20260524_1215`
  returned `ok: true`; preview was `valid: true` with
  `media_shape: [1, 3, 256, 256]` and
  `latents_shape: [1, 2, 16, 32, 32]`; numeric smoke had
  `max_abs_logprob_delta: 0.0`, finite media/logprobs, and
  `trainable_parameters: 4694016`; bounded smoke had `valid: true`,
  `latest.step: 1`, `checkpoint_000001`, final `reward_mean:
  0.5154668688774109`, `rollout_kl_mean: 0.0`, `clipfrac: 0.0`, nonzero
  parameter delta, and before/after preview rewards `0.5285151600837708` ->
  `0.5295270681381226`.
- Run the staged remote SD3.5 full CLI loop for 5 bounded steps. Completed on
  2026-05-24 on `node01` GPU1 with the same command parameters except
  `--stage-name sd3_full_loop_20260524_1215_5step --bounded-steps 5`. Stage
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_full_loop_20260524_1215_5step`
  returned `ok: true`; preview was `valid: true` with
  `media_shape: [1, 3, 256, 256]` and
  `latents_shape: [1, 2, 16, 32, 32]`; numeric smoke again had
  `max_abs_logprob_delta: 0.0`, finite media/logprobs, and
  `trainable_parameters: 4694016`; bounded smoke had `valid: true`,
  `latest.step: 5`, `checkpoint_000005`, 5 metrics rows, final
  `reward_mean: 0.5767983794212341`, `reward_std: 0.06672149896621704`,
  `rollout_kl_mean: 0.0`, `clipfrac: 0.0`,
  `parameter_delta_abs_max: 5.019741365686059e-05`,
  `parameter_delta_l2: 0.0421081454686555`, and
  `parameter_delta_nonzero_count: 4667335`. Main preview, bounded before, and
  bounded after PNGs were verified as 256x256 RGB, `wc -l metrics.jsonl`
  returned `5`, and final `pmon` showed no process. The 5-step before/after
  preview reward was `0.5285151600837708` -> `0.5278942584991455`; this is a
  recorded reward trend, not evidence of improvement.
- Preserve the previous bounded SD3 `VisualRLTrainer` smoke evidence but do not
  overclaim it. Observed run:
  `/home/v-qiaoqifan/visual_rl_experiments/visualrl_sd35_parity_20260524_101348/smoke`
  had `latest.json` at step 1 and a metrics line with `loss: 0.0`,
  `reward_mean: 0.942878246307373`, `reward_std: 0.05712178349494934`, and
  `tempflow_active_timestep_frac: 0.5`. The first two logs in that directory
  captured earlier `return_dict` kwarg and float/bfloat16 mismatch failures
  that local fixes addressed. This is a one-step smoke, not a reward
  improvement or long-training result.

## Latest Local Tiny Validation

- 2026-05-24: `conda run -n visual-rl python -m pytest -q tests` passed with
  95 tests after adding checkpoint inventory, World-R1 reward-server endpoint
  probe coverage, and urllib fallback for the remote pickle reward client.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_checkpoint_inventory.py` passed with 3 tests covering adapter
  classification, required-adapter failure, and malformed
  `model_index.json` structured errors.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_world_r1_reward_server_probe.py tests/test_world_r1_rewards.py
  tests/test_wan_trainer.py` passed with 10 tests before the urllib fallback
  addition; `tests/test_world_r1_rewards.py
  tests/test_world_r1_reward_server_probe.py` then passed with 8 tests after
  adding the no-`requests` fallback case.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`,
  `conda run -n visual-rl python -m compileall -q visual_rl tests`,
  `git diff --check`, and `conda run -n visual-rl python -m visual_rl.cli
  smoke-imports` passed after the checkpoint/probe additions.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  checkpoint-inventory --help` and `world-r1-reward-server-probe --help`
  passed. `checkpoint-inventory visual_rl/configs --require-adapter
  sd3_tempflow` returned structured JSON with `missing_adapters:
  ["sd3_tempflow"]`, as expected for a config directory without checkpoints.
- 2026-05-24: `world-r1-reward-server-probe --reward reward_general --url
  http://127.0.0.1:9 --timeout 0.05 --retries 0 --batch-size 1 --height 4
  --width 4` returned structured JSON failure. Inside the sandbox the network
  layer reported `Operation not permitted`; rerunning with approved loopback
  access returned `Connection refused`, proving the CLI reaches the endpoint
  client path and no longer fails on a missing `requests` dependency. This is
  not a live reward-server validation.
- 2026-05-24: `conda run -n visual-rl python -m compileall -q visual_rl tests`
  passed.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q tests` passed with
  58 tests after the shared TempFlow image numeric-smoke CLI contracts.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q` passed with 65 tests
  after adding `image-preview` CLI coverage.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 17 tests after adding
  before/after preview evidence to `sd3-bounded-trainer-smoke`.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl/cli.py
  tests/test_real_image_adapters.py` passed with `All checks passed!` after the
  same bounded trainer preview evidence update.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q tests` passed with
  67 tests after the bounded trainer preview evidence update.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed with `All checks passed!` after the same update.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_remote_smoke.py` passed with 4 tests after `remote-sd3-cli-smoke`
  was updated to run `image-preview`, `tempflow-image-numeric-smoke`, and
  `sd3-bounded-trainer-smoke` sequentially with `pmon` idle guard and artifact
  checks.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 17 tests after the staged
  runner update.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl/cli.py
  visual_rl/experiments/remote_smoke.py tests/test_remote_smoke.py
  tests/test_real_image_adapters.py` passed with `All checks passed!` after the
  staged runner update.
- 2026-05-24: `conda run -n visual-rl python -m compileall -q visual_rl tests`
  passed after the staged runner update.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  remote-sd3-cli-smoke --stage-name local_dry_run_check --model-path
  /models/sd35` generated a dry-run containing `conda_env: visual-rl-sd35`,
  `remote_preview_dir`, `remote_bounded_dir`, and all three staged commands.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 15 tests after adding
  `image-preview` mocked SD3 contract, local `tiny_diffusion` artifact smoke,
  and help output coverage.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli image-preview
  --adapter tiny_diffusion --model-path /unused/tiny --prompt "a local tiny
  preview" --resolution 4 --num-steps 2 --guidance-scale 1.0 --seed 5 --device
  cpu --output-dir /tmp/visualrl_image_preview_tiny_eval` passed with
  `"valid": true` and `media_shape: [1, 3, 4, 4]`.
- 2026-05-24: `file /tmp/visualrl_image_preview_tiny_eval/preview_000.png`
  passed with `PNG image data, 4 x 4, 8-bit/color RGB, non-interlaced`.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 12 tests after adding SD3
  repo-root, FLUX mocked contract, QwenImage mocked contract, and help tests.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  sd3-numeric-smoke --help` passed and shows `--repo-root`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  tempflow-image-numeric-smoke --help` passed.
- 2026-05-24: `git diff --check` passed.
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
  tests/test_config_v02.py` passed with 12 tests covering shipped preset loads,
  invalid known rollout/algorithm pairs, the `validate-config` JSON CLI,
  missing reward-client alias rejection, and `rollout-probe` JSON contract
  checks, plus `reward-probe` coverage for mock WAN rewards, `prompt_color`,
  and mismatched reward-client alias structured failure.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  validate-config visual_rl/configs/presets/*.yaml` passed with all 10 presets
  valid. This is lightweight config validation only: no model loading, trainer
  construction, reward execution, or output directory writes.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  validate-config --help` passed.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  rollout-probe visual_rl/configs/presets/world_r1_wan_v02_mock.yaml --seed
  123` passed with `valid: true`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  rollout-probe visual_rl/configs/presets/flash_tiny_single_step.yaml
  --batch-size 1 --num-steps 4 --seed 77` passed with `valid: true` and the
  single-step shape contract.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  rollout-probe visual_rl/configs/presets/tempflow_tiny_branching.yaml
  --batch-size 1 --num-steps 4 --seed 88` passed with `valid: true` and
  `branch_ids` shape metadata.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  rollout-probe --help` passed.
- 2026-05-24: rollout probes confirmed lightweight behavior: they load config,
  validate registry/config names, construct only the selected adapter and
  rollout engine, batch prompts, call `rollout.sample()`, strictly validate the
  `RolloutBatch` by default, and emit JSON shapes/metadata. They do not create
  `VisualRLTrainer`, optimizer, rewards, checkpoint/cache writes, or output
  directories.
- 2026-05-24: evaluator confirmed rollout probes did not create
  `runs/world_r1_wan_v02_mock`, `runs/flash_tiny_single_step`, or
  `runs/tempflow_tiny_branching`.
- 2026-05-24: rollout-probe failure cases for a missing config path and the
  SD3 real adapter preset without local dependencies/checkpoint return non-zero
  structured JSON without a Python traceback. This does not prove real SD3.5
  CUDA parity.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  reward-probe world_r1_wan_v02_mock --seed 123` passed with `valid: true`
  and valid mock rewards.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  reward-probe flash_tiny_single_step --batch-size 2 --seed 77` passed with
  `valid: true` and `prompt_color` raw/weighted values `[1.0, 1.0]`.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  reward-probe --help` passed.
- 2026-05-24: reward-probe preset coverage passed locally on 9/10 shipped
  presets. `world_r1_wan_v01_server.yaml` returned structured JSON failure
  with `dataset config must provide either prompts or path`, which is
  acceptable because it is a server/legacy plan preset without local dataset
  prompts.
- 2026-05-24: reward-probe confirmed lightweight behavior: it loads config,
  reuses config/registry/reward alias validation, takes a small
  `PromptDataset` batch, generates deterministic synthetic RGB media, and runs
  `RewardRouter(config.rewards, cache_dir=None)`. It does not construct
  `VisualRLTrainer`, load model adapters, construct rollout engines, run an
  optimizer, write output directories/checkpoints, or write reward-cache
  entries.
- 2026-05-24: invalid reward alias config returns non-zero structured JSON
  without a Python traceback.
- 2026-05-24: reward-probe validates reward routing/config with synthetic
  media only; it does not validate real adapter output shape, CUDA execution,
  or model correctness.
- 2026-05-24: reward-probe synthetic media sizing enhancement completed and
  evaluator validated. `conda run -n visual-rl python -m pytest -q
  tests/test_config_v02.py` passed with 16 tests. Four real image presets now
  probe synthetic 256x256 media: `reward-probe sd3_tempflow_adapter.yaml
  --seed 123` reported `media_shape: [1, 3, 256, 256]`,
  `media_height: 256`, and `media_width: 256`; `reward-probe
  flux_tempflow_adapter.yaml --seed 123`, `reward-probe
  qwenimage_tempflow_adapter.yaml --seed 123`, and `reward-probe
  sd15_lora_rl.yaml --seed 123` also reported 256x256. Tests cover real image
  preset `resolution: 256`, explicit height/width override, top-level
  dict/object helper resolution fields, and invalid-size structured failure.
  The probe remains lightweight: it does not construct `VisualRLTrainer`, load
  model adapters, construct rollout engines, or write cache/output/checkpoint
  artifacts.
- 2026-05-24: positive config load passed for
  `visual_rl/configs/presets/flash_tiny_single_step.yaml` with
  `GOOD flash_grpo/single_step`.
- 2026-05-24: a temporary invalid config with
  `algorithm.name: flash_grpo` and `sample.name: full_trajectory` was rejected
  with `ValueError: Incompatible config: algorithm.name='flash_grpo' requires
  sample.name in {'single_step'}, got sample.name='full_trajectory'.`
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 4 tests, including the mocked
  `sd15-numeric-smoke` and `sd3-numeric-smoke` CLI contracts and explicit
  model-path propagation.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  sd15-numeric-smoke --help` passed and exposes the required `--model-path`
  plus smoke options.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli
  sd3-numeric-smoke --help` passed and exposes the required `--model-path`
  plus smoke options.
- 2026-05-24: `conda run -n visual-rl python -m visual_rl.cli adapter-probe
  --adapter sd15_lora` returned deferred-load status
  `{"adapter": "sd15_lora", "loaded": false, "model_path": ""}`.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed after the SD3 numeric-smoke CLI addition.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q tests` passed with
  52 tests after the SD3 adapter local compatibility hardening.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 9 tests after adding SD3
  `recompute_log_probs()` CPU fake transformer/SDE dtype coverage.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_base_trainer.py` passed with 1 test after the same change.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q tests` passed with
  55 tests after the same change.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed after the same change.
- 2026-05-24: `conda run -n visual-rl python -m compileall -q visual_rl tests`
  passed after the same change.
- 2026-05-24: evaluator independently confirmed the shared TempFlow image
  numeric-smoke CLI update: focused
  `tests/test_real_image_adapters.py` passed with 12 tests in 0.56s, full
  `tests/` passed with 58 tests in 1.22s, Ruff passed, compileall passed,
  `sd3-numeric-smoke --help` passed with `--repo-root`, and
  `tempflow-image-numeric-smoke --help` passed.
- 2026-05-24: evaluator confirmed local GPU is unavailable for real CUDA
  smoke: `nvidia-smi` command not found, `torch.cuda.is_available(): False`,
  and `device_count: 0`.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed after the reward-probe synthetic media sizing enhancement.
- 2026-05-24: `conda run -n visual-rl python -m compileall -q visual_rl tests`
  passed after the reward-probe synthetic media sizing enhancement. The
  existing dirty `visual_rl/model_adapters/sd3.py` was not changed in that
  docs/planning update, but compileall covered it.
- 2026-05-24: `conda run -n visual-rl python -m py_compile visual_rl/cli.py
  visual_rl/model_adapters/sd3.py tests/test_real_image_adapters.py` passed.

## Latest SD3.5 Image Preview and Staged Remote CLI Validation

- 2026-05-24 local gates after `image-preview`: `conda run -n visual-rl python
  -m compileall -q visual_rl tests` passed, `conda run -n visual-rl python -m
  ruff check visual_rl tests` passed with `All checks passed!`, `conda run -n
  visual-rl python -m pytest -q` passed with 65 tests, and `git diff --check`
  passed.
- 2026-05-24 remote GPU availability command:
  `ssh -o BatchMode=yes -o ConnectTimeout=10 v-qiaoqifan@10.130.140.73 'hostname; nvidia-smi; nvidia-smi pmon -c 1'`
  passed on host `node01`; GPU0 was busy, GPU1-7 were `3MiB / 32607MiB`, `0%`,
  and `pmon` showed no process.
- 2026-05-24 remote env checks: `/home/v-qiaoqifan/miniconda3/bin/conda run -n
  visual-rl` failed with `EnvironmentLocationNotFound`; `/home/v-qiaoqifan/miniconda3/bin/conda
  run -n visual-rl-sd35 python -c "import torch; print(torch.__version__);
  print(torch.cuda.is_available())"` passed with `2.11.0+cu130` and `True`.
- 2026-05-24 staging command:
  `rsync -a visual_rl pyproject.toml v-qiaoqifan@10.130.140.73:/home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/framecode/`
  passed. This was a staged copy for validation only, not a remote git checkout
  update.
- 2026-05-24 remote SD3 preview command:
  `cd /home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/framecode && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 /home/v-qiaoqifan/miniconda3/bin/conda run -n visual-rl-sd35 python -m visual_rl.cli image-preview --adapter sd3_tempflow --model-path /home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium --repo-root /home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke --prompt "a small red cube on a white table" --resolution 256 --num-steps 2 --guidance-scale 4.5 --seed 101 --device cuda --output-dir /home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/preview`
  passed with `valid: true`, `media_shape: [1, 3, 256, 256]`, and
  `latents_shape: [1, 2, 16, 32, 32]`.
- 2026-05-24 remote SD3 preview artifact check:
  `file /home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/preview/preview_000.png`
  passed with `PNG image data, 256 x 256, 8-bit/color RGB, non-interlaced`.
- 2026-05-24 remote SD3 numeric command:
  `cd /home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/framecode && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 /home/v-qiaoqifan/miniconda3/bin/conda run -n visual-rl-sd35 python -m visual_rl.cli tempflow-image-numeric-smoke --adapter sd3_tempflow --model-path /home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium --repo-root /home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke --prompt "a small red cube on a white table" --resolution 256 --num-steps 2 --guidance-scale 4.5 --seed 101 --device cuda --dtype bfloat16 --lora-rank 8 --lora-alpha 16 --max-sequence-length 128 --logprob-atol 1e-5`
  passed with `max_abs_logprob_delta: 0.0`, `media_finite: true`,
  `old_log_probs_finite: true`, `recomputed_log_probs_finite: true`, and
  `trainable_parameters: 4694016`.
- 2026-05-24 final GPU check showed GPU1 back at `3, 0` with no `pmon` process.
- 2026-05-24 staged `remote-sd3-cli-smoke --execute` full loop passed on
  `node01` GPU1 for 1-step and 5-step bounded trainer stages. It validates
  bounded trainer plumbing, LoRA checkpoint artifacts, metrics JSONL, before/
  after PNG artifact creation, finite numeric-smoke logprobs, and nonzero
  parameter deltas.
- 2026-05-24 staged `remote-sd3-cli-smoke --execute` then passed one-step
  resume validation and a guarded 20-step trend smoke on `node01` GPU1. This
  validates short-run trend plumbing and resume artifacts, not meaningful
  convergence or long training.

## Latest Local SD3 Recompute Dtype Validation

- 2026-05-24: evaluator validated this as local CPU fake transformer/SDE
  coverage only, not a GPU experiment and not real SD3.5 checkpoint/CUDA
  parity.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 9 tests.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_base_trainer.py` passed with 1 test.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q tests` passed with
  55 tests.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed.
- 2026-05-24: `conda run -n visual-rl python -m compileall -q visual_rl tests`
  passed.
- SD3 recompute coverage added: `SD3TempFlowAdapter.recompute_log_probs()` uses
  `RolloutBatch.model_tensors` embeddings, so it does not invoke text encoders,
  diffusers, checkpoints, or GPU. The direct `guidance_scale == 1.0` path keeps
  the batch size unchanged and casts transformer hidden states to
  `torch.float16`. The CFG `guidance_scale == 3.0` path doubles the batch and
  also casts hidden states to `torch.float16`. The fake SDE step asserts
  float-cast `noise_pred`, `sample`, and `prev_sample`.
- This closes the previous recompute dtype test gap for the current `> 1.0`
  guidance split. Later real SD3.5 checkpoint/CUDA adapter parity and bounded
  `VisualRLTrainer` SD3 smokes passed; further SD3.5 work is guarded trend and
  convergence validation, not first-step plumbing.

## Latest Local SD3 Adapter Compatibility Validation

- 2026-05-24: evaluator validated this as a local compatibility/code coverage
  step only, not a GPU experiment and not real SD3.5 checkpoint/CUDA parity.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_real_image_adapters.py` passed with 8 tests.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q
  tests/test_base_trainer.py` passed with 1 test.
- 2026-05-24: `conda run -n visual-rl python -m pytest -q tests` passed with
  52 tests.
- 2026-05-24: `conda run -n visual-rl python -m ruff check visual_rl tests`
  passed.
- 2026-05-24: `conda run -n visual-rl python -m compileall -q visual_rl tests`
  passed.
- SD3 adapter coverage added: `_call_pipeline_with_logprob()` filters
  unsupported kwargs; it passes `return_dict=False` when the reference pipeline
  supports `return_dict`; `sample()` accepts the TempFlow reference pipeline
  3-tuple no-KL return and zero-fills `kl` to match `old_log_probs`; and
  `_transformer_dtype()` has helper-level dtype coverage.
- BaseTrainer coverage added: `setup_optimizer()` accepts string-like numeric
  config values and builds a valid AdamW optimizer.
- Evaluator checked `reference_code` and found the real TempFlow SD3 pipeline
  signature includes `return_dict` and `kl_reward`, so the helper tests are
  close to the real reference surface.
- Previous non-blocking gap closed on 2026-05-24: CPU fake transformer/SDE
  `recompute_log_probs()` tests now verify dtype casting for guidance off/on.
- Later real SD3.5 checkpoint/CUDA adapter parity and bounded
  `VisualRLTrainer` SD3 smokes passed; remaining SD3.5 scope is guarded trend
  and convergence validation.

## Latest Local SD3 Numeric-Smoke Contract

- 2026-05-24: `sd3-numeric-smoke` now accepts explicit `--repo-root`, writes it
  into adapter `extra["repo_root"]`, and reports `repo_root`/`reference_repo` in
  JSON output.
- 2026-05-24: added and locally validated the shared
  `tempflow-image-numeric-smoke --adapter {sd3_tempflow,flux_tempflow,qwenimage_tempflow}`
  CLI contract. It supports explicit model path, repo root, prompt, resolution,
  denoise steps, guidance, seed, device, dtype, LoRA rank/alpha, logprob
  tolerance, and LoRA disablement; SD3 keeps `--max-sequence-length`.
- 2026-05-24 validation: focused local tests passed with 12 tests, full
  `tests/` passed with 58 tests, Ruff passed, compileall passed,
  `sd3-numeric-smoke --help` passed with `--repo-root`,
  `tempflow-image-numeric-smoke --help` passed, and `git diff --check` passed.
  Evaluator independently confirmed the same focused/full test counts, Ruff,
  compileall, and help commands. Local CUDA was unavailable.
- 2026-05-24 remote direct Python SD3 adapter parity passed on `node01` GPU1
  using the SD3.5 checkpoint and explicit TempFlow `repo_root`: `valid: true`,
  `max_abs_logprob_delta: 0.0`, media shape `[1, 3, 256, 256]`, latents shape
  `[1, 2, 16, 32, 32]`, recomputed logprob shape `[1, 2]`, 382 trainable
  tensors, and 4,694,016 trainable parameters. This proves direct Python
  adapter sample/recompute numeric parity against checkpoint/CUDA.
- 2026-05-24 staged remote CLI smoke passed from
  `/home/v-qiaoqifan/visual_rl_experiments/image_preview_eval_20260524_1130/framecode`.
  The older `/home/v-qiaoqifan/visual_rl_experiments/framecode` tree is still
  not a git checkout and was not overwritten; continue staging current code into
  fresh experiment directories for non-invasive remote validation.

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
  a minimal TempFlow smoke on 2026-05-23; `visual_rl` direct Python SD3 adapter
  sample/recompute parity passed against the real SD3.5 checkpoint on CUDA on
  2026-05-24; staged remote `tempflow-image-numeric-smoke --adapter
  sd3_tempflow` passed; staged remote `remote-sd3-cli-smoke --execute` passed
  with 1-step and 5-step bounded trainer stages.
- Compare Flash selected-step loss against full-trajectory loss on a controlled
  toy scheduler.
- Validate TempFlow branch reward alignment through the SD3.5 `visual_rl`
  adapter path beyond the current 1-step/5-step smoke artifacts; current
  staged runs record reward/logprob/KL/parameter deltas but do not prove reward
  improvement or convergence.
- Extend guarded SD3.5 trend validation only after an idle-GPU check. Suggested
  next command shape:
  `conda run -n visual-rl python -m visual_rl.cli remote-sd3-cli-smoke --execute --gpu 1 --model-path /home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium --repo-root /home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke --conda-env visual-rl-sd35 --bounded-steps 50 --skip-resume-validation --allow-long-run --stage-name sd3_50step_trend_YYYYMMDD_HHMM`.
  Review fixed-prompt before/after PNGs, reward trend, finite logprob/KL fields,
  checkpoint contents, and parameter deltas before extending again.
- Validate FLUX and QwenImage adapters with low-resolution smoke batches only
  after checkpoint paths are found; current status is mocked CLI contract ready.

## Video/Inferix Checks

- Keep Wan/World-R1 runs as smoke-only until small image RL curves are stable.
- Validate World-R1 reward server clients independently before online training.
- For Inferix, validate BlockVid preview/profiling/no-decode paths first; do
  not use it for online RL until logprob/recompute contracts exist.
- 2026-05-24: `InferixEvalBackend` now builds dry-run plans for checkpoint
  preview, profiling, and long-video eval using the vendored Inferix
  self-forcing scripts. Execution remains explicitly disabled in VisualRL; the
  plan records `online_rl_ready: false` until real checkpoint execution and
  logprob/recompute contracts are validated.
- Missing Inferix validation: record a dry-run preview/profile/long-video plan
  against a real checkpoint path, then execute the generated command outside
  VisualRL in the Inferix environment. Do not add an online RL execution path
  until `sample_with_logprob()` and `recompute_logprob()` are implemented and
  tested.
