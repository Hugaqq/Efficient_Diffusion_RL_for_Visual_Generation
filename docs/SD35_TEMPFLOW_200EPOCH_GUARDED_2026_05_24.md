# SD3.5 TempFlow Guarded 200-Epoch Run: 2026-05-24

This run executed the SD3.5-medium TempFlow reference path under a GPU-idle
supervisor on the shared 8x RTX 5090 server.

## Setup

- Server: `v-qiaoqifan@10.130.140.73`
- Environment: `/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35`
- GPU policy: physical GPU1 only
- Model: `/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium`
- Script root: `/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke`
- Run root: `/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded`
- Config base: `config/grpo.py:tempflow_sd3_server_smoke`
- Reward: `jpeg_compressibility`
- Resolution: 256
- Denoising steps: 3
- Branches per timestep: 2
- Batch size: 1 prompt, 2 branch samples for training

## Guard Behavior

The guard was deployed from:

- `scripts/gpu_idle_guard.py`
- `scripts/run_sd35_tempflow_staged.sh`

It started only when GPU1 was idle, watched for foreign compute PIDs on GPU1,
and would stop/terminate its own process group if another user appeared on the
same GPU. No foreign GPU1 process was observed during the successful run.

## Stages

The staged runner used:

```text
5 epoch probe -> 20 epoch probe -> 200 epoch final run
```

Results:

- Guard start: 2026-05-24 00:47:06 CST
- Stage 5 complete: 2026-05-24 00:51:14 CST
- Stage 20 complete: 2026-05-24 00:54:16 CST
- Stage 200 complete: 2026-05-24 00:57:35 CST
- Guard completed: 2026-05-24 00:57:40 CST

The first guarded launch failed before GPU allocation because `flow_grpo` was
not on `PYTHONPATH`. The runner was fixed to export `PYTHONPATH=$ROOT`. A second
launch completed the 5-epoch probe, then failed at the 20-epoch handoff because
`ml_collections` cannot override a `None` `config.train.lora_path`. The staged
runner was changed to treat 5/20 as independent probes and run the final
200-epoch stage from the base model. After that change, the full staged run
completed.

## Outputs

Checkpoints:

```text
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-2/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-4/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-5/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-10/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-15/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-20/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-40/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-60/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-80/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-100/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-120/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-140/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-160/lora
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_200epoch_guarded/checkpoints/checkpoint-180/lora
```

Latest offline W&B run:

```text
/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke/wandb/offline-run-20260524_005422-5ksywgr5
```

Final guard status:

```json
{"gpu": 1, "returncode": 0, "state": "completed"}
```

GPU1 returned to idle after completion:

```text
GPU1: 3 MB / 32607 MB, 0% utilization
```

## Scope

This verifies that the SD3.5-medium TempFlow reference path can run a guarded
200-epoch low-resolution training loop on one RTX 5090 without interfering with
other active GPU jobs. It does not yet prove reward-quality improvement,
adapter-level parity through `visual_rl`, or high-resolution/paper-scale
TempFlow settings.
