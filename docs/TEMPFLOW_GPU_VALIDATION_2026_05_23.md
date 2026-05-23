# TempFlow GPU Validation: 2026-05-23

This validation run checked the current TempFlow training path on the shared
8x5090 server without touching busy GPUs.

## Environment

- Host: `node01`
- Env: `/home/v-qiaoqifan/miniconda3/envs/sglang`
- PyTorch: `2.11.0+cu130`
- GPUs used: `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`
- Busy GPUs avoided: physical GPUs 2-7
- Model: `tiny_diffusion`
- Algorithm: `tempflow_grpo`
- Rollout: branching, main plus 3 branches
- Reward: `prompt_color`

## Two-GPU TempFlow Smoke

Two independent TempFlow runs were launched in parallel, one process per GPU.

| Job | Steps | First Reward | Last Reward | Best Reward | Active Timestep Fraction | Max Reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU0 seed 21 | 16 | 0.5004 | 0.5676 | 0.5676 | 0.25 | 2 MB |
| GPU1 seed 22 | 16 | 0.4969 | 0.6242 | 0.6242 | 0.25 | 2 MB |

This verified that two isolated TempFlow jobs can run concurrently on GPU0 and
GPU1 while leaving the other users' GPU2-7 jobs untouched.

## Correctness Probe

The stronger probe used a fixed red prompt for 100 steps and checked:

- evaluation reward before and after training
- trainable `color_bias` direction
- branch IDs in the cached rollout
- active timestep fraction
- metric line count
- rollout cache, checkpoint, and latest marker
- finite reward/loss metrics

| Job | Steps | Pre Eval Reward | Post Eval Reward | Delta | Final Bias | Passed |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| GPU0 seed 101 | 100 | 0.4997 | 0.5980 | +0.0982 | `[0.2246, 0.0117, -0.3663]` | yes |
| GPU1 seed 202 | 100 | 0.4997 | 0.8078 | +0.3081 | `[0.8082, -0.4181, -0.5682]` | yes |

Both runs satisfied the expected checks:

- reward improved by more than 0.05
- red bias increased from zero
- red bias became the dominant channel
- branch IDs were `[-1, 0, 1, 2]`
- active timestep fraction was `0.25`
- 100 metric lines were written
- rollout cache, checkpoint, and `latest.json` existed
- metrics were finite

## Output Directories

Remote server paths:

```text
/home/v-qiaoqifan/visual_rl_experiments/framecode/runs/tempflow_two_gpu_only/gpu0_seed21
/home/v-qiaoqifan/visual_rl_experiments/framecode/runs/tempflow_two_gpu_only/gpu1_seed22
/home/v-qiaoqifan/visual_rl_experiments/framecode/runs/tempflow_correctness/red_gpu0_seed101
/home/v-qiaoqifan/visual_rl_experiments/framecode/runs/tempflow_correctness/red_gpu1_seed202
```

## Interpretation

This still does not prove real SD3/FLUX/QwenImage TempFlow parity. It does
verify that the current TempFlow infra path has a working RL training circuit:
branching rollout, reward scoring, branch/timestep credit assignment, backward
update, metrics, rollout cache, checkpointing, and isolated two-GPU execution.
