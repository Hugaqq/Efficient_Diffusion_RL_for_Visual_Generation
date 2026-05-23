# SD3.5 TempFlow Validation: 2026-05-23

This validation strengthened the TempFlow correctness check from the local
`tiny_diffusion` adapter to a real SD3.5-medium TempFlow path.

Important scope note: this used the existing reference TempFlow/Flow-GRPO SD3
script on the server. The `visual_rl.model_adapters.sd3` adapter is still a lazy
bridge and is not yet the production SD3.5 training adapter.

## Environment

- Host: `node01`
- GPU used: physical GPU1 only
- Busy GPUs avoided:
  - GPU0 was occupied by another user's `remote_train_v3.py`
  - GPU2-7 were occupied by existing MoT jobs
- Environment: `/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35`
- Base env: cloned from `sglang`
- PyTorch: `2.11.0+cu130`
- CUDA device: RTX 5090
- Reference code: `/home/v-qiaoqifan/visual_rl_experiments/flow_grpo_tempflow_smoke`
- Model: `/home/v-qiaoqifan/flow_grpo/hf_cache/stable-diffusion-3.5-medium`

Why a new env was needed:

- The existing `tempflow-grpo` env has `torch 2.6.0+cu124`.
- That build reports CUDA available, but fails on RTX 5090 with
  `no kernel image is available for execution on the device`.
- The new `visual-rl-sd35` env runs CUDA 13.0 kernels successfully on RTX 5090.

## Config

The run used `config/grpo.py:tempflow_sd3_server_smoke` from the reference
TempFlow smoke tree.

Key settings:

```text
resolution: 256
num_steps: 3
branch_per_timestep: 2
sde_window_size: 2
train_batch_size: 1
train.batch_size: 2
reward: jpeg_compressibility
cfg: false
guidance_scale: 1.0
LoRA: enabled
save_freq: 0
eval_freq: 0
wandb: offline
```

## Runs

### One-Epoch Server Smoke

Remote output:

```text
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_correctness/20260523_212220_gpu1_server_smoke
```

Result:

- SD3.5-medium pipeline loaded.
- Sampling completed for 1 epoch.
- TempFlow per-step rewards completed.
- Advantage tensor was nonzero: mean absolute value `0.9076`.
- Training completed across both trainable timesteps.
- No OOM or CUDA kernel error.

### Three-Epoch Correctness Probe

Remote output:

```text
/home/v-qiaoqifan/visual_rl_experiments/sd35_tempflow_correctness/20260523_212349_gpu1_3epoch
```

Result:

| Epoch | Sampling | Per-Step Reward | Advantage Mean Abs | Training |
| --- | --- | --- | ---: | --- |
| 0 | passed | passed | 0.9076 | passed |
| 1 | passed | passed | 0.9413 | passed |
| 2 | passed | passed | 0.7951 | passed |

This verifies that the real SD3.5 TempFlow reference loop can repeatedly execute:

- SD3.5 model load
- branch rollout
- branch image generation
- per-timestep reward scoring
- advantage construction
- logprob recomputation
- PPO/GRPO-style update
- offline logging

## Interpretation

This is stronger than the `tiny_diffusion` correctness probe because it touches
a real SD3.5-medium policy, real latent/image generation, and the reference
TempFlow per-step branch machinery.

It still does not prove:

- long-horizon SD3.5 reward improvement
- SD3.5/FLUX/QwenImage parity inside the `visual_rl` adapter abstraction
- large group sizes or paper-scale branching
- reward-model correctness for PickScore/OCR/Geneval
- checkpoint/resume correctness for SD3.5, because this smoke used `save_freq=0`

## Next Step

The next meaningful integration step is to wrap this working SD3.5 reference
path behind a `visual_rl` adapter contract:

```python
parameters()
sample()
recompute_log_probs()
save_pretrained()
```

After that, the same SD3.5 smoke should run through `VisualRLTrainer` rather
than through the reference script directly.
