# VisualRL

VisualRL is the integration layer for the four local projects:

- `World-R1-main`: v0.1 trunk for Wan/CogVideoX video RL, camera-aware latents, 3D/general rewards.
- `Flash-GRPO-main`: future Flash single-step GRPO algorithm plugin.
- `TempFlow-GRPO-main`: future branching GRPO algorithm plugin for SD3/FLUX/QwenImage.
- `Inferix-main`: future eval, preview, serving, and profiling backend.

v0.2 keeps the v0.1 isolation and adds a GenRL-inspired runtime layer:
typed config, BaseTrainer lifecycle, epoch-aware sampler utilities, per-reward
advantages, raw/weighted reward logging, and media-aware reward cache.

## Quick Smoke

```bash
conda activate visual-rl
pip install -e ".[dev]"
visual-rl smoke-imports
visual-rl smoke-mock --output-dir runs/smoke --steps 2
visual-rl world-r1-plan --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers --gpus 6,7
```

## Server Safety

Use the GPU probe before running anything on the shared 8x5090 server:

```bash
bash scripts/server_gpu_probe.sh v-qiaoqifan@10.130.140.73
```

The v0.1 launcher is conservative by default and accepts explicit `CUDA_VISIBLE_DEVICES`.
