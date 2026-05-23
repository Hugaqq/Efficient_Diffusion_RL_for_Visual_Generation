# VisualRL

VisualRL is the integration layer for the four local projects:

- `reference_code/World-R1-main`: world/video specialization for Wan/CogVideoX video RL, camera-aware latents, 3D/general rewards.
- `reference_code/GenRL-main`: training runtime reference for typed config, trainer lifecycle, sampling, rewards, and checkpointing.
- `reference_code/Flash-GRPO-main`: single-step GRPO reference for low-cost tiny image RL first, then video.
- `reference_code/TempFlow-GRPO-main`: branching GRPO algorithm reference for tiny image RL first, then SD3/FLUX/QwenImage.
- `reference_code/Inferix-main`: BlockVid-oriented eval, preview, serving, and profiling backend.

v0.2 keeps the v0.1 isolation and adds a GenRL-inspired runtime layer:
typed config, BaseTrainer lifecycle, epoch-aware sampler utilities, per-reward
advantages, raw/weighted reward logging, and media-aware reward cache.

v0.3 adds the first TempFlow implementation: tiny diffusion image rollout,
prompt-color reward, branching rollout expansion, TempFlow-GRPO
branch/timestep credit assignment, noise-aware weighting, lazy SD3/FLUX/QwenImage
bridges, and CPU-only smoke coverage.

v0.4 adds the first Flash-GRPO implementation: single-step rollout,
iso-temporal prompt grouping, selected timestep metadata, scheduler-style
rectification weights, and a tiny diffusion smoke path.

The canonical current roadmap is [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).
It treats GenRL as an engineering reference only, while the actual integration
targets are World-R1, Flash-GRPO, TempFlow-GRPO, and Inferix.

## Quick Smoke

```bash
conda activate visual-rl
pip install -e ".[dev]"
visual-rl smoke-imports
visual-rl smoke-mock --output-dir runs/smoke --steps 2
visual-rl tempflow-smoke --output-dir runs/tempflow_tiny_smoke --steps 2
visual-rl flash-smoke --output-dir runs/flash_tiny_smoke --steps 2
visual-rl world-r1-plan --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers --gpus 6,7
visual-rl wan-plan --output-dir runs/wan_runtime_plan
```

## Server Safety

Use the GPU probe before running anything on the shared 8x5090 server:

```bash
bash scripts/server_gpu_probe.sh v-qiaoqifan@10.130.140.73
```

The v0.1 launcher is conservative by default and accepts explicit `CUDA_VISIBLE_DEVICES`.

For TempFlow tiny smoke on the shared server, prefer CPU-only execution first:

```bash
CUDA_VISIBLE_DEVICES="" python -m visual_rl.cli tempflow-smoke --output-dir runs/server_tempflow_tiny_smoke --steps 2
```
