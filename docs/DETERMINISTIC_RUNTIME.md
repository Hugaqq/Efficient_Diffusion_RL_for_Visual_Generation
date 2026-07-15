# Deterministic runtime mode

VisualRL's real SD3 audit found that independent BF16/CUDA runs were identical
through checkpoint restore, RNG, SDE rollout, rewards, log-probs, and loss, but
first diverged in backward gradients. The validated deterministic runtime made
both one-step tensor probes and two independent five-step continuations exact.

Enable the mode in config:

```yaml
runner:
  deterministic_runtime: true
```

Python hash randomization is fixed before Python starts, so launch with the
training seed in the environment:

```bash
PYTHONHASHSEED=201 python train.py --config path/to/config.yaml
```

For the bounded SD3 CLI, pass the same seed and the explicit flag:

```bash
PYTHONHASHSEED=201 python -m scripts.legacy_cli \
  sd3-bounded-trainer-smoke \
  --seed 201 \
  --deterministic-runtime \
  ...
```

The mode enables deterministic PyTorch algorithms, uses the validated cuBLAS
workspace configuration, disables TF32, makes cuDNN deterministic, and records
the observed runtime in checkpoint implementation identity. Resume therefore
rejects a checkpoint when its runtime identity is incompatible.

Use the exact environment that produced the checkpoint. On the current lab
server the validated SD3 environment is `visual-rl-sd35` with PyTorch
2.11.0+cu130 / CUDA 13.0. A different environment may have the same project
name or model dependencies but a different Torch/CUDA/PEFT stack; the resume
guard will reject it before an optimizer step.

This mode prioritizes reproducibility over throughput. The SD3 diagnostic
rollout was slower under deterministic kernels, so performance claims must be
measured separately.
