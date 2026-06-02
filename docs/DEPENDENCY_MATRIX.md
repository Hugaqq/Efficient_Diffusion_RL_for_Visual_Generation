# Dependency Matrix

VisualRL keeps the default install small.

## Core

| Area | Dependency |
| --- | --- |
| package/runtime | Python 3.10-3.11 |
| config | `pyyaml` |
| numeric helpers | `numpy` |

## Optional Training

The `train` extra is for real diffusion-model adapters:

| Area | Dependency |
| --- | --- |
| tensor/runtime | `torch`, `torchvision` |
| diffusion pipelines | `diffusers`, `transformers` |
| LoRA | `peft` |
| distributed/runtime | `accelerate` |
| logging/media | `wandb`, `pillow`, `imageio` |
| configs/http | `ml-collections`, `requests` |

## Reference Code

| Reference | Role | Runtime Policy |
| --- | --- | --- |
| `Flash-GRPO-main` | selected-step Wan Flash-GRPO behavior | optional, lazy |
| `TempFlow-GRPO-main` | branching rollout behavior and the SD3 image bridge | optional, lazy |
| `World-R1-main` | Wan/world video and reward-server behavior | optional, lazy |
| `GenRL-main` | engineering reference | never a runtime trunk |
