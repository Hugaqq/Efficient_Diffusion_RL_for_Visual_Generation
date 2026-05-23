# Dependency Matrix

Phase 0 keeps the four legacy projects isolated because they share the `flow_grpo`
package name and have partially different dependency assumptions.

| Project | Role | Python | Torch | Diffusers | Accelerate | Transformers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference_code/World-R1-main | World/video specialization | >=3.10 | CUDA build matching server | loosely documented | loosely documented | loosely documented | Reward stack and camera-aware latent source. |
| reference_code/GenRL-main | Training runtime reference | >=3.10 | 2.6.x | see pyproject | see pyproject | see pyproject | Preferred trainer/runtime reference. |
| reference_code/Flash-GRPO-main | Flash single-step plugin | >=3.10 | 2.6.0 | 0.33.1 | 1.4.0 | 4.40.0 | `setup.py` pins many versions. |
| reference_code/TempFlow-GRPO-main | TempFlow branching plugin | >=3.10 | likely same as Flash | likely same as Flash | likely same as Flash | likely same as Flash | Several hardcoded algorithm constants need config. |
| reference_code/Inferix-main | BlockVid eval/preview/profiling backend | >=3.10 | see `requirements-torch.txt` | unpinned | unpinned | unpinned | Inference-first; learn BlockVid/semi-AR block diffusion, not training runtime. |
| visual_rl | Integration layer | >=3.10,<3.12 | optional for smoke/training | optional | optional | optional | Import path is lightweight; heavy imports are lazy. |

v0.1 local smoke environment:

- `conda` env: `visual-rl`
- Python: `3.10`
- `numpy<2`
- local smoke uses `torch` only for the mock train loop

Server training environment target:

- Python 3.10
- CUDA PyTorch matching the 5090 driver
- `numpy==1.26.4`
- World-R1 dependencies installed in an isolated env
- no shared installation of Flash/TempFlow/World-R1 as overlapping `flow_grpo`
