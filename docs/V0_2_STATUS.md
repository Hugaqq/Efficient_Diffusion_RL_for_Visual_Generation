# v0.2 Status

Implemented locally:

- `visual_rl` version bumped to `0.2.0`.
- Typed config dataclasses inspired by GenRL:
  - `ModelConfig`
  - `DatasetConfig`
  - `SampleConfig`
  - `AlgorithmConfig`
  - `RewardConfig`
  - `TrainConfig`
  - `FSDPConfig`
  - `AccelerateConfig`
  - `ProjectPaths`
- `BaseTrainer` with config validation, deterministic output paths, optimizer setup,
  gradient accumulation calculation, and resolved config logging.
- `AdvantageComputer` and `PerPromptStatTracker` with:
  - per-prompt advantages
  - per-reward advantages
  - `weight_advantages`
  - zero-std ratio logging
  - global/max-group std modes
- RewardRouter v2:
  - raw reward dict
  - weighted reward dict
  - weighted total
  - normalized total
  - valid mask
  - media-aware cache key
  - reward version in cache key
- Epoch-aware repeat sampler utility.
- v0.2 mock preset: `visual_rl/configs/presets/world_r1_wan_v02_mock.yaml`.

Validation:

```bash
conda run -n visual-rl visual-rl smoke-imports
conda run -n visual-rl visual-rl smoke-mock --output-dir runs/smoke_v02_mock --steps 2
conda run -n visual-rl visual-rl world-r1-plan --model-path /models/Wan2.1-T2V-1.3B-Diffusers --gpus 6,7
conda run -n visual-rl python -m pytest -q tests
conda run -n visual-rl python -m ruff check visual_rl tests
```

Not run:

- Any server command.
- Any SSH/GPU probe.
- Any real Wan2.1 checkpoint or reward model.

