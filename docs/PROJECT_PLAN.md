# Project Plan

This repo is being simplified around one train loop and three integration
targets.

## Keep

- `visual_rl.trainer.trainer.VisualRLTrainer`
- `visual_rl.core.types.RolloutBatch`
- `visual_rl.core.types.RewardBatch`
- `visual_rl.rollout.full_trajectory`
- `visual_rl.rollout.single_step`
- `visual_rl.rollout.branching`
- `visual_rl.algorithms.grpo`
- `visual_rl.algorithms.flash_grpo`
- `visual_rl.algorithms.tempflow_grpo`
- `visual_rl.rewards.router`
- `visual_rl.model_adapters.tiny_diffusion`
- SD3/TempFlow adapter code that supports the current image-model bridge
- World-R1/Wan planning code until the real video sample/logprob adapter is ready

## Avoid

- Parallel runner stacks
- Manifest/report-only training paths
- New core data contracts beyond `RolloutBatch` and `RewardBatch`
- A new GenRL runtime trunk

## Reference Repos

```text
reference_code/
  Flash-GRPO-main/
  TempFlow-GRPO-main/
  World-R1-main/
  GenRL-main/        # reference only
```

`reference_code/` is ignored by git and is not present in the current local
checkout. Before porting Flash-GRPO, TempFlow-GRPO, or World-R1 behavior, restore
those snapshots under `reference_code/` or configure explicit external paths; do
not make their packages import-time requirements.

## Next Code Work

1. **Flash-GRPO**
   - Port Wan selected-timestep sampling into a `visual_rl` adapter.
   - Port current-model logprob recomputation for the selected transition.
   - Replace hardcoded reference-script behavior with config fields where needed.

2. **TempFlow-GRPO**
   - Keep the tiny branching path as the regression check.
   - Keep SD3 TempFlow as the current real-image bridge.
   - Verify image/prompt/metadata ordering whenever branching is touched.

3. **World-R1**
   - Keep reward-server endpoint probes.
   - Add real Wan checkpoint loading and one bounded sample/logprob smoke before
     claiming video training support.

## Validation

Start with focused tests instead of root-level collection:

```bash
conda run -n visual-rl python -m pytest -q \
  tests/test_flash_grpo.py \
  tests/test_tempflow_branching.py \
  tests/test_mock_trainer.py \
  tests/test_reward_router.py
```

Use full `tests/` only after cleanup passes and reference-code over-collection is
confirmed absent.
