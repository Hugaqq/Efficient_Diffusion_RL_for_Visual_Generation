# VisualRL

VisualRL is a small reinforcement-learning training infra for diffusion-based
visual generation. The active goal is to integrate three reference directions:

- **Flash-GRPO**: selected-timestep / single-step GRPO.
- **TempFlow-GRPO**: branching image rollouts and timestep credit assignment.
- **World-R1**: Wan video/world generation and reward-server integration.

GenRL remains an engineering reference for config, trainer lifecycle, sampling,
rewards, and checkpointing patterns.

## Current Mainline

Keep the runnable training path centered on `VisualRLTrainer`:

```text
PromptDataset
  -> RolloutEngine
  -> RewardRouter
  -> AdvantageComputer
  -> adapter.recompute_log_probs()
  -> algorithm.compute_loss()
  -> loss.backward()
  -> optimizer.step()
  -> checkpoint / metrics
```

The tiny Flash and TempFlow paths are regression gates. The real integration
work should move code from Flash-GRPO, TempFlow-GRPO, and World-R1 into model
adapters and rollout engines that can run through `VisualRLTrainer`.

## What Is In Scope

- Keep `tiny_diffusion` for cheap end-to-end tests.
- Keep Flash-GRPO single-step rollout and loss.
- Keep TempFlow branching rollout and loss.
- Keep SD3 TempFlow adapter work as the current image-model bridge.
- Keep World-R1/Wan planning and probes until a real Wan sample/logprob path is
  implemented.

## Boundaries

- Keep a single training loop.
- Keep `RolloutBatch` and `RewardBatch` as the core data contracts.
- Do not add a parallel artifact/report runner.
- Do not make GenRL a runtime trunk.
- Do not broaden adapter work until Flash, TempFlow, and World-R1 are clean.

## Useful Checks

```bash
conda run -n visual-rl python -m pytest -q tests/test_flash_grpo.py tests/test_tempflow_branching.py tests/test_mock_trainer.py
conda run -n visual-rl python -m visual_rl.cli flash-smoke --output-dir runs/flash_tiny_smoke --steps 2
conda run -n visual-rl python -m visual_rl.cli tempflow-smoke --output-dir runs/tempflow_tiny_smoke --steps 2
```

See [docs/CURRENT_GOAL.md](docs/CURRENT_GOAL.md) and
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the short current plan.
