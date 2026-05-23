# v0.1 Status

Implemented locally:

- Top-level git repository initialized.
- `visual_rl` package skeleton created.
- Shared `RolloutBatch` and `RewardBatch` contracts.
- `RewardRouter` with mock and generic pickle-over-HTTP clients.
- Rollout cache layout under `runs/{run_id}/rollouts`.
- Mock Wan adapter and GRPO trainer smoke path.
- Lazy World-R1 Wan legacy adapter skeleton.
- Conservative World-R1 baseline launcher for later server use.
- `visual-rl world-r1-plan` dry-run plan generator for the legacy baseline.
- Server GPU probe script is present but not run.

Local validation:

```bash
conda run -n visual-rl python -m visual_rl.cli smoke-imports
conda run -n visual-rl python -m visual_rl.cli smoke-mock --output-dir runs/smoke_v01_mock --steps 2
conda run -n visual-rl python -m pytest -q tests
conda run -n visual-rl visual-rl world-r1-plan --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers --gpus 6,7
```

Not run yet:

- Any server SSH/GPU probe.
- Any real Wan2.1 checkpoint load.
- Any real reward server.
