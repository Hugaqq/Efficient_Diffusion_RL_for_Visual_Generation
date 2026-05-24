# Engineering Checkpoint - 2026-05-24

This checkpoint freezes the current VisualRL integration state before adding
new model features or longer SD3.5 runs.

## Scope

- Reviewed the dirty worktree across VisualRL CLI, SD3, Wan, World-R1, Inferix,
  tests, and planning docs.
- Kept the current mainline focused on the real SD3.5 mini-loop. Tiny probes
  remain regression gates only.
- Checked for conflict markers, obvious debug hooks, and temporary subagent
  leftovers in changed source, tests, prompts, and docs.
- Ran local validation under the `visual-rl` conda environment.

## Issue Found And Fixed

`wan-checkpoint-probe` classified a checkpoint as Wan by searching the full
expanded model path. A pytest temp parent directory containing
`wan_checkpoint_probe` made a `StableDiffusionPipeline` manifest look valid.

Fix:

- `visual_rl/experiments/wan_checkpoint_probe.py` now checks only
  `_class_name` and the checkpoint directory name.
- `visual_rl/experiments/checkpoint_inventory.py` uses the same bounded
  classification rule, so parent directory names cannot contaminate model type
  detection.
- `tests/test_checkpoint_inventory.py` now covers parent-directory keyword
  contamination.

## Local Validation

Commands run from the repo root:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache conda run -n visual-rl python -m compileall -q visual_rl tests
conda run -n visual-rl ruff check visual_rl tests
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache conda run -n visual-rl python -m pytest -q tests
```

Results:

- `compileall`: passed.
- `ruff`: passed.
- `pytest tests`: 99 passed.

Targeted regression after the Wan/checkpoint-inventory fix:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache conda run -n visual-rl python -m pytest -q tests/test_wan_checkpoint_probe.py tests/test_checkpoint_inventory.py
```

Result: 7 passed.

## Current Risk Register

- SD3.5 evidence is still smoke/trend evidence. The 20-step run proves bounded
  execution, metrics, artifacts, checkpointing, and parameter updates, not
  reward quality or long-run convergence.
- Current SD3.5 rewards are cheap validation rewards. HPS, PickScore,
  multi-reward, and 3D/world rewards are not in a stable training loop.
- FLUX and QwenImage remain blocked on real checkpoint paths.
- Wan and World-R1 have planning/probe coverage, but no verified real
  sample/logprob/training path and no successful live reward-server call.
- Inferix remains a dry-run preview/profiling backend until real checkpoint
  execution and logprob/recompute contracts are validated.

## Next SD3.5 Experiment Gate

Do not add another feature before the next guarded trend run. Recommended next
experiment:

1. Use a fresh staged remote run, not the shared checkout.
2. Keep prompts fixed and save before/after preview PNGs for the same prompt
   set.
3. Run 50 steps first; only try 100 steps after 50-step artifacts and metrics
   look sane.
4. Record reward mean/std, old/new logprob means, KL, clipfrac, trainable
   parameter count, parameter delta, checkpoint files, and PNG comparisons in a
   single report.

Dry-run command shape:

```bash
conda run -n visual-rl python -m visual_rl.cli remote-sd3-cli-smoke \
  --bounded-steps 50 \
  --allow-long-run \
  --stage-name sd3_50step_guarded_YYYYMMDD_HHMM \
  --dry-run
```

Execution should add the real `--server`, `--gpu`, `--model-path`, and
`--repo-root` values only after confirming the target GPU is idle.
