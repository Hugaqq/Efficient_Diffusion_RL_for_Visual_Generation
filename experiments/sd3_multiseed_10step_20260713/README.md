# SD3 multi-seed 10-step validation

This is the first incremental experiment after the 5-step single-seed pilot.
It contains three matched active/control pairs. GPU2 runs active and GPU3 runs
the zero-learning-rate control for the same training seed; pairs are executed in
three waves to avoid mixing seeds and to respect the two-GPU limit.

The experiment is intentionally bounded at 10 optimizer steps. It is not a
50-step or 100-step effectiveness claim. The frozen inputs, hashes, environment,
and pre-registered gates are in `recipe.json`.

## Promotion rule

Run `scripts/aggregate_sd3_runs.py` over all six summaries. Continue to the
real resume-equivalence experiment regardless of effect size, because resume is
an infra correctness property. Continue toward 20/50-step effectiveness only if
all cross-run gates pass. Failed seeds and failed colors remain in the report.

## Evidence boundaries

- Passing proves repeatability only for this SD3.5 checkpoint, guarded RGB
  reward, prompt distribution, and bounded budget.
- It does not validate semantic image quality, Flash-GRPO/Wan, World-R1, DDP,
  or general convergence.
- Zero-LR controls validate deterministic evaluation and parameter immutability;
  they are not a replacement for the native TempFlow reference comparison.
