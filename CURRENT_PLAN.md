# Current Plan

The canonical VisualRL plan is [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).
The short active-goal summary is [`docs/CURRENT_GOAL.md`](docs/CURRENT_GOAL.md),
and validation gaps are tracked in
[`docs/EXPERIMENT_VALIDATION_BACKLOG.md`](docs/EXPERIMENT_VALIDATION_BACKLOG.md).
The latest dirty-worktree review and local validation checkpoint is
[`docs/ENGINEERING_CHECKPOINT_2026_05_24.md`](docs/ENGINEERING_CHECKPOINT_2026_05_24.md).

This root file exists because some coordination prompts refer to
`CURRENT_PLAN.md`. Keep `docs/PROJECT_PLAN.md` canonical to avoid parallel plan
drift.

## Active Direction

VisualRL is the integration infra for the four active targets:

- `World-R1-main`
- `Flash-GRPO-main`
- `TempFlow-GRPO-main`
- `Inferix-main`

`GenRL-main` remains an engineering reference only, not the runtime trunk.

The current mainline is a small, inspectable SD3.5 image RL loop through
VisualRL:

```text
SD3.5 checkpoint
  -> VisualRL SD3 adapter sample
  -> save preview PNGs
  -> reward router
  -> TempFlow/GRPO update
  -> LoRA checkpoint
  -> before/after preview
  -> metrics and blocker documentation
```

Tiny diffusion checks are regression gates only. Existing bounded SD3.5
1-step, 5-step, resume, and 20-step smoke evidence proves plumbing, metrics,
checkpoint, PNG, parameter-update, and short trend-run behavior; it does not
prove meaningful convergence or paper-scale training.

## Still Open

- Continue SD3.5 beyond smoke scale only with guarded, staged runs; repeat or
  extend the 20-step trend before claiming convergence.
- Keep FLUX and QwenImage real smokes blocked until valid checkpoint paths are
  available; `checkpoint-inventory` now records that only SD3.5 and Wan
  Diffusers paths were found in the scoped remote scan.
- Keep SD1.5 real validation deferred unless a valid Diffusers checkpoint path
  appears; do not let it block SD3.5.
- Validate real Wan/World-R1 loading and live reward-server calls before
  claiming video/world integration. The local plan/probe CLIs now validate
  model-path/readiness metadata and endpoint payload/error contracts, but no
  live Wan sample/logprob path or reward endpoint has passed.
- Keep Inferix execution out of online RL until the dry-run preview/profiling
  plan backend is validated against real checkpoints and logprob contracts.
- Continue hardening shared infra around real reward services, evaluator
  backends, and larger guarded SD3.5 trend runs.
