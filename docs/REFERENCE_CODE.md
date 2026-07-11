# Reference Code Policy

`reference_code/` is a local-only directory for upstream paper/source snapshots.
It is ignored by git and should stay optional.

The current local checkout does not include `reference_code/`. The resolver now
also checks the sibling root `../code_base/reference_code`, which matches the
current local layout:

```text
~/Desktop/Efficient_Diffusion_RL_for_Visual_Generation/code_base/reference_code
```

Set `VISUAL_RL_REFERENCE_CODE_ROOT` to override this location. Before porting
behavior from Flash-GRPO, TempFlow-GRPO, or World-R1, make sure the expected
snapshot exists in one of those roots.

Expected local layout:

```text
reference_code/
  Flash-GRPO-main/
  TempFlow-GRPO-main/
  World-R1-main/
  GenRL-main/
```

Use these repositories as source material for specific behavior:

- Flash-GRPO: selected-timestep Wan sampling, logprob recomputation, temporal
  rectification.
- TempFlow-GRPO: branching rollouts, timestep credit assignment, SD3 bridge.
- World-R1: Wan/world video path and reward servers.
- GenRL: config and trainer-structure reference only.

Do not import reference repos at package import time. Heavy imports should stay
lazy and behind adapters or explicit commands.
