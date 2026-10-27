# VisualRL v0.7 fixed experiment suites

This directory is a historical v0.7 snapshot, not an entry point for current v0.8.
The Python API below has been retired; use the current repository README for training.

This directory contains source preparation only. Every training role is a full
YAML file and all training flows through:

```python
vr.load(config_path).resolve()
experiment.validate()
experiment.run()
vr.inspect_run(output_dir)
vr.audit_run(output_dir)
```

`common_api_run.py` is the only experiment-side `load()/run()` callsite.
`interrupt_resume.py` owns the sole frozen 30-role table, family consistency
checks, readiness barrier, interruption policy, and C20-before-Q100 gates.
Algorithm wrappers pass only one literal family name.

Fixed entry modules:

- `python -m experiments.v0_7.tiny_s100`
- `python -m experiments.v0_7.flow_grpo_sd3`
- `python -m experiments.v0_7.tempflow_sd3`
- `python -m experiments.v0_7.flash_wan`
- `python -m experiments.v0_7.world_r1_wan`
- `python -m experiments.v0_7.mg1_nccl`

These commands require the documented GPU/model/reward environment. Bounded
engineering runs using the same public API and fixed role configurations have
completed Flow, TempFlow, Flash and World-R1 C20
continuous/interrupted-resume parity. They are documented separately in
[the archived operational evidence](https://github.com/Hugaqq/Efficient_Diffusion_RL_for_Visual_Generation/blob/00cee786df783f36cb254a9a666a48a5952dc0a3/docs/V0_7_OPERATIONAL_EVIDENCE.md) because a dirty engineering candidate
cannot populate the clean-candidate final evidence envelope. There is no CLI
option, YAML overlay, experiment Runner, checkpoint loader, or plugin path.

Q100 aggregation is read-only: `offline_aggregate.py` first calls the public
status/audit APIs for exactly the twelve paths in `evidence/q100_inputs.json`,
then reads `weighted_total` from the authoritative manifest.
`verify_reward_improvement.py` consumes only canonical rows and source status.
After authorized runs, the fixed no-argument
`python -m experiments.v0_7.offline_aggregate` entry atomically regenerates the
canonical rows and candidate-bound source envelope; it removes stale generated
outputs before scanning. `verify_evidence.py` then requires the exact six
semantic-family results, 30 roles, Flow native report, MG1 internal results,
all-role environment attempts, Q100 digest, and the current clean Git HEAD.

The old user guide, operational evidence and acceptance matrix were removed
from current docs. Read them in the
[fixed historical snapshot](https://github.com/Hugaqq/Efficient_Diffusion_RL_for_Visual_Generation/tree/00cee786df783f36cb254a9a666a48a5952dc0a3/docs),
not as instructions or support claims for the current framework.
