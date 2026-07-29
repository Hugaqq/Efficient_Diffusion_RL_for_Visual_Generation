# VisualRL v0.7 fixed experiment suites

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

These commands require the documented GPU/model/reference/reward environment.
They are not part of source-preparation tests and have not been run in this
round. There is no CLI option, YAML overlay, experiment Runner, checkpoint
loader, or plugin path.

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

See `docs/V0_7_USER_GUIDE.md` for environment and usage details and
`docs/V0_7_ACCEPTANCE.md` for the explicit `not_run` matrix.
