# VisualRL v0.7 user guide

VisualRL v0.7 has one public entry: the Python API. There is no `visual-rl`
console command, plugin registry, preset resolver, recipe overlay, or second
training Runner.

## Install and import

The release candidate supports Python `>=3.10,<3.12`. The W07 local release
gate verifies one explicit candidate wheel, a fresh base-environment install,
`pip check`, and an outside-repository isolated import. This base-wheel result
does not claim that the optional training stack or real GPU experiments ran.

```python
import visual_rl as vr

print(vr.__version__)
```

Model training additionally needs the `train` extra and the model/reference
repositories selected by the complete YAML. Installing the package does not
download checkpoints or reference repositories.

## Validate and run one complete YAML

Every run uses exactly this path:

```python
from pathlib import Path

import visual_rl as vr

config_path = Path("/absolute/path/to/complete-config.yaml")
experiment = vr.load(config_path)
resolved = experiment.resolve()
report = experiment.validate()

if not report.ok:
    for check in report.errors:
        print(check.code, check.path, check.message)
    raise RuntimeError("preflight failed")

result = experiment.run()
status = vr.inspect_run(result.output_dir)
audit = vr.audit_run(result.output_dir)
if not status.ok or not audit.ok:
    raise RuntimeError("authoritative run audit failed")
```

`load()` reads one UTF-8 YAML with no inheritance. `resolve()` selects only
built-in components and normalizes paths. `validate()` performs bounded
preflight and returns structured checks. An Experiment handle can attempt
`run()` only once. Resume is expressed in another complete YAML whose
`resume.from` is the same run directory as `artifacts.output_dir`.

## Fixed source-prepared suites

The 30 complete experiment configurations live in
[`experiments/v0_7/configs`](../experiments/v0_7/configs). Their sole role table
and C20-before-Q100 gates live in
[`interrupt_resume.py`](../experiments/v0_7/interrupt_resume.py).

The fixed family modules take no user arguments:

```text
python -m experiments.v0_7.tiny_s100
python -m experiments.v0_7.flow_grpo_sd3
python -m experiments.v0_7.tempflow_sd3
python -m experiments.v0_7.flash_wan
python -m experiments.v0_7.world_r1_wan
python -m experiments.v0_7.mg1_nccl
```

Do not run these merely to test installation: they are real training suites.
They require explicit experiment authorization, the frozen checkpoints and
reference source trees, compatible CUDA GPUs, and the declared World-R1
endpoint. W06 did not execute them.

## Read-only run inspection

Use only public artifact readers:

```python
status = vr.inspect_run("/path/to/run")
audit = vr.audit_run("/path/to/run")
```

Do not load checkpoint tensors to decide whether a run succeeded. The
authoritative commit chain, status, and audit establish completion and
resumability.

Q100 quality aggregation first performs these two public checks on exactly the
twelve paths listed in
[`q100_inputs.json`](../experiments/v0_7/evidence/q100_inputs.json), then reads
`weighted_total` from the authoritative sample manifest. Reward improvement is
reported separately from evidence completeness. It requires the same prompt set
across all three seeds, stable and cross-seed-equal samples per step, balanced
early/late prompt counts, and prompt-first then seed/prompt-equal averaging.

After all authorized fixed suites have completed on one clean commit, generate
and verify evidence without launching another training path:

```text
python -m experiments.v0_7.offline_aggregate
python -m experiments.v0_7.verify_evidence
```

The final verifier reads generated evidence only. It checks exactly six
semantic-family results and 30 role results, validates Flow native and MG1
internal reports, and independently compares the evidence candidate with the
current clean Git HEAD.

## Failure and support boundaries

- A validation error means no optimizer update should begin.
- A reward or update failure must not publish the failed step.
- Single-process and DDP runs share the same Runner/data contracts.
- CUDA/NCCL availability is an environment fact, not a skipped pass.
- C20 proves bounded mechanical behavior; it does not prove reward improvement.
- Q100 reward claims require three complete audited seeds and the preregistered
  statistics in the acceptance document.
- No real C20, Q100, Flow native, MG1/NCCL, remote run, or upload was performed
  during W06 source preparation.

See [V0_7_ACCEPTANCE.md](V0_7_ACCEPTANCE.md) for the current evidence matrix.
