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

Model training additionally needs the `train` extra and the local model
checkpoints selected by the complete YAML. All four recipe implementations are
bundled in framecode; no runtime reference source repository is required.
Installing the package does not download checkpoint data.

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

## Read-only callbacks

Callbacks are constructed in Python and passed only to `run()`. They receive
immutable, tensor-free events and cannot stop training or modify the model,
optimizer, reward, loss, or checkpoint state.

```python
import visual_rl as vr


class RewardPrinter(vr.Callback):
    def on_step_end(self, event: vr.CallbackEvent) -> None:
        print(event.step, event.metrics["reward_mean"])


result = vr.load("/absolute/path/to/complete-config.yaml").run(
    callbacks=[RewardPrinter()]
)
```

The minimal lifecycle is `on_run_start`, `on_step_end`, `on_commit`, and
`on_run_end`. Use `on_commit` when an observer needs paths guaranteed to have
passed the authoritative marker and artifact-maintenance boundary. Callback
instances and state are not stored in YAML, manifests, or checkpoints, and
resume does not replay historical events.

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
compatible CUDA GPUs, and the declared local World-R1 reward endpoints.
The reward service implementation and static resources are bundled in the
same wheel under `services.world_r1_strict`; no framecode or World-R1 checkout
is required at runtime. Its HPS, Qwen and DA3 weights remain separately
supplied local model data.

## Precision and frozen-module CPU offload

The bounded operational C20 configurations use BF16 and enable:

```yaml
model:
  params:
    gradient_checkpointing: true
    offload_frozen_modules_during_update: true
```

Wan C20 configurations also enable `vae_tiling`. The offload lifecycle is part
of the existing SD3/Wan adapters:

1. restore the frozen text encoders needed for prompt encoding;
2. keep the trainable transformer/LoRA on the training GPU for rollout;
3. restore the frozen VAE only for media decode;
4. move the text encoders and VAE to CPU before policy recompute/backward;
5. restore them lazily for the next step.

Failure paths perform the same cleanup, repeated restore/offload calls are
idempotent, and checkpoints do not encode the transient device placement.
Callbacks do not manage this lifecycle.

The top-level `configs/flow_grpo_sd3.yaml` intentionally remains FP32 because it
is the frozen input to the separate 14-item native-parity oracle. On a 32 GB
GPU, use the BF16 operational role configuration
`experiments/v0_7/configs/flow_grpo_sd3_c20_continuous.yaml`; do not rewrite an
FP32 parity result as BF16 evidence.

## Read-only run inspection

Use only public artifact readers:

```python
status = vr.inspect_run("/path/to/run")
audit = vr.audit_run("/path/to/run")
```

Do not load checkpoint tensors to decide whether a run succeeded. The
authoritative commit chain, status, and audit establish completion and
resumability.

Each committed metrics row includes the objective statistics `approx_kl`,
`clipfrac`, and `reference_kl`. Their meanings are deliberately different:

- `approx_kl` and `clipfrac` compare the recomputed policy with the policy that
  collected the current rollout. In a strictly on-policy, one-rollout/one-
  update run they are expected to be zero before the optimizer step; they do
  not measure drift accumulated across training steps.
- `reference_kl` compares the current policy with the frozen reference policy
  and is the long-horizon drift signal used by the configured KL penalty.
- `update/gradient_norm_pre_clip` and
  `update/gradient_norm_post_clip` report the actual unscaled gradient norm
  immediately before the optimizer step. When `max_grad_norm` is disabled the
  two values are equal and are read without mutating gradients. When clipping
  is enabled, the first value is the norm returned by `clip_grad_norm_` and the
  second is measured after clipping.

The gradient metrics are diagnostics only. They use the same single/DDP
metric-reduction path as the other update diagnostics and do not create a
second optimizer or update lifecycle.

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
- Flow, TempFlow, Flash and World-R1 operational C20 evidence now exists on a
  dirty engineering wheel. This does not alter the pending clean-candidate,
  Q100, Flow native or MG1 gates.

See [V0_7_OPERATIONAL_EVIDENCE.md](V0_7_OPERATIONAL_EVIDENCE.md) for bounded
real-model results and [V0_7_ACCEPTANCE.md](V0_7_ACCEPTANCE.md) for the formal
gate matrix.
