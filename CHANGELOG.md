# Changelog

## Unreleased

- Removed the unused Phase-A evidence framework, retired v0.7 rollout/reward
  DTOs, migration shims, and their structural test suites. The retained tests
  focus on configuration, model/algorithm binding, rollout, reward, update,
  checkpoint, and end-to-end runtime behavior.
- Removed the custom clean-wheel verification harness; ordinary Python package
  builds remain owned by `pyproject.toml`.

## 0.8.0 — unreleased

- Replaced the v0.7 `load().run()` API, `ExperimentRunner`, and
  `runtime_factory.py` with one schema-v2 module entry backed by the default
  `RunController` composition root.
- Added immutable recipe definitions and seven typed component registries for
  the internal model, trainer, dynamics, rollout, reward, conditioner, and
  credit graph.
- Added the Phase A public model/algorithm axes: an independent algorithm
  registry, `ModelCapabilities × AlgorithmRequirements` compatibility,
  `PolicyRuntimePort`, `AlgorithmModule/BoundAlgorithm`, and runtime/checkpoint
  identities that bind the complete coarse algorithm rather than claiming
  compatibility from an internal rollout or credit helper.
- Made the model/algorithm boundary structural: model sources cannot import or
  branch on algorithm strategies, while algorithm sources may cross only the
  symbol-level `models.interface` and `models.scheduler` ABI and cannot name a
  concrete model through either Python imports or dynamic class paths. Removed
  the unused algorithm-optional compatibility graph and made its module a
  forbidden wheel member.
- Added six official schema-v2 configurations for Flow-GRPO/SD3,
  Flow-GRPO/Wan, TempFlow-GRPO/SD3, Flash-GRPO/Wan, World-R1 core/Wan,
  and World-R1 release-surrogate/Wan. The Flow-GRPO/Wan slice reuses the
  same public algorithm module and Wan adapter instead of adding a recipe or
  pair-specific implementation.
- Added receipt-backed, content-addressed compatibility evidence. Public
  `compatible`, `smoke_update`, `resume_parity`, and `native_parity` gates are
  independent and derived from successful run receipts bound to the exact
  algorithm module, runtime scope, fixture, environment, and result set.
- Added a pass-free subprocess evidence runner and an automatically derived
  capability/evidence matrix. Failed, not-run, stale, malformed, symlinked, or
  identity-mismatched attempts cannot publish a passed record, and evidence is
  never inherited across model-algorithm bindings.
- Added artifact/environment/runtime preflight, bound resource identities,
  launch-security audit, per-rollout dynamics, and a shared six-stage GRPO
  update path.
- Added immutable early `recipe.resolved.json` and redacted
  `launch.resolved.json` manifests with exact-byte fresh/resume checks and
  fail-closed drift/symlink handling.
- Added content-addressed G3 reference-policy state evidence. Flow owns a
  derived LoRA-disable reference, TempFlow retains the same SD3 capability
  without checkpoint ownership, and Wan recipes explicitly own no reference.
- Added TempFlow nonterminal per-timestep branching with an explicit
  deterministic ODE port, B0 mainline prediction/advance followed by B0xK SDE
  expansion and continuation, and preserved timestep/exploration credit axes.
- Made GRPO-family advantage epsilon, population-standard-deviation domain,
  PPO clip, learning rate, and AdamW weight decay explicit recipe identities;
  the six official configurations now preserve their selected upstream
  numeric profiles without model-name branches.
- Added atomic safe-point checkpoints, exact single-process continuation, an
  idempotent terminal finalizer, and the read-only `visual_rl.artifacts.inspection`
  module for v0.8 terminal runs.
- Added a wheel contract that requires `visual_rl/train.py` and rejects the
  retired API, Runner, and runtime-factory modules. A checked-in release
  workflow builds the wheel, records its digest, and verifies a core-only
  isolated install; runtime compatibility exports are lazy so importing the
  package root does not require PyTorch. The installed-surface verifier treats
  a missing parent package as proof that its retired children are absent and
  keeps static recipe compilation independent from concrete runtime
  materializers.

The current automated evidence is limited to strict compilation, fake-leaf
single-process updates/resume, and terminal inspection. Real SD3.5/Wan
one-update/native parity, DDP, multi-node execution, quality, and throughput
claims remain outside this release evidence.

## 0.7.0 — historical development snapshot

- Replaced earlier command-line/config construction paths with one high-level
  Python API: complete YAML `load -> resolve -> validate -> run`, followed by
  public `inspect_run` and `audit_run`.
- Added minimal read-only Callback observers for run start, completed policy
  steps, authoritative commits, and successful run end.
- Unified GRPO, Flash-GRPO, and TempFlow-GRPO policy updates behind one
  clipped-surrogate objective and one Runner/update path.
- Added fixed source preparation and a standard-library wheel checker for the
  v0.7 experiment envelope.

The v0.7 files under `docs/V0_7_*` and `experiments/v0_7/` are retained only as
historical records; they are not v0.8 usage or support contracts.
