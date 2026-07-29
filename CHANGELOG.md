# Changelog

## 0.7.0 — unreleased

- Replaced legacy command-line/config construction paths with one high-level
  Python API: complete YAML `load -> resolve -> validate -> run`, followed by
  public `inspect_run` and `audit_run`.
- Added minimal read-only Callback observers for run start, completed policy
  steps, authoritative commits, and successful run end without exposing
  mutable training objects or adding a second execution path.
- Unified GRPO, Flash-GRPO, and TempFlow-GRPO policy updates behind one
  clipped-surrogate objective and one Runner/update path.
- Added fixed source preparation for 30 Tiny/SD3/Wan/MG1 roles, with
  pre-launch family checks, exact all-rank readiness, interruption/resume
  controls, fail-closed C20/native/NCCL gates, and candidate-bound evidence.
- Added read-only Q100 reward aggregation and preregistered multi-seed quality
  verification with fixed canonical generation and a 3,600-row synthetic
  contract fixture.
- Added a standard-library wheel content/metadata/RECORD checker with exact
  core/extra dependency and forbidden-payload enforcement.

The full local automation, Tiny public-API smoke, and base candidate-wheel
build/install gate pass. Real C20, Q100, Flow native, MG1/NCCL, remote
execution, and upload remain `not_run`; see
[`docs/V0_7_ACCEPTANCE.md`](docs/V0_7_ACCEPTANCE.md).
