# VisualRL v0.7 evidence directory

This directory tracks only curated, machine-readable evidence from an authorized
run of the fixed v0.7 suites. Source preparation does not create placeholder
results.

Currently tracked:

- `q100_inputs.json`: the fixed twelve-run read-only aggregation index.
- this README.

Current formal-envelope status: Q100, Flow native, MG1/NCCL and the
clean-candidate 30-role result are `not_run`. Separate dirty-candidate
operational C20 runs are recorded in `docs/V0_7_OPERATIONAL_EVIDENCE.md`; they
must not be copied into this directory or represented as final release
evidence. Generated final filenames are documented in
`docs/V0_7_ACCEPTANCE.md` and ignored until an authorized, clean-commit run
produces them.
