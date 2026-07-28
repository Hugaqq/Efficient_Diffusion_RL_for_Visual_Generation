# VisualRL v0.7 acceptance

Updated: 2026-07-29

This document separates source readiness from real execution evidence. `not_run`
is not a pass, a skip interpreted as success, or a quality claim.

## Current matrix

| Gate | Required evidence | Current status |
|---|---|---|
| W06 source/config/controller | 30 resolving full YAML files, sole API callsite, exact readiness schemas, fail-closed gates/evidence writers, process cleanup tests | `verified locally` |
| Q100 offline tooling | 3,600-row synthetic fixture, fixed generator, boundary/non-finite/missing-step/sample-count/prompt-balance/isolation/byte-stability tests | `verified locally` |
| Wheel checker source | standard-library synthetic archive/metadata/RECORD tests | `verified locally` |
| Real Flow-GRPO C20 correctness | continuous, interrupted, fresh resume, public audit and semantic parity | `not_run` |
| Real Flow native parity | W04 14-item CUDA one-shot numerical result | `not_run` |
| Real TempFlow-GRPO C20 correctness | continuous, interrupted, fresh resume, public audit and semantic parity | `not_run` |
| Real Flash-GRPO C20 correctness | continuous, interrupted, fresh resume, public audit and semantic parity | `not_run` |
| Real World-R1 C20 correctness | continuous, interrupted, fresh resume, public audit and semantic parity | `not_run` |
| Q100 evidence completeness | four algorithms x seeds 17/29/43, 100 committed audited steps | `not_run` |
| Q100 reward improvement | preregistered pooled/seed/Theil-Sen gates after evidence completeness | `not_run` |
| MG1 internal NCCL | three fixed two-rank NCCL failure/correctness tests | `not_run` |
| MG1 Tiny C20 | two-GPU continuous, interrupted, fresh resume | `not_run` |
| Candidate wheel build/install | unique wheel, checker pass, clean install, outside-repository import; Tiny single/resume/Gloo public API smoke | `verified locally` |
| Remote execution/upload | authorized clean-commit execution and curated evidence transfer | `not_run` |

## Frozen correctness order

For each algorithm:

```text
C20 continuous + interrupted + fresh resume
-> all exit/status/audit gates
-> continuous/resume semantic parity
-> Flow only: 14-item native parity
-> Q100 seeds 17, 29, 43
```

A failed C20/native gate leaves that algorithm's Q100 launch count at zero.
Other algorithm families remain independent.

## Frozen Q100 quality gate

- Early artifact steps: `0..35`; late artifact steps: `64..99`.
- All three seeds must use the same non-empty prompt set.
- Every seed must have a constant positive sample count at every step, and that
  count must be identical across seeds.
- Early and late record counts must be exactly balanced across prompts and
  identical across seeds.
- Average within prompt first, then weight seeds and prompts equally.
- Require `pooled_delta > max(0, 0.1 * pooled_early_std)`.
- Require at least two of three positive seed deltas and positive median delta.
- Require positive Theil-Sen slope over 100 points and exactly 4,950 pairs.
- Require all three seeds complete and public-audit clean.

Every algorithm reports `evidence_complete` and `reward_pass` separately. A
reward failure does not erase C20 mechanical correctness and cannot be rewritten
as a positive quality result.

## Final evidence identity

The final read-only gate accepts no placeholder or existence-only evidence.
`role_results.json`, `flow_native.json`, `mg1_internal.json`,
`q100_source_status.json`, and every attempt in `environment.jsonl` bind to the
same full candidate commit with `clean=true` and `tested=true`. The gate probes
the live Git HEAD and working-tree cleanliness instead of trusting those
reported fields. The Q100 envelope also records the SHA-256 of
`q100_reward_rows.jsonl`.

The gate requires exactly 30 role results, exactly six independently recorded
continuous/resume semantic-family passes, all 14 passing Flow-native items, the
three fixed MG1/NCCL nodeids, environment attempts covering all roles, twelve
complete audited Q100 sources, and a passing reward verdict for all four
algorithms. Missing/unknown fields, duplicate identities, commit drift, dirty
candidates, incomplete audit, digest drift, structural reward imbalance, or
reward failure are rejected.

## Commands run for W06 source readiness

```text
conda run -n visual-rl python -m pytest -q \
  tests/test_v0_7_experiment_tools.py \
  tests/test_v0_7_offline_aggregate.py \
  tests/test_wheel_contract.py

conda run -n visual-rl ruff check experiments/v0_7 \
  scripts/verify_wheel_contract.py \
  tests/test_v0_7_experiment_tools.py \
  tests/test_v0_7_offline_aggregate.py \
  tests/test_wheel_contract.py

conda run -n visual-rl python -m compileall -q \
  experiments/v0_7 scripts/verify_wheel_contract.py

git diff --check
```

The real evidence gate
`python experiments/v0_7/verify_evidence.py` is intentionally not executed
during source preparation because the required files do not yet exist.

## Commands run for W07 local release readiness

The final candidate is accepted locally only when this whole sequence passes on
one clean commit:

```text
compileall + Ruff + git diff --check
full pytest excluding the dedicated API smoke
dedicated Tiny single/resume/Gloo API smoke
python -m build
explicit wheel content/metadata/RECORD check
fresh base venv install + pip check
outside-repository python -I import
```

This verifies local orchestration and base-package installation only. It does
not change any real GPU/NCCL, C20/Q100, reward-improvement, remote-execution, or
upload row above from `not_run`.
