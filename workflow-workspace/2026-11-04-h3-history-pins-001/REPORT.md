# Rewritten-history pin repair — completed locally

## Intent and delivery

Restore Git lookups after the public 2026-11-04 history rewrite without migrating
recorded artifact identities. Base: `b6648f102d5bb6f875f61c9a17c13a31da346462`.
Code/tests commit: `78d0d1c7cf90856d900c48e39810a09063c0cf59`.
Branch: `fix/h3-rewritten-history-pins-20261104`. No push or deployment.

## Pin audit

Tracked code, tests, configs, JSON fixtures and archived workflow definitions
were scanned against the supplied full commit map, including abbreviated pins.

| Old pin | Classification | Disposition |
|---|---|---|
| `00cee786` | Both (a/b) | SMALL40 Git ancestry/diff/object lookups use `2f995fda`; owner, audit and upgrade CLI keep old identity |
| `e835fe39` | Both (a/b) | H3 Git proofs and legacy checkout verification use `a8a6cd7a`; recorded base and CLI keep old identity |
| `5ce1ef7d` | (b) recorded | Characterization fixture provenance unchanged |
| `b758540` | (b) recorded | Experiment candidate and archive name unchanged |

Full pinned old/new table: [history rewrite note](../../docs/history-rewrite-20261104.md).
Archived workflow SHAs/command examples remain historical, not startup instructions.
External repository pins and non-Git content hashes are unaffected.

## Changes and boundaries

- Shared exact map in `scripts/rewritten_history.py`; four affected launch/upgrade
  scripts distinguish recorded and Git identities. Direct script invocation works.
- All ancestry, allowlist, owner and content-hash checks remain enforced. Current
  HEAD still fails the historical launcher-only allowlist, as intended.
- Seven GitHub snapshot URLs repaired across CHANGELOG, docs and experiment READMEs;
  all seven target Git objects verified locally.
- `visual_rl/`, the original seven tests, existing metadata and historical workflow
  records are unchanged. No SSH, network, GPU, service operations or git-config edits.

## Validation

- Requested original `tests/test_flash_collection_runtime.py` command, with the
  authorized author-data environment and shared local venv: **6 passed, 1 failed**
  (50.50s). Sole failure: macOS lacks Linux `/proc` process `cpu_rss_bytes` at line502.
- New `tests/test_rewritten_history.py`: **22 passed** (0.67s). Covers exact mapping,
  real rewritten ancestry, rejecting old revisions at the Git subprocess boundary,
  preserved owner/CLI/audit identities, direct entrypoints and fail-closed checks.
- Additional `test_h3_mvp_cpu_{entry,checkpoint_state,bound_checkpoint,checkpoint_publish}.py`:
  **9 passed, 13 failed** (15.46s). Same 13 failures reproduced on a clean local
  `git archive b6648f1` snapshot (11.43s): stale distributed mocks, CUDA-unavailable
  checkpoint publication and related assertions. Not regressions from these scripts;
  no out-of-scope fixture/runtime repairs.
- `git diff --check` passed; original tests and `visual_rl/` have no diff.

These are local engineering checks, not Linux seven-test acceptance, legacy
checkpoint admission or GPU performance evidence.
