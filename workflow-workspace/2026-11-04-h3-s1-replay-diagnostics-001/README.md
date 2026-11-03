# S1 replay warning diagnostics — local CPU delivery

Base: `a38a5fd1d3ac078488541f1da144d382c7941572`.
Branch: `perf/h3-s1-tolerance-warn-20261104`.
Atomic commit: resolve with `git log -1 --format=%H -- tests/test_h3_rollout_replay_diagnostics.py`.

## Implementation and identity

- `visual_rl/algorithms/optimization/h3_dual_anchor.py`: fixed observation-only warning constant `1e-4`; accepted S1 deltas above it emit flushed `h3_rollout_replay_score_warning` before the row result gate. Existing check events and declared abort tolerance remain. Successful capture returns local counters/max; B=1 still returns None and has no new events.
- `visual_rl/algorithms/trainer/h3_dual_stages.py`: retains each successful S1 capture's diagnostics with its zero-based optimizer step. PPO still uses the B=1 anchor score; O1 remains rejected with S1.
- `visual_rl/runtime/h3_lifecycle.py`: carries diagnostics in the existing loss exchange, validates/aggregates within the existing metrics error gate, and emits one flushed rank-0 world summary per completed update. No added collective.
- **Addendum fallback:** terminal metrics have an exact checked schema (`visual_rl/runtime/h3_lifecycle.py:521`). Both metrics files remain unchanged; watchdogs read the aggregated `h3_rollout_replay_score_summary` in train.log. No rank-local summary events.
- Defaults and semantic/resume identity fields are unchanged. Explicit `--rollout-batch-replay-tolerance 1e-3` remains necessary and already identity-bearing. Shared launcher help (also used by formal) and `docs/h3_mvp.md` explain this. Source commit/content hashes naturally change; this is not a same-code resume claim.

## Verification

New file: `tests/test_h3_rollout_replay_diagnostics.py` (24 cases); original seven untouched.

Using `../../framecode/.venv/bin/python -m pytest -q`:
- New diagnostics + existing `test_h3_dual_anchor.py`: **25 passed**.
- Original seven, with `VISUALRL_FLASH_AUTHOR_DATA="$PWD/../flash-grpo/source/dataset/video"`: **6 passed, 1 expected macOS `/proc` failure (`cpu_rss_bytes`)**. Includes existing formal-launcher coverage. Initial run without the dataset variable had two additional missing-data failures; resolved by configuration only.
- Combined final acceptance: **31 passed, 1 expected macOS failure**.
- Extra historical suites: dual stages **1 passed/1 failed** (3-step fixture vs 5-step window); lifecycle entry **1 passed/4 failed** (obsolete distributed mocks). All five failures reproduced on an untouched archive of base `a38a5fd`; no unrelated fixes.
- `git diff --check` passed. Raw pytest output is preserved losslessly in `*.log.gz`.

## Read-only precision findings

`x`/`action` are detached CPU FP16 clones; `old_joint` is detached CPU FP32, one element (`visual_rl/algorithms/trainer/h3_dual_rollout.py:118-130`). Live states move to the execution device without casting (same file:285-289). Euler returns the input dtype; sampling casts the FP32 action to that dtype **before** scoring (`visual_rl/algorithms/dynamics/h3_dual_flow_sde.py:101,110-114`). Batched rollout evaluates these already-quantized x/action values, then copies them to CPU (`h3_dual_rollout.py:337-340`); anchor transfers them back without dtype conversion (`h3_dual_anchor.py:82-91`). Thus there is no replay-only FP16 round-trip mismatch. Shared quantization may interact with differing predictions, but this audit does not measure GPU causality. Joint score is an element-count-weighted **mean**, not a latent-dimension sum (`h3_dual_flow_sde.py:121-125`).

No push, network, SSH, GPU execution, or numerical/scientific protocol change.
