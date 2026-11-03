# C1s — opt-in frozen-base stat sentinel

## Verified conclusion

Implemented one opt-in boundary-verification change on `perf/h3-training-speed-20261103-002-c1s`, starting at `7d944c3`. The original seven-function Mac gate has the expected result: **1 failed, 6 passed in 19.12s**. The sole failure is the original Linux `/proc` dependency, `tests/test_flash_collection_runtime.py:292`, `KeyError: 'cpu_rss_bytes'`. This is CPU engineering evidence, not GPU acceptance or measured speedup.

## Premise audit (line references at input commit 7d944c3)

- `visual_rl/runtime/h3_binding.py:396-417`: `validate_live()` always invokes the full model/code verification.
- `visual_rl/runtime/h3_binding.py:66-85`: `verify_h3_content()` hashes the complete model with `all-files.v1` and the code tree with `python-code.v1`.
- `visual_rl/composition/preflight/artifacts.py:190-224,332-391`: tree enumeration and streamed per-file SHA256; `:186-187,386-390` expose the dev/inode/size/mtime_ns/ctime_ns/mode observations from that same read.
- `visual_rl/runtime/h3_binding.py:141-143`: initial observer retained only `vae/` records.
- `visual_rl/runtime/h3_file_phase.py:13-42`: every rank runs its own local file worker; the collective shares completion status, not a rank-zero content result.
- `visual_rl/runtime/h3_binding.py:471,504`: both `stages.checkpoint_boundary()` calls are confirmed and preserved. They bracket live validation with read-only tail/source checks (`visual_rl/algorithms/trainer/h3_stages.py:384-402`), not two full content hashes. I1's single live-validation result is still passed directly to publication.
- `workflow-workspace/2026-11-03-h3-training-speed-001/inputs/baseline-MEMORY-REPORT.md:70`: checkpoint-validation timings are 1009.135 / 1007.942 seconds. These are historical whole-phase timings, not a measurement of this change or proof all those seconds can be removed.

## Implementation (current source references)

- `scripts/launch_h3_resolution_memory.py:56-57,159-161`: strict `--boundary-content-check {full,sentinel}`, default `full`, threaded through `H3LifecycleBackend` into the binding. Invalid values fail rather than falling back.
- `visual_rl/runtime/h3_binding.py:145-160`: initial full hashing is mandatory in both modes. Sentinel retains observations for every model file from the same byte-reading pass. The optional directory observer in `visual_rl/composition/preflight/artifacts.py:293-297` captures directory metadata in that traversal, including empty directories; the content codec/identity is unchanged.
- `visual_rl/runtime/h3_content_sentinel.py:34-92`: immutable initial snapshot, canonical snapshot digest, and an all-node `lstat` traversal. Missing/extra files or directories, inode/dev/size/time/mode changes, symlinks, unsupported node types, access failures, and observed traversal races fail closed. There is no rehash-and-accept or snapshot refresh.
- `visual_rl/runtime/h3_binding.py:416-453,503-560`: non-terminal sentinel boundaries use that traversal plus the unchanged full code hash and resource/source/model checks. `validate_live()` without a boundary argument remains the full-hash restore API. The configured terminal step always fully hashes; an early safe pause also requests full verification. Provenance is produced only after successful tail revalidation, and cleared before each attempt.
- `visual_rl/runtime/h3_content_sentinel.py:27-31`: WORLD8 gate labels contain the configured mode at binding and both configured/effective mode and step at boundaries. Every rank performs its own local sentinel check.
- `visual_rl/runtime/h3_lifecycle.py:380-393`: each opted-in rank atomically writes `h3-content-check-step-N-rank-R.json` in the run output before the checkpoint transaction. It records actual mode, configured mode, file count, snapshot SHA256, initial model content SHA256, rank, step, contract ID, and corresponding checkpoint path. A sidecar proves boundary validation, **not** successful checkpoint publication; the authoritative complete checkpoint must still exist. Receipt failure prevents publication.

The receipt stays outside checkpoint experiment identity and outside DCP. The base content SHA, live bindings, contract, RNG/data envelope, optimizer math, restore validation, DCP writer, decode/VAE reuse implementation, HPS, and I1 tail/deduplication logic are unchanged. Full mode retains its original content-hash results and emits no new provenance sidecars (apart from the required mode-consistency gate).

## Validation

Exact gate command:

```sh
PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin VISUALRL_FLASH_AUTHOR_DATA=/Users/user/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/.scratch/flash-grpo/source/dataset/video PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /Users/user/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest -q tests/test_flash_collection_runtime.py
```

Final summary: `1 failed, 6 passed in 19.12s`; only the expected first-test `/proc` KeyError remains. An initial CPU fixture transport mismatch from the new gate was repaired in this same unit's test leaves, without altering production transport or any existing assertion.

Coverage added inside the existing last test (`:1665` onward and `:2563` onward): initial one-pass file/byte counters; default full behavior; zero model-content hasher calls at a successful sentinel boundary; continued code and cache checks; unchanged contract/live/state identities; terminal, explicit early-terminal, and default/restore full hashing; actual binding rejection without fallback even after restoring file bytes; filesystem size/mtime/ctime/inode/mode/missing/extra/node-type/symlink/directory failures; no file opens by the stat checker; incomplete snapshot rejection; real rank-receipt decoder mismatch rejection; receipt persistence before publication and missing-receipt failure; strict CLI/backend values, lifecycle threading, and help output.

- `git diff --check`: empty.
- Import checks passed from this worktree with the specified venv for all six changed/new Python modules (including launcher and test module).
- AST check: same seven test function names, and every original assertion retained.
- No GPU, SSH, network, push, dependency, other-checkout, or tool-metadata changes. Linux 7/7 and real GPU/end-to-end validation remain controller integration work.

## Limitations and progress handoff

A same-size, same-inode content rewrite with **both mtime and ctime restored** is not detected by the sentinel; ctime restoration needs root/clock manipulation. This is an explicitly weaker opt-in integrity check, not an adversarial tamper-proof guarantee or filesystem lock. Metadata-only changes intentionally fail, even if bytes are unchanged. There is still a check-to-use race, as with a content scan without a filesystem lock.

The zero-byte assertion is scoped to the replaced model-content hashing pass. Existing model config/weight-header reads in `preflight_h3_profile` (`visual_rl/composition/config/h3_profile.py:294-380`) and text-cache shard checks remain active; zero total model-file I/O at a checkpoint is **not** claimed.

Completed locally: audit, scoped implementation, original-seven Mac gate, import/assertion-preservation checks, and diff check. The implementation commit is the commit containing this report; resolve it with `git log -1 --format=%H -- workflow-workspace/2026-11-03-h3-training-speed-002/c1s/REPORT.md`. No deployment or measured speedup is claimed. Controller should carry this scoped progress record into the shared progress log during integration; this unit does not edit outside its worktree.
