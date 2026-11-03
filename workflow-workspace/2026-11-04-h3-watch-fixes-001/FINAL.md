# H3 formal watch review fixes — 2026-11-04

## Current result — findings 1–3 from re-review of `61df2df`

All three findings are fixed locally, each with failing tests observed before implementation and a separate atomic commit. **275 watch tests pass; standalone offline drill 9/9; `git diff --check` passes.** No push, SSH/network, GPU, systemd or git-config changes. Only sanitized fixtures and fake local processes were used; no private data was copied into artifacts. No real-training or hardware-acceptance claim.

| Finding | Commit | Red → green evidence |
|---|---|---|
| 1: real torchrun TERM exits 1 | `aa23760` | 17 failures / 56 targeted cases → all pass; full suite 254 passed. Exact signal-15 exception plus same-launch watchdog TERM (`phase=term`, `last_signal=SIGTERM`, checkpoint saved) and matching receipt required. Tests exercise real watchdog bookkeeping before the log/receipt, monitor/disk stops and restart. Code 1 without that evidence, extra application/NCCL error, OOM, signal 9, stale ownership/phase and missing receipt fail closed. Direct -15/143 receipts remain supported with owned TERM evidence. |
| 2: ambiguous NCCL async-error diagnostics | `d562b87` | 2 failures / 6 targeted cases → all pass; full suite 259 passed. The isolated error-or-timeout line and combined failed-or-timed-out tail are insufficient; explicit collective-operation-timeout or watchdog timed-out evidence is required. |
| 3: mid-line application log levels | `e453c81` | 6 failures / 16 targeted cases → all pass; full suite 275 passed. Timestamp/rank/logger-prefixed ERROR/CRITICAL/FATAL records veto recovery with NCCL or TERM tails. Existing ValueError and OOM vetoes remain unchanged. |

### Current validation and probes

- `../../framecode/.venv/bin/python -m pytest -q tests/test_h3_formal_watch.py`: **275 passed in 53.71s** (68 more cases than `61df2df`).
- `../../framecode/.venv/bin/python ops/h3_formal_watch/drill.py --parent <scratch-output>`: **9/9 scenarios passed**.
- All six original rev2 scripts and the original rev3 adversarial script reran unchanged. Their TERM fixtures omit `stop.phase`, so those original TERM cases now correctly fail closed; this is not evidence that valid owned TERM is rejected. Isolated ambiguous NCCL diagnostics now return `unknown_exit`; mid-line ERROR returns `correctness_or_permanent_error`.
- Adapted copies of rev2 `probe_term.py` and rev3 `probe_adv.py` add the required phase to their stop fixtures, retain original cases, add real code-1 positive/negative cases and assert results: **4/4 and 26/26 pass**. Originals were not modified. Scripts, outputs and fake drill artifacts are under the supplied scratchpad's new `watch-fix3-TqIl30/` directory.
- Valid code-1 TERM → `watchdog_checkpoint_stop`; no stop record, extra application error or signal 9 → permanent. Application ValueError then NCCL and both OOM class names remain permanent. Existing conservative vetoes for leading `error bars` and `loss_error:` were deliberately not relaxed.
- Completion acknowledgement reaches DONE; dead-process acknowledgement performs zero resumes; ledger compaction retains 32 active rows/968 dedup records; dead-man recovery and distinct human-latch alerts remain intact.
- `git diff --check`: passed. No training source/tests changed; the original seven training tests and Linux gate were not rerun in this follow-up. Previous-round results below are historical only.

Tests below are in `tests/test_h3_formal_watch.py`. The earlier claim that all original 17 items were fixed was too strong: `100c3ea` left original item 3 partial, and `61df2df` still mishandled real torchrun exit code 1. The current corrections above supersede those classification claims. Legacy STOPPING behavior remains documented and fail-closed.

## Historical original-review mapping

| Item | Status / commit | Regression test(s) |
|---|---|---|
| 1 | Fixed `d2710e0` | `test_human_alerts_follow_token_not_polled_state` |
| 2 | Fixed `9fa5a67` | `test_new_launch_adoption_clears_stale_pause_ownership`, `test_pause_ownership_requires_launch_and_token` |
| 3 | Previously partial; further corrected by `aa23760`, `d562b87`, `e453c81` after `5eca896`, `e355f4a`, `8c17c60` | `test_failure_records_veto_transient_retry`, `test_launcher_identity_json_does_not_veto_nccl`, `test_unrecognized_error_cannot_borrow_same_line_nccl_signature` |
| 4 | Fixed `9fa5a67` | `test_cancelled_operator_pause_latches_human` |
| 5 | Fixed `9fa5a67` | `test_acknowledge_dead_process_cannot_resume` |
| 6 | Fixed `9fa5a67` | `test_unknown_launch_reconcile_adopt_or_abandon`, `test_unknown_launch_abandon_prevents_delayed_submission`, `test_unknown_launch_reconcile_rejects_foreign_unit` |
| 7 | Fixed `9fa5a67` | `test_unpublished_pause_replayed_before_escalation` |
| 8 | Fixed `d2710e0` | `test_persisted_deadman_job_gets_outbox_after_recovery` |
| 9 | Fixed `e355f4a` | `test_verification_deadline_survives_restart` |
| 10 | Fixed `196104a` | `test_observer_history_bounded_pending_and_dedup_preserved`, `test_human_observations_only_on_health_change`, `test_rotated_segments_have_retention_cap`, `test_legacy_observer_migration_cannot_replay_newer_claims` |
| 11 | Fixed `e355f4a` | `test_exact_tree_nested_metadata_names` |
| 12 | Fixed `e355f4a` | `test_proc_scan_skips_disappearing_pid`, `test_ownership_unknown_is_debounced` |
| 13 | Fixed `e355f4a` | `test_hps_unicode_producer_digest` |
| 14 | Fixed `e355f4a` | `test_verification_done_on_deadline_is_accepted` |
| 15 | Fixed `d2710e0` | `test_repeat_alert_window_key_stable_across_failed_retry` |
| 16 | Fixed `b402b93` | `test_progress_log_uses_append_preserves_concurrent_edits_and_mode`, `test_failed_atomic_report_write_leaves_no_temp` |
| 17 | Fixed `196104a` | `test_observer_unit_covers_bounded_command_budget` |

## Historical first re-review corrections

| Defect | Commit / resolution | Red → green evidence |
|---|---|---|
| 1: real torch timeout/TERM veto (incomplete at this stage) | `5eca896`; superseded by current findings 1–3 above | 9 failing cases before fix; 22 new cases pass, including reviewed torch 2.x lines, sanitized NCCL/torchrun tails, receipt gates and sampled/replayed-score ValueError veto |
| 2: acknowledged completion re-latches | `78b52a2`; SUCCESS validation precedes dead-process veto | 4 failing cases before fix; 10 cases pass, covering restart, delayed receipt and mismatched evidence; no launch/signal |
| 3: failed-before-execute abandonment | `45b017a`; stable inactive/failed empty unit without ownership receipts can be abandoned | 12 failing cases before fix; 26 cases pass, including negative ownership/read/race cases and delayed-submission prevention |
| 4: legacy STOPPING | `08862de`; README note, fail-closed behavior unchanged | 2 guard cases pass for missing launch ID/token; existing request preserved |

Fixtures under `tests/fixtures/h3_formal_watch/` use sanitized log text from the re-review probes, not a newly collected GPU run. Unit reconciliation tests mock manager/cgroup observations; they are not systemd acceptance.

## Historical validation at `61df2df`

- Watch tests: **207 passed in 11.52s** (previous round: 147; 60 additional cases).
- Standalone offline drill: **9/9 scenarios passed**; local fake sleep processes only.
- All six original rev2 scripts ran unchanged: `probe_cls`, `probe_term`, `probe_done`, `probe_ledger`, `probe`, `probeA`. Torch transients now classify correctly; application veto remains permanent; acknowledged completion reaches DONE. Ledger retains 32 rows/968 dedup records; dead-man recovery and distinct human-latch alerts remain intact. Dead-process acknowledgement still performs zero resumes.
- `git diff --check`: passed.
- Interpreter: `../../framecode/.venv/bin/python`.
- Additional original seven-test Mac gate: **6 passed, 1 failed**, the known Linux `/proc` CPU-RSS dependency. Initial missing author-data configuration produced 4/7; supplying the existing dataset restored expected 6/7 before each implementation commit. No Linux gate was run; no training source/tests changed.
- Previous-round producer canonicalization check remains historical evidence, not rerun this round.

## Safety choices and remaining risks

Cancelled operator pauses latch NEEDS_HUMAN: explicit rearming is safer than guessing intent. Reconciliation never submits/signals training; adoption requires durable matching receipts, abandonment requires proven absence/death and blocks delayed submission. Legacy unknown units without proof still need manual investigation unless the strict failed-before-execute absence/death gate succeeds. Legacy STOPPING without launch ID/token cannot establish pause ownership and remains fail-closed.

SQLite keeps pending deliveries and permanent dedup keys outside bounded JSON; these necessarily consume disk proportional to backlog/unique work. Diagnostic retention is eight segments plus active; export longer history separately. Human observations record meaningful changes only. Observer helper budget is capped at 3825 seconds under a 90-minute unit timeout.

Linux/systemd, real Feishu delivery and GPU acceptance remain untested. Log classification is intentionally conservative. Progress append preserves concurrent appends/mode/inode, not uncoordinated whole-file replacements; interrupted short writes may leave a partial line. No real-run performance or training claims.
