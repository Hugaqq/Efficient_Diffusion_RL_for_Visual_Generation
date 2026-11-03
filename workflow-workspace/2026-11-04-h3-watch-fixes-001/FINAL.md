# H3 formal watch review fixes — 2026-11-04

All 17 items fixed; none rejected. Regression tests reproduced defects before fixes. Six atomic implementation commits; no push, network, SSH, GPU or systemd operations. README synchronized.

Tests below are in `tests/test_h3_formal_watch.py`.

| Item | Status / commit | Regression test(s) |
|---|---|---|
| 1 | Fixed `d2710e0` | `test_human_alerts_follow_token_not_polled_state` |
| 2 | Fixed `9fa5a67` | `test_new_launch_adoption_clears_stale_pause_ownership`, `test_pause_ownership_requires_launch_and_token` |
| 3 | Fixed `e355f4a`, `8c17c60` | `test_failure_records_veto_transient_retry`, `test_launcher_identity_json_does_not_veto_nccl`, `test_unrecognized_error_cannot_borrow_same_line_nccl_signature` |
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

## Validation

- Watch tests: **147 passed** (baseline 114; 33 additional cases).
- Standalone offline drill: **9/9 scenarios passed**; local fake sleep processes only.
- `git diff --check`: passed.
- Interpreter: `../../framecode/.venv/bin/python`.
- Producer canonicalization checked against `../../framecode/visual_rl/core/protocols/flash_hpsv3.py`.

## Safety choices and remaining risks

Cancelled operator pauses latch NEEDS_HUMAN: explicit rearming is safer than guessing intent. Reconciliation never submits/signals training; adoption requires durable matching receipts, abandonment requires proven absence/death and blocks delayed submission. Legacy unknown units without proof still need manual investigation.

SQLite keeps pending deliveries and permanent dedup keys outside bounded JSON; these necessarily consume disk proportional to backlog/unique work. Diagnostic retention is eight segments plus active; export longer history separately. Human observations record meaningful changes only. Observer helper budget is capped at 3825 seconds under a 90-minute unit timeout.

Linux/systemd, real Feishu delivery and GPU acceptance remain untested. Log classification is intentionally conservative. Progress append preserves concurrent appends/mode/inode, not uncoordinated whole-file replacements; interrupted short writes may leave a partial line. No real-run performance or training claims.
