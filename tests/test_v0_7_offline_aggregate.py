"""Synthetic contracts for Q100 aggregation and reward verification."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from experiments.v0_7 import offline_aggregate
from experiments.v0_7.offline_aggregate import (
    ALGORITHMS,
    canonical_rows_bytes,
    load_rows,
    load_source_status,
)
from experiments.v0_7.interrupt_resume import FAMILY_ORDER, ROLE_SPECS
from experiments.v0_7.mg1_nccl import INTERNAL_NCCL_COMMANDS
from experiments.v0_7.verify_evidence import verify_evidence
from experiments.v0_7.verify_reward_improvement import (
    theil_sen_slope,
    verify_reward_improvement,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures/v0_7"


def _loaded():
    rows = load_rows(FIXTURES / "q100_reward_rows.jsonl")
    statuses = load_source_status(FIXTURES / "q100_source_status.json")
    return rows, statuses


def _candidate(commit: str = "a" * 40) -> dict[str, object]:
    return {"clean": True, "commit": commit, "tested": True}


def _native_report() -> dict[str, object]:
    names = {
        "checkpoint_resume",
        "current_log_prob",
        "gradient",
        "group_advantage",
        "initial_latent",
        "old_log_prob",
        "parameter_delta",
        "policy_loss",
        "prompt_encoding",
        "reference_kl",
        "rollout_latent",
        "timestep",
        "total_loss",
        "transition_statistics",
    }
    resume = {
        "adapter_tensors",
        "global_step",
        "grad_scaler_state",
        "next_step_inputs",
        "non_timing_metrics",
        "optimizer_state",
        "rng_state",
    }
    comparison = {
        "atol": 0.0,
        "dtype": "torch.float32",
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "passed": True,
        "rtol": 0.0,
        "shape": [1],
        "tensor_name": "tensor",
    }
    items = {
        name: {
            "comparisons": (
                {key: True for key in sorted(resume)}
                if name == "checkpoint_resume"
                else [comparison]
            ),
            "passed": True,
        }
        for name in names
    }
    return {
        "case": "flow_grpo_sd3_case_v1",
        "config_path": "configs/flow_grpo_sd3.yaml",
        "items": items,
        "overall_pass": True,
        "precision": "fp32",
        "schema_version": 1,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _valid_evidence(directory: Path) -> None:
    directory.mkdir()
    rows_path = directory / "q100_reward_rows.jsonl"
    shutil.copyfile(FIXTURES / "q100_reward_rows.jsonl", rows_path)
    source_status = json.loads(
        (FIXTURES / "q100_source_status.json").read_text(encoding="utf-8")
    )
    source_status["candidate"] = _candidate()
    source_status["rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    _write_json(directory / "q100_source_status.json", source_status)
    _write_json(
        directory / "role_results.json",
        {
            "candidate": _candidate(),
            "families": {
                family: {"semantic_parity": True}
                for family in FAMILY_ORDER
            },
            "roles": [
                {
                    "audit_ok": True,
                    "committed_steps": spec.target_steps,
                    "exit_code": spec.expected_exit,
                    "output_dir": f"runs/{spec.name}",
                    "role": spec.name,
                }
                for spec in ROLE_SPECS
            ],
            "schema_version": 1,
        },
    )
    environment_lines = [
        json.dumps(
            {
                "attempt_id": f"attempt-{spec.name}",
                "clean": True,
                "commit": "a" * 40,
                "cuda": None,
                "devices": [],
                "packages": {"visual-rl": "0.7.0"},
                "platform": "test-platform",
                "python": "3.10.0",
                "role": spec.name,
                "tested": True,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for spec in ROLE_SPECS
    ]
    (directory / "environment.jsonl").write_text(
        "\n".join(environment_lines) + "\n",
        encoding="utf-8",
    )
    _write_json(
        directory / "flow_native.json",
        {
            "candidate": _candidate(),
            "report": _native_report(),
            "schema_version": 1,
        },
    )
    _write_json(
        directory / "mg1_internal.json",
        {
            "candidate": _candidate(),
            "schema_version": 1,
            "tests": [
                {"nodeid": command[-1], "passed": True}
                for command in INTERNAL_NCCL_COMMANDS
            ],
        },
    )


def test_fixed_3600_row_fixture_passes_all_four_algorithms() -> None:
    rows, statuses = _loaded()
    expected = json.loads(
        (FIXTURES / "q100_reward_expected.json").read_text(encoding="utf-8")
    )["expectations"]
    report = verify_reward_improvement(rows, statuses)
    assert len(rows) == expected["row_count"] == 3600
    assert len(report) == expected["algorithm_count"] == 4
    assert set(report) == set(ALGORITHMS)
    assert all(item["evidence_complete"] for item in report.values())
    assert all(item["reward_pass"] for item in report.values())
    for item in report.values():
        assert item["positive_seed_count"] == expected["positive_seed_count"]
        assert item["pooled_delta"] == pytest.approx(expected["pooled_delta"])
        assert item["theil_sen_slope"] == pytest.approx(
            expected["theil_sen_slope"]
        )
        assert item["theil_sen_pair_count"] == expected["theil_sen_pair_count"]


def test_windows_are_prompt_balanced_and_use_exact_boundaries() -> None:
    rows, _ = _loaded()
    subset = [
        row
        for row in rows
        if row.algorithm == "flow_grpo_sd3" and row.seed == 17
    ]
    for steps in (set(range(0, 36)), set(range(64, 100))):
        counts = {
            prompt: sum(row.step in steps and row.prompt_id == prompt for row in subset)
            for prompt in {"prompt-0", "prompt-1", "prompt-2"}
        }
        assert set(counts.values()) == {36}


def test_canonical_jsonl_is_byte_stable_under_input_reordering() -> None:
    rows, _ = _loaded()
    original = (FIXTURES / "q100_reward_rows.jsonl").read_bytes()
    assert canonical_rows_bytes(rows) == original
    assert canonical_rows_bytes(reversed(rows)) == original


def test_fixed_offline_generator_writes_rows_and_candidate_bound_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows, statuses = _loaded()
    index_path = tmp_path / "q100_inputs.json"
    index_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        offline_aggregate,
        "scan_q100_inputs",
        lambda received: (
            rows,
            statuses,
        )
        if received == index_path
        else pytest.fail("generator scanned an unexpected index"),
    )
    rows_path, status_path = offline_aggregate.generate_q100_evidence(
        index_path=index_path,
        evidence_dir=tmp_path,
        git_probe=lambda _root: {"clean": True, "commit": "a" * 40},
    )
    assert rows_path.read_bytes() == canonical_rows_bytes(rows)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["candidate"] == _candidate()
    assert status["rows_sha256"] == hashlib.sha256(
        rows_path.read_bytes()
    ).hexdigest()
    assert status["sources"] == [asdict(item) for item in statuses]


def test_fixed_offline_generator_removes_stale_outputs_before_scan_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "q100_reward_rows.jsonl"
    status_path = tmp_path / "q100_source_status.json"
    rows_path.write_text("stale\n", encoding="utf-8")
    status_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        offline_aggregate,
        "scan_q100_inputs",
        lambda _path: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )
    with pytest.raises(RuntimeError, match="scan failed"):
        offline_aggregate.generate_q100_evidence(
            index_path=tmp_path / "q100_inputs.json",
            evidence_dir=tmp_path,
            git_probe=lambda _root: {"clean": True, "commit": "a" * 40},
        )
    assert not rows_path.exists()
    assert not status_path.exists()


def test_nonfinite_duplicate_and_blank_rows_fail_closed(tmp_path: Path) -> None:
    valid = {
        "algorithm": "flow_grpo_sd3",
        "prompt_id": "p",
        "sample_id": "s",
        "seed": 17,
        "source_run": "run",
        "step": 0,
        "weighted_total": 1.0,
    }
    for text in (
        json.dumps({**valid, "weighted_total": float("nan")}) + "\n",
        '{"algorithm":"flow_grpo_sd3","algorithm":"tempflow_sd3"}\n',
        json.dumps(valid) + "\n\n",
    ):
        path = tmp_path / "rows.jsonl"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_rows(path)


def test_missing_step_only_blocks_its_algorithm() -> None:
    rows, statuses = _loaded()
    reduced = tuple(
        row
        for row in rows
        if not (
            row.algorithm == "flow_grpo_sd3"
            and row.seed == 17
            and row.step == 99
        )
    )
    report = verify_reward_improvement(reduced, statuses)
    assert report["flow_grpo_sd3"]["evidence_complete"] is False
    for algorithm in set(ALGORITHMS) - {"flow_grpo_sd3"}:
        assert report[algorithm]["evidence_complete"] is True
        assert report[algorithm]["reward_pass"] is True


def test_incomplete_audit_blocks_only_its_algorithm() -> None:
    rows, statuses = _loaded()
    changed = tuple(
        replace(status, audit_ok=False)
        if status.algorithm == "flash_wan" and status.seed == 29
        else status
        for status in statuses
    )
    report = verify_reward_improvement(rows, changed)
    assert report["flash_wan"]["evidence_complete"] is False
    assert report["flow_grpo_sd3"]["reward_pass"] is True


def test_reward_gate_rejects_fewer_samples_in_one_step() -> None:
    rows, statuses = _loaded()
    reduced = tuple(
        row
        for row in rows
        if row.sample_id != "flow_grpo_sd3-17-000-0"
    )
    report = verify_reward_improvement(reduced, statuses)["flow_grpo_sd3"]
    assert report["evidence_complete"] is False
    assert "constant positive sample count" in report["reason"]


def test_reward_gate_rejects_stable_but_different_sample_counts_across_seeds() -> None:
    rows, statuses = _loaded()
    reduced = tuple(
        row
        for row in rows
        if not (
            row.algorithm == "flow_grpo_sd3"
            and row.seed == 17
            and row.sample_id.endswith("-0")
        )
    )
    report = verify_reward_improvement(reduced, statuses)["flow_grpo_sd3"]
    assert report["evidence_complete"] is False
    assert "identical across seeds" in report["reason"]


def test_reward_gate_rejects_different_prompt_sets_across_seeds() -> None:
    rows, statuses = _loaded()
    changed = tuple(
        replace(row, prompt_id=f"seed43-{row.prompt_id}")
        if row.algorithm == "flow_grpo_sd3" and row.seed == 43
        else row
        for row in rows
    )
    report = verify_reward_improvement(changed, statuses)["flow_grpo_sd3"]
    assert report["evidence_complete"] is False
    assert "same non-empty prompt set" in report["reason"]


def test_reward_gate_rejects_early_prompt_imbalance() -> None:
    rows, statuses = _loaded()
    changed = tuple(
        replace(row, prompt_id="prompt-1")
        if (
            row.algorithm == "flow_grpo_sd3"
            and row.seed == 17
            and row.step == 0
        )
        else row
        for row in rows
    )
    report = verify_reward_improvement(changed, statuses)["flow_grpo_sd3"]
    assert report["evidence_complete"] is False
    assert "balanced across prompts" in report["reason"]


def test_theil_sen_requires_exactly_100_finite_points() -> None:
    assert theil_sen_slope(tuple(float(step) for step in range(100))) == 1.0
    with pytest.raises(ValueError, match="exactly 100"):
        theil_sen_slope(tuple(float(step) for step in range(99)))
    with pytest.raises(ValueError, match="finite"):
        theil_sen_slope((float("inf"),) + tuple(float(step) for step in range(99)))


def test_fixture_rows_have_exact_schema() -> None:
    rows, _ = _loaded()
    assert set(asdict(rows[0])) == {
        "algorithm",
        "prompt_id",
        "sample_id",
        "seed",
        "source_run",
        "step",
        "weighted_total",
    }


def test_final_evidence_gate_validates_all_evidence_classes(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    _valid_evidence(evidence)
    report = verify_evidence(
        evidence,
        git_probe=lambda _root: {"clean": True, "commit": "a" * 40},
    )
    assert report["overall_pass"] is True
    assert report["candidate"] == _candidate()
    assert report["role_count"] == 30
    assert report["semantic_family_count"] == 6
    assert report["flow_native_pass"] is True
    assert report["mg1_fixed_test_count"] == 3
    assert all(item["reward_pass"] for item in report["reward"].values())


@pytest.mark.parametrize(
    "case",
    (
        "missing_role",
        "missing_family",
        "failed_family",
        "dirty_candidate",
        "candidate_mismatch",
        "native_failed",
        "missing_mg1",
        "q100_digest_mismatch",
        "unknown_field",
    ),
)
def test_final_evidence_gate_rejects_incomplete_or_drifted_schema(
    tmp_path: Path,
    case: str,
) -> None:
    evidence = tmp_path / "evidence"
    _valid_evidence(evidence)
    if case in {
        "missing_role",
        "missing_family",
        "failed_family",
        "dirty_candidate",
        "unknown_field",
    }:
        path = evidence / "role_results.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if case == "missing_role":
            value["roles"].pop()
        elif case == "missing_family":
            value["families"].pop(FAMILY_ORDER[-1])
        elif case == "failed_family":
            value["families"][FAMILY_ORDER[-1]]["semantic_parity"] = False
        elif case == "dirty_candidate":
            value["candidate"]["clean"] = False
        else:
            value["unexpected"] = True
        _write_json(path, value)
    elif case in {"candidate_mismatch", "native_failed"}:
        path = evidence / "flow_native.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if case == "candidate_mismatch":
            value["candidate"]["commit"] = "b" * 40
        else:
            value["report"]["overall_pass"] = False
        _write_json(path, value)
    elif case == "missing_mg1":
        path = evidence / "mg1_internal.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["tests"].pop()
        _write_json(path, value)
    else:
        path = evidence / "q100_source_status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["rows_sha256"] = "0" * 64
        _write_json(path, value)
    with pytest.raises((FileNotFoundError, RuntimeError, TypeError, ValueError)):
        verify_evidence(
            evidence,
            git_probe=lambda _root: {"clean": True, "commit": "a" * 40},
        )


def test_final_evidence_gate_rejects_reward_failure(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    _valid_evidence(evidence)
    rows_path = evidence / "q100_reward_rows.jsonl"
    declining = tuple(
        replace(row, weighted_total=-row.weighted_total)
        for row in load_rows(rows_path)
    )
    rows_path.write_bytes(canonical_rows_bytes(declining))
    status_path = evidence / "q100_source_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    _write_json(status_path, status)
    with pytest.raises(RuntimeError, match="reward-improvement"):
        verify_evidence(
            evidence,
            git_probe=lambda _root: {"clean": True, "commit": "a" * 40},
        )


@pytest.mark.parametrize(
    "live",
    (
        {"clean": True, "commit": "b" * 40},
        {"clean": False, "commit": "a" * 40},
    ),
)
def test_final_evidence_gate_does_not_trust_reported_git_identity(
    tmp_path: Path,
    live: dict[str, object],
) -> None:
    evidence = tmp_path / "evidence"
    _valid_evidence(evidence)
    with pytest.raises(ValueError, match="current clean Git HEAD"):
        verify_evidence(
            evidence,
            git_probe=lambda _root: live,
        )
