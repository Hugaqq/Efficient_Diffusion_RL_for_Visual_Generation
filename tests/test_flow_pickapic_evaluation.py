"""Contracts for the read-only Flow/Pick-a-Pic paired evaluator."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.flow_pickapic_20260801.evaluate_hps import (
    EVAL_SEEDS,
    compare_evaluations,
)


def _write_scores(
    root: Path,
    *,
    reward_offset: float,
    condition: str,
    mismatched_sample: bool = False,
) -> None:
    root.mkdir()
    rows = []
    for eval_seed in EVAL_SEEDS:
        for prompt_index in range(64):
            sample_id = f"sample-{eval_seed}-{prompt_index}"
            if mismatched_sample and eval_seed == EVAL_SEEDS[0] and prompt_index == 0:
                sample_id += "-wrong"
            rows.append(
                {
                    "condition": condition,
                    "eval_seed": eval_seed,
                    "prompt_index": prompt_index,
                    "prompt_sha256": f"prompt-{prompt_index}",
                    "sample_id": sample_id,
                    "reward": 0.2 + reward_offset,
                }
            )
    (root / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_paired_hps_comparison_uses_prompt_cluster_acceptance(tmp_path: Path) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    output = tmp_path / "comparison.json"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(trained, reward_offset=0.01, condition="trained")

    compare_evaluations(
        Namespace(base_dir=base, trained_dir=trained, output=output)
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["pair_count"] == 128
    assert result["prompt_count"] == 64
    assert result["mean_paired_delta"] == pytest.approx(0.01)
    assert result["prompt_win_rate"] == 1.0
    assert result["cluster_bootstrap_95_ci"][0] > 0.0
    assert result["acceptance"] == {
        "lower_95_ci_gt_zero": True,
        "prompt_win_rate_gt_half": True,
        "passed": True,
    }


def test_paired_hps_comparison_rejects_identity_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(
        trained,
        reward_offset=0.01,
        condition="trained",
        mismatched_sample=True,
    )

    with pytest.raises(ValueError, match="sample_id"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


def test_paired_hps_comparison_rejects_wrong_condition(tmp_path: Path) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="trained")
    _write_scores(trained, reward_offset=0.01, condition="trained")

    with pytest.raises(ValueError, match="condition must be 'base'"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


def test_paired_hps_comparison_rejects_noncanonical_grid(tmp_path: Path) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(trained, reward_offset=0.01, condition="trained")
    rows = [
        json.loads(line)
        for line in (trained / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["eval_seed"] = 9999
    (trained / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen seed/prompt grid"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


def test_paired_hps_comparison_rejects_nonfinite_reward(tmp_path: Path) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(trained, reward_offset=0.01, condition="trained")
    rows = [
        json.loads(line)
        for line in (trained / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["reward"] = float("nan")
    (trained / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reward must be finite numeric"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )
