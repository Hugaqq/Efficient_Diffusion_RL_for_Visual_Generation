"""Contracts for the read-only Flow/Pick-a-Pic paired evaluator."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.flow_pickapic_20260801.evaluate_hps import (
    EVAL_SEEDS,
    compare_evaluations,
)

CONFIG_SHA256 = "1" * 64
PROMPT_SHA256 = "2" * 64
SERVER_REVISION = "world-r1-e156b02bc171"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_scores(
    root: Path,
    *,
    reward_offset: float,
    condition: str,
    mismatched_sample: bool = False,
    config_sha256: str = CONFIG_SHA256,
    prompt_sha256: str = PROMPT_SHA256,
    server_revision: str = SERVER_REVISION,
    include_scorer_identity: bool = True,
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
    checkpoint = None
    if condition == "trained":
        checkpoint = {
            "path": f"/{condition}/adapter",
            "adapter_json_sha256": "3" * 64,
            "adapter_state_sha256": "4" * 64,
        }
    manifest = {
        "schema_version": 1,
        "protocol": "flow_pickapic_paired_hps_v1",
        "condition": condition,
        "config_path": f"/{condition}/different/config.yaml",
        "config_sha256": config_sha256,
        "prompt_path": f"/{condition}/different/prompts.txt",
        "prompt_sha256": prompt_sha256,
        "prompt_count": 64,
        "eval_seeds": list(EVAL_SEEDS),
        "batch_size": 8,
        "num_diffusion_steps": 20,
        "precision": "bf16",
        "adapter_checkpoint": checkpoint,
    }
    if include_scorer_identity:
        manifest["reward_general"] = {
            "name": "reward_general",
            "params": {"server_revision": server_revision},
        }
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "summary.json",
        {
            "schema_version": 1,
            "protocol": "flow_pickapic_paired_hps_v1",
            "condition": condition,
            "sample_count": 128,
            "scores_sha256": _sha256(root / "scores.jsonl"),
        },
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _refresh_scores_sha256(root: Path) -> None:
    summary = _read_json(root / "summary.json")
    summary["scores_sha256"] = _sha256(root / "scores.jsonl")
    _write_json(root / "summary.json", summary)


def test_paired_hps_comparison_uses_prompt_cluster_acceptance(tmp_path: Path) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    output = tmp_path / "comparison.json"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(trained, reward_offset=0.01, condition="trained")

    compare_evaluations(Namespace(base_dir=base, trained_dir=trained, output=output))

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

    with pytest.raises(FileExistsError):
        compare_evaluations(
            Namespace(base_dir=base, trained_dir=trained, output=output)
        )


def test_paired_hps_comparison_accepts_two_legacy_manifests_without_scorer(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    output = tmp_path / "comparison.json"
    _write_scores(
        base,
        reward_offset=0.0,
        condition="base",
        include_scorer_identity=False,
    )
    _write_scores(
        trained,
        reward_offset=0.01,
        condition="trained",
        include_scorer_identity=False,
    )

    compare_evaluations(Namespace(base_dir=base, trained_dir=trained, output=output))

    assert json.loads(output.read_text(encoding="utf-8"))["acceptance"]["passed"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("config_sha256", "a" * 64),
        ("prompt_sha256", "b" * 64),
    ),
)
def test_paired_hps_comparison_rejects_manifest_identity_drift(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="base")
    kwargs = {field: replacement}
    _write_scores(
        trained,
        reward_offset=0.01,
        condition="trained",
        **kwargs,
    )

    with pytest.raises(ValueError, match=field):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


def test_paired_hps_comparison_rejects_scorer_revision_drift(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(
        trained,
        reward_offset=0.01,
        condition="trained",
        server_revision="world-r1-different-revision",
    )

    with pytest.raises(ValueError, match="server_revision"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


def test_paired_hps_comparison_rejects_partial_scorer_identity(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(
        base,
        reward_offset=0.0,
        condition="base",
        include_scorer_identity=False,
    )
    _write_scores(trained, reward_offset=0.01, condition="trained")

    with pytest.raises(ValueError, match="scorer identity availability"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


@pytest.mark.parametrize("condition", ("base", "trained"))
def test_paired_hps_comparison_rejects_checkpoint_role_mismatch(
    tmp_path: Path,
    condition: str,
) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(trained, reward_offset=0.01, condition="trained")
    root = base if condition == "base" else trained
    manifest = _read_json(root / "manifest.json")
    manifest["adapter_checkpoint"] = (
        {
            "path": "/wrong/base/adapter",
            "adapter_json_sha256": "3" * 64,
            "adapter_state_sha256": "4" * 64,
        }
        if condition == "base"
        else None
    )
    _write_json(root / "manifest.json", manifest)

    with pytest.raises(ValueError, match=f"{condition} evaluation manifest"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


def test_paired_hps_comparison_rejects_scores_digest_mismatch(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    _write_scores(base, reward_offset=0.0, condition="base")
    _write_scores(trained, reward_offset=0.01, condition="trained")
    with (trained / "scores.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="SHA-256 does not match summary"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )


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
    _refresh_scores_sha256(trained)

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
    _refresh_scores_sha256(trained)

    with pytest.raises(ValueError, match="reward must be finite numeric"):
        compare_evaluations(
            Namespace(
                base_dir=base,
                trained_dir=trained,
                output=tmp_path / "comparison.json",
            )
        )
