"""Canonical-config and Preflight boundary coverage for TempFlow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import visual_rl as vr
from visual_rl.errors import ConfigError


ROOT = Path(__file__).resolve().parents[1]
TEMPFLOW_CONFIG = ROOT / "configs/tempflow_sd3.yaml"


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    destination = tmp_path / "tempflow_sd3.yaml"
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def test_tempflow_public_config_resolves_to_canonical_components() -> None:
    config = vr.load(TEMPFLOW_CONFIG).resolve()

    assert config.model.name == "sd3_tempflow"
    assert config.rollout.name == "branching"
    assert config.algorithm.name == "tempflow_grpo"
    assert config.algorithm.params["clip_range"] == pytest.approx(0.001)
    assert config.rollout.params["branch_count"] == 6


def test_retired_objective_version_is_an_unknown_canonical_field(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(TEMPFLOW_CONFIG.read_text(encoding="utf-8"))
    payload["algorithm"]["objective_version"] = "reference_v1"
    experiment = vr.load(_write_config(tmp_path, payload))

    with pytest.raises(
        ConfigError,
        match=r"unknown keys.*objective_version",
    ):
        experiment.resolve()

    report = experiment.validate()
    assert not report.ok
    assert len(report.errors) == 1
    assert report.errors[0].code == "config.resolve"
    assert report.errors[0].path == "algorithm"
