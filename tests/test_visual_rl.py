"""Flow-GRPO Tiny closes the public API run/resume path with reference KL."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

import visual_rl as vr
from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256


ROOT = Path(__file__).resolve().parents[1]
FLOW_TINY = (
    ROOT / "tests" / "fixtures" / "configs" / "flow_grpo_sd3_tiny.yaml"
)


def _config(
    tmp_path: Path,
    *,
    output_name: str,
    max_steps: int,
    resume: bool,
) -> Path:
    payload = yaml.safe_load(FLOW_TINY.read_text(encoding="utf-8"))
    output_dir = (tmp_path / output_name).resolve()
    payload["runtime"]["max_steps"] = max_steps
    payload["artifacts"]["output_dir"] = str(output_dir)
    payload["artifacts"]["checkpoint_every"] = 1
    payload["resume"]["from"] = str(output_dir) if resume else None
    path = tmp_path / f"{output_name}-{max_steps}-{int(resume)}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _assert_flow_metrics(result: vr.RunResult) -> None:
    assert result.committed_steps == 2
    for name in (
        "loss",
        "policy_loss",
        "reference_kl",
        "approx_kl",
        "clipfrac",
        "reward_mean",
        "reward_std",
    ):
        assert name in result.last_metrics
        assert math.isfinite(float(result.last_metrics[name]))
    assert float(result.last_metrics["reference_kl"]) > 0.0


def test_flow_grpo_reference_run_resume_matches_continuous(tmp_path: Path) -> None:
    split_one = vr.load(
        _config(
            tmp_path,
            output_name="split",
            max_steps=1,
            resume=False,
        )
    ).run()
    assert split_one.committed_steps == 1

    resumed = vr.load(
        _config(
            tmp_path,
            output_name="split",
            max_steps=2,
            resume=True,
        )
    ).run()
    continuous = vr.load(
        _config(
            tmp_path,
            output_name="continuous",
            max_steps=2,
            resume=False,
        )
    ).run()

    _assert_flow_metrics(resumed)
    _assert_flow_metrics(continuous)
    assert resumed.last_metrics == pytest.approx(
        continuous.last_metrics,
        abs=1e-9,
    )
    assert checkpoint_tree_sha256(
        resumed.authoritative_checkpoint
    ) == checkpoint_tree_sha256(continuous.authoritative_checkpoint)
    assert vr.inspect_run(resumed.output_dir).ok
    assert vr.audit_run(resumed.output_dir).ok
