"""Public-API lifecycle coverage for the sole Tiny Runner path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import visual_rl as vr
from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256
from visual_rl.errors import ArtifactError, RunError


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"


def _config(
    tmp_path: Path,
    *,
    name: str,
    max_steps: int,
    resume: bool,
    checkpoint_keep_last: int = 2,
) -> Path:
    payload = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    output_dir = (tmp_path / name).resolve()
    payload["runtime"]["max_steps"] = max_steps
    payload["artifacts"]["output_dir"] = str(output_dir)
    payload["artifacts"]["checkpoint_every"] = 1
    payload["artifacts"]["checkpoint_keep_last"] = checkpoint_keep_last
    payload["resume"]["from"] = str(output_dir) if resume else None
    path = tmp_path / f"{name}-{max_steps}-{int(resume)}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_single_tiny_run_has_one_committed_training_chain(tmp_path: Path) -> None:
    experiment = vr.load(
        _config(
            tmp_path,
            name="single",
            max_steps=1,
            resume=False,
        )
    )
    assert experiment.validate().ok

    result = experiment.run()

    assert result.committed_steps == 1
    assert result.authoritative_checkpoint.name == "checkpoint_000001"
    assert result.last_metrics["step"] == 0
    assert result.last_metrics["sample_count"] == 4
    assert result.last_metrics["active_transition_count"] == 8
    assert vr.inspect_run(result.output_dir).ok
    assert vr.audit_run(result.output_dir).ok
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 4
    assert {record["rank"] for record in manifest["records"]} == {0}


def test_tiny_checkpoint_resume_matches_continuous_two_steps(
    tmp_path: Path,
) -> None:
    split_first = vr.load(
        _config(
            tmp_path,
            name="split",
            max_steps=1,
            resume=False,
        )
    ).run()
    assert split_first.committed_steps == 1

    resumed = vr.load(
        _config(
            tmp_path,
            name="split",
            max_steps=2,
            resume=True,
        )
    ).run()
    continuous = vr.load(
        _config(
            tmp_path,
            name="continuous",
            max_steps=2,
            resume=False,
        )
    ).run()

    assert resumed.committed_steps == continuous.committed_steps == 2
    assert resumed.last_metrics == pytest.approx(
        continuous.last_metrics,
        abs=1e-9,
    )
    assert checkpoint_tree_sha256(
        resumed.authoritative_checkpoint
    ) == checkpoint_tree_sha256(continuous.authoritative_checkpoint)
    assert vr.inspect_run(resumed.output_dir).committed_steps == 2
    assert vr.audit_run(resumed.output_dir).checked_commit_count == 2


def test_resume_at_target_is_a_noop_on_the_authoritative_head(
    tmp_path: Path,
) -> None:
    first = vr.load(
        _config(
            tmp_path,
            name="noop",
            max_steps=1,
            resume=False,
        )
    ).run()
    resumed = vr.load(
        _config(
            tmp_path,
            name="noop",
            max_steps=1,
            resume=True,
        )
    ).run()

    assert resumed == first
    assert vr.audit_run(first.output_dir).checked_commit_count == 1


@pytest.mark.parametrize(
    ("method_name", "phase"),
    (
        ("rebuild_projections", "projection"),
        ("cleanup_published_staging", "cleanup"),
        ("apply_checkpoint_retention", "retention"),
    ),
)
def test_single_post_marker_failure_is_run_error_with_head_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    phase: str,
) -> None:
    from visual_rl.artifacts.manager import ArtifactManager

    output_dir = tmp_path / f"post-marker-single-{phase}"
    config = _config(
        tmp_path,
        name=output_dir.name,
        max_steps=1,
        resume=False,
    )
    original = getattr(ArtifactManager, method_name)

    def fail_after_marker(
        manager: ArtifactManager,
        *args,
        **kwargs,
    ) -> None:
        if method_name == "rebuild_projections" and manager.head is None:
            original(manager, *args, **kwargs)
            return
        raise ArtifactError(f"injected post-marker {phase} failure")

    monkeypatch.setattr(
        ArtifactManager,
        method_name,
        fail_after_marker,
    )

    with pytest.raises(
        RunError,
        match="post-commit artifact maintenance failed",
    ) as caught:
        vr.load(config).run()

    assert isinstance(caught.value.__cause__, ArtifactError)
    assert f"injected post-marker {phase} failure" in str(
        caught.value.__cause__
    )
    assert (output_dir / "commits" / "commit_000001.json").is_file()
    assert (output_dir / "checkpoint_000001").is_dir()

    monkeypatch.setattr(ArtifactManager, method_name, original)
    status = vr.inspect_run(output_dir)
    assert status.committed_steps == 1
    assert status.authoritative_checkpoint == output_dir / "checkpoint_000001"
    resumed = vr.load(
        _config(
            tmp_path,
            name=output_dir.name,
            max_steps=1,
            resume=True,
        )
    ).run()
    assert resumed.committed_steps == 1
    assert list((output_dir / ".staging").iterdir()) == []


def test_fresh_noop_resume_retries_failed_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visual_rl.artifacts.manager import ArtifactManager

    output_dir = tmp_path / "retention-retry"
    config = _config(
        tmp_path,
        name=output_dir.name,
        max_steps=3,
        resume=False,
        checkpoint_keep_last=2,
    )
    original = ArtifactManager.apply_checkpoint_retention
    failed = False

    def fail_once_at_third_commit(
        manager: ArtifactManager,
        *,
        keep_last: int | None,
    ) -> None:
        nonlocal failed
        head = manager.head
        if (
            not failed
            and head is not None
            and int(head["completed_steps"]) == 3
        ):
            failed = True
            raise ArtifactError("injected third-commit retention failure")
        original(manager, keep_last=keep_last)

    monkeypatch.setattr(
        ArtifactManager,
        "apply_checkpoint_retention",
        fail_once_at_third_commit,
    )
    with pytest.raises(
        RunError,
        match="post-commit artifact maintenance failed",
    ) as caught:
        vr.load(config).run()

    assert isinstance(caught.value.__cause__, ArtifactError)
    assert failed
    assert vr.inspect_run(output_dir).committed_steps == 3
    assert sorted(path.name for path in output_dir.glob("checkpoint_*")) == [
        "checkpoint_000001",
        "checkpoint_000002",
        "checkpoint_000003",
    ]

    retry_calls = 0

    def record_retry(
        manager: ArtifactManager,
        *,
        keep_last: int | None,
    ) -> None:
        nonlocal retry_calls
        retry_calls += 1
        original(manager, keep_last=keep_last)

    monkeypatch.setattr(
        ArtifactManager,
        "apply_checkpoint_retention",
        record_retry,
    )
    resumed = vr.load(
        _config(
            tmp_path,
            name=output_dir.name,
            max_steps=3,
            resume=True,
            checkpoint_keep_last=2,
        )
    ).run()

    assert retry_calls == 1
    assert resumed.committed_steps == 3
    assert resumed.authoritative_checkpoint.name == "checkpoint_000003"
    assert sorted(path.name for path in output_dir.glob("checkpoint_*")) == [
        "checkpoint_000002",
        "checkpoint_000003",
    ]
    assert vr.audit_run(output_dir).checked_commit_count == 3
