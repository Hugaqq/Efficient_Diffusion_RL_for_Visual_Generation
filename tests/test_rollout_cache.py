"""Final format-v3 rollout-cache contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.rollout.cache import CACHE_SCHEMA, CACHE_VERSION, RolloutCache


def _batch(
    marker: int = 0,
    *,
    video: bool = False,
    step: int = 7,
) -> RolloutBatch:
    media = (
        torch.full((2, 3, 3, 4, 4), float(marker))
        if video
        else torch.full((2, 3, 4, 4), float(marker))
    )
    camera = (
        torch.eye(4, dtype=torch.float64)
        .reshape(1, 1, 4, 4)
        .repeat(2, 3, 1, 1)
        if video
        else None
    )
    return RolloutBatch(
        prompts=(f"alpha-{marker}", f"beta-{marker}"),
        metadata=(
            {"source": "fixture", "marker": marker},
            {"source": "fixture", "marker": marker},
        ),
        media=media,
        latents=torch.full((2, 2, 1, 2, 2), float(marker)),
        next_latents=torch.full((2, 2, 1, 2, 2), float(marker + 1)),
        timesteps=torch.tensor([[2, 1], [2, 1]], dtype=torch.int64),
        old_log_probs=torch.full((2, 2), float(marker)),
        transition_mask=torch.ones(2, 2, dtype=torch.bool),
        sample_id=(f"sample-0-{marker}", f"sample-1-{marker}"),
        prompt_id=(f"prompt-0-{marker}", f"prompt-1-{marker}"),
        group_id=(f"group-0-{marker}", f"group-1-{marker}"),
        branch_id=None,
        media_layout="BFCHW" if video else "BCHW",
        camera_trajectory=camera,
        context=StepContext(
            step=step,
            seed=17 + marker,
            rank=1,
            world_size=2,
        ),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={
            "conditioning": torch.full((2, 3), float(marker))
        },
        artifact_metadata={"adapter": "fixture", "marker": marker},
    )


def _cache(tmp_path: Path) -> tuple[RolloutCache, Path, Path]:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    root = output_dir / "cache" / "rank_1"
    return RolloutCache(root, output_dir=output_dir), output_dir, root


def test_v3_save_returns_output_relative_paths_and_round_trips_all_fields(
    tmp_path: Path,
) -> None:
    cache, output_dir, root = _cache(tmp_path)
    batch = _batch(video=True)

    media_path, rollout_path = cache.save(batch)
    loaded = cache.load(7)

    assert CACHE_SCHEMA == "visual_rl.rollout_cache"
    assert CACHE_VERSION == 3
    assert media_path == "cache/rank_1/batch_000007.media.pt"
    assert rollout_path == "cache/rank_1/batch_000007.pt"
    assert (output_dir / media_path).is_file()
    assert (output_dir / rollout_path).is_file()
    assert loaded.context == batch.context
    assert loaded.sample_id == batch.sample_id
    assert loaded.prompt_id == batch.prompt_id
    assert loaded.group_id == batch.group_id
    assert loaded.artifact_metadata == batch.artifact_metadata
    assert torch.equal(loaded.media, batch.media)
    assert torch.equal(loaded.latents, batch.latents)
    assert torch.equal(
        loaded.recompute_payload["conditioning"],
        batch.recompute_payload["conditioning"],
    )
    assert loaded.camera_trajectory is not None
    assert loaded.camera_trajectory.dtype == torch.float64
    assert tuple(loaded.camera_trajectory.shape) == (2, 3, 4, 4)
    assert torch.equal(loaded.camera_trajectory, batch.camera_trajectory)

    metadata = json.loads(
        (root / "batch_000007.json").read_text(encoding="utf-8")
    )
    assert metadata["context"] == {
        "rank": 1,
        "seed": 17,
        "step": 7,
        "world_size": 2,
    }
    assert "camera_trajectory" not in metadata
    tensors = torch.load(
        root / "batch_000007.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert tensors["camera_trajectory"].dtype == torch.float64


def test_disabled_cache_has_same_surface_and_performs_zero_writes(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    before = tuple(output_dir.iterdir())
    cache = RolloutCache(None, output_dir=output_dir)

    assert cache.save(_batch()) == (None, None)
    assert tuple(output_dir.iterdir()) == before
    with pytest.raises(RuntimeError, match="disabled"):
        cache.load(7)
    cache.truncate_from_step(0)
    assert tuple(output_dir.iterdir()) == before


def test_load_uses_only_cpu_weights_only_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _output_dir, _root = _cache(tmp_path)
    cache.save(_batch())
    calls: list[dict[str, object]] = []
    original = torch.load

    def observed(path, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", observed)
    cache.load(7)

    assert calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu", "weights_only": True},
    ]


@pytest.mark.parametrize("version", [1, 2, 99])
def test_old_or_unknown_versions_are_rejected_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    cache, _output_dir, root = _cache(tmp_path)
    cache.save(_batch())
    metadata_path = root / "batch_000007.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = version
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(RuntimeError, match="version"):
        cache.load(7)
    assert calls == 0


def test_digest_generation_and_exact_keys_fail_closed(tmp_path: Path) -> None:
    cache, _output_dir, root = _cache(tmp_path)
    cache.save(_batch())
    tensor_path = root / "batch_000007.pt"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="digest"):
        cache.load(7)

    cache.save(_batch())
    payload = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=True,
    )
    payload["legacy"] = True
    torch.save(payload, tensor_path)
    metadata_path = root / "batch_000007.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tensor_sha256"] = _sha256(tensor_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact key set"):
        cache.load(7)


def test_declared_paths_and_symlink_payloads_are_rejected(tmp_path: Path) -> None:
    cache, _output_dir, root = _cache(tmp_path)
    cache.save(_batch())
    metadata_path = root / "batch_000007.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["media_path"] = "../outside.pt"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="step-local"):
        cache.load(7)

    cache.save(_batch())
    media_path = root / "batch_000007.media.pt"
    outside = tmp_path / "outside.pt"
    torch.save(torch.zeros(2, 3, 4, 4), outside)
    media_path.unlink()
    media_path.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink"):
        cache.load(7)


def test_constructor_rejects_root_outside_output_or_through_symlink(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="below output_dir"):
        RolloutCache(tmp_path / "outside", output_dir=output_dir)

    outside = output_dir / "outside"
    outside.mkdir()
    linked = output_dir / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinks"):
        RolloutCache(linked / "rank_0", output_dir=output_dir)
    assert tuple(outside.iterdir()) == ()


def test_failed_save_removes_only_owned_random_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _output_dir, root = _cache(tmp_path)

    def fail_save(_payload, handle):
        handle.write(b"partial")
        raise OSError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        cache.save(_batch())
    assert tuple(root.glob(".*.tmp-*")) == ()


def test_truncate_removes_only_validated_v3_step_files(tmp_path: Path) -> None:
    cache, _output_dir, root = _cache(tmp_path)
    cache.save(_batch(step=6))
    cache.save(_batch(step=7))
    ignored = root / "batch_notes.txt"
    ignored.write_text("keep", encoding="utf-8")

    cache.truncate_from_step(7)

    assert sorted(path.name for path in root.glob("batch_000006*")) == [
        "batch_000006.json",
        "batch_000006.media.pt",
        "batch_000006.pt",
    ]
    assert tuple(root.glob("batch_000007*")) == ()
    assert ignored.read_text(encoding="utf-8") == "keep"


def test_truncate_validates_every_target_before_deleting(tmp_path: Path) -> None:
    cache, _output_dir, root = _cache(tmp_path)
    cache.save(_batch())
    tensor_path = root / "batch_000007.pt"
    tensor_path.unlink()
    tensor_path.mkdir()

    with pytest.raises(RuntimeError, match="regular file"):
        cache.truncate_from_step(7)
    assert (root / "batch_000007.json").is_file()
    assert tensor_path.is_dir()


def test_public_surface_contains_no_retired_v1_v2_aliases() -> None:
    assert tuple(inspect.signature(RolloutCache).parameters) == (
        "root",
        "output_dir",
    )
    assert tuple(inspect.signature(RolloutCache.save).parameters) == (
        "self",
        "batch",
    )
    assert not hasattr(RolloutCache, "load_batch")
    source = inspect.getsource(RolloutCache)
    for retired in (
        '"kl"',
        "model_tensors",
        "model_metadata",
        "epoch_tag",
        "branch_ids",
    ):
        assert retired not in source


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
