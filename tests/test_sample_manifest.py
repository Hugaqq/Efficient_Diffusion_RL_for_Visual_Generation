import json

import pytest

from visual_rl.artifacts import SampleManifest, SampleRecord


def _record(
    *, run_id: str, sample_id: str, sample_index: int, media_type: str
) -> SampleRecord:
    return SampleRecord(
        run_id=run_id,
        sample_id=sample_id,
        sample_index=sample_index,
        step=3,
        prompt="一辆红色汽车"
        if media_type == "image"
        else "a camera moves through a city",
        media_type=media_type,
        prompt_metadata={"source": "test"},
        seed=42 + sample_index,
        rollout_type="full_trajectory",
        timestep_summary={"count": 4},
        reward_values={"mock": 0.5 + sample_index},
        media_path=f"media/{sample_id}",
        rollout_cache_path="rollouts/batch_000003.pt",
        checkpoint_path="checkpoint_000004",
        model_metadata={"adapter": "mock"},
    )


def test_manifest_round_trip_preserves_image_and_video_records(tmp_path):
    manifest = SampleManifest(run_id="run-001")
    manifest.add(
        _record(
            run_id="run-001", sample_id="image-0", sample_index=0, media_type="image"
        )
    )
    manifest.add(
        _record(
            run_id="run-001", sample_id="video-1", sample_index=1, media_type="video"
        )
    )

    path = tmp_path / "nested" / "sample_manifest.json"
    manifest.save(path)
    loaded = SampleManifest.load(path)

    assert path.exists()
    assert loaded.to_dict() == manifest.to_dict()
    assert [record.media_type for record in loaded.records] == ["image", "video"]
    assert "一辆红色汽车" in path.read_text(encoding="utf-8")


def test_manifest_rejects_duplicate_sample_id():
    manifest = SampleManifest(run_id="run-001")
    manifest.add(
        _record(
            run_id="run-001", sample_id="sample-0", sample_index=0, media_type="image"
        )
    )

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        manifest.add(
            _record(
                run_id="run-001",
                sample_id="sample-0",
                sample_index=1,
                media_type="video",
            )
        )


def test_manifest_rejects_record_from_another_run():
    manifest = SampleManifest(run_id="run-001")

    with pytest.raises(ValueError, match="record run_id does not match"):
        manifest.add(
            _record(
                run_id="run-002",
                sample_id="sample-0",
                sample_index=0,
                media_type="image",
            )
        )


def test_manifest_validate_rejects_unsupported_media_type():
    manifest = SampleManifest(
        run_id="run-001",
        records=[
            _record(
                run_id="run-001",
                sample_id="sample-0",
                sample_index=0,
                media_type="audio",
            )
        ],
    )

    with pytest.raises(ValueError, match="media type"):
        manifest.validate()


def test_manifest_load_rejects_non_object_json(tmp_path):
    path = tmp_path / "sample_manifest.json"
    path.write_text(json.dumps(["not", "a", "manifest"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        SampleManifest.load(path)
