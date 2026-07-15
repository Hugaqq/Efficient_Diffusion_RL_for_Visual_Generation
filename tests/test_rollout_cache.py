"""Safe serialization and path-boundary tests for rollout caches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest
import torch

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.rollout.cache import CACHE_SCHEMA, CACHE_VERSION, RolloutCache


def _batch(marker: int = 0) -> RolloutBatch:
    return RolloutBatch(
        prompts=[f"alpha-{marker}", f"beta-{marker}"],
        metadata=[
            {"prompt_id": f"p0-{marker}", "marker": marker},
            {"prompt_id": f"p1-{marker}", "marker": marker},
        ],
        media=torch.full((2, 3, 4, 4), float(marker)),
        latents=torch.full((2, 2, 1, 2, 2), float(marker)),
        next_latents=torch.full((2, 2, 1, 2, 2), float(marker + 1)),
        timesteps=torch.tensor([[2, 1], [2, 1]]),
        old_log_probs=torch.full((2, 2), float(marker)),
        kl=torch.full((2, 2), float(marker)),
        context=StepContext(step=7, seed=17 + marker, epoch_tag=7),
        model_tensors={
            "nested": {"conditioning": torch.full((2, 3), float(marker))}
        },
    )


def _metadata_path(root, step=7):
    return root / f"batch_{step:06d}.json"


def test_new_cache_has_versioned_weights_only_safe_payloads(tmp_path):
    cache = RolloutCache(tmp_path)
    batch = _batch()
    cache.save(7, batch)

    metadata = json.loads(_metadata_path(tmp_path).read_text(encoding="utf-8"))
    assert metadata["schema"] == CACHE_SCHEMA
    assert metadata["version"] == CACHE_VERSION
    assert len(metadata["generation"]) == 32
    assert len(metadata["tensor_sha256"]) == 64
    assert len(metadata["media_sha256"]) == 64
    assert metadata["media_path"] == "batch_000007.media.pt"
    tensor_payload = torch.load(
        tmp_path / "batch_000007.pt",
        map_location="cpu",
        weights_only=True,
    )
    media_payload = torch.load(
        tmp_path / "batch_000007.media.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert tensor_payload["kind"] == "tensors"
    assert media_payload["kind"] == "media"
    assert tensor_payload["generation"] == metadata["generation"]
    assert media_payload["generation"] == metadata["generation"]

    loaded = cache.load(7)
    assert loaded.context == batch.context
    assert loaded.sample_id == batch.sample_id
    assert torch.equal(loaded.media, batch.media)
    assert torch.equal(
        loaded.model_tensors["nested"]["conditioning"],
        batch.model_tensors["nested"]["conditioning"],
    )


def test_legacy_tensor_cache_loads_only_through_weights_only(tmp_path):
    batch = _batch()
    base = tmp_path / "batch_000008"
    torch.save(
        {
            "latents": batch.latents,
            "next_latents": batch.next_latents,
            "timesteps": batch.timesteps,
            "old_log_probs": batch.old_log_probs,
            "kl": batch.kl,
            "branch_ids": torch.tensor([3, 4]),
            "model_tensors": {},
        },
        base.with_suffix(".pt"),
    )
    torch.save(batch.media, base.with_suffix(".media.pt"))
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "prompts": batch.prompts,
                "metadata": batch.metadata,
                "model_metadata": {},
                "media_path": base.with_suffix(".media.pt").name,
            }
        ),
        encoding="utf-8",
    )

    loaded = RolloutCache(tmp_path).load(8)
    assert loaded.branch_id.tolist() == [3, 4]


def test_version_one_cache_remains_loadable(tmp_path):
    batch = _batch()
    base = tmp_path / "batch_000009"
    torch.save(
        {
            "schema": CACHE_SCHEMA,
            "version": 1,
            "kind": "tensors",
            "latents": batch.latents,
            "next_latents": batch.next_latents,
            "timesteps": batch.timesteps,
            "old_log_probs": batch.old_log_probs,
            "kl": batch.kl,
            "transition_mask": batch.transition_mask,
            "model_tensors": batch.model_tensors,
        },
        base.with_suffix(".pt"),
    )
    torch.save(
        {
            "schema": CACHE_SCHEMA,
            "version": 1,
            "kind": "media",
            "media": batch.media,
        },
        base.with_suffix(".media.pt"),
    )
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "version": 1,
                "kind": "metadata",
                "step": 9,
                "tensor_path": base.with_suffix(".pt").name,
                "media_path": base.with_suffix(".media.pt").name,
                "prompts": batch.prompts,
                "metadata": batch.metadata,
                "model_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = RolloutCache(tmp_path).load(9)

    assert loaded.prompts == batch.prompts
    assert torch.equal(loaded.media, batch.media)


@pytest.mark.parametrize(
    "declared",
    [
        "/tmp/batch_000007.media.pt",
        "../batch_000007.media.pt",
        "batch_000008.media.pt",
        "nested/batch_000007.media.pt",
    ],
)
def test_media_path_must_be_the_expected_step_local_file(tmp_path, declared):
    cache = RolloutCache(tmp_path)
    cache.save(7, _batch())
    metadata_path = _metadata_path(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["media_path"] = declared
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected step-local file"):
        cache.load(7)


def test_media_symlink_is_rejected_before_load(tmp_path):
    cache = RolloutCache(tmp_path)
    cache.save(7, _batch())
    media_path = tmp_path / "batch_000007.media.pt"
    media_path.unlink()
    outside = tmp_path / "outside.pt"
    torch.save(torch.zeros(2, 3, 4, 4), outside)
    media_path.symlink_to(outside)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        cache.load(7)


def test_cache_rejects_invalid_tensor_and_media_containers(tmp_path):
    cache = RolloutCache(tmp_path)
    cache.save(7, _batch())
    tensor_path = tmp_path / "batch_000007.pt"
    tensor_payload = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=True,
    )
    tensor_payload["latents"] = [1, 2]
    torch.save(tensor_payload, tensor_path)
    with pytest.raises(RuntimeError, match="latents.*tensor or None"):
        cache.load(7)

    cache.save(7, _batch())
    media_path = tmp_path / "batch_000007.media.pt"
    media_payload = torch.load(
        media_path,
        map_location="cpu",
        weights_only=True,
    )
    media_payload["media"] = torch.zeros(2, 3)
    torch.save(media_payload, media_path)
    with pytest.raises(RuntimeError, match="BCHW or BFCHW"):
        cache.load(7)


def test_cache_never_falls_back_to_unsafe_pickle(tmp_path):
    cache = RolloutCache(tmp_path)
    cache.save(7, _batch())
    np_state = np.random.get_state()
    torch.save(np_state, tmp_path / "batch_000007.pt")

    with pytest.raises(RuntimeError, match="weights_only=True") as error:
        cache.load(7)
    assert np_state[0] == "MT19937"
    assert error.value.__cause__ is None


def test_unknown_cache_version_fails_closed(tmp_path):
    cache = RolloutCache(tmp_path)
    cache.save(7, _batch())
    metadata_path = _metadata_path(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported rollout cache version"):
        cache.load(7)


def test_constructor_rejects_symlink_ancestor_without_creating_outside_child(
    tmp_path,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must not contain symlinks"):
        RolloutCache(linked_parent / "cache" / "nested")

    assert list(outside.iterdir()) == []


def test_preexisting_legacy_fixed_temp_files_do_not_block_save(tmp_path):
    cache_root = tmp_path / "cache"
    cache = RolloutCache(cache_root)
    legacy_temps = [
        cache_root / "batch_000007.pt.tmp",
        cache_root / "batch_000007.media.pt.tmp",
        cache_root / "batch_000007.json.tmp",
    ]
    for path in legacy_temps:
        path.write_bytes(b"attacker-owned")

    cache.save(7, _batch())

    assert cache.load(7).context == _batch().context
    assert all(path.read_bytes() == b"attacker-owned" for path in legacy_temps)


def test_concurrent_distinct_saves_never_load_a_mixed_batch(tmp_path):
    cache_root = tmp_path / "cache"
    cache = RolloutCache(cache_root)
    batches = [_batch(marker) for marker in range(8)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda batch: cache.save(7, batch), batches))

    assert all(
        result["rollout_cache_path"].endswith("batch_000007.pt")
        for result in results
    )
    assert list(cache_root.glob(".*.tmp-*")) == []
    try:
        loaded = cache.load(7)
    except RuntimeError as exc:
        assert "metadata publication" in str(exc)
    else:
        marker = loaded.metadata[0]["marker"]
        assert loaded.prompts == [f"alpha-{marker}", f"beta-{marker}"]
        assert loaded.context == StepContext(
            step=7,
            seed=17 + marker,
            epoch_tag=7,
        )
        assert torch.all(loaded.media == marker)
        assert torch.all(loaded.latents == marker)
        assert torch.all(loaded.old_log_probs == marker)
        assert torch.all(
            loaded.model_tensors["nested"]["conditioning"] == marker
        )


def test_partial_distinct_save_fails_closed_against_prior_metadata(
    tmp_path,
    monkeypatch,
):
    cache = RolloutCache(tmp_path)
    cache.save(7, _batch(1))
    real_save = torch.save

    def fail_media_save(payload, handle):
        if isinstance(payload, dict) and payload.get("kind") == "media":
            handle.write(b"partial")
            raise OSError("injected media save failure")
        real_save(payload, handle)

    monkeypatch.setattr(torch, "save", fail_media_save)
    with pytest.raises(OSError, match="injected media save failure"):
        cache.save(7, _batch(2))

    with pytest.raises(RuntimeError, match="metadata publication"):
        cache.load(7)
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_failed_save_removes_only_its_own_random_temp(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache = RolloutCache(cache_root)

    def fail_save(_payload, handle):
        handle.write(b"partial")
        raise OSError("injected torch save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="injected torch save failure"):
        cache.save(7, _batch())

    assert list(cache_root.glob(".*.tmp-*")) == []


def test_failed_save_does_not_delete_replacement_at_temp_name(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache = RolloutCache(cache_root)
    replacement_paths = []

    def replace_temp_then_fail(_payload, _handle):
        [owned_temp] = list(cache_root.glob(".*.tmp-*"))
        owned_temp.unlink()
        owned_temp.write_bytes(b"attacker-owned")
        replacement_paths.append(owned_temp)
        raise OSError("injected replacement race")

    monkeypatch.setattr(torch, "save", replace_temp_then_fail)
    with pytest.raises(OSError, match="injected replacement race"):
        cache.save(7, _batch())

    assert len(replacement_paths) == 1
    assert replacement_paths[0].read_bytes() == b"attacker-owned"


def test_truncate_rejects_symlink_cache_root_without_touching_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "batch_000007.pt"
    protected.write_bytes(b"keep")
    cache_root = tmp_path / "cache"
    cache_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must not contain symlinks"):
        RolloutCache(cache_root).truncate_from_step(7)

    assert protected.read_bytes() == b"keep"


def test_truncate_rejects_symlink_cache_root_ancestor(tmp_path):
    outside = tmp_path / "outside"
    cache_root = outside / "cache"
    cache_root.mkdir(parents=True)
    protected = cache_root / "batch_000007.pt"
    protected.write_bytes(b"keep")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must not contain symlinks"):
        RolloutCache(linked_parent / "cache").truncate_from_step(7)

    assert protected.read_bytes() == b"keep"


def test_truncate_rejects_symlink_entry_without_touching_target(tmp_path):
    cache_root = tmp_path / "cache"
    cache = RolloutCache(cache_root)
    retained = cache_root / "batch_000006.pt"
    retained.write_bytes(b"old")
    normal = cache_root / "batch_000007.json"
    normal.write_text("cache", encoding="utf-8")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"keep")
    (cache_root / "batch_000007.pt").symlink_to(outside)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        cache.truncate_from_step(7)

    assert retained.read_bytes() == b"old"
    assert normal.read_text(encoding="utf-8") == "cache"
    assert outside.read_bytes() == b"keep"


def test_truncate_rejects_non_regular_entry_before_deleting(tmp_path):
    cache_root = tmp_path / "cache"
    cache = RolloutCache(cache_root)
    normal = cache_root / "batch_000007.json"
    normal.write_text("cache", encoding="utf-8")
    (cache_root / "batch_000007.pt").mkdir()

    with pytest.raises(RuntimeError, match="must be a regular file"):
        cache.truncate_from_step(7)

    assert normal.read_text(encoding="utf-8") == "cache"
    assert (cache_root / "batch_000007.pt").is_dir()


def test_truncate_only_removes_validated_files_at_or_after_step(tmp_path):
    cache_root = tmp_path / "cache"
    cache = RolloutCache(cache_root)
    retained = cache_root / "batch_000006.pt"
    removed = cache_root / "batch_000007.pt"
    ignored = cache_root / "batch_notes.txt"
    retained.write_bytes(b"old")
    removed.write_bytes(b"new")
    ignored.write_bytes(b"notes")

    cache.truncate_from_step(7)

    assert retained.exists()
    assert not removed.exists()
    assert ignored.exists()
