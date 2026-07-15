"""Security and deterministic-resume contracts for training checkpoints."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import random
import stat

import numpy as np
import pytest
import torch

import visual_rl.artifacts.checkpoint as checkpoint_module
from visual_rl.artifacts.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    apply_training_state,
    load_json,
    load_training_state,
    migrate_legacy_checkpoint_to_v4,
    read_and_validate_training_state,
    save_json,
    save_training_state,
)


class _Optimizer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {
            "state": {0: {"step": torch.tensor(3), "exp_avg": torch.ones(2)}},
            "param_groups": [{"params": [0], "lr": 0.01}],
        }

    def load_state_dict(self, state) -> None:
        self.loaded = state


class _Plugin:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"history": [1, 2, 3]}

    def load_state_dict(self, state) -> None:
        self.loaded = state


def _save_checkpoint(path):
    path.mkdir(parents=True)
    (path / "adapter.bin").write_bytes(b"adapter-state")
    optimizer = _Optimizer()
    plugin = _Plugin()
    metadata = save_training_state(
        path,
        optimizer=optimizer,
        plugin=plugin,
        step=4,
        config={},
        implementation={},
    )
    return optimizer, plugin, metadata


def _rewrite_formats(path, state, metadata, version: int) -> None:
    state["format_version"] = version
    metadata["format_version"] = version
    metadata.pop("training_state_sha256", None)
    torch.save(state, path / "training_state.pt")
    (path / "checkpoint.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def _refresh_training_state_hash(path) -> None:
    state_path = path / "training_state.pt"
    metadata_path = path / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_state_sha256"] = hashlib.sha256(
        state_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _rewrite_fingerprint_bundle(path, bundle) -> None:
    state_path = path / "training_state.pt"
    metadata_path = path / "checkpoint.json"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state.update(bundle)
    torch.save(state, state_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "config_fingerprint_version",
        "config_fingerprint_scheme",
        "config_fingerprint",
        "training_semantics_fingerprint",
        "data_identity_fingerprint",
        "implementation_identity_fingerprint",
        "data_identity",
        "data_source",
    ):
        if key in bundle:
            metadata[key] = bundle[key]
        else:
            metadata.pop(key, None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _refresh_training_state_hash(path)


def test_v2_scheme_preserves_legacy_async_audit_and_fails_closed_on_ambiguity(
    tmp_path,
):
    checkpoint = tmp_path / "legacy-v2-async"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")
    config = {
        "runner": {
            "reward_executor": {
                "mode": "async",
                "microbatch_size": 1,
            }
        }
    }
    save_training_state(
        checkpoint,
        optimizer=_Optimizer(),
        plugin=_Plugin(),
        step=4,
        config=config,
        implementation={},
    )
    legacy_bundle = checkpoint_module._build_v2_fingerprint_bundle(
        config,
        {},
        validate_data=True,
        fingerprint_scheme="component-sha256-v1",
    )
    _rewrite_fingerprint_bundle(checkpoint, legacy_bundle)

    assert read_and_validate_training_state(
        checkpoint,
        config=config,
        implementation={},
    ).step == 4
    audited = read_and_validate_training_state(
        checkpoint,
        config={
            "runner": {
                "reward_executor": {
                    "mode": "async",
                    "microbatch_size": 2,
                }
            }
        },
        use_checkpoint_implementation_identity=True,
    )
    assert audited.step == 4
    assert audited.applicable is False
    with pytest.raises(RuntimeError, match="read-only audit identity"):
        apply_training_state(
            audited,
            optimizer=_Optimizer(),
            plugin=_Plugin(),
        )
    legacy_explicit_partition = deepcopy(config)
    legacy_explicit_partition["runner"]["reward_executor"]["microbatch_size"] = 2
    assert read_and_validate_training_state(
        checkpoint,
        config=legacy_explicit_partition,
        implementation={},
    ).step == 4

    ambiguous = deepcopy(config)
    ambiguous["runner"]["reward_executor"]["microbatch_size"] = None
    with pytest.raises(RuntimeError, match="did not bind the async reward batch"):
        read_and_validate_training_state(
            checkpoint,
            config=ambiguous,
            implementation={},
        )


def test_current_v2_scheme_binds_explicit_partition_but_not_full_batch_mode(
    tmp_path,
):
    explicit = {
        "runner": {
            "reward_executor": {
                "mode": "async",
                "microbatch_size": 1,
            }
        }
    }
    checkpoint = tmp_path / "current-v2-explicit"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")
    metadata = save_training_state(
        checkpoint,
        optimizer=_Optimizer(),
        plugin=_Plugin(),
        step=4,
        config=explicit,
        implementation={},
    )
    assert metadata["config_fingerprint_scheme"] == "component-sha256-v2"
    assert read_and_validate_training_state(
        checkpoint,
        config=explicit,
        implementation={},
    ).step == 4

    changed = deepcopy(explicit)
    changed["runner"]["reward_executor"]["microbatch_size"] = 2
    with pytest.raises(
        RuntimeError,
        match=r"training_semantics\.reward_batch_partition\.microbatch_size",
    ):
        read_and_validate_training_state(
            checkpoint,
            config=changed,
            implementation={},
        )

    sync_full = {
        "runner": {
            "reward_executor": {
                "mode": "sync",
                "microbatch_size": None,
            }
        }
    }
    full_checkpoint = tmp_path / "current-v2-full"
    full_checkpoint.mkdir()
    (full_checkpoint / "adapter.bin").write_bytes(b"adapter-state")
    save_training_state(
        full_checkpoint,
        optimizer=_Optimizer(),
        plugin=_Plugin(),
        step=4,
        config=sync_full,
        implementation={},
    )
    async_full = deepcopy(sync_full)
    async_full["runner"]["reward_executor"]["mode"] = "async"
    assert read_and_validate_training_state(
        full_checkpoint,
        config=async_full,
        implementation={},
    ).step == 4


@pytest.mark.parametrize("invalid_microbatch", [None, 0, -1, True, 1.5, "2"])
def test_v1_rejects_invalid_legacy_async_partition_on_save_and_load(
    tmp_path,
    invalid_microbatch,
):
    config = {
        "runner": {
            "reward_executor": {
                "mode": "async",
                "microbatch_size": 1,
            }
        }
    }
    checkpoint = tmp_path / "legacy-v1-async"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")
    save_training_state(
        checkpoint,
        optimizer=_Optimizer(),
        plugin=_Plugin(),
        step=4,
        config=config,
        implementation={},
        config_fingerprint_version=1,
    )
    ambiguous = deepcopy(config)
    ambiguous["runner"]["reward_executor"][
        "microbatch_size"
    ] = invalid_microbatch

    with pytest.raises(RuntimeError, match="did not bind the async reward batch"):
        read_and_validate_training_state(
            checkpoint,
            config=ambiguous,
            implementation={},
        )

    fresh = tmp_path / "invalid-new-v1"
    fresh.mkdir()
    (fresh / "adapter.bin").write_bytes(b"adapter-state")
    with pytest.raises(RuntimeError, match="did not bind the async reward batch"):
        save_training_state(
            fresh,
            optimizer=_Optimizer(),
            plugin=_Plugin(),
            step=4,
            config=ambiguous,
            implementation={},
            config_fingerprint_version=1,
        )


@pytest.mark.parametrize("invalid_scheme", [[], {}])
def test_v2_rejects_non_string_fingerprint_scheme_without_type_leak(
    tmp_path,
    invalid_scheme,
):
    checkpoint = tmp_path / "invalid-scheme"
    _save_checkpoint(checkpoint)
    state_path = checkpoint / "training_state.pt"
    metadata_path = checkpoint / "checkpoint.json"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state["config_fingerprint_scheme"] = invalid_scheme
    metadata["config_fingerprint_scheme"] = invalid_scheme
    torch.save(state, state_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _refresh_training_state_hash(checkpoint)

    with pytest.raises(RuntimeError, match="Unsupported config fingerprint scheme"):
        read_and_validate_training_state(checkpoint, config={})
    with pytest.raises(ValueError, match="Unsupported config fingerprint scheme"):
        checkpoint_module._build_v2_fingerprint_bundle(
            {},
            {},
            validate_data=True,
            fingerprint_scheme=invalid_scheme,
        )


def test_v4_checkpoint_hashes_state_and_restores_all_cpu_rngs(tmp_path):
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    checkpoint = tmp_path / "checkpoint_000004"
    optimizer, plugin, metadata = _save_checkpoint(checkpoint)

    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    state_bytes = (checkpoint / "training_state.pt").read_bytes()
    assert metadata["format_version"] == CHECKPOINT_FORMAT_VERSION == 4
    assert metadata["training_state_sha256"] == hashlib.sha256(state_bytes).hexdigest()
    assert "training_state_sha256" not in state
    assert isinstance(state["rng"]["numpy"], dict)
    assert isinstance(state["rng"]["numpy"]["state"], list)

    expected = (
        random.random(),
        np.random.random(5),
        torch.rand(5),
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    validated = read_and_validate_training_state(checkpoint, config={})
    assert apply_training_state(
        validated,
        optimizer=optimizer,
        plugin=plugin,
    ) == 4
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.random(5), expected[1])
    assert torch.equal(torch.rand(5), expected[2])
    assert optimizer.loaded["param_groups"][0]["lr"] == 0.01
    assert plugin.loaded == {"history": [1, 2, 3]}


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("optimizer", {}, "optimizer state is missing keys"),
        ("plugin", [], "plugin state must be a dictionary"),
    ],
)
def test_v4_rejects_malformed_optimizer_and_plugin_state(
    tmp_path,
    field,
    replacement,
    message,
):
    checkpoint = tmp_path / "checkpoint_000004"
    _save_checkpoint(checkpoint)
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state[field] = replacement
    torch.save(state, state_path)
    _refresh_training_state_hash(checkpoint)

    with pytest.raises(RuntimeError, match=message):
        read_and_validate_training_state(checkpoint, config={})


def test_v4_rejects_raw_training_state_tamper_before_load_or_apply(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint_000004"
    optimizer, plugin, _metadata = _save_checkpoint(checkpoint)
    state_path = checkpoint / "training_state.pt"
    payload = bytearray(state_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    state_path.write_bytes(payload)

    load_calls = []
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )
    with pytest.raises(RuntimeError, match="training_state.pt SHA256 mismatch"):
        load_training_state(
            checkpoint,
            optimizer=optimizer,
            plugin=plugin,
            config={},
        )
    assert load_calls == []
    assert optimizer.loaded is None
    assert plugin.loaded is None


def test_v4_rejects_optimizer_state_tamper_before_load_or_apply(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint_000004"
    optimizer, plugin, _metadata = _save_checkpoint(checkpoint)
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["optimizer"]["param_groups"][0]["lr"] = 99.0
    torch.save(state, state_path)

    load_calls = []
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )
    with pytest.raises(RuntimeError, match="training_state.pt SHA256 mismatch"):
        load_training_state(
            checkpoint,
            optimizer=optimizer,
            plugin=plugin,
            config={},
        )
    assert load_calls == []
    assert optimizer.loaded is None
    assert plugin.loaded is None


def test_legacy_unsafe_fallback_requires_explicit_opt_in(tmp_path):
    checkpoint = tmp_path / "checkpoint_000004"
    optimizer, plugin, metadata = _save_checkpoint(checkpoint)
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    state["rng"]["numpy"] = np.random.get_state()
    _rewrite_formats(checkpoint, state, metadata, version=2)

    with pytest.raises(RuntimeError, match="could not be loaded safely") as error:
        read_and_validate_training_state(checkpoint, config={})
    assert error.value.__cause__ is None

    validated = read_and_validate_training_state(
        checkpoint,
        config={},
        allow_unsafe_legacy=True,
    )
    assert apply_training_state(
        validated,
        optimizer=optimizer,
        plugin=plugin,
    ) == 4


def test_prior_safe_v3_without_training_state_hash_remains_supported(tmp_path):
    checkpoint = tmp_path / "checkpoint_000004"
    optimizer, plugin, metadata = _save_checkpoint(checkpoint)
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    _rewrite_formats(checkpoint, state, metadata, version=3)

    validated = read_and_validate_training_state(checkpoint, config={})
    assert apply_training_state(
        validated,
        optimizer=optimizer,
        plugin=plugin,
    ) == 4


def test_v1_rejects_symlinked_payload_before_training_state_load(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint_000004"
    _optimizer, _plugin, metadata = _save_checkpoint(checkpoint)
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    _rewrite_formats(checkpoint, state, metadata, version=1)
    outside = tmp_path / "outside-adapter.bin"
    outside.write_bytes(b"outside-checkpoint")
    (checkpoint / "nested-adapter.bin").symlink_to(outside)

    load_calls = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: load_calls.append(args))
    with pytest.raises(RuntimeError, match="payload must not contain symlinks"):
        read_and_validate_training_state(
            checkpoint,
            config={},
            allow_unsafe_legacy=True,
        )
    assert load_calls == []


def test_v3_never_uses_unsafe_fallback(tmp_path):
    checkpoint = tmp_path / "checkpoint_000004"
    _optimizer, _plugin, metadata = _save_checkpoint(checkpoint)
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    state["rng"]["numpy"] = np.random.get_state()
    _rewrite_formats(checkpoint, state, metadata, version=3)

    with pytest.raises(RuntimeError, match="could not be loaded safely") as error:
        read_and_validate_training_state(
            checkpoint,
            config={},
            allow_unsafe_legacy=True,
        )
    assert error.value.__cause__ is None


def test_unknown_metadata_format_fails_before_pickle_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_000004"
    _save_checkpoint(checkpoint)
    metadata_path = checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    load_calls = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: load_calls.append(args))
    with pytest.raises(RuntimeError, match="Unsupported checkpoint format_version"):
        read_and_validate_training_state(
            checkpoint,
            config={},
            allow_unsafe_legacy=True,
        )
    assert load_calls == []


def test_trusted_root_rejects_outside_and_symlinked_checkpoint_paths(tmp_path):
    trusted = tmp_path / "trusted"
    checkpoint = trusted / "runs" / "checkpoint_000004"
    _save_checkpoint(checkpoint)
    assert read_and_validate_training_state(
        checkpoint,
        config={},
        trusted_root=trusted,
    ).step == 4

    outside = tmp_path / "outside" / "checkpoint_000004"
    _save_checkpoint(outside)
    with pytest.raises(RuntimeError, match="escapes checkpoint root"):
        read_and_validate_training_state(
            outside,
            config={},
            trusted_root=trusted,
        )

    linked_parent = trusted / "linked-runs"
    linked_parent.symlink_to(checkpoint.parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="must not contain symlinks"):
        read_and_validate_training_state(
            linked_parent / checkpoint.name,
            config={},
            trusted_root=trusted,
        )


def test_clear_error_when_torch_lacks_weights_only_support(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_000004"
    _save_checkpoint(checkpoint)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("old torch detail")),
    )

    with pytest.raises(RuntimeError, match="requires a PyTorch version") as error:
        read_and_validate_training_state(checkpoint, config={})
    assert error.value.__cause__ is None
    assert "old torch detail" not in str(error.value)


def test_explicit_non_inplace_legacy_migration_produces_safe_v4(tmp_path):
    source = tmp_path / "legacy" / "checkpoint_000004"
    _optimizer, _plugin, metadata = _save_checkpoint(source)
    state_path = source / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    numpy_state = state["rng"]["numpy"]
    state["rng"]["numpy"] = (
        numpy_state["bit_generator"],
        np.asarray(numpy_state["state"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    )
    _rewrite_formats(source, state, metadata, 1)

    with pytest.raises(RuntimeError, match="could not be loaded safely"):
        read_and_validate_training_state(source, config={})

    destination_root = tmp_path / "migrated"
    destination_root.mkdir()
    destination = destination_root / "checkpoint_000004"
    migrated = migrate_legacy_checkpoint_to_v4(
        source,
        destination,
        config={},
        implementation={},
        trusted_root=tmp_path,
        destination_root=destination_root,
    )

    assert migrated["format_version"] == CHECKPOINT_FORMAT_VERSION == 4
    assert migrated["migrated_from_format_version"] == 1
    assert len(migrated["training_state_sha256"]) == 64
    assert len(migrated["adapter_payload_sha256"]) == 64
    assert len(migrated["checkpoint_tree_sha256"]) == 64
    validated = read_and_validate_training_state(
        destination,
        config={},
        implementation={},
        trusted_root=destination_root,
    )
    assert validated.step == 4
    assert validated.state["format_version"] == 4

    with pytest.raises(ValueError, match="must not run in place"):
        migrate_legacy_checkpoint_to_v4(
            source,
            source,
            config={},
            implementation={},
            trusted_root=tmp_path,
            destination_root=tmp_path,
        )


def test_save_json_ignores_prepositioned_fixed_temp_symlink(tmp_path):
    target = tmp_path / "state.json"
    outside = tmp_path / "outside.txt"
    outside.write_text("do-not-overwrite", encoding="utf-8")
    fixed_temp = target.with_suffix(".tmp")
    fixed_temp.symlink_to(outside)

    save_json(target, {"step": 7})

    assert json.loads(target.read_text(encoding="utf-8")) == {"step": 7}
    assert outside.read_text(encoding="utf-8") == "do-not-overwrite"
    assert fixed_temp.is_symlink()
    assert list(tmp_path.glob(".state.json.tmp-*")) == []


def test_save_json_rejects_symlink_paths_and_cleans_failed_temps(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        save_json(linked_parent / "state.json", {"step": 1})

    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    target = real_parent / "state.json"
    target.symlink_to(outside)
    with pytest.raises(RuntimeError, match="regular file"):
        save_json(target, {"step": 2})
    assert outside.read_text(encoding="utf-8") == "outside"

    target.unlink()
    with pytest.raises(TypeError):
        save_json(target, {"serializable": True, "bad": object()})
    assert not target.exists()
    assert list(real_parent.glob(".state.json.tmp-*")) == []


def test_save_json_fsyncs_parent_directory_after_replace(tmp_path, monkeypatch):
    observed_modes = []
    real_fsync = checkpoint_module.os.fsync

    def observed_fsync(fd):
        observed_modes.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(checkpoint_module.os, "fsync", observed_fsync)
    save_json(tmp_path / "state.json", {"step": 1})

    assert any(stat.S_ISREG(mode) for mode in observed_modes)
    assert any(stat.S_ISDIR(mode) for mode in observed_modes)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_save_json_rejects_non_finite_numbers(tmp_path, value):
    target = tmp_path / "state.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        save_json(target, {"value": value})

    assert not target.exists()
    assert list(tmp_path.glob(".state.json.tmp-*")) == []


@pytest.mark.parametrize(
    "payload",
    [
        '{"step": 1, "step": 2}',
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": 1e9999}',
    ],
)
def test_load_json_rejects_duplicate_keys_and_non_finite_numbers(tmp_path, payload):
    target = tmp_path / "state.json"
    target.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate|Non-finite"):
        load_json(target)


def test_checkpoint_persists_only_redacted_identity_but_rejects_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint_000004"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")
    config = {
        "algorithm": {"clip_range": 0.2},
        "rewards": {
            "provider": {
                "api_key": "api-key-plaintext",
                "token": "token-plaintext",
                "password": "password-plaintext",
                "url": (
                    "https://url-user:url-password@reward.example/private/"
                    "path?token=url-query-secret"
                ),
            }
        },
    }
    implementation = {"backend": {"access_token": "implementation-secret"}}
    save_training_state(
        checkpoint,
        optimizer=_Optimizer(),
        plugin=_Plugin(),
        step=4,
        config=config,
        implementation=implementation,
    )

    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    persisted_strings = []

    def collect_strings(value):
        if isinstance(value, str):
            persisted_strings.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect_strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_strings(item)

    collect_strings(state)
    persisted_strings.append((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    persisted = "\n".join(persisted_strings)
    for secret in (
        "api-key-plaintext",
        "token-plaintext",
        "password-plaintext",
        "url-user",
        "url-password",
        "url-query-secret",
        "implementation-secret",
    ):
        assert secret not in persisted
    assert (
        state["identity_payload"]["training_semantics"]["rewards"]["provider"]
        ["api_key"]
        == "[REDACTED]"
    )
    assert state["implementation"]["backend"]["access_token"] == "[REDACTED]"
    assert read_and_validate_training_state(
        checkpoint,
        config=config,
        implementation=implementation,
    ).step == 4

    changed = deepcopy(config)
    changed["algorithm"]["clip_range"] = 0.5
    with pytest.raises(RuntimeError, match="training_semantics.algorithm.clip_range"):
        read_and_validate_training_state(
            checkpoint,
            config=changed,
            implementation=implementation,
        )

    changed_secret = deepcopy(config)
    changed_secret["rewards"]["provider"]["token"] = "different-runtime-token"
    with pytest.raises(RuntimeError, match="fingerprint v2 mismatch"):
        read_and_validate_training_state(
            checkpoint,
            config=changed_secret,
            implementation=implementation,
        )
