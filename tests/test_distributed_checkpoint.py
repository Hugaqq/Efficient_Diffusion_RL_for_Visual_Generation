"""Rank-local runtime-state checkpoint contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from visual_rl.artifacts.checkpoint import (
    apply_training_state,
    capture_rng_state,
    read_and_validate_training_state,
    restore_rng_state,
    save_training_state,
)


class _Optimizer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {
            "state": {0: {"step": torch.tensor(2)}},
            "param_groups": [{"params": [0], "lr": 0.01}],
        }

    def load_state_dict(self, state) -> None:
        self.loaded = state


class _Plugin:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {"updates": 2}

    def load_state_dict(self, state) -> None:
        self.loaded = state


def _rng_state(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return capture_rng_state()


def _draw_rng_values() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


def _rank_entry(rank: int, seed: int) -> dict:
    return {
        "rank": rank,
        "rng": _rng_state(seed),
        "sampler_cursor": {"epoch": 3, "offset": rank + 4},
        "runtime_identity": {"device": "cpu", "rank_seed": seed},
    }


def _checkpoint(path, distributed_state=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter.bin").write_bytes(b"adapter-state")
    optimizer = _Optimizer()
    plugin = _Plugin()
    metadata = save_training_state(
        path,
        optimizer=optimizer,
        plugin=plugin,
        step=8,
        config={},
        implementation={},
        distributed_state=distributed_state,
    )
    return optimizer, plugin, metadata


def _refresh_training_state_hash(path) -> None:
    state_path = path / "training_state.pt"
    metadata_path = path / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_state_sha256"] = hashlib.sha256(
        state_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_two_rank_roundtrip_is_sorted_weights_only_safe_and_rank_local(tmp_path):
    rank_zero = _rank_entry(0, 101)
    rank_one = _rank_entry(1, 202)
    checkpoint = tmp_path / "checkpoint_000008"
    optimizer, plugin, metadata = _checkpoint(
        checkpoint,
        {
            "world_size": 2,
            "backend": "gloo",
            "entries": [rank_one, rank_zero],
        },
    )

    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert metadata["distributed_state"] == {
        "world_size": 2,
        "backend": "gloo",
    }
    assert [entry["rank"] for entry in state["distributed_state"]["entries"]] == [
        0,
        1,
    ]
    assert [
        entry["runtime_identity"]
        for entry in state["distributed_state"]["entries"]
    ] == [rank_zero["runtime_identity"], rank_one["runtime_identity"]]

    restore_rng_state(rank_zero["rng"])
    expected_zero = _draw_rng_values()
    restore_rng_state(rank_one["rng"])
    expected_one = _draw_rng_values()
    validated = read_and_validate_training_state(
        checkpoint,
        config={},
        expected_world_size=2,
        expected_rank=1,
    )
    assert [
        entry["runtime_identity"]
        for entry in validated.state["distributed_state"]["entries"]
    ] == [rank_zero["runtime_identity"], rank_one["runtime_identity"]]

    assert apply_training_state(
        validated,
        optimizer=optimizer,
        plugin=plugin,
        rank=0,
    ) == 8
    actual_zero = _draw_rng_values()
    assert actual_zero[0:2] == expected_zero[0:2]
    assert torch.equal(actual_zero[2], expected_zero[2])

    apply_training_state(validated, optimizer=optimizer, plugin=plugin, rank=1)
    actual_one = _draw_rng_values()
    assert actual_one[0:2] == expected_one[0:2]
    assert torch.equal(actual_one[2], expected_one[2])
    assert actual_zero[0:2] != actual_one[0:2]
    assert not torch.equal(actual_zero[2], actual_one[2])


def test_world_size_and_missing_rank_changes_are_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint_000008"
    _checkpoint(
        checkpoint,
        {
            "world_size": 2,
            "backend": "gloo",
            "entries": [_rank_entry(0, 11), _rank_entry(1, 22)],
        },
    )

    with pytest.raises(RuntimeError, match="world size changed"):
        read_and_validate_training_state(
            checkpoint,
            config={},
            expected_world_size=3,
        )
    with pytest.raises(RuntimeError, match="does not contain rank 2"):
        read_and_validate_training_state(
            checkpoint,
            config={},
            expected_world_size=2,
            expected_rank=2,
        )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            lambda: [_rank_entry(0, 1), _rank_entry(0, 2)],
            "duplicate rank 0",
        ),
        (
            lambda: [_rank_entry(0, 1)],
            "missing ranks: \\[1\\]",
        ),
    ],
)
def test_malformed_rank_sets_fail_closed(tmp_path, entries, message):
    checkpoint = tmp_path / "checkpoint_000008"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")

    with pytest.raises(RuntimeError, match=message):
        save_training_state(
            checkpoint,
            optimizer=_Optimizer(),
            plugin=_Plugin(),
            step=8,
            config={},
            implementation={},
            distributed_state={
                "world_size": 2,
                "backend": "gloo",
                "entries": entries(),
            },
        )
    assert not (checkpoint / "training_state.pt").exists()


@pytest.mark.parametrize(
    "unsafe_identity",
    [object(), float("nan")],
    ids=["object", "nan"],
)
def test_unsafe_runtime_identity_values_fail_before_persistence(
    tmp_path,
    unsafe_identity,
):
    checkpoint = tmp_path / "checkpoint_000008"
    entry_zero = _rank_entry(0, 1)
    entry_zero["runtime_identity"]["nested"] = {"value": unsafe_identity}
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")

    with pytest.raises(RuntimeError, match="runtime identity must be JSON-safe"):
        save_training_state(
            checkpoint,
            optimizer=_Optimizer(),
            plugin=_Plugin(),
            step=8,
            config={},
            implementation={},
            distributed_state={
                "world_size": 1,
                "backend": "gloo",
                "entries": [entry_zero],
            },
        )
    assert not (checkpoint / "training_state.pt").exists()


def test_unsafe_rank_local_values_fail_before_persistence(tmp_path):
    checkpoint = tmp_path / "checkpoint_000008"
    entry_zero = _rank_entry(0, 1)
    entry_zero["sampler_cursor"] = object()
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")

    with pytest.raises(RuntimeError, match="unsafe value type"):
        save_training_state(
            checkpoint,
            optimizer=_Optimizer(),
            plugin=_Plugin(),
            step=8,
            config={},
            implementation={},
            distributed_state={
                "world_size": 1,
                "backend": "gloo",
                "entries": [entry_zero],
            },
        )
    assert not (checkpoint / "training_state.pt").exists()


def test_nested_runtime_identity_secrets_are_redacted_before_persistence(tmp_path):
    checkpoint = tmp_path / "checkpoint_000008"
    entry_zero = _rank_entry(0, 1)
    entry_zero["runtime_identity"] = {
        "device": "cpu",
        "runtime": {
            "worker": "worker-0",
            "credentials": {
                "api_token": "nested-api-token-plaintext",
                "password": "nested-password-plaintext",
            },
            "headers": [
                {"authorization": "Bearer nested-authorization-plaintext"},
                {"cookie": "session=nested-cookie-plaintext"},
            ],
        },
    }
    _checkpoint(
        checkpoint,
        {
            "world_size": 1,
            "backend": "gloo",
            "entries": [entry_zero],
        },
    )

    secrets = (
        "nested-api-token-plaintext",
        "nested-password-plaintext",
        "nested-authorization-plaintext",
        "nested-cookie-plaintext",
    )
    checkpoint_bytes = (checkpoint / "training_state.pt").read_bytes()
    for secret in secrets:
        assert secret.encode() not in checkpoint_bytes

    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    identity = state["distributed_state"]["entries"][0]["runtime_identity"]
    assert identity == {
        "device": "cpu",
        "runtime": {
            "worker": "worker-0",
            "credentials": {
                "api_token": "[REDACTED]",
                "password": "[REDACTED]",
            },
            "headers": [
                {"authorization": "[REDACTED]"},
                {"cookie": "[REDACTED]"},
            ],
        },
    }
    validated = read_and_validate_training_state(
        checkpoint,
        config={},
        expected_world_size=1,
        expected_rank=0,
    )
    assert (
        validated.state["distributed_state"]["entries"][0]["runtime_identity"]
        == identity
    )


def test_unredacted_runtime_identity_is_rejected_on_read(tmp_path):
    checkpoint = tmp_path / "checkpoint_000008"
    _checkpoint(
        checkpoint,
        {
            "world_size": 1,
            "backend": "gloo",
            "entries": [_rank_entry(0, 1)],
        },
    )
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["distributed_state"]["entries"][0]["runtime_identity"]["nested"] = {
        "api_token": "tampered-token-plaintext"
    }
    torch.save(state, state_path)
    _refresh_training_state_hash(checkpoint)

    with pytest.raises(RuntimeError, match="contains unredacted secrets"):
        read_and_validate_training_state(checkpoint, config={})


def test_tampered_unsorted_entries_are_rejected_on_read(tmp_path):
    checkpoint = tmp_path / "checkpoint_000008"
    _checkpoint(
        checkpoint,
        {
            "world_size": 2,
            "backend": "gloo",
            "entries": [_rank_entry(0, 1), _rank_entry(1, 2)],
        },
    )
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["distributed_state"]["entries"].reverse()
    torch.save(state, state_path)
    _refresh_training_state_hash(checkpoint)

    with pytest.raises(RuntimeError, match="sorted by rank"):
        read_and_validate_training_state(checkpoint, config={})


def test_single_process_checkpoint_remains_implicit_rank_zero(tmp_path):
    checkpoint = tmp_path / "checkpoint_000008"
    optimizer, plugin, metadata = _checkpoint(checkpoint)
    assert "distributed_state" not in metadata

    validated = read_and_validate_training_state(
        checkpoint,
        config={},
        expected_world_size=1,
        expected_rank=0,
    )
    assert apply_training_state(
        validated,
        optimizer=optimizer,
        plugin=plugin,
    ) == 8
    with pytest.raises(RuntimeError, match="does not contain rank 1"):
        read_and_validate_training_state(
            checkpoint,
            config={},
            expected_rank=1,
        )


def test_training_state_ignores_prepositioned_fixed_temp_symlink(tmp_path):
    checkpoint = tmp_path / "checkpoint_000008"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter-state")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"do-not-overwrite")
    fixed_temp = checkpoint / "training_state.pt.tmp"
    fixed_temp.symlink_to(outside)

    before = deepcopy(outside.read_bytes())
    _checkpoint(checkpoint)

    assert outside.read_bytes() == before
    assert fixed_temp.is_symlink()
    assert torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )["step"] == 8
    assert read_and_validate_training_state(checkpoint, config={}).step == 8
    assert list(checkpoint.glob(".training_state.pt.tmp-*")) == []
