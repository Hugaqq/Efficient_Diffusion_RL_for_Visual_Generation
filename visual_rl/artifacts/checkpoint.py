"""Mechanical format-v5 training checkpoints.

This module owns one producer, one side-effect-free reader/validator and one
atomic mutation boundary.  It deliberately stores no config, data, model,
reference, source or plugin identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import stat
from typing import Any

import numpy as np

from visual_rl.configs.schema import OptimizerConfig
from visual_rl.errors import ResumeError

CHECKPOINT_FORMAT_VERSION = 5
_CHECKPOINT_ROOT_ENTRIES = frozenset(
    {"adapter", "training_state.pt", "checkpoint.json"}
)
_METADATA_KEYS = frozenset(
    {
        "format_version",
        "global_step",
        "world_size",
        "training_contract",
        "adapter_dir",
        "adapter_tree_sha256",
        "training_state",
        "training_state_sha256",
    }
)
_STATE_KEYS = frozenset(
    {
        "format_version",
        "global_step",
        "world_size",
        "training_contract",
        "optimizer_topology",
        "optimizer",
        "grad_scaler",
        "rank_states",
    }
)
_TOPOLOGY_KEYS = frozenset({"parameter_names", "group_roles"})
_OPTIMIZER_KEYS = frozenset({"state", "param_groups"})
_CONTRACT_KEYS = frozenset({"algorithm", "version"})
_RANK_STATE_KEYS = frozenset({"rank", "rng"})
_RNG_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda"})
_NUMPY_RNG_KEYS = frozenset(
    {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
)
_GROUP_ROLES = ("trainable_adapter",)
_SHA256_CHUNK_SIZE = 1024 * 1024

__all__ = [
    "CheckpointMetadata",
    "RankState",
    "TrainingContract",
    "ValidatedTrainingState",
    "apply_training_state",
    "checkpoint_tree_sha256",
    "load_json",
    "read_and_validate_training_state",
    "save_training_state",
    "strict_json_loads",
]


@dataclass(frozen=True)
class TrainingContract:
    """Mechanical objective/update compatibility identifier."""

    algorithm: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, str) or not self.algorithm:
            raise TypeError("training contract algorithm must be a non-empty string")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("training contract version must be a positive integer")

    def to_payload(self) -> dict[str, object]:
        return {"algorithm": self.algorithm, "version": self.version}


@dataclass(frozen=True)
class RankState:
    """One rank's immutable Python/NumPy/Torch RNG snapshot."""

    rank: int
    python_state: tuple[Any, ...]
    numpy_bit_generator: str
    numpy_state: tuple[int, ...]
    numpy_position: int
    numpy_has_gauss: int
    numpy_cached_gaussian: float
    torch_cpu: Any
    torch_cuda: Any | None

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if type(self.python_state) is not tuple:
            raise TypeError("python RNG state must be a tuple")
        try:
            probe = random.Random()
            probe.setstate(self.python_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("python RNG state is invalid") from exc
        if self.numpy_bit_generator != "MT19937":
            raise ValueError("NumPy RNG bit_generator must be MT19937")
        if (
            type(self.numpy_state) is not tuple
            or len(self.numpy_state) != 624
            or any(type(item) is not int or not 0 <= item <= 0xFFFF_FFFF for item in self.numpy_state)
        ):
            raise ValueError("NumPy RNG state must contain 624 uint32 integers")
        if type(self.numpy_position) is not int or not 0 <= self.numpy_position <= 624:
            raise ValueError("NumPy RNG position must be in [0, 624]")
        if type(self.numpy_has_gauss) is not int or self.numpy_has_gauss not in {0, 1}:
            raise ValueError("NumPy RNG has_gauss must be integer 0 or 1")
        if (
            isinstance(self.numpy_cached_gaussian, bool)
            or not isinstance(self.numpy_cached_gaussian, (int, float))
            or not math.isfinite(float(self.numpy_cached_gaussian))
        ):
            raise ValueError("NumPy RNG cached_gaussian must be finite")
        cpu = _validated_rng_tensor(self.torch_cpu, label="torch_cpu")
        cuda = (
            None
            if self.torch_cuda is None
            else _validated_rng_tensor(self.torch_cuda, label="torch_cuda")
        )
        object.__setattr__(self, "torch_cpu", cpu.clone())
        object.__setattr__(
            self,
            "torch_cuda",
            None if cuda is None else cuda.clone(),
        )

    @classmethod
    def from_rng(
        cls,
        *,
        rank: int,
        python_state: tuple[Any, ...],
        numpy_state: tuple[Any, ...],
        torch_cpu: Any,
        torch_cuda: Any | None,
    ) -> RankState:
        """Build the typed contract from standard library RNG snapshots."""

        if type(numpy_state) is not tuple or len(numpy_state) != 5:
            raise ValueError("numpy_state must be the five-item np.random state tuple")
        words = np.asarray(numpy_state[1], dtype=np.uint32)
        if words.shape != (624,):
            raise ValueError("NumPy RNG state must contain 624 uint32 words")
        return cls(
            rank=rank,
            python_state=python_state,
            numpy_bit_generator=str(numpy_state[0]),
            numpy_state=tuple(int(item) for item in words.tolist()),
            numpy_position=int(numpy_state[2]),
            numpy_has_gauss=int(numpy_state[3]),
            numpy_cached_gaussian=float(numpy_state[4]),
            torch_cpu=torch_cpu,
            torch_cuda=torch_cuda,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "rng": {
                "python": self.python_state,
                "numpy": {
                    "bit_generator": self.numpy_bit_generator,
                    "state": list(self.numpy_state),
                    "position": self.numpy_position,
                    "has_gauss": self.numpy_has_gauss,
                    "cached_gaussian": float(self.numpy_cached_gaussian),
                },
                "torch_cpu": self.torch_cpu.clone(),
                "torch_cuda": (
                    None if self.torch_cuda is None else self.torch_cuda.clone()
                ),
            },
        }


@dataclass(frozen=True)
class CheckpointMetadata:
    """Integrity summary returned to the authoritative commit owner."""

    checkpoint_dir: Path
    global_step: int
    world_size: int
    training_contract: TrainingContract
    adapter_tree_sha256: str
    training_state_sha256: str
    tree_sha256: str


class _ApplyToken:
    __slots__ = ("consumed",)

    def __init__(self) -> None:
        self.consumed = False


@dataclass(frozen=True)
class ValidatedTrainingState:
    """A fully preflighted format-v5 state, applicable exactly once."""

    checkpoint_dir: Path
    global_step: int
    world_size: int
    training_contract: TrainingContract
    metadata: Mapping[str, Any] = field(repr=False, compare=False)
    state: Mapping[str, Any] = field(repr=False, compare=False)
    _token: _ApplyToken = field(
        default_factory=_ApplyToken,
        repr=False,
        compare=False,
    )


def strict_json_loads(value: str) -> Any:
    """Load finite JSON while rejecting duplicate object keys."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON constant {constant!r} is not allowed")

    return json.loads(
        value,
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Read one regular UTF-8 JSON object without following a final symlink."""

    target = Path(path)
    _require_regular_file(target, label="JSON file")
    try:
        value = strict_json_loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Cannot read finite JSON object: {target}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {target}")
    return value


def checkpoint_tree_sha256(
    checkpoint_dir: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> str:
    """Hash a symlink-free directory tree by relative path, kind and bytes."""

    root = _safe_directory(Path(checkpoint_dir), label="checkpoint tree")
    if trusted_root is not None:
        boundary = _safe_directory(Path(trusted_root), label="trusted root")
        try:
            root.relative_to(boundary)
        except ValueError as exc:
            raise RuntimeError("checkpoint tree escapes trusted root") from exc
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"checkpoint tree must not contain symlinks: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            digest.update(b"D\0")
            digest.update(relative)
            digest.update(b"\0")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"checkpoint tree entries must be files/directories: {entry}"
            )
        digest.update(b"F\0")
        digest.update(relative)
        digest.update(b"\0")
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_SHA256_CHUNK_SIZE), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def save_training_state(
    checkpoint_dir: str | Path,
    *,
    adapter: Any,
    optimizer: Any,
    scaler: Any | None,
    global_step: int,
    training_contract: TrainingContract,
    rank_states: tuple[RankState, ...],
    writer_rank: int,
    writer_device: Any,
) -> CheckpointMetadata:
    """Write the exact three-entry v5 checkpoint and restore writer RNG."""

    step = _positive_int(global_step, label="global_step")
    contract = _require_training_contract(training_contract)
    states = _validate_rank_states(rank_states)
    world_size = len(states)
    if type(writer_rank) is not int or not 0 <= writer_rank < world_size:
        raise ValueError("writer_rank must identify one rank state")
    writer_state = states[writer_rank]
    if writer_state.rank != writer_rank:
        raise ValueError("writer_rank must identify the same ordered rank state")
    device = _validate_device(writer_device)
    _validate_rank_device_topology(states, device)
    names, _parameters = _validate_live_topology(adapter, optimizer)
    optimizer_state = _normalized_optimizer_state(
        optimizer.state_dict(),
        parameter_names=names,
        adapter=adapter,
        live_optimizer=optimizer,
    )
    scaler_state = _normalized_scaler_state(scaler)

    try:
        root = _prepare_checkpoint_target(Path(checkpoint_dir))
        adapter_dir = root / "adapter"
        adapter.save_checkpoint(adapter_dir)
        _safe_directory(adapter_dir, label="adapter checkpoint")
        adapter.validate_checkpoint(adapter_dir)
        adapter_digest = checkpoint_tree_sha256(
            adapter_dir,
            trusted_root=root,
        )

        state = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "global_step": step,
            "world_size": world_size,
            "training_contract": contract.to_payload(),
            "optimizer_topology": {
                "parameter_names": names,
                "group_roles": _GROUP_ROLES,
            },
            "optimizer": optimizer_state,
            "grad_scaler": scaler_state,
            "rank_states": tuple(item.to_payload() for item in states),
        }
        state_path = root / "training_state.pt"
        _save_torch_payload(state_path, state)
        state_digest = _sha256_file(state_path)

        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "global_step": step,
            "world_size": world_size,
            "training_contract": contract.to_payload(),
            "adapter_dir": "adapter",
            "adapter_tree_sha256": adapter_digest,
            "training_state": "training_state.pt",
            "training_state_sha256": state_digest,
        }
        _write_canonical_json(root / "checkpoint.json", metadata)
        _require_exact_checkpoint_root(root)
        _fsync_tree(root)
        tree_digest = checkpoint_tree_sha256(root, trusted_root=root)
        return CheckpointMetadata(
            checkpoint_dir=root,
            global_step=step,
            world_size=world_size,
            training_contract=contract,
            adapter_tree_sha256=adapter_digest,
            training_state_sha256=state_digest,
            tree_sha256=tree_digest,
        )
    finally:
        _restore_rank_rng(writer_state, device=device)


def read_and_validate_training_state(
    checkpoint_dir: str | Path,
    *,
    adapter: Any,
    optimizer: Any,
    scaler: Any | None,
    expected_global_step: int,
    expected_world_size: int,
    expected_training_contract: TrainingContract,
) -> ValidatedTrainingState:
    """Safely load and mechanically validate v5 without mutating live state."""

    import torch

    expected_step = _positive_int(
        expected_global_step,
        label="expected_global_step",
    )
    world_size = _world_size(expected_world_size, label="expected_world_size")
    expected_contract = _require_training_contract(expected_training_contract)
    root = _safe_directory(Path(checkpoint_dir), label="checkpoint")
    _require_exact_checkpoint_root(root)
    metadata = load_json(root / "checkpoint.json")
    parsed = _validate_metadata(metadata)
    if parsed["global_step"] != expected_step:
        raise ResumeError(
            "checkpoint global_step does not match authoritative marker",
            path=str(root),
        )
    if parsed["world_size"] != world_size:
        raise ResumeError(
            "checkpoint world_size does not match the current topology",
            path=str(root),
        )
    if parsed["training_contract"] != expected_contract:
        raise ResumeError(
            "checkpoint training_contract is mechanically incompatible",
            path=str(root),
        )

    adapter_dir = root / "adapter"
    actual_adapter_digest = checkpoint_tree_sha256(
        adapter_dir,
        trusted_root=root,
    )
    if not secrets.compare_digest(
        actual_adapter_digest,
        parsed["adapter_tree_sha256"],
    ):
        raise ResumeError("adapter checkpoint tree digest mismatch", path=str(root))

    state_path = root / "training_state.pt"
    actual_state_digest = _sha256_file(state_path)
    if not secrets.compare_digest(
        actual_state_digest,
        parsed["training_state_sha256"],
    ):
        raise ResumeError("training_state.pt digest mismatch", path=str(root))
    try:
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ResumeError(
            "training_state.pt cannot be safely loaded with weights_only=True",
            path=str(root),
        ) from exc
    _validate_safe_value(state, label="training_state")
    if not isinstance(state, dict) or set(state) != set(_STATE_KEYS):
        raise ResumeError("training_state.pt has an invalid exact key set")
    shared = _validate_state_shared_fields(state)
    if (
        shared["global_step"] != parsed["global_step"]
        or shared["world_size"] != parsed["world_size"]
        or shared["training_contract"] != parsed["training_contract"]
    ):
        raise ResumeError(
            "checkpoint.json and training_state.pt shared fields disagree"
        )

    names, parameters = _validate_live_topology(adapter, optimizer)
    topology = state["optimizer_topology"]
    if not isinstance(topology, dict) or set(topology) != set(_TOPOLOGY_KEYS):
        raise ResumeError("optimizer_topology has an invalid exact key set")
    parameter_names = topology["parameter_names"]
    group_roles = topology["group_roles"]
    if type(parameter_names) is not tuple or parameter_names != names:
        raise ResumeError("optimizer parameter names/order do not match live Adapter")
    if type(group_roles) is not tuple or group_roles != _GROUP_ROLES:
        raise ResumeError("optimizer group_roles must be ('trainable_adapter',)")
    _validate_optimizer_payload(
        state["optimizer"],
        parameter_names=names,
        parameters=parameters,
        live_optimizer=optimizer,
    )
    _validate_scaler_payload(state["grad_scaler"], scaler)
    rank_states = _rank_states_from_payload(
        state["rank_states"],
        expected_world_size=world_size,
    )
    device = parameters[0].device
    _validate_rank_device_topology(rank_states, device)

    # Adapter validation intentionally happens only after the whole safe payload
    # and both mechanical digests have passed.
    try:
        adapter.validate_checkpoint(adapter_dir)
    except Exception as exc:
        raise ResumeError(
            "adapter checkpoint failed mechanical validation",
            path=str(adapter_dir),
        ) from exc
    return ValidatedTrainingState(
        checkpoint_dir=root,
        global_step=expected_step,
        world_size=world_size,
        training_contract=expected_contract,
        metadata=deepcopy(metadata),
        state=_clone_safe_value(state),
    )


def _audit_checkpoint_artifacts(
    checkpoint_dir: str | Path,
    *,
    expected_global_step: int,
) -> None:
    """Deeply inspect one committed v5 payload without constructing live state."""

    import torch

    expected_step = _positive_int(
        expected_global_step,
        label="expected_global_step",
    )
    root = _safe_directory(Path(checkpoint_dir), label="checkpoint")
    _require_exact_checkpoint_root(root)
    parsed = _validate_metadata(load_json(root / "checkpoint.json"))
    if parsed["global_step"] != expected_step:
        raise ResumeError(
            "checkpoint global_step does not match authoritative marker",
            path=str(root),
        )
    if not secrets.compare_digest(
        checkpoint_tree_sha256(root / "adapter", trusted_root=root),
        parsed["adapter_tree_sha256"],
    ):
        raise ResumeError("adapter checkpoint tree digest mismatch", path=str(root))
    state_path = root / "training_state.pt"
    if not secrets.compare_digest(
        _sha256_file(state_path),
        parsed["training_state_sha256"],
    ):
        raise ResumeError("training_state.pt digest mismatch", path=str(root))
    try:
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ResumeError(
            "training_state.pt cannot be safely loaded with weights_only=True",
            path=str(root),
        ) from exc
    _validate_safe_value(state, label="training_state")
    if not isinstance(state, dict) or set(state) != set(_STATE_KEYS):
        raise ResumeError("training_state.pt has an invalid exact key set")
    shared = _validate_state_shared_fields(state)
    if (
        shared["global_step"] != parsed["global_step"]
        or shared["world_size"] != parsed["world_size"]
        or shared["training_contract"] != parsed["training_contract"]
    ):
        raise ResumeError(
            "checkpoint.json and training_state.pt shared fields disagree"
        )
    names = _audit_optimizer_topology(state["optimizer_topology"])
    _audit_optimizer_payload(state["optimizer"], parameter_names=names)
    _audit_scaler_payload(state["grad_scaler"])
    _rank_states_from_payload(
        state["rank_states"],
        expected_world_size=parsed["world_size"],
    )


def apply_training_state(
    validated: ValidatedTrainingState,
    *,
    adapter: Any,
    optimizer: Any,
    scaler: Any | None,
    optimizer_config: OptimizerConfig,
    rank: int,
) -> None:
    """Atomically apply Adapter/AdamW/scaler/RNG and current hyperparameters."""

    import torch

    if not isinstance(validated, ValidatedTrainingState):
        raise TypeError("validated must be a ValidatedTrainingState")
    if validated._token.consumed:
        raise ResumeError("validated training state has already been applied")
    if not isinstance(optimizer_config, OptimizerConfig):
        raise TypeError("optimizer_config must be an OptimizerConfig")
    if type(rank) is not int or not 0 <= rank < validated.world_size:
        raise ValueError("rank is outside the validated checkpoint world_size")
    validated._token.consumed = True

    root = _safe_directory(validated.checkpoint_dir, label="checkpoint")
    _require_exact_checkpoint_root(root)
    parsed = _validate_metadata(load_json(root / "checkpoint.json"))
    if (
        parsed["global_step"] != validated.global_step
        or parsed["world_size"] != validated.world_size
        or parsed["training_contract"] != validated.training_contract
    ):
        raise ResumeError("checkpoint metadata changed after validation")
    if (
        checkpoint_tree_sha256(root / "adapter", trusted_root=root)
        != parsed["adapter_tree_sha256"]
        or _sha256_file(root / "training_state.pt")
        != parsed["training_state_sha256"]
    ):
        raise ResumeError("checkpoint changed after validation", path=str(root))
    try:
        adapter.validate_checkpoint(root / "adapter")
    except Exception as exc:
        raise ResumeError("adapter checkpoint changed after validation") from exc

    names, parameters = _validate_live_topology(adapter, optimizer)
    state = validated.state
    topology = state["optimizer_topology"]
    if topology["parameter_names"] != names:
        raise ResumeError("live Adapter topology changed after validation")
    _validate_optimizer_payload(
        state["optimizer"],
        parameter_names=names,
        parameters=parameters,
        live_optimizer=optimizer,
    )
    _validate_scaler_payload(state["grad_scaler"], scaler)
    rank_states = _rank_states_from_payload(
        state["rank_states"],
        expected_world_size=validated.world_size,
    )
    device = parameters[0].device
    _validate_rank_device_topology(rank_states, device)
    selected_rank_state = rank_states[rank]

    adapter_snapshot = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer_snapshot = _clone_safe_value(optimizer.state_dict())
    scaler_snapshot = (
        None if scaler is None else _clone_safe_value(scaler.state_dict())
    )
    rng_snapshot = _capture_rank_rng(rank=rank, device=device)
    try:
        adapter.load_checkpoint(root / "adapter")
        optimizer.load_state_dict(_clone_safe_value(state["optimizer"]))
        _apply_current_optimizer_config(optimizer, optimizer_config)
        if scaler is not None:
            scaler.load_state_dict(_clone_safe_value(state["grad_scaler"]))
        _restore_rank_rng(selected_rank_state, device=device)
    except BaseException as exc:
        rollback_error: BaseException | None = None
        try:
            with torch.no_grad():
                for parameter, original in zip(
                    parameters,
                    adapter_snapshot,
                    strict=True,
                ):
                    parameter.copy_(original)
            optimizer.load_state_dict(optimizer_snapshot)
            if scaler is not None:
                assert scaler_snapshot is not None
                scaler.load_state_dict(scaler_snapshot)
            _restore_rank_rng(rng_snapshot, device=device)
        except BaseException as rollback_exc:
            rollback_error = rollback_exc
        if rollback_error is not None:
            raise ResumeError(
                "training-state apply failed and rollback also failed",
                path=str(root),
            ) from rollback_error
        raise ResumeError(
            "training-state apply failed; live state was rolled back",
            path=str(root),
        ) from exc


def _require_training_contract(value: Any) -> TrainingContract:
    if not isinstance(value, TrainingContract):
        raise TypeError("training_contract must be a TrainingContract")
    return value


def _contract_from_payload(value: Any) -> TrainingContract:
    if not isinstance(value, dict) or set(value) != set(_CONTRACT_KEYS):
        raise ResumeError("training_contract has an invalid exact key set")
    try:
        return TrainingContract(
            algorithm=value["algorithm"],
            version=value["version"],
        )
    except (TypeError, ValueError) as exc:
        raise ResumeError("training_contract values are invalid") from exc


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _world_size(value: Any, *, label: str) -> int:
    if type(value) is not int or value not in {1, 2}:
        raise ValueError(f"{label} must be integer 1 or 2")
    return value


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict) or set(metadata) != set(_METADATA_KEYS):
        raise ResumeError("checkpoint.json has an invalid exact key set")
    if metadata["format_version"] != 5 or type(metadata["format_version"]) is not int:
        raise ResumeError("checkpoint format_version must be integer 5")
    try:
        step = _positive_int(metadata["global_step"], label="global_step")
        world_size = _world_size(metadata["world_size"], label="world_size")
    except ValueError as exc:
        raise ResumeError("checkpoint metadata step/world_size is invalid") from exc
    contract = _contract_from_payload(metadata["training_contract"])
    if metadata["adapter_dir"] != "adapter":
        raise ResumeError("checkpoint adapter_dir must be 'adapter'")
    if metadata["training_state"] != "training_state.pt":
        raise ResumeError("checkpoint training_state must be 'training_state.pt'")
    adapter_digest = _sha256(metadata["adapter_tree_sha256"], label="adapter digest")
    state_digest = _sha256(
        metadata["training_state_sha256"],
        label="training-state digest",
    )
    return {
        "global_step": step,
        "world_size": world_size,
        "training_contract": contract,
        "adapter_tree_sha256": adapter_digest,
        "training_state_sha256": state_digest,
    }


def _validate_state_shared_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    if state["format_version"] != 5 or type(state["format_version"]) is not int:
        raise ResumeError("training state format_version must be integer 5")
    try:
        global_step = _positive_int(state["global_step"], label="global_step")
        world_size = _world_size(state["world_size"], label="world_size")
    except ValueError as exc:
        raise ResumeError("training state step/world_size is invalid") from exc
    return {
        "global_step": global_step,
        "world_size": world_size,
        "training_contract": _contract_from_payload(state["training_contract"]),
    }


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ResumeError(f"{label} must be lowercase SHA-256")
    return value


def _validate_rank_states(value: Any) -> tuple[RankState, ...]:
    if type(value) is not tuple or len(value) not in {1, 2}:
        raise ValueError("rank_states must be a tuple of length 1 or 2")
    if any(not isinstance(item, RankState) for item in value):
        raise TypeError("rank_states must contain only RankState values")
    expected = tuple(range(len(value)))
    if tuple(item.rank for item in value) != expected:
        raise ValueError("rank_states must be rank-sorted and cover 0..world_size-1")
    cuda_presence = {item.torch_cuda is not None for item in value}
    if len(cuda_presence) != 1:
        raise ValueError("all rank states must use the same CPU/CUDA RNG topology")
    return value


def _rank_states_from_payload(
    value: Any,
    *,
    expected_world_size: int,
) -> tuple[RankState, ...]:
    if type(value) is not tuple or len(value) != expected_world_size:
        raise ResumeError("rank_states tuple length does not match world_size")
    states: list[RankState] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != set(_RANK_STATE_KEYS):
            raise ResumeError("rank state has an invalid exact key set")
        rng = entry["rng"]
        if not isinstance(rng, dict) or set(rng) != set(_RNG_KEYS):
            raise ResumeError("rank RNG has an invalid exact key set")
        numpy_rng = rng["numpy"]
        if not isinstance(numpy_rng, dict) or set(numpy_rng) != set(_NUMPY_RNG_KEYS):
            raise ResumeError("NumPy RNG state has an invalid exact key set")
        words = numpy_rng["state"]
        if type(words) is not list:
            raise ResumeError("NumPy RNG state words must be a list")
        try:
            state = RankState(
                rank=entry["rank"],
                python_state=rng["python"],
                numpy_bit_generator=numpy_rng["bit_generator"],
                numpy_state=tuple(words),
                numpy_position=numpy_rng["position"],
                numpy_has_gauss=numpy_rng["has_gauss"],
                numpy_cached_gaussian=numpy_rng["cached_gaussian"],
                torch_cpu=rng["torch_cpu"],
                torch_cuda=rng["torch_cuda"],
            )
        except (TypeError, ValueError) as exc:
            raise ResumeError("rank RNG state is invalid") from exc
        states.append(state)
    try:
        return _validate_rank_states(tuple(states))
    except (TypeError, ValueError) as exc:
        raise ResumeError("rank_states ordering/topology is invalid") from exc


def _validated_rng_tensor(value: Any, *, label: str) -> Any:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() < 1
        or not value.is_contiguous()
    ):
        raise ValueError(f"{label} must be contiguous CPU uint8 with shape [N]")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{label} must be detached")
    return value


def _validate_device(value: Any) -> Any:
    import torch

    if not isinstance(value, torch.device) or value.type not in {"cpu", "cuda"}:
        raise TypeError("writer_device must be a CPU or CUDA torch.device")
    if value.type == "cuda" and value.index is None:
        raise ValueError("CUDA writer_device must have an explicit index")
    return value


def _validate_rank_device_topology(
    states: tuple[RankState, ...],
    device: Any,
) -> None:
    device = _validate_device(device)
    has_cuda = all(item.torch_cuda is not None for item in states)
    if device.type == "cpu" and has_cuda:
        raise ResumeError("CUDA RNG checkpoint cannot be used by a CPU runtime")
    if device.type == "cuda" and not has_cuda:
        raise ResumeError("CPU RNG checkpoint cannot be used by a CUDA runtime")
    import torch

    expected_cpu_words = torch.get_rng_state().numel()
    if any(item.torch_cpu.numel() != expected_cpu_words for item in states):
        raise ResumeError("Torch CPU RNG state length is incompatible")
    if device.type == "cuda":
        expected_cuda_words = torch.cuda.get_rng_state(device=device).numel()
        if any(
            item.torch_cuda is None
            or item.torch_cuda.numel() != expected_cuda_words
            for item in states
        ):
            raise ResumeError("Torch CUDA RNG state length is incompatible")


def _validate_live_topology(
    adapter: Any,
    optimizer: Any,
) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    import torch

    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("optimizer must be torch.optim.AdamW")
    named = adapter.named_parameters()
    if type(named) is not tuple or not named:
        raise ValueError("adapter.named_parameters() must return a non-empty tuple")
    names = tuple(item[0] for item in named)
    parameters = tuple(item[1] for item in named)
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or len({id(parameter) for parameter in parameters}) != len(parameters)
    ):
        raise ValueError("adapter trainable parameter names/identities are invalid")
    if len(optimizer.param_groups) != 1:
        raise ValueError("v0.7 AdamW must contain exactly one parameter group")
    if tuple(optimizer.param_groups[0]["params"]) != parameters:
        raise ValueError(
            "AdamW parameter identity/order must match adapter.named_parameters()"
        )
    if any(parameter.device != parameters[0].device for parameter in parameters):
        raise ValueError("all trainable Adapter parameters must share one device")
    return names, parameters


def _audit_optimizer_topology(value: Any) -> tuple[str, ...]:
    if not isinstance(value, dict) or set(value) != set(_TOPOLOGY_KEYS):
        raise ResumeError("optimizer_topology has an invalid exact key set")
    names = value["parameter_names"]
    roles = value["group_roles"]
    if (
        type(names) is not tuple
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ResumeError("optimizer parameter_names are invalid")
    if type(roles) is not tuple or roles != _GROUP_ROLES:
        raise ResumeError("optimizer group_roles must be ('trainable_adapter',)")
    return names


def _audit_optimizer_payload(
    value: Any,
    *,
    parameter_names: tuple[str, ...],
) -> None:
    import torch

    if not isinstance(value, dict) or set(value) != set(_OPTIMIZER_KEYS):
        raise ResumeError("optimizer state has an invalid exact key set")
    state = value["state"]
    groups = value["param_groups"]
    if not isinstance(state, dict) or type(groups) is not list or len(groups) != 1:
        raise ResumeError("optimizer must contain one state mapping and one group")
    group = groups[0]
    if not isinstance(group, dict):
        raise ResumeError("optimizer parameter group must be a mapping")
    ids = group.get("params")
    if type(ids) is not list or ids != list(range(len(parameter_names))):
        raise ResumeError("optimizer parameter ids must be canonical 0..N-1")
    if any(type(item) is not int for item in state) or set(state).difference(ids):
        raise ResumeError("optimizer state references an unknown parameter id")
    for parameter_id, entry in state.items():
        if not isinstance(entry, dict):
            raise ResumeError("optimizer per-parameter state must be a mapping")
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        if bool(group.get("amsgrad", False)):
            expected_keys.add("max_exp_avg_sq")
        if set(entry) != expected_keys:
            raise ResumeError("AdamW per-parameter state has incompatible keys")
        step = entry["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.dtype != torch.float32
            or step.ndim != 0
            or not step.is_contiguous()
            or not bool(torch.isfinite(step))
            or float(step) < 0
        ):
            raise ResumeError("AdamW step must be a finite non-negative CPU scalar")
        moments = tuple(
            entry[name] for name in sorted(expected_keys.difference({"step"}))
        )
        first = moments[0]
        if (
            not isinstance(first, torch.Tensor)
            or first.device.type != "cpu"
            or not first.is_contiguous()
            or not bool(torch.isfinite(first).all())
        ):
            raise ResumeError("AdamW moment tensor is invalid")
        if any(
            not isinstance(moment, torch.Tensor)
            or moment.device.type != "cpu"
            or moment.shape != first.shape
            or moment.dtype != first.dtype
            or not moment.is_contiguous()
            or not bool(torch.isfinite(moment).all())
            for moment in moments[1:]
        ):
            raise ResumeError(
                f"AdamW moment tensors disagree for parameter {parameter_id}"
            )
    for name, item in group.items():
        if name == "params":
            continue
        if isinstance(item, float) and not math.isfinite(item):
            raise ResumeError(
                f"optimizer parameter-group field {name!r} must be finite"
            )
        if isinstance(item, tuple) and any(
            isinstance(part, float) and not math.isfinite(part) for part in item
        ):
            raise ResumeError(
                f"optimizer parameter-group field {name!r} must be finite"
            )


def _audit_scaler_payload(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ResumeError("checkpoint GradScaler state must be a mapping or None")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ResumeError("checkpoint GradScaler keys must be non-empty strings")
        if isinstance(item, float) and not math.isfinite(item):
            raise ResumeError(
                f"checkpoint GradScaler field {key!r} must be finite"
            )


def _normalized_optimizer_state(
    value: Any,
    *,
    parameter_names: tuple[str, ...],
    adapter: Any,
    live_optimizer: Any,
) -> dict[str, Any]:
    normalized = _clone_safe_value(value)
    _names, parameters = _validate_live_topology(adapter, live_optimizer)
    _validate_optimizer_payload(
        normalized,
        parameter_names=parameter_names,
        parameters=parameters,
        live_optimizer=live_optimizer,
    )
    return normalized


def _validate_optimizer_payload(
    value: Any,
    *,
    parameter_names: tuple[str, ...],
    parameters: tuple[Any, ...],
    live_optimizer: Any,
) -> None:
    import torch

    if not isinstance(value, dict) or set(value) != set(_OPTIMIZER_KEYS):
        raise ResumeError("optimizer state has an invalid exact key set")
    state = value["state"]
    groups = value["param_groups"]
    if not isinstance(state, dict) or type(groups) is not list or len(groups) != 1:
        raise ResumeError("optimizer must contain one state mapping and one group")
    group = groups[0]
    live_group = live_optimizer.state_dict()["param_groups"][0]
    if not isinstance(group, dict) or set(group) != set(live_group):
        raise ResumeError("optimizer parameter-group key set is incompatible")
    ids = group["params"]
    if type(ids) is not list or len(ids) != len(parameter_names):
        raise ResumeError("optimizer parameter ids do not match Adapter topology")
    if ids != list(range(len(parameter_names))):
        raise ResumeError("optimizer parameter ids must be canonical 0..N-1")
    if any(type(item) is not int for item in state):
        raise ResumeError("optimizer state keys must be non-bool integers")
    if set(state).difference(ids):
        raise ResumeError("optimizer state references an unknown parameter id")
    mutable_hyperparameters = {"lr", "betas", "eps", "weight_decay", "params"}
    for key in set(group).difference(mutable_hyperparameters):
        if group[key] != live_group[key]:
            raise ResumeError(
                f"optimizer fixed parameter-group field {key!r} is incompatible"
            )
    for index, parameter_id in enumerate(ids):
        entry = state.get(parameter_id)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ResumeError("optimizer per-parameter state must be a mapping")
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        if bool(group.get("amsgrad", False)):
            expected_keys.add("max_exp_avg_sq")
        if set(entry) != expected_keys:
            raise ResumeError("AdamW per-parameter state has incompatible keys")
        step = entry["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.dtype != torch.float32
            or step.ndim != 0
            or not step.is_contiguous()
            or not bool(torch.isfinite(step))
            or float(step) < 0
        ):
            raise ResumeError("AdamW step must be a finite non-negative CPU scalar")
        for name in expected_keys.difference({"step"}):
            tensor = entry[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tuple(tensor.shape) != tuple(parameters[index].shape)
                or tensor.dtype != parameters[index].dtype
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ResumeError(
                    f"AdamW {name} tensor is incompatible with parameter "
                    f"{parameter_names[index]!r}"
                )


def _normalized_scaler_state(scaler: Any | None) -> dict[str, Any] | None:
    if scaler is None:
        return None
    state = _clone_safe_value(scaler.state_dict())
    _validate_scaler_payload(state, scaler)
    return state


def _validate_scaler_payload(value: Any, scaler: Any | None) -> None:
    if scaler is None:
        if value is not None:
            raise ResumeError("checkpoint GradScaler topology does not match runtime")
        return
    if not isinstance(value, dict):
        raise ResumeError("checkpoint GradScaler state must be a mapping")
    live = scaler.state_dict()
    if set(value) != set(live):
        raise ResumeError("checkpoint GradScaler key set does not match runtime")
    for key, item in value.items():
        expected = live[key]
        if type(item) is not type(expected):
            raise ResumeError(f"checkpoint GradScaler field {key!r} has wrong type")
        if isinstance(item, float) and not math.isfinite(item):
            raise ResumeError(f"checkpoint GradScaler field {key!r} must be finite")


def _apply_current_optimizer_config(
    optimizer: Any,
    config: OptimizerConfig,
) -> None:
    if len(optimizer.param_groups) != 1:
        raise ResumeError("AdamW group topology changed while applying checkpoint")
    group = optimizer.param_groups[0]
    group["lr"] = config.learning_rate
    group["betas"] = (config.adam_beta1, config.adam_beta2)
    group["weight_decay"] = config.adam_weight_decay
    group["eps"] = config.adam_epsilon


def _capture_rank_rng(*, rank: int, device: Any) -> RankState:
    import torch

    device = _validate_device(device)
    cuda = (
        None
        if device.type == "cpu"
        else torch.cuda.get_rng_state(device=device).cpu().contiguous()
    )
    return RankState.from_rng(
        rank=rank,
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu=torch.get_rng_state().cpu().contiguous(),
        torch_cuda=cuda,
    )


def _restore_rank_rng(rank_state: RankState, *, device: Any) -> None:
    import torch

    if not isinstance(rank_state, RankState):
        raise TypeError("rank_state must be a RankState")
    device = _validate_device(device)
    _validate_rank_device_topology((rank_state,), device)
    random.setstate(rank_state.python_state)
    np.random.set_state(
        (
            rank_state.numpy_bit_generator,
            np.asarray(rank_state.numpy_state, dtype=np.uint32),
            rank_state.numpy_position,
            rank_state.numpy_has_gauss,
            float(rank_state.numpy_cached_gaussian),
        )
    )
    torch.set_rng_state(rank_state.torch_cpu)
    if device.type == "cuda":
        assert rank_state.torch_cuda is not None
        torch.cuda.set_rng_state(rank_state.torch_cuda, device=device)


def _prepare_checkpoint_target(path: Path) -> Path:
    absolute = path.absolute()
    if absolute.exists() or absolute.is_symlink():
        if absolute.is_symlink() or not absolute.is_dir():
            raise ValueError("checkpoint_dir must be a real directory")
        if any(absolute.iterdir()):
            raise ValueError("checkpoint_dir must be empty before save")
    else:
        parent = _safe_directory(absolute.parent, label="checkpoint parent")
        absolute = parent / absolute.name
        absolute.mkdir()
    return _safe_directory(absolute, label="checkpoint")


def _require_exact_checkpoint_root(root: Path) -> None:
    root = _safe_directory(root, label="checkpoint")
    children = tuple(root.iterdir())
    if {item.name for item in children} != set(_CHECKPOINT_ROOT_ENTRIES):
        raise ResumeError(
            "v5 checkpoint must contain exactly adapter/, training_state.pt "
            "and checkpoint.json",
            path=str(root),
        )
    for item in children:
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ResumeError("checkpoint root entries must not be symlinks")
        if item.name == "adapter":
            if not stat.S_ISDIR(metadata.st_mode):
                raise ResumeError("checkpoint adapter entry must be a directory")
        elif not stat.S_ISREG(metadata.st_mode):
            raise ResumeError("checkpoint control entries must be regular files")


def _safe_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} must be a real directory: {path}")
    resolved = path.resolve(strict=True)
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} path must not contain symlinks: {current}")
    return resolved


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {path}")


def _save_torch_payload(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(16)}"
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(16)}"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    _require_regular_file(path, label="checkpoint file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree(root: Path) -> None:
    directories = [root, *(item for item in root.rglob("*") if item.is_dir())]
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clone_safe_value(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) and type(key) is not int:
                raise TypeError("checkpoint mapping keys must be strings or integers")
            result[key] = _clone_safe_value(item)
        return result
    if isinstance(value, list):
        return [_clone_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_safe_value(item) for item in value)
    if isinstance(value, (str, bool, int, type(None))):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint values must be finite")
        return value
    raise TypeError(f"unsupported checkpoint value type: {type(value).__name__}")


def _validate_safe_value(value: Any, *, label: str) -> None:
    try:
        _clone_safe_value(value)
    except (TypeError, ValueError) as exc:
        raise ResumeError(f"{label} contains an unsafe value") from exc
