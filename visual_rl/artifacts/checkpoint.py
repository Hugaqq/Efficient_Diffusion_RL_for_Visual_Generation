"""Atomic metadata and complete training-state checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import json
import math
import os
import random
import secrets
import shutil
import stat
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from visual_rl.artifacts.serialization import redact_artifact_config, to_jsonable

CHECKPOINT_FORMAT_VERSION = 4
CONFIG_FINGERPRINT_VERSION = 2
_LEGACY_CHECKPOINT_FORMAT_VERSIONS = {1, 2}
_PRIOR_SAFE_CHECKPOINT_FORMAT_VERSIONS = {3}
_SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = {
    *_LEGACY_CHECKPOINT_FORMAT_VERSIONS,
    *_PRIOR_SAFE_CHECKPOINT_FORMAT_VERSIONS,
    CHECKPOINT_FORMAT_VERSION,
}
_SAFE_CHECKPOINT_FORMAT_VERSIONS = {
    *_PRIOR_SAFE_CHECKPOINT_FORMAT_VERSIONS,
    CHECKPOINT_FORMAT_VERSION,
}
_HASHED_CHECKPOINT_FORMAT_VERSIONS = {
    2,
    *_PRIOR_SAFE_CHECKPOINT_FORMAT_VERSIONS,
    CHECKPOINT_FORMAT_VERSION,
}
_TRAINING_STATE_HASHED_CHECKPOINT_FORMAT_VERSIONS = {CHECKPOINT_FORMAT_VERSION}
_DISTRIBUTED_CHECKPOINT_FORMAT_VERSIONS = _SAFE_CHECKPOINT_FORMAT_VERSIONS
_SUPPORTED_CONFIG_FINGERPRINT_VERSIONS = {1, CONFIG_FINGERPRINT_VERSION}
_CHECKPOINT_CONTROL_FILES = {"checkpoint.json", "training_state.pt"}
_ADAPTER_HASH_CHUNK_SIZE = 1024 * 1024
_PRIOR_SAFE_CONFIG_FINGERPRINT_SCHEMES = frozenset({"component-sha256-v1"})
_SAFE_CONFIG_FINGERPRINT_SCHEME = "component-sha256-v2"
_SUPPORTED_SAFE_CONFIG_FINGERPRINT_SCHEMES = frozenset(
    {*_PRIOR_SAFE_CONFIG_FINGERPRINT_SCHEMES, _SAFE_CONFIG_FINGERPRINT_SCHEME}
)
_SEMANTIC_CONFIG_KEYS = {
    "seed",
    "use_lora",
    "per_prompt_stat_tracking",
    "model",
    "sample",
    "rollout",
    "algorithm",
    "rewards",
    "optimizer",
    "train",
    "paths",
}


@dataclass(frozen=True)
class ValidatedTrainingState:
    """A checkpoint state that has passed all side-effect-free validation."""

    checkpoint_dir: Path
    state: dict[str, Any]
    metadata: dict[str, Any]
    applicable: bool = True

    @property
    def step(self) -> int:
        return int(self.state["step"])


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path, parent_fd = _open_safe_json_parent(Path(path))
    temp_name: str | None = None
    temp_fd: int | None = None
    try:
        _validate_file_target(parent_fd, path.name, label="JSON target")
        temp_name, temp_fd = _create_json_temp(parent_fd, path.name)
        handle = os.fdopen(temp_fd, "w", encoding="utf-8")
        temp_fd = None
        with handle:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_file_target(parent_fd, path.name, label="JSON target")
        os.replace(
            temp_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temp_name = None
        os.fsync(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _open_safe_json_parent(path: Path) -> tuple[Path, int]:
    """Create and open a parent path without following symlink components."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("Secure JSON writes require O_NOFOLLOW and O_DIRECTORY")
    absolute = path.absolute()
    if not absolute.name or absolute.name in {".", ".."}:
        raise ValueError(f"JSON target must name a file: {path}")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parent.parts[1:]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=current_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return absolute, current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _validate_file_target(parent_fd: int, name: str, *, label: str) -> None:
    try:
        target_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(target_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular file, not a symlink: {name}")


def _create_json_temp(parent_fd: int, target_name: str) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    prefix = f".{target_name[:80]}.tmp-"
    for _ in range(128):
        name = f"{prefix}{secrets.token_hex(16)}"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate a unique temporary file for {target_name}")


def _save_training_state_payload(
    torch: Any,
    path: Path,
    state: dict[str, Any],
) -> None:
    path, parent_fd = _open_safe_json_parent(path)
    temp_name: str | None = None
    temp_fd: int | None = None
    try:
        _validate_file_target(parent_fd, path.name, label="Training state target")
        temp_name, temp_fd = _create_json_temp(parent_fd, path.name)
        handle = os.fdopen(temp_fd, "wb")
        temp_fd = None
        with handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_file_target(parent_fd, path.name, label="Training state target")
        os.replace(
            temp_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temp_name = None
        os.fsync(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number is not allowed: {value}")
    return parsed


def strict_json_loads(value: str) -> Any:
    """Parse RFC-style JSON while rejecting duplicate keys and non-finite values."""

    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Secure JSON reads require O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"JSON document must be a regular file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
                parse_float=_parse_finite_json_float,
            )
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return value


def config_fingerprint(
    config: dict[str, Any],
    implementation: dict[str, Any] | None = None,
    *,
    version: int = CONFIG_FINGERPRINT_VERSION,
) -> str:
    """Hash checkpoint compatibility semantics for the requested version.

    The public helper is side-effect free and trusts declared content hashes.
    Checkpoint save/load paths additionally validate file-backed data before
    using the v2 identity.
    """

    if int(version) == 1:
        return _payload_sha256(_legacy_config_payload(config, implementation))
    if int(version) != CONFIG_FINGERPRINT_VERSION:
        raise ValueError(f"Unsupported config fingerprint version: {version}")
    return _build_v2_fingerprint_bundle(
        config,
        implementation,
        validate_data=False,
    )["config_fingerprint"]


def _legacy_config_payload(
    config: dict[str, Any],
    implementation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reproduce the original path-sensitive v1 fingerprint payload."""

    source = deepcopy(to_jsonable(config))
    semantic_keys = {
        "seed",
        "use_lora",
        "per_prompt_stat_tracking",
        "model",
        "dataset",
        "evaluation",
        "sample",
        "rollout",
        "algorithm",
        "rewards",
        "optimizer",
        "train",
        "paths",
    }
    payload = {key: source[key] for key in semantic_keys if key in source}
    paths = payload.get("paths", {})
    for key in ("output_dir", "resume_from"):
        paths.pop(key, None)
    train = payload.get("train", {})
    for key in ("max_steps", "save_every"):
        train.pop(key, None)
    payload.pop("runner", None)
    if implementation is not None:
        payload["implementation"] = to_jsonable(implementation)
    return payload


def _build_v2_fingerprint_bundle(
    config: dict[str, Any],
    implementation: dict[str, Any] | None,
    *,
    validate_data: bool,
    fingerprint_scheme: str | None = _SAFE_CONFIG_FINGERPRINT_SCHEME,
) -> dict[str, Any]:
    if fingerprint_scheme is not None and (
        not isinstance(fingerprint_scheme, str)
        or fingerprint_scheme not in _SUPPORTED_SAFE_CONFIG_FINGERPRINT_SCHEMES
    ):
        raise ValueError(f"Unsupported config fingerprint scheme: {fingerprint_scheme}")
    source = deepcopy(to_jsonable(config))
    training_semantics = _training_semantics_payload(
        source,
        include_reward_batch_partition=(
            fingerprint_scheme == _SAFE_CONFIG_FINGERPRINT_SCHEME
        ),
    )
    data_identity, data_source = _data_identity_payload(
        source,
        validate_data=validate_data,
    )
    implementation_identity = to_jsonable(implementation or {})
    raw_identity_payload = {
        "training_semantics": training_semantics,
        "data_identity": data_identity,
        "implementation": implementation_identity,
    }
    component_fingerprints = {
        "training_semantics_fingerprint": _payload_sha256(training_semantics),
        "data_identity_fingerprint": _payload_sha256(data_identity),
        "implementation_identity_fingerprint": _payload_sha256(implementation_identity),
    }
    if fingerprint_scheme in _SUPPORTED_SAFE_CONFIG_FINGERPRINT_SCHEMES:
        fingerprint = _payload_sha256(component_fingerprints)
    elif fingerprint_scheme is None:
        fingerprint = _payload_sha256(raw_identity_payload)
    else:
        raise ValueError(f"Unsupported config fingerprint scheme: {fingerprint_scheme}")

    persisted_identity = redact_artifact_config(raw_identity_payload)
    bundle = {
        "config_fingerprint_version": CONFIG_FINGERPRINT_VERSION,
        "config_fingerprint": fingerprint,
        **component_fingerprints,
        "identity_payload": persisted_identity,
        "data_identity": persisted_identity["data_identity"],
        "data_source": redact_artifact_config(data_source),
    }
    if fingerprint_scheme is not None:
        bundle["config_fingerprint_scheme"] = fingerprint_scheme
    return bundle


def _training_semantics_payload(
    source: dict[str, Any],
    *,
    include_reward_batch_partition: bool = False,
) -> dict[str, Any]:
    payload = {
        key: deepcopy(source[key]) for key in _SEMANTIC_CONFIG_KEYS if key in source
    }
    if include_reward_batch_partition:
        _normalize_world_r1_reward_defaults(payload)
    paths = payload.get("paths", {})
    for key in ("output_dir", "resume_from"):
        paths.pop(key, None)
    train = payload.get("train", {})
    for key in ("max_steps", "save_every"):
        train.pop(key, None)
    runner = source.get("runner") or {}
    reward_executor = runner.get("reward_executor") or {}
    if (
        include_reward_batch_partition
        and reward_executor.get("mode") == "async"
        and reward_executor.get("microbatch_size") is not None
    ):
        payload["reward_batch_partition"] = {
            "microbatch_size": reward_executor["microbatch_size"]
        }
    return payload


def _normalize_world_r1_reward_defaults(payload: dict[str, Any]) -> None:
    rewards = payload.get("rewards")
    if not isinstance(rewards, dict):
        return
    clients = rewards.get("clients")
    if not isinstance(clients, dict):
        return
    for key, client in clients.items():
        if not isinstance(client, dict):
            continue
        name = client.get("name", key)
        if name == "reward_general":
            default_batch_size = 64
        elif name == "reward_3d":
            default_batch_size = 8
        else:
            continue
        if client.get("protocol_mode") == "reference_v1":
            client.pop("protocol_mode")
        if client.get("batch_size") == default_batch_size:
            client.pop("batch_size")
        if client.get("server_revision", object()) is None:
            client.pop("server_revision")


def _data_identity_payload(
    source: dict[str, Any],
    *,
    validate_data: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = deepcopy(source.get("dataset") or {})
    evaluation = deepcopy(source.get("evaluation") or {})

    # ``empty_prompt_policy`` was added after v2 fingerprints shipped.  Its
    # default reproduces the old strict behavior, so omit only that exact value
    # to keep existing checkpoints resumable.  Non-default policies remain part
    # of the data identity and therefore cannot silently cross a resume boundary.
    if dataset.get("empty_prompt_policy") == "error":
        dataset.pop("empty_prompt_policy")

    train_path = dataset.pop("path", None)
    train_prompts = dataset.pop("prompts", None)
    train_declared_hash = dataset.pop("content_sha256", None)
    train_hash = _resolve_prompt_content_hash(
        label="dataset",
        path=train_path,
        prompts=train_prompts,
        declared_hash=train_declared_hash,
        validate_data=validate_data,
    )
    dataset["content_sha256"] = train_hash

    evaluation_path = evaluation.pop("path", None)
    evaluation_declared_hash = evaluation.pop("content_sha256", None)
    evaluation_hash = _resolve_prompt_content_hash(
        label="evaluation",
        path=evaluation_path,
        prompts=None,
        declared_hash=evaluation_declared_hash,
        validate_data=validate_data,
    )
    evaluation["content_sha256"] = evaluation_hash

    return (
        {
            "train": dataset,
            "evaluation": evaluation,
        },
        {
            "train_path": train_path,
            "evaluation_path": evaluation_path,
        },
    )


def _resolve_prompt_content_hash(
    *,
    label: str,
    path: str | None,
    prompts: list[Any] | None,
    declared_hash: str | None,
    validate_data: bool,
) -> str | None:
    from visual_rl.datasets.prompt_dataset import (
        prompt_content_sha256,
        read_prompt_file,
    )

    if path and prompts:
        raise RuntimeError(
            f"Cannot fingerprint {label}: both path and inline prompts are set"
        )

    actual_hash: str | None = None
    if prompts:
        actual_hash = prompt_content_sha256(prompts)
    elif path and validate_data:
        try:
            actual_hash = prompt_content_sha256(read_prompt_file(path))
        except OSError as exc:
            raise RuntimeError(
                f"Cannot validate {label} data source {path!r}: {exc}"
            ) from exc

    expected = None if declared_hash is None else str(declared_hash)
    if expected and actual_hash and actual_hash != expected:
        raise RuntimeError(
            f"{label} content SHA256 mismatch: actual {actual_hash} != "
            f"declared {expected}"
        )
    return actual_hash or expected


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        to_jsonable(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_implementation_identity(
    adapter: Any,
    plugin: Any,
    *,
    rollout: Any | None = None,
    feedback: Any | None = None,
) -> dict[str, Any]:
    """Describe the code and trainable parameter contract used by a checkpoint."""

    algorithm = getattr(plugin, "algorithm", None)
    return {
        "adapter": _object_identity(adapter),
        "optimizer_plugin": _object_identity(plugin),
        "algorithm": _object_identity(algorithm) if algorithm is not None else None,
        "advantage": _object_identity(getattr(plugin, "advantage_computer", None)),
        "rollout": _object_identity(rollout),
        "feedback": _object_identity(feedback),
        "trainable_parameters": _parameter_signature(adapter),
        "git_commit": _git_commit(),
        "git_diff_sha256": _git_diff_hash(),
        "runtime_tree_sha256": _runtime_tree_hash(),
        "reference_patch_sha256": _reference_patch_hash(adapter),
    }


def adapter_payload_sha256(checkpoint_dir: str | Path) -> str:
    """Hash adapter files by normalized relative path and streamed contents."""

    root = validated_checkpoint_directory(
        checkpoint_dir,
        checkpoint_dir,
        label="checkpoint root",
    )
    files = sorted(
        (
            path
            for path in _validated_checkpoint_tree_files(root)
            if path.is_file()
            and not _is_checkpoint_control_file(path.relative_to(root))
            and not _is_temporary_checkpoint_path(path.relative_to(root))
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise RuntimeError(f"Checkpoint adapter payload is empty: {root}")

    digest = hashlib.sha256(b"visual-rl-adapter-payload-tree-v1\0")
    for path in files:
        path = validated_checkpoint_file(root, path, label="adapter payload file")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(16, "big"))
        bytes_read = 0
        with path.open("rb") as handle:
            while chunk := handle.read(_ADAPTER_HASH_CHUNK_SIZE):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != size:
            raise RuntimeError(f"Adapter payload changed while hashing: {path}")
    return digest.hexdigest()


def _file_sha256(path: Path, *, label: str) -> str:
    digest = hashlib.sha256()
    expected_size = path.stat().st_size
    bytes_read = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_ADAPTER_HASH_CHUNK_SIZE):
            digest.update(chunk)
            bytes_read += len(chunk)
    if bytes_read != expected_size or path.stat().st_size != expected_size:
        raise RuntimeError(f"{label} changed while hashing: {path}")
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA256 digest")
    return value


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validated_checkpoint_directory(
    checkpoint_root: str | Path,
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve a real directory without following checkpoint symlinks."""

    root = Path(checkpoint_root)
    candidate = Path(path)
    try:
        root_stat = root.lstat()
        candidate_stat = candidate.lstat()
    except OSError as exc:
        raise RuntimeError(f"Missing {label}: {candidate}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or stat.S_ISLNK(candidate_stat.st_mode):
        raise RuntimeError(f"Checkpoint {label} must not be a symlink: {candidate}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot resolve checkpoint {label}: {candidate}") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
        raise RuntimeError(f"Checkpoint {label} must be a directory: {candidate}")
    if not _is_within(resolved_candidate, resolved_root):
        raise RuntimeError(
            f"Checkpoint {label} escapes checkpoint root {resolved_root}: {candidate}"
        )
    return resolved_candidate


def validated_checkpoint_file(
    checkpoint_root: str | Path,
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve a regular checkpoint file without following symlinks."""

    root = validated_checkpoint_directory(
        checkpoint_root,
        checkpoint_root,
        label="root",
    )
    candidate = Path(path)
    try:
        candidate_stat = candidate.lstat()
    except OSError as exc:
        raise RuntimeError(f"Missing checkpoint {label}: {candidate}") from exc
    if stat.S_ISLNK(candidate_stat.st_mode):
        raise RuntimeError(f"Checkpoint {label} must not be a symlink: {candidate}")
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot resolve checkpoint {label}: {candidate}") from exc
    if not stat.S_ISREG(candidate_stat.st_mode):
        raise RuntimeError(f"Checkpoint {label} must be a regular file: {candidate}")
    if not _is_within(resolved_candidate, root):
        raise RuntimeError(
            f"Checkpoint {label} escapes checkpoint root {root}: {candidate}"
        )
    return resolved_candidate


def _validated_checkpoint_tree_files(root: Path) -> list[Path]:
    files: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in directory.iterdir():
            relative = child.relative_to(root)
            if _is_temporary_checkpoint_path(relative):
                continue
            try:
                child_stat = child.lstat()
            except OSError as exc:
                raise RuntimeError(f"Cannot inspect checkpoint payload: {child}") from exc
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(
                    f"Checkpoint payload must not contain symlinks: {child}"
                )
            resolved_child = child.resolve(strict=True)
            if not _is_within(resolved_child, root):
                raise RuntimeError(
                    f"Checkpoint payload escapes checkpoint root {root}: {child}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                pending.append(resolved_child)
            elif stat.S_ISREG(child_stat.st_mode):
                files.append(resolved_child)
            else:
                raise RuntimeError(
                    f"Checkpoint payload contains a non-regular entry: {child}"
                )
    return files


def checkpoint_tree_sha256(
    checkpoint_dir: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> str:
    """Hash a complete checkpoint tree without following links or special files.

    The byte format intentionally matches the v1 artifact commit-marker digest so
    existing markers remain verifiable.  ``trusted_root`` additionally confines
    the checkpoint path before any traversal occurs.
    """

    root = _validated_training_state_directory(
        checkpoint_dir,
        trusted_root=trusted_root,
    )
    entries: list[tuple[str, Path, bool]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = list(directory.iterdir())
        except OSError as exc:
            raise RuntimeError(
                f"Cannot inspect checkpoint tree directory: {directory}"
            ) from exc
        for child in children:
            relative = child.relative_to(root).as_posix()
            try:
                child_stat = child.lstat()
            except OSError as exc:
                raise RuntimeError(f"Cannot inspect checkpoint tree: {child}") from exc
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(
                    f"Checkpoint tree must not contain symlinks: {child}"
                )
            if stat.S_ISDIR(child_stat.st_mode):
                resolved = validated_checkpoint_directory(
                    root,
                    child,
                    label="tree directory",
                )
                entries.append((relative, resolved, False))
                pending.append(resolved)
            elif stat.S_ISREG(child_stat.st_mode):
                resolved = validated_checkpoint_file(
                    root,
                    child,
                    label="tree file",
                )
                entries.append((relative, resolved, True))
            else:
                raise RuntimeError(
                    f"Checkpoint tree contains a non-regular entry: {child}"
                )

    digest = hashlib.sha256()
    for relative, path, is_file in sorted(entries, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if is_file:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if not hasattr(os, "O_NOFOLLOW"):
                raise RuntimeError("Secure checkpoint hashing requires O_NOFOLLOW")
            flags |= os.O_NOFOLLOW
            fd = os.open(path, flags)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError(
                        f"Checkpoint tree file is not a regular file: {path}"
                    )
                bytes_read = 0
                with os.fdopen(fd, "rb") as handle:
                    fd = -1
                    while chunk := handle.read(_ADAPTER_HASH_CHUNK_SIZE):
                        digest.update(chunk)
                        bytes_read += len(chunk)
                if bytes_read != before.st_size:
                    raise RuntimeError(
                        f"Checkpoint tree file changed while hashing: {path}"
                    )
            finally:
                if fd >= 0:
                    os.close(fd)
        digest.update(b"\0")
    return digest.hexdigest()


def save_training_state(
    checkpoint_dir: str | Path,
    *,
    optimizer: Any,
    plugin: Any,
    step: int,
    config: dict[str, Any],
    implementation: dict[str, Any] | None = None,
    config_fingerprint_version: int = CONFIG_FINGERPRINT_VERSION,
    distributed_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    adapter_payload_hash = adapter_payload_sha256(path)
    version = int(config_fingerprint_version)
    if version not in _SUPPORTED_CONFIG_FINGERPRINT_VERSIONS:
        raise ValueError(f"Unsupported config fingerprint version: {version}")
    if version == 1:
        _reject_ambiguous_legacy_reward_partition(
            config,
            fingerprint_scheme=None,
            read_only_audit=False,
        )
        fingerprint = config_fingerprint(
            config,
            implementation,
            version=1,
        )
        fingerprint_bundle: dict[str, Any] | None = None
    else:
        fingerprint_bundle = _build_v2_fingerprint_bundle(
            config,
            implementation,
            validate_data=True,
        )
        fingerprint = fingerprint_bundle["config_fingerprint"]
    optimizer_state = optimizer.state_dict()
    plugin_state = plugin.state_dict()
    _validate_optimizer_state(optimizer_state, safe=True)
    _validate_plugin_state(plugin_state, safe=True)
    normalized_distributed_state = None
    if distributed_state is not None:
        normalized_distributed_state = _validate_distributed_state(
            distributed_state,
            require_sorted=False,
            redact_runtime_secrets=True,
        )
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "optimizer": optimizer_state,
        "plugin": plugin_state,
        "rng": capture_rng_state(),
        "config_fingerprint": fingerprint,
        "implementation": redact_artifact_config(implementation or {}),
        "adapter_payload_sha256": adapter_payload_hash,
    }
    if normalized_distributed_state is not None:
        state["distributed_state"] = normalized_distributed_state
    if fingerprint_bundle is not None:
        state.update(fingerprint_bundle)
    target = path / "training_state.pt"
    _save_training_state_payload(torch, target, state)
    training_state_hash = _file_sha256(
        validated_checkpoint_file(
            path,
            target,
            label="training_state.pt",
        ),
        label="Checkpoint training state",
    )
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "config_fingerprint": fingerprint,
        "training_state": target.name,
        "training_state_sha256": training_state_hash,
        "adapter_payload_sha256": adapter_payload_hash,
    }
    if normalized_distributed_state is not None:
        metadata["distributed_state"] = {
            "world_size": normalized_distributed_state["world_size"],
            "backend": normalized_distributed_state["backend"],
        }
    if fingerprint_bundle is not None:
        metadata.update(
            {
                key: fingerprint_bundle[key]
                for key in (
                    "config_fingerprint_version",
                    "config_fingerprint_scheme",
                    "training_semantics_fingerprint",
                    "data_identity_fingerprint",
                    "implementation_identity_fingerprint",
                    "data_identity",
                    "data_source",
                )
            }
        )
    save_json(path / "checkpoint.json", metadata)
    return metadata


def load_training_state(
    checkpoint_dir: str | Path,
    *,
    optimizer: Any,
    plugin: Any,
    config: dict[str, Any],
    implementation: dict[str, Any] | None = None,
    allow_unsafe_legacy: bool = False,
    trusted_root: str | Path | None = None,
    expected_world_size: int | None = None,
    expected_rank: int | None = None,
) -> int:
    """Compatibility entrypoint that validates and then applies training state."""

    validated = read_and_validate_training_state(
        checkpoint_dir,
        config=config,
        implementation=implementation,
        allow_unsafe_legacy=allow_unsafe_legacy,
        trusted_root=trusted_root,
        expected_world_size=expected_world_size,
        expected_rank=expected_rank,
    )
    return apply_training_state(
        validated,
        optimizer=optimizer,
        plugin=plugin,
        rank=expected_rank,
    )


def migrate_legacy_checkpoint_to_v4(
    source_checkpoint: str | Path,
    destination_checkpoint: str | Path,
    *,
    config: dict[str, Any],
    trusted_root: str | Path,
    implementation: dict[str, Any] | None = None,
    destination_root: str | Path | None = None,
) -> dict[str, Any]:
    """Explicitly migrate a trusted v1/v2 checkpoint into safe format v4.

    This is intentionally non-in-place and requires a caller-declared trusted
    source root.  The only ``weights_only=False`` fallback remains inside the
    named legacy reader reached with ``allow_unsafe_legacy=True``.  Arbitrary
    legacy objects that cannot satisfy the v4 safe-value contract are rejected.
    """

    source = _validated_training_state_directory(
        source_checkpoint,
        trusted_root=trusted_root,
    )
    destination = Path(destination_checkpoint).absolute()
    try:
        resolved_destination = destination.resolve(strict=False)
        resolved_source = source.resolve(strict=True)
        if (
            resolved_destination == resolved_source
            or resolved_source in resolved_destination.parents
            or resolved_destination in resolved_source.parents
        ):
            raise ValueError("Legacy checkpoint migration must not run in place")
    except OSError as exc:
        raise RuntimeError(f"Cannot resolve migration destination: {destination}") from exc
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Legacy checkpoint migration destination already exists: {destination}"
        )

    destination_boundary = Path(destination_root or destination.parent)
    try:
        boundary = destination_boundary.resolve(strict=True)
        boundary_stat = destination_boundary.lstat()
    except OSError as exc:
        raise RuntimeError(
            "Legacy checkpoint migration destination root must already exist: "
            f"{destination_boundary}"
        ) from exc
    if stat.S_ISLNK(boundary_stat.st_mode) or not stat.S_ISDIR(boundary_stat.st_mode):
        raise RuntimeError(
            f"Legacy checkpoint destination root is not a safe directory: {boundary}"
        )
    try:
        relative_destination = destination.relative_to(boundary)
    except ValueError as exc:
        raise RuntimeError(
            f"Legacy checkpoint destination escapes destination root {boundary}: "
            f"{destination}"
        ) from exc
    if any(part in {"", ".", ".."} for part in relative_destination.parts):
        raise RuntimeError(
            f"Legacy checkpoint destination has an unsafe component: {destination}"
        )
    current = boundary
    for part in relative_destination.parts[:-1]:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        current_stat = current.lstat()
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(
            current_stat.st_mode
        ):
            raise RuntimeError(
                f"Legacy checkpoint destination parent is unsafe: {current}"
            )

    from visual_rl.scaling import build_scaling_trigger_decision

    runner_config = config.get("runner") or {}
    if not isinstance(runner_config, dict):
        raise TypeError("Migration config runner section must be a dictionary")
    conditional_scaling = runner_config.get("conditional_scaling") or {}
    scaling_decision = build_scaling_trigger_decision(conditional_scaling)
    scaling_path = destination.parent / "trigger_decision.json"
    if scaling_path.exists() or scaling_path.is_symlink():
        if load_json(scaling_path) != scaling_decision:
            raise RuntimeError(
                "Migration destination has a conflicting scaling trigger decision"
            )

    validated = read_and_validate_training_state(
        source,
        config=config,
        implementation=implementation,
        allow_unsafe_legacy=True,
        trusted_root=trusted_root,
    )
    source_version = int(validated.state["format_version"])
    if source_version not in _LEGACY_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError(
            "Legacy checkpoint migration accepts only format v1/v2 sources, "
            f"got v{source_version}"
        )
    if validated.state.get("distributed_state") is not None:
        raise RuntimeError(
            "Legacy distributed checkpoints require a dedicated audited migration"
        )

    destination.mkdir(mode=0o700)
    try:
        for source_file in _validated_checkpoint_tree_files(source):
            relative = source_file.relative_to(source)
            if _is_checkpoint_control_file(relative) or _is_temporary_checkpoint_path(
                relative
            ):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target, follow_symlinks=False)
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            fd = os.open(target, flags)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        adapter_hash = adapter_payload_sha256(destination)
        state = deepcopy(validated.state)
        state["format_version"] = CHECKPOINT_FORMAT_VERSION
        state["rng"] = _legacy_rng_to_safe(state["rng"])
        state["adapter_payload_sha256"] = adapter_hash
        state["implementation"] = redact_artifact_config(
            state.get("implementation") or implementation or {}
        )
        state["migrated_from_format_version"] = source_version
        _validate_optimizer_state(state.get("optimizer"), safe=True)
        _validate_plugin_state(state.get("plugin"), safe=True)
        _validate_rng_state(state.get("rng"), safe=True)
        _validate_safe_checkpoint_value(state, label="migrated training state")

        state_path = destination / "training_state.pt"
        import torch

        _save_training_state_payload(torch, state_path, state)
        training_state_hash = _file_sha256(
            validated_checkpoint_file(
                destination,
                state_path,
                label="training_state.pt",
            ),
            label="Migrated checkpoint training state",
        )
        metadata = deepcopy(validated.metadata)
        metadata.update(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "step": int(state["step"]),
                "config_fingerprint": state["config_fingerprint"],
                "training_state": "training_state.pt",
                "training_state_sha256": training_state_hash,
                "adapter_payload_sha256": adapter_hash,
                "migrated_from_format_version": source_version,
            }
        )
        save_json(destination / "checkpoint.json", metadata)
        _fsync_checkpoint_directories(destination)
        result = dict(metadata)
        result["checkpoint_tree_sha256"] = checkpoint_tree_sha256(
            destination,
            trusted_root=boundary,
        )
        save_json(scaling_path, scaling_decision)
        return result
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _legacy_rng_to_safe(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Legacy checkpoint RNG state must be a dictionary")
    migrated = deepcopy(value)
    numpy_state = migrated.get("numpy")
    if isinstance(numpy_state, tuple) and len(numpy_state) == 5:
        vector = numpy_state[1]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        migrated["numpy"] = {
            "bit_generator": str(numpy_state[0]),
            "state": [int(item) for item in vector],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        }
    _validate_rng_state(migrated, safe=True)
    return migrated


def _fsync_checkpoint_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(directory, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(root.parent, flags)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def read_and_validate_training_state(
    checkpoint_dir: str | Path,
    *,
    config: dict[str, Any],
    implementation: dict[str, Any] | None = None,
    allow_unsafe_legacy: bool = False,
    trusted_root: str | Path | None = None,
    expected_world_size: int | None = None,
    expected_rank: int | None = None,
    use_checkpoint_implementation_identity: bool = False,
) -> ValidatedTrainingState:
    """Read and validate a checkpoint without mutating optimizer, plugin, or RNG.

    ``use_checkpoint_implementation_identity`` is for read-only artifact audits:
    it checks a safe-format checkpoint against its integrity-protected persisted
    identity without claiming that identity matches the current runtime code.
    Runtime resume callers must keep the default and supply their live identity.
    """

    import torch

    checkpoint_dir = _validated_training_state_directory(
        checkpoint_dir,
        trusted_root=trusted_root,
    )
    path = validated_checkpoint_file(
        checkpoint_dir,
        checkpoint_dir / "training_state.pt",
        label="training_state.pt",
    )
    metadata_path = validated_checkpoint_file(
        checkpoint_dir,
        checkpoint_dir / "checkpoint.json",
        label="checkpoint.json",
    )
    _validated_checkpoint_tree_files(checkpoint_dir)

    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise RuntimeError("Checkpoint metadata must be a JSON object")
    metadata_required = {
        "format_version",
        "step",
        "config_fingerprint",
        "training_state",
    }
    missing_metadata = sorted(metadata_required.difference(metadata))
    if missing_metadata:
        raise RuntimeError(f"Checkpoint metadata is missing keys: {missing_metadata}")
    if metadata.get("training_state") != path.name:
        raise RuntimeError(
            "checkpoint.json training_state does not reference training_state.pt"
        )
    metadata_format = _checkpoint_format_version(
        metadata.get("format_version"),
        label="checkpoint metadata",
    )
    if metadata_format in _TRAINING_STATE_HASHED_CHECKPOINT_FORMAT_VERSIONS:
        expected_training_state_hash = _required_sha256(
            metadata.get("training_state_sha256"),
            label=(
                f"Checkpoint format v{metadata_format} metadata "
                "training_state_sha256"
            ),
        )
        actual_training_state_hash = _file_sha256(
            path,
            label="Checkpoint training state",
        )
        if not secrets.compare_digest(
            actual_training_state_hash,
            expected_training_state_hash,
        ):
            raise RuntimeError(
                "Checkpoint training_state.pt SHA256 mismatch: "
                f"actual {actual_training_state_hash}, metadata "
                f"{expected_training_state_hash}"
            )
    actual_adapter_hash: str | None = None
    if metadata_format in _HASHED_CHECKPOINT_FORMAT_VERSIONS:
        expected_adapter_hash = metadata.get("adapter_payload_sha256")
        if not expected_adapter_hash:
            raise RuntimeError(
                f"Checkpoint format v{metadata_format} metadata is missing "
                "adapter_payload_sha256"
            )
        actual_adapter_hash = adapter_payload_sha256(checkpoint_dir)
        if actual_adapter_hash != expected_adapter_hash:
            raise RuntimeError(
                "Checkpoint adapter payload SHA256 mismatch: "
                f"actual {actual_adapter_hash}, metadata "
                f"{expected_adapter_hash}"
            )

    state = _load_training_state_payload(
        torch,
        path,
        metadata_format=metadata_format,
        allow_unsafe_legacy=allow_unsafe_legacy,
    )
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint training state must be a dictionary")
    required = {
        "format_version",
        "step",
        "optimizer",
        "plugin",
        "rng",
        "config_fingerprint",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(f"Checkpoint training state is missing keys: {missing}")
    state_format = _checkpoint_format_version(
        state.get("format_version"),
        label="checkpoint training state",
    )
    if state_format != metadata_format:
        raise RuntimeError(
            "checkpoint.json format_version does not match training_state.pt"
        )
    if int(metadata.get("step", -1)) != int(state["step"]):
        raise RuntimeError("checkpoint.json step does not match training_state.pt")
    if metadata.get("config_fingerprint") != state.get("config_fingerprint"):
        raise RuntimeError(
            "checkpoint.json fingerprint does not match training_state.pt"
        )
    _validate_optimizer_state(
        state.get("optimizer"),
        safe=state_format in _SAFE_CHECKPOINT_FORMAT_VERSIONS,
    )
    _validate_plugin_state(
        state.get("plugin"),
        safe=state_format in _SAFE_CHECKPOINT_FORMAT_VERSIONS,
    )
    _validate_rng_state(
        state.get("rng"),
        safe=state_format in _SAFE_CHECKPOINT_FORMAT_VERSIONS,
    )
    if state_format in _SAFE_CHECKPOINT_FORMAT_VERSIONS:
        _validate_safe_checkpoint_value(state, label="training state")
    distributed_state = _validate_checkpoint_distributed_state(
        state,
        metadata,
        state_format=state_format,
        expected_world_size=expected_world_size,
        expected_rank=expected_rank,
    )
    if distributed_state is not None:
        state["distributed_state"] = distributed_state
    if state_format in _HASHED_CHECKPOINT_FORMAT_VERSIONS:
        state_adapter_hash = state.get("adapter_payload_sha256")
        if not state_adapter_hash:
            raise RuntimeError(
                f"Checkpoint format v{state_format} training state is missing "
                "adapter_payload_sha256"
            )
        if state_adapter_hash != metadata.get("adapter_payload_sha256"):
            raise RuntimeError(
                "checkpoint.json adapter_payload_sha256 does not match "
                "training_state.pt"
            )
        if state_adapter_hash != actual_adapter_hash:
            raise RuntimeError(
                "Checkpoint adapter payload SHA256 does not match training_state.pt"
            )

    validation_implementation = implementation
    if use_checkpoint_implementation_identity:
        if implementation is not None:
            raise ValueError(
                "Cannot combine a live implementation identity with the "
                "checkpoint audit identity"
            )
        if state_format not in _SAFE_CHECKPOINT_FORMAT_VERSIONS:
            raise RuntimeError(
                "Checkpoint-provided implementation identity is only allowed for "
                "safe checkpoint formats"
            )
        identity_payload = state.get("identity_payload")
        persisted_implementation = (
            identity_payload.get("implementation")
            if isinstance(identity_payload, dict)
            else None
        )
        if not isinstance(persisted_implementation, dict):
            raise RuntimeError(
                "Checkpoint has no integrity-protected implementation identity"
            )
        validation_implementation = persisted_implementation

    state_version = int(state.get("config_fingerprint_version", 1))
    metadata_version = int(metadata.get("config_fingerprint_version", 1))
    if state_version != metadata_version:
        raise RuntimeError(
            "checkpoint.json config fingerprint version does not match "
            "training_state.pt"
        )
    if state_version not in _SUPPORTED_CONFIG_FINGERPRINT_VERSIONS:
        raise RuntimeError(f"Unsupported config fingerprint version: {state_version}")

    if state_version == 1:
        _reject_ambiguous_legacy_reward_partition(
            config,
            fingerprint_scheme=None,
            read_only_audit=use_checkpoint_implementation_identity,
        )
        expected = config_fingerprint(config, validation_implementation, version=1)
        actual = state.get("config_fingerprint")
        if actual != expected:
            raise RuntimeError(
                "Resume config does not match checkpoint training semantics. "
                "Resume rejected: checkpoint uses config fingerprint v1. "
                "Fingerprint mismatch: v1 binds absolute data paths and "
                "implementation identity. Reuse the original data paths and "
                "implementation, or use an audited migration workflow. "
                f"Checkpoint {actual}, current {expected}."
            )
    else:
        _validate_v2_checkpoint_metadata(state, metadata)
        fingerprint_scheme = state.get("config_fingerprint_scheme")
        _reject_ambiguous_legacy_reward_partition(
            config,
            fingerprint_scheme=fingerprint_scheme,
            read_only_audit=use_checkpoint_implementation_identity,
        )
        current = _build_v2_fingerprint_bundle(
            config,
            validation_implementation,
            validate_data=True,
            fingerprint_scheme=fingerprint_scheme,
        )
        actual = state.get("config_fingerprint")
        expected = current["config_fingerprint"]
        if actual != expected:
            differences = _identity_differences(
                state.get("identity_payload"),
                current["identity_payload"],
            )
            changed = ", ".join(differences[:8]) or "unknown identity field"
            if len(differences) > 8:
                changed += f", and {len(differences) - 8} more"
            raise RuntimeError(
                "Resume config does not match checkpoint training semantics. "
                "Resume rejected: checkpoint config fingerprint v2 mismatch. "
                f"Changed fields: {changed}. Checkpoint {actual}, current "
                f"{expected}."
            )
    return ValidatedTrainingState(
        checkpoint_dir=checkpoint_dir,
        state=state,
        metadata=metadata,
        applicable=not use_checkpoint_implementation_identity,
    )


def apply_training_state(
    validated: ValidatedTrainingState,
    *,
    optimizer: Any,
    plugin: Any,
    rank: int | None = None,
) -> int:
    """Apply optimizer, plugin, and RNG state after adapter restoration."""

    if not isinstance(validated, ValidatedTrainingState):
        raise TypeError("apply_training_state requires ValidatedTrainingState")
    if not validated.applicable:
        raise RuntimeError(
            "Cannot apply a checkpoint validated with read-only audit identity"
        )
    state = validated.state
    distributed_state = state.get("distributed_state")
    if distributed_state is None:
        if rank not in {None, 0}:
            raise RuntimeError(
                "Single-process checkpoint only contains rank 0 runtime state"
            )
        rng_state = state["rng"]
    else:
        world_size = int(distributed_state["world_size"])
        if rank is None:
            if world_size != 1:
                raise RuntimeError(
                    "Distributed checkpoint restore requires an explicit rank"
                )
            rank = 0
        _validate_expected_rank(rank, world_size=world_size)
        rng_state = next(
            entry["rng"]
            for entry in distributed_state["entries"]
            if entry["rank"] == rank
        )
    optimizer.load_state_dict(state["optimizer"])
    plugin.load_state_dict(dict(state.get("plugin") or {}))
    if int(state["format_version"]) in _SAFE_CHECKPOINT_FORMAT_VERSIONS:
        restore_rng_state(dict(rng_state))
    else:
        _restore_rng_state(dict(rng_state))
    return validated.step


def _checkpoint_format_version(value: Any, *, label: str) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label} format_version: {value!r}") from exc
    if version not in _SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise RuntimeError(f"Unsupported checkpoint format_version: {version}")
    return version


def _validated_training_state_directory(
    checkpoint_dir: str | Path,
    *,
    trusted_root: str | Path | None,
) -> Path:
    if trusted_root is None:
        return validated_checkpoint_directory(
            checkpoint_dir,
            checkpoint_dir,
            label="root",
        )

    trusted = validated_checkpoint_directory(
        trusted_root,
        trusted_root,
        label="trusted root",
    )
    candidate = validated_checkpoint_directory(
        trusted,
        checkpoint_dir,
        label="root",
    )
    _reject_symlinked_path_components(trusted, checkpoint_dir)
    return candidate


def _reject_symlinked_path_components(
    trusted_root: str | Path,
    checkpoint_dir: str | Path,
) -> None:
    root = Path(trusted_root).absolute()
    candidate = Path(checkpoint_dir).absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Checkpoint root escapes trusted root {root}: {candidate}"
        ) from exc

    current = root
    for part in relative.parts:
        current = current / part
        try:
            current_stat = current.lstat()
        except OSError as exc:
            raise RuntimeError(f"Missing checkpoint path component: {current}") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise RuntimeError(
                f"Checkpoint path must not contain symlinks: {current}"
            )


def _load_training_state_payload(
    torch: Any,
    path: Path,
    *,
    metadata_format: int,
    allow_unsafe_legacy: bool,
) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        if not (
            allow_unsafe_legacy
            and metadata_format in _LEGACY_CHECKPOINT_FORMAT_VERSIONS
        ):
            raise RuntimeError(
                "Safe checkpoint loading requires a PyTorch version whose "
                "torch.load supports weights_only=True"
            ) from None
    except Exception:
        if not (
            allow_unsafe_legacy
            and metadata_format in _LEGACY_CHECKPOINT_FORMAT_VERSIONS
        ):
            legacy_hint = (
                " Pass allow_unsafe_legacy=True only for a trusted, audited "
                "format v1/v2 checkpoint."
                if metadata_format in _LEGACY_CHECKPOINT_FORMAT_VERSIONS
                else ""
            )
            raise RuntimeError(
                "Checkpoint training state could not be loaded safely with "
                f"weights_only=True.{legacy_hint}"
            ) from None

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        raise RuntimeError(
            "Trusted legacy checkpoint could not be loaded with the explicit "
            "unsafe fallback"
        ) from None


def _validate_optimizer_state(value: Any, *, safe: bool) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("Checkpoint optimizer state must be a dictionary")
    missing = sorted({"state", "param_groups"}.difference(value))
    if missing:
        raise RuntimeError(
            f"Checkpoint optimizer state is missing keys: {missing}"
        )
    if not isinstance(value["state"], dict):
        raise RuntimeError("Checkpoint optimizer state['state'] must be a dictionary")
    if not isinstance(value["param_groups"], list):
        raise RuntimeError(
            "Checkpoint optimizer state['param_groups'] must be a list"
        )
    if safe:
        _validate_safe_checkpoint_value(value, label="optimizer state")


def _validate_plugin_state(value: Any, *, safe: bool) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("Checkpoint plugin state must be a dictionary")
    if safe:
        _validate_safe_checkpoint_value(value, label="plugin state")


def _validate_safe_checkpoint_value(value: Any, *, label: str) -> None:
    import torch

    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (bool, int, float, str)):
                raise RuntimeError(
                    f"Checkpoint {label} has an unsafe dictionary key type: "
                    f"{type(key).__name__}"
                )
            _validate_safe_checkpoint_value(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_safe_checkpoint_value(item, label=label)
        return
    raise RuntimeError(
        f"Checkpoint {label} contains an unsafe value type: {type(value).__name__}"
    )


def _validate_distributed_state(
    value: Any,
    *,
    require_sorted: bool,
    redact_runtime_secrets: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Checkpoint distributed state must be a dictionary")
    if not all(isinstance(key, str) for key in value):
        raise RuntimeError("Checkpoint distributed state keys must be strings")
    required = {"world_size", "backend", "entries"}
    missing = sorted(required.difference(value))
    if missing:
        raise RuntimeError(
            f"Checkpoint distributed state is missing keys: {missing}"
        )
    extra = sorted(set(value).difference(required))
    if extra:
        raise RuntimeError(
            f"Checkpoint distributed state has unexpected keys: {extra}"
        )

    world_size = value["world_size"]
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise RuntimeError("Checkpoint distributed world_size must be an integer")
    if world_size < 1:
        raise RuntimeError("Checkpoint distributed world_size must be at least 1")

    backend = value["backend"]
    if (
        not isinstance(backend, str)
        or not backend
        or backend != backend.strip().lower()
    ):
        raise RuntimeError(
            "Checkpoint distributed backend must be a lowercase non-empty string"
        )

    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        raise RuntimeError("Checkpoint distributed entries must be a list")
    normalized_entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    input_ranks: list[int] = []
    entry_required = {"rank", "rng", "sampler_cursor", "runtime_identity"}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError(
                "Checkpoint distributed entries must be dictionaries"
            )
        if not all(isinstance(key, str) for key in raw_entry):
            raise RuntimeError(
                "Checkpoint distributed entry keys must be strings"
            )
        missing_entry = sorted(entry_required.difference(raw_entry))
        if missing_entry:
            raise RuntimeError(
                "Checkpoint distributed entry is missing keys: "
                f"{missing_entry}"
            )
        extra_entry = sorted(set(raw_entry).difference(entry_required))
        if extra_entry:
            raise RuntimeError(
                "Checkpoint distributed entry has unexpected keys: "
                f"{extra_entry}"
            )
        rank = raw_entry["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RuntimeError("Checkpoint distributed rank must be an integer")
        if rank < 0 or rank >= world_size:
            raise RuntimeError(
                f"Checkpoint distributed rank {rank} is outside world_size "
                f"{world_size}"
            )
        if rank in seen:
            raise RuntimeError(
                f"Checkpoint distributed entries contain duplicate rank {rank}"
            )
        runtime_identity = raw_entry["runtime_identity"]
        if not isinstance(runtime_identity, dict):
            raise RuntimeError(
                "Checkpoint distributed runtime_identity must be a dictionary"
            )
        runtime_identity = _validated_runtime_identity(
            runtime_identity,
            rank=rank,
            redact_secrets=redact_runtime_secrets,
        )
        _validate_rng_state(raw_entry["rng"], safe=True)
        _validate_safe_checkpoint_value(
            raw_entry["sampler_cursor"],
            label=f"distributed rank {rank} sampler cursor",
        )
        _validate_safe_checkpoint_value(
            runtime_identity,
            label=f"distributed rank {rank} runtime identity",
        )
        seen.add(rank)
        input_ranks.append(rank)
        normalized_entries.append(
            {
                "rank": rank,
                "rng": deepcopy(raw_entry["rng"]),
                "sampler_cursor": deepcopy(raw_entry["sampler_cursor"]),
                "runtime_identity": runtime_identity,
            }
        )

    expected_ranks = list(range(world_size))
    missing_ranks = sorted(set(expected_ranks).difference(seen))
    if missing_ranks:
        raise RuntimeError(
            f"Checkpoint distributed entries are missing ranks: {missing_ranks}"
        )
    if len(normalized_entries) != world_size:
        raise RuntimeError(
            "Checkpoint distributed entries must contain exactly one entry per rank"
        )
    if require_sorted and input_ranks != expected_ranks:
        raise RuntimeError(
            "Checkpoint distributed entries must be sorted by rank"
        )
    normalized_entries.sort(key=lambda entry: entry["rank"])
    normalized = {
        "world_size": world_size,
        "backend": backend,
        "entries": normalized_entries,
    }
    _validate_safe_checkpoint_value(normalized, label="distributed state")
    return normalized


def _validated_runtime_identity(
    value: dict[str, Any],
    *,
    rank: int,
    redact_secrets: bool,
) -> dict[str, Any]:
    label = f"distributed rank {rank} runtime identity"
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        normalized = json.loads(serialized)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError(f"Checkpoint {label} must be JSON-safe") from None
    if normalized != value:
        raise RuntimeError(
            f"Checkpoint {label} must round-trip through JSON without changes"
        )

    redacted = redact_artifact_config(normalized)
    if redact_secrets:
        return redacted
    if redacted != normalized:
        raise RuntimeError(f"Checkpoint {label} contains unredacted secrets")
    return normalized


def _validate_checkpoint_distributed_state(
    state: dict[str, Any],
    metadata: dict[str, Any],
    *,
    state_format: int,
    expected_world_size: int | None,
    expected_rank: int | None,
) -> dict[str, Any] | None:
    if "distributed_state" not in state:
        if "distributed_state" in metadata:
            raise RuntimeError(
                "checkpoint.json marks distributed state missing from "
                "training_state.pt"
            )
        world_size = 1
        distributed_state = None
    else:
        raw_state = state["distributed_state"]
        if state_format not in _DISTRIBUTED_CHECKPOINT_FORMAT_VERSIONS:
            raise RuntimeError(
                "Distributed runtime state requires checkpoint format v3 or newer"
            )
        distributed_state = _validate_distributed_state(
            raw_state,
            require_sorted=True,
            redact_runtime_secrets=False,
        )
        world_size = int(distributed_state["world_size"])
        marker = metadata.get("distributed_state")
        expected_marker = {
            "world_size": world_size,
            "backend": distributed_state["backend"],
        }
        if marker != expected_marker:
            raise RuntimeError(
                "checkpoint.json distributed_state does not match "
                "training_state.pt"
            )

    if expected_world_size is not None:
        if isinstance(expected_world_size, bool) or not isinstance(
            expected_world_size, int
        ):
            raise TypeError("expected_world_size must be an integer")
        if expected_world_size < 1:
            raise ValueError("expected_world_size must be at least 1")
        if expected_world_size != world_size:
            raise RuntimeError(
                "Checkpoint world size changed: "
                f"checkpoint {world_size}, expected {expected_world_size}; "
                "same-world-size resume is required"
            )
    if expected_rank is not None:
        _validate_expected_rank(expected_rank, world_size=world_size)
    return distributed_state


def _validate_expected_rank(rank: Any, *, world_size: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("expected rank must be an integer")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"Checkpoint does not contain rank {rank}; world_size is {world_size}"
        )


def _validate_rng_state(value: Any, *, safe: bool) -> None:
    import torch

    if not isinstance(value, dict):
        raise RuntimeError("Checkpoint RNG state must be a dictionary")
    missing = sorted({"python", "numpy", "torch"}.difference(value))
    if missing:
        raise RuntimeError(f"Checkpoint RNG state is missing keys: {missing}")
    if not isinstance(value["torch"], torch.Tensor):
        raise RuntimeError("Checkpoint torch RNG state must be a tensor")
    cuda_state = value.get("cuda")
    if cuda_state is not None and (
        not isinstance(cuda_state, list)
        or not all(isinstance(item, torch.Tensor) for item in cuda_state)
    ):
        raise RuntimeError("Checkpoint CUDA RNG state must be a list of tensors")
    numpy_state = value["numpy"]
    if safe:
        extra = sorted(set(value).difference({"python", "numpy", "torch", "cuda"}))
        if extra:
            raise RuntimeError(f"Checkpoint RNG state has unexpected keys: {extra}")
        required = {
            "bit_generator",
            "state",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        if not isinstance(numpy_state, dict):
            raise RuntimeError("Checkpoint NumPy RNG state must be a dictionary")
        missing_numpy = sorted(required.difference(numpy_state))
        if missing_numpy:
            raise RuntimeError(
                f"Checkpoint NumPy RNG state is missing keys: {missing_numpy}"
            )
        if not isinstance(numpy_state["state"], list) or not all(
            isinstance(item, int) for item in numpy_state["state"]
        ):
            raise RuntimeError(
                "Checkpoint NumPy RNG state vector must be a list of integers"
            )
        _validate_safe_checkpoint_value(value, label="RNG state")
        try:
            random.Random().setstate(value["python"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Checkpoint Python RNG state is malformed") from exc
        try:
            np.random.RandomState().set_state(
                (
                    str(numpy_state["bit_generator"]),
                    np.asarray(numpy_state["state"], dtype=np.uint32),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Checkpoint NumPy RNG state is malformed") from exc
        try:
            torch.Generator(device="cpu").set_state(value["torch"])
        except RuntimeError as exc:
            raise RuntimeError("Checkpoint torch RNG state is malformed") from exc


def _is_checkpoint_control_file(relative: Path) -> bool:
    return len(relative.parts) == 1 and relative.name in _CHECKPOINT_CONTROL_FILES


def _is_temporary_checkpoint_path(relative: Path) -> bool:
    for part in relative.parts:
        if (
            part.endswith((".tmp", ".temp", ".part", "~"))
            or ".tmp-" in part
            or part.startswith((".tmp-", ".temp-"))
        ):
            return True
    return False


def _validate_v2_checkpoint_metadata(
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    state_required = {
        "training_semantics_fingerprint",
        "data_identity_fingerprint",
        "implementation_identity_fingerprint",
        "identity_payload",
        "data_identity",
        "data_source",
    }
    metadata_required = {
        "training_semantics_fingerprint",
        "data_identity_fingerprint",
        "implementation_identity_fingerprint",
        "data_identity",
        "data_source",
    }
    missing_state = sorted(state_required.difference(state))
    if missing_state:
        raise RuntimeError(
            f"Checkpoint v2 training state is missing keys: {missing_state}"
        )
    missing_metadata = sorted(metadata_required.difference(metadata))
    if missing_metadata:
        raise RuntimeError(
            f"Checkpoint v2 metadata is missing keys: {missing_metadata}"
        )
    for key in (
        "config_fingerprint_scheme",
        "training_semantics_fingerprint",
        "data_identity_fingerprint",
        "implementation_identity_fingerprint",
        "data_identity",
        "data_source",
    ):
        if metadata.get(key) != state.get(key):
            raise RuntimeError(
                f"checkpoint.json {key} does not match training_state.pt"
            )
    payload = state["identity_payload"]
    if not isinstance(payload, Mapping):
        raise RuntimeError("Checkpoint v2 identity_payload must be a dictionary")
    if payload.get("data_identity") != state.get("data_identity"):
        raise RuntimeError(
            "Checkpoint v2 identity payload data_identity does not match "
            "training_state.pt"
        )
    fingerprint_scheme = state.get("config_fingerprint_scheme")
    if fingerprint_scheme is not None and not isinstance(fingerprint_scheme, str):
        raise RuntimeError(
            f"Unsupported config fingerprint scheme: {fingerprint_scheme!r}"
        )
    if fingerprint_scheme in _SUPPORTED_SAFE_CONFIG_FINGERPRINT_SCHEMES:
        if redact_artifact_config(payload) != payload:
            raise RuntimeError("Checkpoint v2 identity payload contains unredacted secrets")
        if redact_artifact_config(state.get("data_source")) != state.get("data_source"):
            raise RuntimeError("Checkpoint v2 data source contains unredacted secrets")
        component_fingerprints = {
            key: state.get(key)
            for key in (
                "training_semantics_fingerprint",
                "data_identity_fingerprint",
                "implementation_identity_fingerprint",
            )
        }
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in component_fingerprints.values()
        ):
            raise RuntimeError("Checkpoint v2 contains an invalid identity fingerprint")
        expected_components = {
            "config_fingerprint": _payload_sha256(component_fingerprints)
        }
    elif fingerprint_scheme is None:
        expected_components = {
            "training_semantics_fingerprint": _payload_sha256(
                payload.get("training_semantics")
            ),
            "data_identity_fingerprint": _payload_sha256(payload.get("data_identity")),
            "implementation_identity_fingerprint": _payload_sha256(
                payload.get("implementation")
            ),
            "config_fingerprint": _payload_sha256(payload),
        }
    else:
        raise RuntimeError(
            f"Unsupported config fingerprint scheme: {fingerprint_scheme}"
        )
    for key, expected in expected_components.items():
        if state.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint v2 {key} does not match its canonical identity payload"
            )


def _reject_ambiguous_legacy_reward_partition(
    config: Mapping[str, Any],
    *,
    fingerprint_scheme: Any,
    read_only_audit: bool,
) -> None:
    if fingerprint_scheme is not None and not isinstance(fingerprint_scheme, str):
        raise RuntimeError(
            f"Unsupported config fingerprint scheme: {fingerprint_scheme!r}"
        )
    if read_only_audit or fingerprint_scheme not in {
        None,
        *_PRIOR_SAFE_CONFIG_FINGERPRINT_SCHEMES,
    }:
        return
    runner = config.get("runner") or {}
    if not isinstance(runner, Mapping):
        return
    reward_executor = runner.get("reward_executor") or {}
    if not isinstance(reward_executor, Mapping):
        return
    # Legacy schemas accepted only positive integer microbatch sizes. Preserve
    # their historical validation rule, while refusing ``None`` because it is a
    # new full-batch sentinel that could otherwise reinterpret the old default 1.
    microbatch_size = reward_executor.get("microbatch_size")
    if reward_executor.get("mode") == "async" and (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size < 1
    ):
        raise RuntimeError(
            "Resume rejected: this legacy checkpoint fingerprint did not bind the async "
            "reward batch partition. Older VisualRL releases defaulted "
            "microbatch_size to a positive integer. Explicitly restore the "
            "original resolved microbatch_size, or use an audited checkpoint "
            "migration."
        )


def _identity_differences(
    checkpoint: Any,
    current: Any,
    path: str = "",
) -> list[str]:
    if type(checkpoint) is not type(current):
        return [path or "identity_payload"]
    if isinstance(checkpoint, dict):
        differences: list[str] = []
        for key in sorted(set(checkpoint) | set(current)):
            child = f"{path}.{key}" if path else str(key)
            if key not in checkpoint or key not in current:
                differences.append(child)
            else:
                differences.extend(
                    _identity_differences(checkpoint[key], current[key], child)
                )
        return differences
    if isinstance(checkpoint, list):
        if checkpoint == current:
            return []
        return [path]
    if checkpoint != current:
        return [path]
    return []


def _object_identity(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    cls = type(value)
    class_name = f"{cls.__module__}.{cls.__qualname__}"
    try:
        source = inspect.getsource(cls).encode("utf-8")
    except (OSError, TypeError):
        source = class_name.encode("utf-8")
    identity: dict[str, Any] = {
        "class": class_name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "module_sha256": _module_hash(cls),
    }
    details = getattr(value, "_visual_rl_identity", None)
    if details is None:
        implementation_identity = getattr(value, "implementation_identity", None)
        if callable(implementation_identity):
            details = implementation_identity()
    if details is None:
        return identity
    if not isinstance(details, Mapping):
        raise TypeError("implementation identity details must be a JSON-safe mapping")
    try:
        json_details = json.loads(
            json.dumps(dict(details), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"implementation identity details must be JSON-safe: {exc}"
        ) from exc
    for reserved in ("class", "module_sha256"):
        if reserved in json_details and json_details[reserved] != identity[reserved]:
            raise ValueError(
                f"implementation identity details cannot override {reserved!r}"
            )
    identity.update(json_details)
    return identity


def _parameter_signature(adapter: Any) -> list[dict[str, Any]]:
    signature = []
    for name, parameter in adapter.named_parameters():
        if not getattr(parameter, "requires_grad", True):
            continue
        signature.append(
            {
                "name": str(name),
                "shape": list(getattr(parameter, "shape", ())),
                "dtype": str(getattr(parameter, "dtype", "unknown")),
            }
        )
    return signature


@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@lru_cache(maxsize=1)
def _git_diff_hash() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--", "visual_rl"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return hashlib.sha256(result.stdout).hexdigest()


@lru_cache(maxsize=1)
def _runtime_tree_hash() -> str:
    package_root = Path(__file__).resolve().parents[1]
    return _hash_files(package_root, package_root.rglob("*.py"))


def _reference_patch_hash(adapter: Any) -> str | None:
    repo_root = getattr(adapter, "repo_root", None)
    if repo_root is None:
        return None
    root = Path(repo_root)
    patch_root = root / "flow_grpo" / "diffusers_patch"
    if not patch_root.exists():
        return None
    return _hash_files(root, patch_root.rglob("*.py"))


def _module_hash(cls: type[Any]) -> str:
    module = inspect.getmodule(cls)
    module_path = Path(getattr(module, "__file__", "")) if module else Path()
    if not module_path.is_file():
        return hashlib.sha256(
            f"{cls.__module__}.{cls.__qualname__}".encode("utf-8")
        ).hexdigest()
    return hashlib.sha256(module_path.read_bytes()).hexdigest()


def _hash_files(root: Path, files) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in files if item.is_file()), key=str):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, torch, and available CUDA RNG state safely."""

    import torch

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": numpy_state[1].astype(np.uint32, copy=False).tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    _validate_rng_state(state, safe=True)
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a RNG snapshot produced by :func:`capture_rng_state`."""

    _validate_rng_state(state, safe=True)
    _restore_rng_state(state)


def _rng_state() -> dict[str, Any]:
    return capture_rng_state()


def _restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    else:
        np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
