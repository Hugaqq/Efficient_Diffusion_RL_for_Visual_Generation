"""Safe, rank-local format-v3 rollout cache."""

from __future__ import annotations

from collections.abc import Mapping
import errno
import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Any

from visual_rl.artifacts.serialization import (
    canonical_json_text,
    strict_json_load,
)
from visual_rl.core.types import RolloutBatch, StepContext, to_plain_dict


CACHE_SCHEMA = "visual_rl.rollout_cache"
CACHE_VERSION = 3
_SUPPORTED_CACHE_VERSIONS = frozenset({CACHE_VERSION})
_DIGEST_CHUNK_SIZE = 1024 * 1024
_METADATA_KEYS = frozenset(
    {
        "schema",
        "version",
        "kind",
        "step",
        "generation",
        "tensor_path",
        "tensor_sha256",
        "media_path",
        "media_sha256",
        "prompts",
        "metadata",
        "sample_id",
        "prompt_id",
        "group_id",
        "branch_id",
        "media_layout",
        "context",
        "artifact_metadata",
    }
)
_TENSOR_KEYS = frozenset(
    {
        "schema",
        "version",
        "kind",
        "generation",
        "latents",
        "next_latents",
        "timesteps",
        "old_log_probs",
        "transition_mask",
        "camera_trajectory",
        "selected_timestep_index",
        "flash_coefficient",
        "branch_step_index",
        "trajectory_step_index",
        "transition_std_dev",
        "recompute_payload",
    }
)
_MEDIA_KEYS = frozenset(
    {"schema", "version", "kind", "generation", "media"}
)
_TENSOR_VALUE_KEYS = _TENSOR_KEYS.difference(
    {"schema", "version", "kind", "generation", "recompute_payload"}
)


class RolloutCache:
    """Persist one exact ``RolloutBatch`` per logical step."""

    def __init__(
        self,
        root: str | Path | None,
        *,
        output_dir: str | Path,
    ) -> None:
        self.output_dir = _validated_existing_directory(
            Path(output_dir),
            label="rollout cache output_dir",
        )
        self.root: Path | None
        if root is None:
            self.root = None
            return
        requested = Path(root).absolute()
        _require_lexically_below(requested, self.output_dir)
        _initialize_cache_root(requested)
        resolved = _validated_existing_directory(
            requested,
            label="rollout cache root",
        )
        if resolved == self.output_dir or not resolved.is_relative_to(
            self.output_dir
        ):
            raise ValueError("rollout cache root must be below output_dir")
        self.root = resolved

    def save(
        self,
        batch: RolloutBatch,
    ) -> tuple[str | None, str | None]:
        """Save v3 media/tensors and return output-relative POSIX paths."""

        if not isinstance(batch, RolloutBatch):
            raise TypeError("RolloutCache.save() requires a RolloutBatch")
        if self.root is None:
            return None, None
        import torch

        root = _validated_cache_root(self.root)
        step = batch.context.step
        base = root / f"batch_{step:06d}"
        generation = secrets.token_hex(16)
        tensor_path = base.with_suffix(".pt")
        media_path = base.with_suffix(".media.pt")
        metadata_path = base.with_suffix(".json")
        tensor_payload = {
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION,
            "kind": "tensors",
            "generation": generation,
            "latents": _portable_tensor(batch.latents),
            "next_latents": _portable_tensor(batch.next_latents),
            "timesteps": _portable_tensor(batch.timesteps),
            "old_log_probs": _portable_tensor(batch.old_log_probs),
            "transition_mask": _portable_tensor(batch.transition_mask),
            "camera_trajectory": _portable_optional_tensor(
                batch.camera_trajectory
            ),
            "selected_timestep_index": _portable_optional_tensor(
                batch.selected_timestep_index
            ),
            "flash_coefficient": _portable_optional_tensor(
                batch.flash_coefficient
            ),
            "branch_step_index": _portable_optional_tensor(
                batch.branch_step_index
            ),
            "trajectory_step_index": _portable_optional_tensor(
                batch.trajectory_step_index
            ),
            "transition_std_dev": _portable_optional_tensor(
                batch.transition_std_dev
            ),
            "recompute_payload": {
                name: _portable_tensor(value)
                for name, value in batch.recompute_payload.items()
            },
        }
        media_payload = {
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION,
            "kind": "media",
            "generation": generation,
            "media": _portable_tensor(batch.media),
        }
        _validate_tensor_payload(tensor_payload)
        _validate_media_payload(media_payload)
        tensor_digest = _atomic_torch_save(
            torch,
            tensor_payload,
            tensor_path,
            root=root,
        )
        media_digest = _atomic_torch_save(
            torch,
            media_payload,
            media_path,
            root=root,
        )
        metadata = {
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION,
            "kind": "metadata",
            "step": step,
            "generation": generation,
            "tensor_path": tensor_path.name,
            "tensor_sha256": tensor_digest,
            "media_path": media_path.name,
            "media_sha256": media_digest,
            "prompts": to_plain_dict(batch.prompts),
            "metadata": to_plain_dict(batch.metadata),
            "sample_id": to_plain_dict(batch.sample_id),
            "prompt_id": to_plain_dict(batch.prompt_id),
            "group_id": to_plain_dict(batch.group_id),
            "branch_id": to_plain_dict(batch.branch_id),
            "media_layout": batch.media_layout,
            "context": to_plain_dict(batch.context),
            "artifact_metadata": to_plain_dict(batch.artifact_metadata),
        }
        _validate_metadata_payload(metadata, expected_step=step)
        _atomic_json_save(metadata, metadata_path, root=root)
        _fsync_directory(root)
        return (
            _output_relative(media_path, self.output_dir),
            _output_relative(tensor_path, self.output_dir),
        )

    def load(self, step: int) -> RolloutBatch:
        """Safely load one exact v3 batch on CPU."""

        step = _non_negative_step(step)
        if self.root is None:
            raise RuntimeError("rollout cache is disabled")
        import torch

        root = _validated_cache_root(self.root)
        base = root / f"batch_{step:06d}"
        metadata_path = _validated_cache_file(
            root,
            base.with_suffix(".json"),
            label="metadata",
        )
        try:
            metadata = strict_json_load(metadata_path)
        except ValueError as exc:
            raise RuntimeError("rollout cache metadata is invalid") from exc
        _validate_metadata_payload(metadata, expected_step=step)
        tensor_path = _declared_step_file(
            root,
            base.with_suffix(".pt"),
            metadata["tensor_path"],
            label="tensor payload",
        )
        media_path = _declared_step_file(
            root,
            base.with_suffix(".media.pt"),
            metadata["media_path"],
            label="media payload",
        )
        tensor_digest = _sha256_path(tensor_path)
        media_digest = _sha256_path(media_path)
        if tensor_digest != metadata["tensor_sha256"]:
            raise RuntimeError("rollout cache tensor payload digest mismatch")
        if media_digest != metadata["media_sha256"]:
            raise RuntimeError("rollout cache media payload digest mismatch")
        tensor_payload = _safe_torch_load(
            torch,
            tensor_path,
            label="tensor payload",
        )
        media_payload = _safe_torch_load(
            torch,
            media_path,
            label="media payload",
        )
        _validate_tensor_payload(tensor_payload)
        _validate_media_payload(media_payload)
        if (
            tensor_payload["generation"] != metadata["generation"]
            or media_payload["generation"] != metadata["generation"]
        ):
            raise RuntimeError(
                "rollout cache generation does not match metadata publication"
            )
        context_payload = metadata["context"]
        try:
            context = StepContext(**context_payload)
            batch = RolloutBatch(
                prompts=tuple(metadata["prompts"]),
                metadata=tuple(metadata["metadata"]),
                media=media_payload["media"],
                latents=tensor_payload["latents"],
                next_latents=tensor_payload["next_latents"],
                timesteps=tensor_payload["timesteps"],
                old_log_probs=tensor_payload["old_log_probs"],
                transition_mask=tensor_payload["transition_mask"],
                sample_id=tuple(metadata["sample_id"]),
                prompt_id=tuple(metadata["prompt_id"]),
                group_id=tuple(metadata["group_id"]),
                branch_id=(
                    None
                    if metadata["branch_id"] is None
                    else tuple(metadata["branch_id"])
                ),
                media_layout=metadata["media_layout"],
                camera_trajectory=tensor_payload["camera_trajectory"],
                context=context,
                selected_timestep_index=tensor_payload[
                    "selected_timestep_index"
                ],
                flash_coefficient=tensor_payload["flash_coefficient"],
                branch_step_index=tensor_payload["branch_step_index"],
                trajectory_step_index=tensor_payload[
                    "trajectory_step_index"
                ],
                transition_std_dev=tensor_payload["transition_std_dev"],
                recompute_payload=tensor_payload["recompute_payload"],
                artifact_metadata=metadata["artifact_metadata"],
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "rollout cache does not contain a valid RolloutBatch"
            ) from exc
        if batch.context.step != step:
            raise RuntimeError("rollout cache context step disagrees with filename")
        return batch

    def truncate_from_step(self, start_step: int) -> None:
        """Remove only validated v3 step files at or after ``start_step``."""

        start = _non_negative_step(start_step)
        if self.root is None:
            return
        root = _validated_cache_root(self.root)
        candidates: list[Path] = []
        for path in root.iterdir():
            step = _cache_entry_step(path.name)
            if step is not None and step >= start:
                candidates.append(path)
        validated = tuple(
            _validated_cache_file(root, path, label="truncate entry")
            for path in candidates
        )
        for path in validated:
            path.unlink()
        _fsync_directory(root)


def _validate_metadata_payload(payload: Any, *, expected_step: int) -> None:
    if not isinstance(payload, dict) or set(payload) != set(_METADATA_KEYS):
        raise RuntimeError("rollout cache metadata has an invalid exact key set")
    _validate_header(payload, expected_kind="metadata")
    if payload["step"] != expected_step or type(payload["step"]) is not int:
        raise RuntimeError("rollout cache metadata step does not match filename")
    if payload["tensor_path"] != f"batch_{expected_step:06d}.pt":
        raise RuntimeError("rollout cache tensor_path is not the step-local file")
    if payload["media_path"] != f"batch_{expected_step:06d}.media.pt":
        raise RuntimeError("rollout cache media_path is not the step-local file")
    _hex(payload["generation"], length=32, label="generation")
    _hex(payload["tensor_sha256"], length=64, label="tensor_sha256")
    _hex(payload["media_sha256"], length=64, label="media_sha256")
    for name in ("prompts", "metadata", "sample_id", "prompt_id", "group_id"):
        if not isinstance(payload[name], list) or not payload[name]:
            raise RuntimeError(f"rollout cache {name} must be a non-empty list")
    count = len(payload["prompts"])
    if any(
        len(payload[name]) != count
        for name in ("metadata", "sample_id", "prompt_id", "group_id")
    ):
        raise RuntimeError("rollout cache row metadata lengths disagree")
    if any(not isinstance(item, str) for item in payload["prompts"]):
        raise RuntimeError("rollout cache prompts must contain strings")
    if any(not isinstance(item, dict) for item in payload["metadata"]):
        raise RuntimeError("rollout cache metadata rows must be objects")
    for name in ("sample_id", "prompt_id", "group_id"):
        if any(not isinstance(item, str) or not item for item in payload[name]):
            raise RuntimeError(f"rollout cache {name} entries must be strings")
    branch_id = payload["branch_id"]
    if branch_id is not None and (
        not isinstance(branch_id, list)
        or len(branch_id) != count
        or any(
            isinstance(item, bool)
            or not isinstance(item, (str, int, type(None)))
            for item in branch_id
        )
    ):
        raise RuntimeError("rollout cache branch_id rows are invalid")
    if payload["media_layout"] not in {"BCHW", "BFCHW", "BFHWC"}:
        raise RuntimeError("rollout cache media_layout is invalid")
    context = payload["context"]
    if not isinstance(context, dict) or set(context) != {
        "step",
        "seed",
        "rank",
        "world_size",
    }:
        raise RuntimeError("rollout cache context has an invalid exact key set")
    try:
        StepContext(**context)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("rollout cache context is invalid") from exc
    if not isinstance(payload["artifact_metadata"], dict):
        raise RuntimeError("rollout cache artifact_metadata must be an object")


def _validate_tensor_payload(payload: Any) -> None:
    import torch

    if not isinstance(payload, dict) or set(payload) != set(_TENSOR_KEYS):
        raise RuntimeError(
            "rollout cache tensor payload has an invalid exact key set"
        )
    _validate_header(payload, expected_kind="tensors")
    _hex(payload["generation"], length=32, label="tensor generation")
    for name in _TENSOR_VALUE_KEYS:
        value = payload[name]
        if value is not None and not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"rollout cache tensor field {name!r} must be a tensor or None"
            )
        if isinstance(value, torch.Tensor):
            _validate_tensor(value, label=name)
    recompute = payload["recompute_payload"]
    if not isinstance(recompute, dict):
        raise RuntimeError("rollout cache recompute_payload must be an object")
    for name, value in recompute.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, torch.Tensor)
        ):
            raise RuntimeError(
                "rollout cache recompute_payload must map names to tensors"
            )
        _validate_tensor(value, label=f"recompute_payload.{name}")


def _validate_media_payload(payload: Any) -> None:
    import torch

    if not isinstance(payload, dict) or set(payload) != set(_MEDIA_KEYS):
        raise RuntimeError("rollout cache media payload has an invalid exact key set")
    _validate_header(payload, expected_kind="media")
    _hex(payload["generation"], length=32, label="media generation")
    media = payload["media"]
    if not isinstance(media, torch.Tensor) or media.ndim not in {4, 5}:
        raise RuntimeError(
            "rollout cache media payload must contain BCHW/BFCHW/BFHWC tensor"
        )
    _validate_tensor(media, label="media")


def _validate_header(payload: Mapping[str, Any], *, expected_kind: str) -> None:
    if payload.get("schema") != CACHE_SCHEMA:
        raise RuntimeError(
            f"unsupported rollout cache schema: {payload.get('schema')!r}"
        )
    version = payload.get("version")
    if type(version) is not int or version not in _SUPPORTED_CACHE_VERSIONS:
        raise RuntimeError(f"unsupported rollout cache version: {version!r}")
    if payload.get("kind") != expected_kind:
        raise RuntimeError(
            f"rollout cache kind must be {expected_kind!r}"
        )


def _validate_tensor(value: Any, *, label: str) -> None:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise RuntimeError(
            f"rollout cache {label} must be detached contiguous CPU tensor"
        )


def _portable_tensor(value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("rollout cache tensor fields must be torch.Tensor values")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError("rollout cache tensor fields must be detached")
    return value.detach().to(device="cpu").contiguous().clone()


def _portable_optional_tensor(value: Any | None) -> Any | None:
    return None if value is None else _portable_tensor(value)


def _safe_torch_load(torch: Any, path: Path, *, label: str) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        raise RuntimeError(
            "safe rollout cache loading requires weights_only=True support"
        ) from None
    except Exception:
        raise RuntimeError(
            f"rollout cache {label} could not be loaded with weights_only=True"
        ) from None


def _declared_step_file(
    root: Path,
    expected: Path,
    declared: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(declared, str) or declared != expected.name:
        raise RuntimeError(
            f"rollout cache {label} must name the expected step-local file"
        )
    return _validated_cache_file(root, expected, label=label)


def _non_negative_step(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("rollout cache step must be a non-negative integer")
    return value


def _cache_entry_step(name: str) -> int | None:
    if not name.startswith("batch_"):
        return None
    stem, separator, suffix = name.partition(".")
    if not separator or suffix not in {"pt", "media.pt", "json"}:
        return None
    digits = stem.removeprefix("batch_")
    if len(digits) != 6 or not digits.isdigit():
        return None
    return int(digits)


def _output_relative(path: Path, output_dir: Path) -> str:
    try:
        relative = path.relative_to(output_dir)
    except ValueError as exc:
        raise RuntimeError("rollout cache artifact escapes output_dir") from exc
    value = relative.as_posix()
    if value.startswith("/") or ".." in relative.parts:
        raise RuntimeError("rollout cache artifact path is unsafe")
    return value


def _require_lexically_below(path: Path, output_dir: Path) -> None:
    try:
        relative = path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("rollout cache root must be below output_dir") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("rollout cache root must be below output_dir")


def _validated_existing_directory(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise RuntimeError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} path must not contain symlinks: {current}")
    if not stat.S_ISDIR(absolute.lstat().st_mode):
        raise RuntimeError(f"{label} must be a directory")
    return absolute.resolve(strict=True)


def _validated_cache_root(root: Path) -> Path:
    return _validated_existing_directory(root, label="rollout cache root")


def _initialize_cache_root(root: Path) -> None:
    absolute = root.absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            try:
                metadata = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
            else:
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeError(
                        "rollout cache root path must not contain symlinks: "
                        f"{current}"
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeError(
                        f"rollout cache root must be a directory: {current}"
                    )
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise RuntimeError(
                        "rollout cache root path must not contain symlinks: "
                        f"{current}"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def _validated_cache_file(root: Path, path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"missing rollout cache {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"rollout cache {label} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"rollout cache {label} must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"rollout cache {label} escapes cache root")
    return resolved


def _prepare_cache_output(
    root: Path,
    path: Path,
) -> tuple[int, Path, tuple[int, int]]:
    if path.parent.resolve(strict=True) != root:
        raise RuntimeError("rollout cache output must be inside its root")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(100):
        temporary = root / f".{path.name}.tmp-{secrets.token_hex(12)}"
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
        metadata = os.fstat(descriptor)
        return descriptor, temporary, (metadata.st_dev, metadata.st_ino)
    raise FileExistsError("unable to allocate rollout cache temporary file")


def _cleanup_owned_temp(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        return
    if stat.S_ISLNK(metadata.st_mode) or (
        metadata.st_dev,
        metadata.st_ino,
    ) != identity:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _atomic_torch_save(
    torch: Any,
    payload: Mapping[str, Any],
    path: Path,
    *,
    root: Path,
) -> str:
    descriptor, temporary, identity = _prepare_cache_output(root, path)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(descriptor)
        digest = _sha256_path(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        _cleanup_owned_temp(temporary, identity)
        os.close(descriptor)


def _atomic_json_save(
    payload: Mapping[str, Any],
    path: Path,
    *,
    root: Path,
) -> None:
    descriptor, temporary, identity = _prepare_cache_output(root, path)
    try:
        encoded = (canonical_json_text(payload) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(descriptor)
        os.replace(temporary, path)
    finally:
        _cleanup_owned_temp(temporary, identity)
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DIGEST_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: Any, *, length: int, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            f"rollout cache {label} must be {length} lowercase hex digits"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ("CACHE_SCHEMA", "CACHE_VERSION", "RolloutCache")
