"""Rollout cache for recovering reward/training work without rerolling."""

from __future__ import annotations

import errno
import hashlib
import json
from dataclasses import asdict
import os
from pathlib import Path
import secrets
import stat
from typing import Any

from visual_rl.artifacts.serialization import to_jsonable
from visual_rl.core.types import RolloutBatch, StepContext

CACHE_SCHEMA = "visual_rl.rollout_cache"
CACHE_VERSION = 2
_SUPPORTED_CACHE_VERSIONS = {1, CACHE_VERSION}
_DIGEST_CHUNK_SIZE = 1024 * 1024
_TENSOR_FIELDS = {
    "latents",
    "next_latents",
    "timesteps",
    "old_log_probs",
    "kl",
    "transition_mask",
    "model_tensors",
}
_LEGACY_TENSOR_FIELDS = _TENSOR_FIELDS | {
    "branch_id",
    "branch_ids",
    "context",
    "seed",
    "epoch_tag",
}


class RolloutCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root:
            _initialize_cache_root(self.root)

    def save(self, step: int, batch, rewards: Any | None = None) -> dict[str, str]:
        if self.root is None:
            return {}
        import torch

        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Rollout cache step must be a non-negative integer")
        if not isinstance(batch, RolloutBatch):
            raise TypeError("Rollout cache can only save RolloutBatch values")
        batch.validate_lightweight(strict=True)
        root = _validated_cache_root(self.root)
        base = root / f"batch_{step:06d}"
        generation = secrets.token_hex(16)
        tensor_payload = {
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION,
            "kind": "tensors",
            "generation": generation,
            "latents": batch.latents,
            "next_latents": batch.next_latents,
            "timesteps": batch.timesteps,
            "old_log_probs": batch.old_log_probs,
            "kl": batch.kl,
            "transition_mask": batch.transition_mask,
            "model_tensors": batch.model_tensors,
        }
        _validate_tensor_payload(tensor_payload)
        tensor_path = base.with_suffix(".pt")
        media_path = base.with_suffix(".media.pt")
        media_payload = {
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION,
            "kind": "media",
            "generation": generation,
            "media": batch.media,
        }
        _validate_media_payload(media_payload)
        tensor_sha256 = _atomic_torch_save(
            torch,
            tensor_payload,
            tensor_path,
            root=root,
        )
        media_sha256 = _atomic_torch_save(
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
            "tensor_sha256": tensor_sha256,
            "prompts": batch.prompts,
            "metadata": batch.metadata,
            "model_metadata": batch.model_metadata,
            "media_path": media_path.name,
            "media_sha256": media_sha256,
            "sample_id": to_jsonable(batch.sample_id),
            "prompt_id": to_jsonable(batch.prompt_id),
            "group_id": to_jsonable(batch.group_id),
            "branch_id": to_jsonable(batch.branch_id),
            "media_layout": batch.media_layout,
            "context": asdict(batch.context) if batch.context is not None else None,
        }
        if rewards is not None:
            metadata["reward_metadata"] = rewards.metadata
            metadata["weighted_total"] = rewards.weighted_total.detach().cpu().tolist()
        metadata = to_jsonable(metadata)
        _validate_metadata_payload(metadata, expected_step=step)
        metadata_path = base.with_suffix(".json")
        _atomic_json_save(metadata, metadata_path, root=root)
        return {
            "rollout_cache_path": str(tensor_path),
            "media_path": str(media_path),
            "metadata_path": str(metadata_path),
        }

    def load(self, step: int, *, map_location: Any = "cpu") -> RolloutBatch:
        """Load a cached batch, including caches written before formal identity."""

        if self.root is None:
            raise ValueError("RolloutCache has no root directory")
        import torch

        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Rollout cache step must be a non-negative integer")
        root = _validated_cache_root(self.root)
        base = root / f"batch_{step:06d}"
        metadata_path = _validated_cache_file(
            root,
            base.with_suffix(".json"),
            label="metadata",
        )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata_payload = json.load(handle)
        metadata_version = _validate_metadata_payload(
            metadata_payload,
            expected_step=step,
        )
        tensor_path = _validated_cache_file(
            root,
            base.with_suffix(".pt"),
            label="tensor payload",
        )
        tensor_payload, tensor_sha256 = _safe_torch_load(
            torch,
            tensor_path,
            map_location=map_location,
            label="tensor payload",
            calculate_digest=metadata_version == CACHE_VERSION,
        )
        tensor_version = _validate_tensor_payload(tensor_payload)
        _validate_payload_publication(
            metadata_payload,
            metadata_version=metadata_version,
            payload=tensor_payload,
            payload_version=tensor_version,
            actual_sha256=tensor_sha256,
            digest_key="tensor_sha256",
            label="tensor payload",
        )
        media_path = _resolve_media_path(
            root,
            base,
            metadata_payload.get("media_path"),
        )
        media_payload, media_sha256 = _safe_torch_load(
            torch,
            media_path,
            map_location=map_location,
            label="media payload",
            calculate_digest=metadata_version == CACHE_VERSION,
        )
        media = _validate_media_payload(media_payload)
        media_version = _payload_version(media_payload)
        _validate_payload_publication(
            metadata_payload,
            metadata_version=metadata_version,
            payload=media_payload,
            payload_version=media_version,
            actual_sha256=media_sha256,
            digest_key="media_sha256",
            label="media payload",
        )

        context_payload = metadata_payload.get("context") or tensor_payload.get(
            "context"
        )
        context = (
            StepContext(**context_payload)
            if isinstance(context_payload, dict)
            else _legacy_context(metadata_payload, tensor_payload)
        )
        branch_id = metadata_payload.get("branch_id")
        if branch_id is None:
            branch_id = tensor_payload.get(
                "branch_id",
                tensor_payload.get("branch_ids"),
            )
        batch = RolloutBatch(
            prompts=metadata_payload["prompts"],
            metadata=metadata_payload["metadata"],
            media=media,
            latents=tensor_payload.get("latents"),
            next_latents=tensor_payload.get("next_latents"),
            timesteps=tensor_payload.get("timesteps"),
            old_log_probs=tensor_payload.get("old_log_probs"),
            kl=tensor_payload.get("kl"),
            sample_id=metadata_payload.get("sample_id"),
            prompt_id=metadata_payload.get("prompt_id"),
            group_id=metadata_payload.get("group_id"),
            branch_id=branch_id,
            transition_mask=tensor_payload.get("transition_mask"),
            media_layout=metadata_payload.get("media_layout"),
            context=context,
            model_metadata=metadata_payload.get("model_metadata") or {},
            model_tensors=tensor_payload.get("model_tensors") or {},
        )
        batch.validate_lightweight(strict=True)
        return batch

    def load_batch(self, step: int, *, map_location: Any = "cpu") -> RolloutBatch:
        """Explicit alias for callers that distinguish batch and reward caches."""

        return self.load(step, map_location=map_location)

    def validate_step(
        self,
        step: int,
        *,
        map_location: Any = "cpu",
    ) -> dict[str, Any]:
        """Read-only audit of one published cache generation.

        ``load`` remains the authority for schema, safe ``weights_only`` tensor
        loading, generation matching, digest verification and ``RolloutBatch``
        validation. This wrapper only adds a stable inspection report; it never
        falls back to the pre-v2 triplet assumptions or repairs cache files.
        """

        if self.root is None:
            raise RuntimeError("rollout cache is disabled")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Rollout cache step must be a non-negative integer")
        try:
            batch = self.load(step, map_location=map_location)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"rollout cache step {step} metadata is corrupt or invalid"
            ) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "media payload" in message:
                raise RuntimeError(
                    f"rollout cache step {step} media payload is corrupt or invalid"
                ) from exc
            if "tensor payload" in message:
                raise RuntimeError(
                    f"rollout cache step {step} tensor payload is corrupt or invalid"
                ) from exc
            if "metadata" in message:
                raise RuntimeError(
                    f"rollout cache step {step} metadata is corrupt or invalid"
                ) from exc
            raise

        root = _validated_cache_root(self.root)
        base = root / f"batch_{step:06d}"
        paths = {
            "tensor": _validated_cache_file(
                root,
                base.with_suffix(".pt"),
                label="tensor payload",
            ),
            "media": _validated_cache_file(
                root,
                base.with_suffix(".media.pt"),
                label="media payload",
            ),
            "metadata": _validated_cache_file(
                root,
                base.with_suffix(".json"),
                label="metadata",
            ),
        }
        try:
            with paths["metadata"].open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"rollout cache step {step} metadata is corrupt or invalid"
            ) from exc
        version = _validate_metadata_payload(metadata, expected_step=step)
        digests = {name: _sha256_path(path) for name, path in paths.items()}
        if version == CACHE_VERSION:
            if digests["tensor"] != metadata["tensor_sha256"]:
                raise RuntimeError(
                    f"rollout cache step {step} tensor payload is corrupt or invalid"
                )
            if digests["media"] != metadata["media_sha256"]:
                raise RuntimeError(
                    f"rollout cache step {step} media payload is corrupt or invalid"
                )

        return {
            "step": step,
            "valid": True,
            "schema": metadata.get("schema"),
            "version": version,
            "generation": metadata.get("generation"),
            "prompt_count": len(batch.prompts),
            "sample_count": batch.batch_size,
            "files": {
                name: {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": digests[name],
                }
                for name, path in paths.items()
            },
        }

    def truncate_from_step(self, start_step: int) -> None:
        if self.root is None:
            return
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        root = _validated_cache_root(self.root)
        candidates: list[Path] = []
        for path in root.glob("batch_*"):
            prefix = path.name.split(".", maxsplit=1)[0]
            try:
                step = int(prefix.removeprefix("batch_"))
            except ValueError:
                continue
            if step >= start_step:
                candidates.append(path)

        validated = [
            _validated_cache_file(root, path, label="truncate entry")
            for path in candidates
        ]
        for path in validated:
            path.unlink()


def _validated_cache_root(root: Path) -> Path:
    absolute = Path(os.path.abspath(root))
    current = Path(absolute.anchor)
    components = [current]
    for part in absolute.parts[1:]:
        current /= part
        components.append(current)

    for component in components:
        try:
            component_stat = component.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"Rollout cache root path does not exist: {component}"
            ) from exc
        if stat.S_ISLNK(component_stat.st_mode):
            raise RuntimeError(
                f"Rollout cache root path must not contain symlinks: {component}"
            )

    root_stat = components[-1].lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"Rollout cache root must be a directory: {root}")
    try:
        return absolute.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot resolve rollout cache root: {root}") from exc


def _initialize_cache_root(root: Path) -> None:
    absolute = Path(os.path.abspath(root))
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW

    current_fd = os.open(absolute.anchor, directory_flags)
    try:
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            try:
                component_stat = os.stat(
                    part,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=current_fd)
                except FileExistsError:
                    pass
            else:
                if stat.S_ISLNK(component_stat.st_mode):
                    raise RuntimeError(
                        "Rollout cache root path must not contain symlinks: "
                        f"{current}"
                    )
                if not stat.S_ISDIR(component_stat.st_mode):
                    raise RuntimeError(
                        f"Rollout cache root must be a directory: {current}"
                    )

            try:
                child_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise RuntimeError(
                        "Rollout cache root path must not contain symlinks: "
                        f"{current}"
                    ) from exc
                raise
            os.close(current_fd)
            current_fd = child_fd
    finally:
        os.close(current_fd)


def _validated_cache_file(root: Path, path: Path, *, label: str) -> Path:
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"Missing rollout cache {label}: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise RuntimeError(f"Rollout cache {label} must not be a symlink: {path}")
    if not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(
            f"Rollout cache {label} must be a regular file: {path}"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot resolve rollout cache {label}: {path}") from exc
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Rollout cache {label} escapes cache root: {path}")
    return resolved


def _prepare_cache_output(
    root: Path,
    path: Path,
    *,
    label: str,
) -> tuple[int, Path, tuple[int, int]]:
    parent = path.parent.resolve(strict=True)
    if parent != root:
        raise RuntimeError(f"Rollout cache {label} must be written in {root}")

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _attempt in range(100):
        tmp = root / f".{path.name}.tmp-{secrets.token_hex(12)}"
        try:
            fd = os.open(tmp, flags, 0o600)
        except FileExistsError:
            continue
        opened_stat = os.fstat(fd)
        return fd, tmp, (opened_stat.st_dev, opened_stat.st_ino)
    raise FileExistsError(
        f"Unable to allocate rollout cache temporary file for {path}"
    )


def _cleanup_owned_temp(path: Path, identity: tuple[int, int]) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISLNK(path_stat.st_mode):
        return
    if (path_stat.st_dev, path_stat.st_ino) != identity:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _atomic_torch_save(
    torch: Any,
    payload: Any,
    path: Path,
    *,
    root: Path,
) -> str:
    fd, tmp, identity = _prepare_cache_output(root, path, label=path.name)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(fd)
        digest = _sha256_path(tmp)
        os.replace(tmp, path)
        return digest
    finally:
        _cleanup_owned_temp(tmp, identity)
        os.close(fd)


def _atomic_json_save(payload: dict[str, Any], path: Path, *, root: Path) -> None:
    fd, tmp, identity = _prepare_cache_output(root, path, label=path.name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(fd)
        os.replace(tmp, path)
    finally:
        _cleanup_owned_temp(tmp, identity)
        os.close(fd)


def _safe_torch_load(
    torch: Any,
    path: Path,
    *,
    map_location: Any,
    label: str,
    calculate_digest: bool,
) -> tuple[Any, str | None]:
    digest = None
    try:
        if calculate_digest:
            with path.open("rb") as handle:
                digest = _sha256_handle(handle)
                handle.seek(0)
                payload = torch.load(
                    handle,
                    map_location=map_location,
                    weights_only=True,
                )
        else:
            payload = torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )
    except TypeError:
        raise RuntimeError(
            "Safe rollout cache loading requires a PyTorch version whose "
            "torch.load supports weights_only=True"
        ) from None
    except Exception:
        raise RuntimeError(
            f"Rollout cache {label} could not be loaded safely with "
            "weights_only=True"
        ) from None
    return payload, digest


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_handle(handle)


def _sha256_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_DIGEST_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _schema_kind(payload: Any, *, expected_kind: str, label: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    schema_keys = {"schema", "version", "kind"}
    present = schema_keys.intersection(payload)
    if not present:
        return None
    if present != schema_keys:
        raise RuntimeError(f"Rollout cache {label} has an incomplete schema header")
    if payload["schema"] != CACHE_SCHEMA:
        raise RuntimeError(f"Unsupported rollout cache schema: {payload['schema']!r}")
    version = payload["version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _SUPPORTED_CACHE_VERSIONS
    ):
        raise RuntimeError(
            f"Unsupported rollout cache version: {version!r}"
        )
    if payload["kind"] != expected_kind:
        raise RuntimeError(
            f"Rollout cache {label} kind must be {expected_kind!r}"
        )
    return version


def _validate_metadata_payload(
    payload: Any,
    *,
    expected_step: int,
) -> int | None:
    if not isinstance(payload, dict):
        raise RuntimeError("Rollout cache metadata must be a JSON object")
    version = _schema_kind(
        payload,
        expected_kind="metadata",
        label="metadata",
    )
    required = {"prompts", "metadata", "model_metadata"}
    if version is not None:
        required.update({"step", "tensor_path", "media_path"})
        if payload.get("step") != expected_step:
            raise RuntimeError("Rollout cache metadata step does not match filename")
        expected_tensor = f"batch_{expected_step:06d}.pt"
        if payload.get("tensor_path") != expected_tensor:
            raise RuntimeError(
                "Rollout cache metadata tensor_path does not match the expected "
                "step-local file"
            )
    if version == CACHE_VERSION:
        required.update(
            {
                "generation",
                "tensor_sha256",
                "media_sha256",
            }
        )
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Rollout cache metadata is missing keys: {missing}")
    prompts = payload["prompts"]
    metadata = payload["metadata"]
    if not isinstance(prompts, list) or not all(
        isinstance(item, str) for item in prompts
    ):
        raise RuntimeError("Rollout cache prompts must be a list of strings")
    if not isinstance(metadata, list) or not all(
        isinstance(item, dict) for item in metadata
    ):
        raise RuntimeError("Rollout cache metadata entries must be dictionaries")
    if len(prompts) != len(metadata):
        raise RuntimeError(
            "Rollout cache prompts and metadata must have the same length"
        )
    if not isinstance(payload["model_metadata"], dict):
        raise RuntimeError("Rollout cache model_metadata must be a dictionary")
    if version == CACHE_VERSION:
        _validate_hex_identity(
            payload["generation"],
            length=32,
            label="generation",
        )
        for key in ("tensor_sha256", "media_sha256"):
            _validate_hex_identity(payload[key], length=64, label=key)
    return version


def _validate_tensor_payload(payload: Any) -> int | None:
    import torch

    if not isinstance(payload, dict):
        raise RuntimeError("Rollout cache tensor payload must be a dictionary")
    version = _schema_kind(
        payload,
        expected_kind="tensors",
        label="tensor payload",
    )
    allowed = _TENSOR_FIELDS | (
        {"schema", "version", "kind"} if version is not None else set()
    )
    if version == CACHE_VERSION:
        allowed.add("generation")
    if version is None:
        allowed = _LEGACY_TENSOR_FIELDS
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise RuntimeError(f"Rollout cache tensor payload has unknown keys: {unknown}")
    if version is not None:
        missing = sorted(_TENSOR_FIELDS.difference(payload))
        if missing:
            raise RuntimeError(
                f"Rollout cache tensor payload is missing keys: {missing}"
            )
    if version == CACHE_VERSION:
        if "generation" not in payload:
            raise RuntimeError(
                "Rollout cache tensor payload is missing key: generation"
            )
        _validate_hex_identity(
            payload["generation"],
            length=32,
            label="tensor payload generation",
        )
    for name in _TENSOR_FIELDS.difference({"model_tensors"}):
        value = payload.get(name)
        if value is not None and not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"Rollout cache tensor field {name!r} must be a tensor or None"
            )
    model_tensors = payload.get("model_tensors", {})
    if not isinstance(model_tensors, dict):
        raise RuntimeError("Rollout cache model_tensors must be a dictionary")
    _validate_safe_tensor_tree(model_tensors, label="model_tensors")
    for name in ("branch_id", "branch_ids"):
        value = payload.get(name)
        if value is not None:
            _validate_safe_tensor_tree(value, label=name)
    context = payload.get("context")
    if context is not None and not isinstance(context, dict):
        raise RuntimeError("Rollout cache legacy context must be a dictionary")
    if context is not None:
        _validate_safe_tensor_tree(context, label="context")
    return version


def _validate_media_payload(payload: Any) -> Any:
    import torch

    media = payload
    if isinstance(payload, dict):
        if not _schema_kind(payload, expected_kind="media", label="media payload"):
            raise RuntimeError("Rollout cache media payload has no valid schema header")
        version = _payload_version(payload)
        allowed = {"schema", "version", "kind", "media"}
        if version == CACHE_VERSION:
            allowed.add("generation")
        unknown = sorted(set(payload).difference(allowed))
        if unknown:
            raise RuntimeError(f"Rollout cache media payload has unknown keys: {unknown}")
        if "media" not in payload:
            raise RuntimeError("Rollout cache media payload is missing key: media")
        if version == CACHE_VERSION:
            if "generation" not in payload:
                raise RuntimeError(
                    "Rollout cache media payload is missing key: generation"
                )
            _validate_hex_identity(
                payload["generation"],
                length=32,
                label="media payload generation",
            )
        media = payload["media"]
    if not isinstance(media, torch.Tensor):
        raise RuntimeError("Rollout cache media payload must contain a tensor")
    if media.ndim not in {4, 5}:
        raise RuntimeError("Rollout cache media tensor must have BCHW or BFCHW shape")
    return media


def _payload_version(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    return version if isinstance(version, int) and not isinstance(version, bool) else None


def _validate_payload_publication(
    metadata: dict[str, Any],
    *,
    metadata_version: int | None,
    payload: Any,
    payload_version: int | None,
    actual_sha256: str | None,
    digest_key: str,
    label: str,
) -> None:
    if payload_version != metadata_version:
        raise RuntimeError(
            f"Rollout cache {label} version does not match metadata publication"
        )
    if metadata_version != CACHE_VERSION:
        return
    if not isinstance(payload, dict) or (
        payload.get("generation") != metadata["generation"]
    ):
        raise RuntimeError(
            f"Rollout cache {label} generation does not match metadata publication"
        )
    if actual_sha256 != metadata[digest_key]:
        raise RuntimeError(
            f"Rollout cache {label} digest does not match metadata publication"
        )


def _validate_hex_identity(value: Any, *, length: int, label: str) -> None:
    if not isinstance(value, str) or len(value) != length:
        raise RuntimeError(f"Rollout cache {label} must be {length} lowercase hex digits")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Rollout cache {label} must be {length} lowercase hex digits")


def _validate_safe_tensor_tree(value: Any, *, label: str) -> None:
    import torch

    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(
                    f"Rollout cache {label} keys must be strings"
                )
            _validate_safe_tensor_tree(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_safe_tensor_tree(item, label=label)
        return
    raise RuntimeError(
        f"Rollout cache {label} contains an unsafe value type: "
        f"{type(value).__name__}"
    )


def _resolve_media_path(
    root: Path,
    base: Path,
    declared: Any,
) -> Path:
    expected = base.with_suffix(".media.pt")
    if declared is not None:
        if not isinstance(declared, str):
            raise RuntimeError("Rollout cache media_path must be a string")
        candidate = Path(declared)
        if candidate.is_absolute() or candidate.parts != (expected.name,):
            raise RuntimeError(
                "Rollout cache media_path must name the expected step-local file"
            )
    return _validated_cache_file(root, expected, label="media payload")


def _legacy_context(
    metadata_payload: dict[str, Any],
    tensor_payload: dict[str, Any],
) -> StepContext | None:
    seed = metadata_payload.get("seed", tensor_payload.get("seed"))
    epoch_tag = metadata_payload.get(
        "epoch_tag",
        tensor_payload.get("epoch_tag"),
    )
    if seed is None and epoch_tag is None:
        return None
    resolved_epoch = int(epoch_tag or 0)
    return StepContext(
        step=int(metadata_payload.get("step", resolved_epoch)),
        seed=int(seed or 0),
        epoch_tag=resolved_epoch,
    )
