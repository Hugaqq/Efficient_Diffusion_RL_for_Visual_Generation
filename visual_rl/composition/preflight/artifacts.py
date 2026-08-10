"""Read-only filesystem artifact identities for environment preflight."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from visual_rl.composition.preflight.types import (
    ArtifactIdentityRequest,
    ArtifactIdentityResolution,
)
from visual_rl.core.filesystem_identity import (
    ALL_FILES_CONTENT_POLICY,
    FILESYSTEM_ARTIFACT_IDENTITY_SCHEMA,
    new_filesystem_artifact_digest,
    update_filesystem_artifact_record,
    update_filesystem_file_record,
)
from visual_rl.core.immutable import FrozenMapping
from visual_rl.data import DatasetArtifactBinding, SourceLocationBinding

__all__ = (
    "ArtifactIdentityError",
    "FilesystemArtifactIdentityResolver",
)

_CHUNK_SIZE = 1024 * 1024
_IDENTITY_SCHEMA = FILESYSTEM_ARTIFACT_IDENTITY_SCHEMA
_ALL_FILES_POLICY = ALL_FILES_CONTENT_POLICY
_PYTHON_CODE_POLICY = "python-code.v1"


class ArtifactIdentityError(RuntimeError):
    """An artifact cannot be identified without weakening identity guarantees."""


class FilesystemArtifactIdentityResolver:
    """Resolve model, dataset, reward, and code filesystem identities.

    Locations are used only to read artifacts.  They are deliberately absent
    from the returned identity, as are mtimes and permission bits.  Hashes are
    cached only for the duration of one resolution call so repeated dataset
    references share I/O while later calls still observe content changes.
    """

    def __init__(self, code_root: Path | None = None) -> None:
        if code_root is not None and not isinstance(code_root, Path):
            raise TypeError("code_root must be a Path or None")
        # The default identity must cover every importable training module, not
        # merely this ``composition`` subtree.  Otherwise model/algorithm code
        # could change while a materialized recipe kept the same code identity.
        selected = Path(__file__).resolve().parents[2] if code_root is None else code_root
        self._code_root = Path(os.path.abspath(selected.expanduser()))

    @property
    def code_root(self) -> Path:
        """Return the location probed for code identity, never its identity."""

        return self._code_root

    def resolve_artifact_identities(
        self,
        request: ArtifactIdentityRequest,
    ) -> ArtifactIdentityResolution:
        """Return the exact artifact identity shape required by a recipe."""

        if not isinstance(request, ArtifactIdentityRequest):
            raise TypeError("request must be an ArtifactIdentityRequest")

        cache: dict[tuple[str, str], dict[str, Any]] = {}

        def identify(
            location: Path,
            *,
            content_policy: str = _ALL_FILES_POLICY,
        ) -> dict[str, Any]:
            key = (_location_cache_key(location), content_policy)
            identity = cache.get(key)
            if identity is None:
                identity = _hash_artifact(
                    location,
                    content_policy=content_policy,
                )
                cache[key] = identity
            return dict(identity)

        dataset_refs = _dataset_artifact_refs(request)
        provided_dataset_refs = tuple(
            artifact_ref for artifact_ref, _location in request.locations.datasets
        )
        if provided_dataset_refs != dataset_refs:
            raise ArtifactIdentityError(
                "dataset artifact locations must exactly cover source plan refs: "
                f"expected={list(dataset_refs)}, "
                f"got={list(provided_dataset_refs)}"
            )
        dataset_bindings: list[DatasetArtifactBinding] = []
        for artifact_ref in dataset_refs:
            try:
                location = request.locations.dataset(artifact_ref)
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactIdentityError(
                    f"dataset plan references unavailable artifact {artifact_ref!r}"
                ) from exc
            dataset_bindings.append(
                DatasetArtifactBinding(
                    artifact_ref=artifact_ref,
                    artifact_location=location,
                    expected_content_identity=FrozenMapping(identify(location)),
                )
            )

        reward_refs = _reward_artifact_refs(request)
        provided_reward_refs = tuple(
            artifact_ref for artifact_ref, _location in request.locations.rewards
        )
        if provided_reward_refs != reward_refs:
            raise ArtifactIdentityError(
                "reward artifact locations must exactly cover physical descriptor "
                f"refs: expected={list(reward_refs)}, "
                f"got={list(provided_reward_refs)}"
            )
        reward_resources = tuple(
            (
                artifact_ref,
                FrozenMapping(identify(request.locations.reward(artifact_ref))),
            )
            for artifact_ref in reward_refs
        )

        return ArtifactIdentityResolution(
            model_artifact_identity=FrozenMapping(identify(request.locations.model)),
            source_locations=SourceLocationBinding(
                source_plan_id=request.resolved.source_plan.plan_id,
                artifacts=tuple(dataset_bindings),
            ),
            reward_artifact_identities=reward_resources,
            code_artifact_identity=FrozenMapping(
                identify(
                    self._code_root,
                    content_policy=_PYTHON_CODE_POLICY,
                )
            ),
        )


def _dataset_artifact_refs(
    request: ArtifactIdentityRequest,
) -> tuple[str, ...]:
    return tuple(
        sorted({source.artifact_ref for source in request.resolved.source_plan.sources})
    )


def _reward_artifact_refs(
    request: ArtifactIdentityRequest,
) -> tuple[str, ...]:
    refs = {
        resource.artifact_ref for resource in request.resolved.reward_plan.resources
    }
    if not refs:
        raise ArtifactIdentityError(
            "resolved recipe must declare at least one reward artifact ref"
        )
    return tuple(sorted(refs))


def _location_cache_key(location: Path) -> str:
    if not isinstance(location, Path):
        raise TypeError("artifact locations must be Paths")
    return os.path.normcase(os.path.abspath(location))


def _hash_artifact(
    location: Path,
    *,
    content_policy: str = _ALL_FILES_POLICY,
) -> dict[str, Any]:
    if content_policy not in {_ALL_FILES_POLICY, _PYTHON_CODE_POLICY}:
        raise ValueError(f"unsupported artifact content policy {content_policy!r}")
    root = Path(os.path.abspath(location))
    metadata = _lstat(root)
    node_type = _node_type(root, metadata)
    digest = new_filesystem_artifact_digest(content_policy=content_policy)
    totals = [0, 0]

    if node_type == "file":
        _hash_file_record(digest, root, ".", metadata, totals)
    else:
        _hash_directory(
            digest,
            root,
            ".",
            metadata,
            totals,
            content_policy=content_policy,
        )

    return {
        "identity_schema": _IDENTITY_SCHEMA,
        "content_policy": content_policy,
        "node_type": node_type,
        "content_sha256": digest.hexdigest(),
        "file_count": totals[0],
        "byte_count": totals[1],
    }


def _hash_directory(
    digest: Any,
    directory: Path,
    relative: str,
    metadata: os.stat_result,
    totals: list[int],
    *,
    content_policy: str,
) -> None:
    _require_directory_access(directory, metadata)
    update_filesystem_artifact_record(
        digest,
        b"directory",
        _relative_bytes(relative),
    )
    before = _stable_metadata(metadata)
    try:
        children = sorted(
            directory.iterdir(),
            key=lambda item: _name_bytes(item.name),
        )
    except OSError as exc:
        raise ArtifactIdentityError(
            f"cannot enumerate artifact directory: {directory}"
        ) from exc

    for child in children:
        child_metadata = _lstat(child)
        child_type = _node_type(child, child_metadata)
        if content_policy == _PYTHON_CODE_POLICY and _excluded_code_node(
            child,
            node_type=child_type,
        ):
            _validate_excluded_node(child, child_metadata, node_type=child_type)
            continue
        child_relative = child.name if relative == "." else f"{relative}/{child.name}"
        if child_type == "tree":
            _hash_directory(
                digest,
                child,
                child_relative,
                child_metadata,
                totals,
                content_policy=content_policy,
            )
        else:
            _hash_file_record(
                digest,
                child,
                child_relative,
                child_metadata,
                totals,
            )

    after = _lstat(directory)
    if _node_type(directory, after) != "tree" or _stable_metadata(after) != before:
        raise ArtifactIdentityError(
            f"artifact directory changed while it was being hashed: {directory}"
        )


def _excluded_code_node(path: Path, *, node_type: str) -> bool:
    if node_type == "tree":
        return path.name == "__pycache__"
    return path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}


def _validate_excluded_node(
    path: Path,
    metadata: os.stat_result,
    *,
    node_type: str,
) -> None:
    """Traverse excluded nodes so a cache directory cannot conceal symlinks."""

    before = _stable_metadata(metadata)
    if node_type == "file":
        after = _lstat(path)
        if _node_type(path, after) != "file" or _stable_metadata(after) != before:
            raise ArtifactIdentityError(
                f"excluded runtime artifact changed during inspection: {path}"
            )
        return

    _require_directory_access(path, metadata)
    try:
        children = sorted(path.iterdir(), key=lambda item: _name_bytes(item.name))
    except OSError as exc:
        raise ArtifactIdentityError(
            f"cannot enumerate excluded runtime artifact directory: {path}"
        ) from exc
    for child in children:
        child_metadata = _lstat(child)
        child_type = _node_type(child, child_metadata)
        _validate_excluded_node(child, child_metadata, node_type=child_type)
    after = _lstat(path)
    if _node_type(path, after) != "tree" or _stable_metadata(after) != before:
        raise ArtifactIdentityError(
            f"excluded runtime artifact directory changed during inspection: {path}"
        )


def _hash_file_record(
    digest: Any,
    path: Path,
    relative: str,
    metadata: os.stat_result,
    totals: list[int],
) -> None:
    _require_file_access(path, metadata)
    before = _stable_metadata(metadata)
    content_digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
            ) != (metadata.st_dev, metadata.st_ino):
                raise ArtifactIdentityError(
                    f"artifact file changed before it could be read: {path}"
                )
            while True:
                chunk = handle.read(_CHUNK_SIZE)
                if not chunk:
                    break
                content_digest.update(chunk)
                byte_count += len(chunk)
            closed_metadata = os.fstat(handle.fileno())
    except ArtifactIdentityError:
        raise
    except OSError as exc:
        raise ArtifactIdentityError(f"cannot read artifact file: {path}") from exc

    after = _lstat(path)
    if (
        _node_type(path, after) != "file"
        or _stable_metadata(opened) != before
        or _stable_metadata(closed_metadata) != before
        or _stable_metadata(after) != before
        or byte_count != metadata.st_size
    ):
        raise ArtifactIdentityError(
            f"artifact file changed while it was being hashed: {path}"
        )

    update_filesystem_file_record(
        digest,
        relative=_relative_bytes(relative),
        byte_count=byte_count,
        content_sha256=content_digest.digest(),
    )
    totals[0] += 1
    totals[1] += byte_count


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise ArtifactIdentityError(f"artifact does not exist: {path}") from exc
    except OSError as exc:
        raise ArtifactIdentityError(f"cannot inspect artifact: {path}") from exc


def _node_type(path: Path, metadata: os.stat_result) -> str:
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactIdentityError(f"artifact paths must not be symlinks: {path}")
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    if stat.S_ISDIR(metadata.st_mode):
        return "tree"
    raise ArtifactIdentityError(
        f"artifact entries must be regular files or directories: {path}"
    )


def _require_file_access(path: Path, metadata: os.stat_result) -> None:
    if metadata.st_mode & 0o444 == 0:
        raise ArtifactIdentityError(f"artifact file is not readable: {path}")


def _require_directory_access(path: Path, metadata: os.stat_result) -> None:
    if metadata.st_mode & 0o444 == 0 or metadata.st_mode & 0o111 == 0:
        raise ArtifactIdentityError(f"artifact directory is not readable: {path}")


def _stable_metadata(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _relative_bytes(relative: str) -> bytes:
    return relative.encode("utf-8", errors="surrogateescape")


def _name_bytes(name: str) -> bytes:
    return name.encode("utf-8", errors="surrogateescape")
