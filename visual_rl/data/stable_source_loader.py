"""Canonical stable-byte loader for path-free dataset source plans.

The public loader accepts only :class:`SourceLoadRequest`.  Every bound file is
opened exactly once, read into one immutable bytes snapshot, checked against its
preflight ``filesystem-artifact.v1`` identity, and only then parsed according
to the plan's explicit format.  Absolute locations are I/O coordinates only.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from visual_rl.core.filesystem_identity import (
    filesystem_file_identity_from_snapshot,
)
from visual_rl.data.source_plan import (
    DatasetArtifactBinding,
    SourceLoadError,
    SourceLoadRequest,
)
from visual_rl.data.source_records import (
    DatasetSourceDescriptor,
    build_source_sequence,
    parse_source_records,
    source_descriptor,
)
from visual_rl.data.source_sampler import SourceSequence

__all__ = ("load_stable_source_sequences",)


def load_stable_source_sequences(
    request: SourceLoadRequest,
) -> tuple[SourceSequence, ...]:
    """Load a validated semantic plan from identity-bound stable snapshots."""

    if not isinstance(request, SourceLoadRequest):
        raise TypeError("request must be a SourceLoadRequest")

    # Re-run the immutable join invariant at the I/O boundary.  This rejects a
    # forged/subclassed request before any artifact is opened.
    validated = SourceLoadRequest(plan=request.plan, locations=request.locations)
    descriptors: dict[str, DatasetSourceDescriptor] = {
        source.source_id: source_descriptor(
            source.selector,
            source_id=source.source_id,
        )
        for source in validated.plan.sources
    }

    # Verify the complete content binding before parsing any source.  Shared
    # artifact refs intentionally share this one immutable bytes object.
    snapshots = {
        binding.artifact_ref: _read_verified_snapshot(binding)
        for binding in validated.locations.artifacts
    }

    return tuple(
        build_source_sequence(
            source_id=source.source_id,
            descriptor=descriptors[source.source_id],
            records=parse_source_records(
                snapshots[source.artifact_ref],
                source_id=source.source_id,
                file_format=source.format,
            ),
        )
        for source in validated.plan.sources
    )


def _read_verified_snapshot(binding: DatasetArtifactBinding) -> bytes:
    path = binding.artifact_location
    before = _lstat_regular_file(path, artifact_ref=binding.artifact_ref)
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
            ) != (before.st_dev, before.st_ino):
                raise SourceLoadError(
                    f"dataset artifact {binding.artifact_ref!r} changed before "
                    "its stable snapshot could be read"
                )
            snapshot = handle.read()
            closed = os.fstat(handle.fileno())
    except SourceLoadError:
        raise
    except OSError as exc:
        raise SourceLoadError(
            f"cannot read dataset artifact {binding.artifact_ref!r}: {path}"
        ) from exc

    after = _lstat_after_read(path, artifact_ref=binding.artifact_ref)
    stable = _stable_metadata(before)
    if (
        _stable_metadata(opened) != stable
        or _stable_metadata(closed) != stable
        or _stable_metadata(after) != stable
        or len(snapshot) != before.st_size
    ):
        raise SourceLoadError(
            f"dataset artifact {binding.artifact_ref!r} changed while its "
            "stable snapshot was being read"
        )

    observed_identity = filesystem_file_identity_from_snapshot(snapshot)
    expected_identity = dict(binding.expected_content_identity.items())
    if observed_identity != expected_identity:
        differing_fields = tuple(
            key
            for key in sorted(expected_identity)
            if observed_identity.get(key) != expected_identity[key]
        )
        raise SourceLoadError(
            f"dataset artifact {binding.artifact_ref!r} content identity "
            f"mismatch before parsing; fields={list(differing_fields)}"
        )
    return snapshot


def _lstat_regular_file(path: Path, *, artifact_ref: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise SourceLoadError(
            f"dataset artifact {artifact_ref!r} does not exist: {path}"
        ) from exc
    except OSError as exc:
        raise SourceLoadError(
            f"cannot inspect dataset artifact {artifact_ref!r}: {path}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise SourceLoadError(
            f"dataset artifact {artifact_ref!r} must not be a symlink: {path}"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise SourceLoadError(
            f"dataset artifact {artifact_ref!r} is not a regular file: {path}"
        )
    if metadata.st_mode & 0o444 == 0:
        raise SourceLoadError(
            f"dataset artifact {artifact_ref!r} is not readable: {path}"
        )
    return metadata


def _lstat_after_read(path: Path, *, artifact_ref: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SourceLoadError(
            f"dataset artifact {artifact_ref!r} changed while its stable "
            "snapshot was being read"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SourceLoadError(
            f"dataset artifact {artifact_ref!r} changed while its stable "
            "snapshot was being read"
        )
    return metadata


def _stable_metadata(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
