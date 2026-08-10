"""Path-free codec for ``filesystem-artifact.v1`` identity records.

Filesystem traversal and stable I/O belong to callers.  This module owns only
the domain-separated byte encoding shared by streaming preflight hashing and
single-file in-memory snapshot verification.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

__all__ = (
    "ALL_FILES_CONTENT_POLICY",
    "FILESYSTEM_ARTIFACT_IDENTITY_SCHEMA",
    "filesystem_file_identity_from_snapshot",
    "new_filesystem_artifact_digest",
    "update_filesystem_artifact_record",
    "update_filesystem_file_record",
)

FILESYSTEM_ARTIFACT_IDENTITY_SCHEMA = "filesystem-artifact.v1"
ALL_FILES_CONTENT_POLICY = "all-files.v1"
_TREE_HASH_DOMAIN = b"visual-rl/filesystem-artifact/v1\0"


class _Digest(Protocol):
    def update(self, value: bytes, /) -> object: ...

    def digest(self) -> bytes: ...

    def hexdigest(self) -> str: ...


def update_filesystem_artifact_record(digest: _Digest, *fields: bytes) -> None:
    """Append one length-delimited record to an artifact digest."""

    if not fields or any(not isinstance(field, bytes) for field in fields):
        raise TypeError("filesystem artifact record fields must be bytes")
    for field in fields:
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)


def new_filesystem_artifact_digest(*, content_policy: str) -> _Digest:
    """Start one domain-separated artifact identity stream."""

    if not isinstance(content_policy, str) or not content_policy:
        raise ValueError("content_policy must be a non-empty string")
    try:
        encoded_policy = content_policy.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("content_policy must contain only ASCII characters") from exc
    digest = hashlib.sha256()
    digest.update(_TREE_HASH_DOMAIN)
    update_filesystem_artifact_record(digest, encoded_policy)
    return digest


def update_filesystem_file_record(
    digest: _Digest,
    *,
    relative: bytes,
    byte_count: int,
    content_sha256: bytes,
) -> None:
    """Append the canonical record for one already-hashed regular file."""

    if not isinstance(relative, bytes):
        raise TypeError("relative must be bytes")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError("byte_count must be a non-negative integer")
    if not isinstance(content_sha256, bytes) or len(content_sha256) != 32:
        raise ValueError("content_sha256 must be a 32-byte SHA-256 digest")
    update_filesystem_artifact_record(
        digest,
        b"file",
        relative,
        str(byte_count).encode("ascii"),
        content_sha256,
    )


def filesystem_file_identity_from_snapshot(
    snapshot: bytes,
    *,
    content_policy: str = ALL_FILES_CONTENT_POLICY,
) -> dict[str, object]:
    """Identify one root file from the exact bytes that a caller will consume."""

    if not isinstance(snapshot, bytes):
        raise TypeError("snapshot must be bytes")
    digest = new_filesystem_artifact_digest(content_policy=content_policy)
    update_filesystem_file_record(
        digest,
        relative=b".",
        byte_count=len(snapshot),
        content_sha256=hashlib.sha256(snapshot).digest(),
    )
    return {
        "identity_schema": FILESYSTEM_ARTIFACT_IDENTITY_SCHEMA,
        "content_policy": content_policy,
        "node_type": "file",
        "content_sha256": digest.hexdigest(),
        "file_count": 1,
        "byte_count": len(snapshot),
    }
