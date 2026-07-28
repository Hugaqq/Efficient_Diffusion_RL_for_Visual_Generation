"""Small content hashes for frozen local model and scorer assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: str | Path) -> str:
    file_path = Path(path).expanduser().resolve(strict=True)
    if not file_path.is_file():
        raise ValueError(f"Expected a local file, got {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    """Hash relative filenames and bytes, independent of the tree location."""

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected a local directory, got {root}")
    files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def mapping_sha256(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(values),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
