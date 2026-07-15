"""Verify the pinned 19-file Wan snapshot after laptop-to-server transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PINNED_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.model / "wan_snapshot_manifest.sha256"
    revision_path = args.model / "wan_snapshot_revision.txt"
    entries = []
    seen = set()
    for line in manifest_path.read_text().splitlines():
        expected_hash, expected_size, relative_name = line.split(" ", maxsplit=2)
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or relative_name in seen:
            raise ValueError(f"unsafe or duplicate manifest path: {relative_name}")
        seen.add(relative_name)
        path = args.model / relative
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = file_sha256(path) if exists else None
        entries.append(
            {
                "path": relative_name,
                "exists": exists,
                "expected_size": int(expected_size),
                "actual_size": actual_size,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "valid": exists
                and actual_size == int(expected_size)
                and actual_hash == expected_hash,
            }
        )
    revision = revision_path.read_text().strip() if revision_path.is_file() else None
    all_files = {
        str(path.relative_to(args.model))
        for path in args.model.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    auxiliary = {manifest_path.name, revision_path.name}
    extras = sorted(all_files.difference(seen).difference(auxiliary))
    result = {
        "valid": len(entries) == 19
        and all(entry["valid"] for entry in entries)
        and revision == PINNED_REVISION,
        "pinned_revision": PINNED_REVISION,
        "observed_revision": revision,
        "required_file_count": len(entries),
        "required_total_bytes": sum(entry["expected_size"] for entry in entries),
        "entries": entries,
        "extra_non_cache_files": extras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
