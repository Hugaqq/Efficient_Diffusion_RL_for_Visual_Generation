"""Lightweight Diffusers checkpoint inventory helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ADAPTER_KEYS = ("sd3_tempflow", "world_r1_wan_legacy")


@dataclass
class CheckpointRecord:
    path: str
    model_index_path: str
    class_name: str
    model_type: str
    adapter_keys: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative_depth(path: Path, root: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return len(path.parts)


def iter_model_index_files(root: str | Path, *, max_depth: int = 5) -> list[Path]:
    resolved = Path(root).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {resolved}")
    if resolved.is_file():
        if resolved.name != "model_index.json":
            raise ValueError(f"Checkpoint file must be model_index.json, got: {resolved}")
        return [resolved.resolve()]

    matches = []
    for path in sorted(resolved.rglob("model_index.json")):
        if _relative_depth(path, resolved) <= max_depth + 1:
            matches.append(path.resolve())
    return matches


def _classify_checkpoint(model_index: dict[str, Any], checkpoint_dir: Path) -> tuple[str, list[str]]:
    class_name = str(model_index.get("_class_name", ""))
    haystack = " ".join([class_name, checkpoint_dir.name]).lower()
    if "stable-diffusion-3" in haystack or "stablediffusion3" in haystack or "sd3" in haystack:
        return "sd3", ["sd3_tempflow"]
    if "wan" in haystack:
        return "wan", ["world_r1_wan_legacy"]
    return "unknown", []


def read_checkpoint_record(model_index_path: str | Path) -> CheckpointRecord:
    path = Path(model_index_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        model_index = json.load(handle)
    checkpoint_dir = path.parent
    model_type, adapter_keys = _classify_checkpoint(model_index, checkpoint_dir)
    return CheckpointRecord(
        path=str(checkpoint_dir),
        model_index_path=str(path),
        class_name=str(model_index.get("_class_name", "")),
        model_type=model_type,
        adapter_keys=adapter_keys,
    )


def build_checkpoint_inventory(
    roots: list[str | Path],
    *,
    required_adapters: list[str] | None = None,
    max_depth: int = 5,
) -> dict[str, Any]:
    records: list[CheckpointRecord] = []
    errors = []
    for root in roots:
        try:
            for model_index_path in iter_model_index_files(root, max_depth=max_depth):
                records.append(read_checkpoint_record(model_index_path))
        except Exception as exc:  # noqa: BLE001 - inventory reports all roots together
            errors.append({"root": str(root), "message": str(exc)})

    found_adapters = sorted({adapter for record in records for adapter in record.adapter_keys})
    required = sorted(required_adapters or [])
    unknown_required = sorted(set(required) - set(ADAPTER_KEYS))
    missing_adapters = sorted(set(required) - set(found_adapters))
    valid = not errors and not unknown_required and not missing_adapters
    return {
        "valid": valid,
        "roots": [str(root) for root in roots],
        "records": [record.to_dict() for record in records],
        "found_adapters": found_adapters,
        "required_adapters": required,
        "missing_adapters": missing_adapters,
        "unknown_required_adapters": unknown_required,
        "errors": errors,
    }
