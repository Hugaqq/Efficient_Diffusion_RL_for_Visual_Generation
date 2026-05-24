"""Wan checkpoint probe helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter
from visual_rl.third_party.legacy import resolve_legacy_repo


@dataclass
class WanCheckpointProbeConfig:
    model_path: str
    repo_root: str = "reference_code/World-R1-main"
    torch_dtype: str = "auto"
    device: str = ""
    local_files_only: bool = True
    low_cpu_mem_usage: bool = True
    manifest_only: bool = False


def _read_model_index(model_path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    model_index_path = model_path / "model_index.json"
    if not model_index_path.exists():
        return {}, [f"model_index.json not found under {model_path}"]
    try:
        with model_index_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - convert file/JSON failures to structured probe output
        return {}, [f"failed to read {model_index_path}: {exc}"]
    return data, errors


def run_wan_checkpoint_probe(config: WanCheckpointProbeConfig) -> dict[str, Any]:
    model_path_text = str(config.model_path).strip()
    if not model_path_text:
        raise ValueError("model_path must not be empty.")

    model_path = Path(model_path_text).expanduser()
    repo_root = resolve_legacy_repo(config.repo_root)
    model_index, manifest_errors = _read_model_index(model_path)
    class_name = str(model_index.get("_class_name", ""))
    manifest_valid = "wan" in " ".join([class_name, model_path.name]).lower()

    payload: dict[str, Any] = {
        "valid": False,
        "model_path": model_path_text,
        "repo_root": str(repo_root),
        "model_index_path": str(model_path / "model_index.json"),
        "class_name": class_name,
        "manifest_valid": manifest_valid,
        "loaded": False,
        "load_requested": not config.manifest_only,
        "torch_dtype": config.torch_dtype,
        "device": config.device,
        "local_files_only": bool(config.local_files_only),
        "low_cpu_mem_usage": bool(config.low_cpu_mem_usage),
        "errors": manifest_errors.copy(),
        "warnings": [],
        "side_effects": {
            "trainer_constructed": False,
            "sample_called": False,
            "checkpoint_written": False,
            "output_dir_written": False,
        },
    }
    if class_name and not manifest_valid:
        payload["errors"].append(f"checkpoint class does not look like Wan: {class_name!r}")
    if config.manifest_only:
        payload["valid"] = manifest_valid and not payload["errors"]
        return payload
    if payload["errors"]:
        return payload

    adapter = WorldR1WanLegacyAdapter(
        {
            "model_path": model_path_text,
            "repo_root": str(repo_root),
            "torch_dtype": config.torch_dtype,
            "device": config.device,
            "local_files_only": config.local_files_only,
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
        }
    )
    adapter.load()
    payload.update(
        {
            "valid": True,
            "loaded": True,
            "pipeline_class": type(adapter.pipeline).__name__ if adapter.pipeline is not None else "",
            "transformer_class": type(adapter.transformer).__name__ if adapter.transformer is not None else "",
            "transformer_present": adapter.transformer is not None,
        }
    )
    return payload
