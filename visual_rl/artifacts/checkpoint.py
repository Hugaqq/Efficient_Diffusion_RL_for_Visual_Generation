"""Atomic metadata and complete training-state checkpoints."""

from __future__ import annotations

import json
import hashlib
import inspect
import random
import subprocess
from functools import lru_cache
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from visual_rl.artifacts.serialization import to_jsonable

CHECKPOINT_FORMAT_VERSION = 1


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_fingerprint(
    config: dict[str, Any],
    implementation: dict[str, Any] | None = None,
) -> str:
    """Hash training semantics while allowing run location/length to change."""

    source = deepcopy(to_jsonable(config))
    semantic_keys = {
        "seed",
        "use_lora",
        "per_prompt_stat_tracking",
        "model",
        "dataset",
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
    encoded = json.dumps(
        payload,
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


def save_training_state(
    checkpoint_dir: str | Path,
    *,
    optimizer: Any,
    plugin: Any,
    step: int,
    config: dict[str, Any],
    implementation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config, implementation)
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "plugin": plugin.state_dict(),
        "rng": _rng_state(),
        "config_fingerprint": fingerprint,
        "implementation": to_jsonable(implementation or {}),
    }
    target = path / "training_state.pt"
    tmp = target.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(target)
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "config_fingerprint": fingerprint,
        "training_state": target.name,
    }
    save_json(path / "checkpoint.json", metadata)
    return metadata


def load_training_state(
    checkpoint_dir: str | Path,
    *,
    optimizer: Any,
    plugin: Any,
    config: dict[str, Any],
    implementation: dict[str, Any] | None = None,
) -> int:
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / "training_state.pt"
    metadata_path = checkpoint_dir / "checkpoint.json"
    if not path.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"Checkpoint is missing complete training state: {checkpoint_dir}"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = load_json(metadata_path)
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
    if int(state["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint format_version: {state['format_version']}"
        )
    if int(metadata.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint metadata format_version: {metadata.get('format_version')}"
        )
    if int(metadata.get("step", -1)) != int(state["step"]):
        raise RuntimeError("checkpoint.json step does not match training_state.pt")
    if metadata.get("config_fingerprint") != state.get("config_fingerprint"):
        raise RuntimeError(
            "checkpoint.json fingerprint does not match training_state.pt"
        )

    expected = config_fingerprint(config, implementation)
    actual = state.get("config_fingerprint")
    if actual != expected:
        raise RuntimeError(
            "Resume config does not match checkpoint training semantics: "
            f"expected {actual}, got {expected}"
        )
    optimizer.load_state_dict(state["optimizer"])
    plugin.load_state_dict(dict(state.get("plugin") or {}))
    _restore_rng_state(dict(state["rng"]))
    return int(state["step"])


def _object_identity(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    cls = type(value)
    class_name = f"{cls.__module__}.{cls.__qualname__}"
    try:
        source = inspect.getsource(cls).encode("utf-8")
    except (OSError, TypeError):
        source = class_name.encode("utf-8")
    return {
        "class": class_name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "module_sha256": _module_hash(cls),
    }


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


def _rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
