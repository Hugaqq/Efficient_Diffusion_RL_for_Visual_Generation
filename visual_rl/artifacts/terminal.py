"""Filesystem-safe terminal run artifact primitives.

This module owns bytes, paths, digests, and atomic publication. Runtime code
supplies already validated semantic payloads and never implements a second
artifact transaction or JSON parser.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from visual_rl.core.serialization import canonical_json_text
from visual_rl.composition.config.specs import LaunchSpec, RewardRuntimeBindingSpec
from visual_rl.core.types import FrozenMapping, to_plain_dict

__all__ = (
    "artifact_paths",
    "atomic_write",
    "build_launch_runtime_audit",
    "canonical_line",
    "finalize_terminal_run",
    "prepare_output_dir",
    "read_exact_json",
    "read_single_json_line",
    "regular_file",
    "sha256_bytes",
    "sha256_file",
    "TerminalArtifactError",
    "TerminalFinalizationRequest",
    "TerminalFinalizationResult",
)


class TerminalArtifactError(RuntimeError):
    """Committed terminal sidecars cannot form one authoritative run result."""


@dataclass(frozen=True, slots=True)
class TerminalFinalizationRequest:
    """Runtime-projected primitive inputs for terminal sidecar finalization."""

    current_output_dir: Path
    resume_from: Path | None
    checkpoint_path: Path
    run_id: str
    committed_steps: int
    checkpoint_contract_id: str
    progress_id: str
    state_tree_id: str
    terminal_artifacts: Mapping[str, str]
    current_recipe_payload: Mapping[str, object]
    last_metrics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TerminalFinalizationResult:
    """Validated paths and success receipt for one completed run."""

    output_dir: Path
    checkpoint_path: Path
    resolved_path: Path
    manifest_path: Path
    metrics_path: Path
    marker_path: Path
    success_payload: FrozenMapping


def finalize_terminal_run(
    request: TerminalFinalizationRequest,
) -> TerminalFinalizationResult:
    """Validate committed sidecars, publish SUCCESS, and return exact paths."""

    if not isinstance(request, TerminalFinalizationRequest):
        raise TypeError("request must be TerminalFinalizationRequest")
    artifacts = dict(request.terminal_artifacts)
    required = {
        "checkpoint_relative_path",
        "resolved_recipe_relative_path",
        "run_manifest_relative_path",
        "metrics_relative_path",
        "marker_relative_path",
        "resolved_recipe_sha256",
        "run_manifest_sha256",
        "metrics_sha256",
    }
    if set(artifacts) != required:
        raise TerminalArtifactError("terminal artifact receipt has unexpected fields")
    for name in required:
        if not isinstance(artifacts[name], str) or not artifacts[name]:
            raise TerminalArtifactError(
                f"terminal artifact field {name!r} must be non-empty"
            )
    output_dir = _terminal_output_dir(request, artifacts)
    paths = artifact_paths(output_dir)
    expected_names = {
        "resolved": artifacts["resolved_recipe_relative_path"],
        "manifest": artifacts["run_manifest_relative_path"],
        "metrics": artifacts["metrics_relative_path"],
        "marker": artifacts["marker_relative_path"],
    }
    for name, relative in expected_names.items():
        if paths[name] != output_dir / relative:
            raise TerminalArtifactError("terminal artifact filename changed")
    expected_checkpoint = output_dir / artifacts["checkpoint_relative_path"]
    if expected_checkpoint != request.checkpoint_path:
        raise TerminalArtifactError(
            "final checkpoint path differs from output artifacts"
        )
    for name in ("resolved", "manifest", "metrics"):
        regular_file(paths[name])
    for name, digest_key in (
        ("resolved", "resolved_recipe_sha256"),
        ("manifest", "run_manifest_sha256"),
        ("metrics", "metrics_sha256"),
    ):
        if sha256_file(paths[name]) != artifacts[digest_key]:
            raise TerminalArtifactError(f"{name} artifact digest changed")

    recorded_recipe = read_exact_json(paths["resolved"])
    current_recipe = dict(request.current_recipe_payload)
    if set(recorded_recipe) != set(current_recipe):
        raise TerminalArtifactError("resolved recipe artifact schema changed")
    diagnostic_fields = {"compatibility_inspection"}
    if {
        key: value
        for key, value in recorded_recipe.items()
        if key not in diagnostic_fields
    } != {
        key: value
        for key, value in current_recipe.items()
        if key not in diagnostic_fields
    }:
        raise TerminalArtifactError("resolved recipe artifact semantic payload changed")
    metric_row = read_single_json_line(paths["metrics"])
    if metric_row != {
        "schema_version": 1,
        **to_plain_dict(request.last_metrics),
    }:
        raise TerminalArtifactError("metrics artifact payload changed")

    success = FrozenMapping(
        {
            "schema_version": 1,
            "kind": "visual_rl_run_success",
            "run_id": request.run_id,
            "committed_steps": request.committed_steps,
            "checkpoint_relative_path": artifacts["checkpoint_relative_path"],
            "checkpoint_contract_id": request.checkpoint_contract_id,
            "progress_id": request.progress_id,
            "state_tree_id": request.state_tree_id,
            "resolved_recipe_sha256": artifacts["resolved_recipe_sha256"],
            "run_manifest_sha256": artifacts["run_manifest_sha256"],
            "metrics_sha256": artifacts["metrics_sha256"],
        }
    )
    success_payload = to_plain_dict(success)
    if paths["marker"].exists() or paths["marker"].is_symlink():
        if read_exact_json(paths["marker"]) != success_payload:
            raise TerminalArtifactError(
                "SUCCESS marker differs from checkpoint receipt"
            )
    else:
        atomic_write(paths["marker"], canonical_line(success_payload))
    return TerminalFinalizationResult(
        output_dir=output_dir,
        checkpoint_path=request.checkpoint_path,
        resolved_path=paths["resolved"],
        manifest_path=paths["manifest"],
        metrics_path=paths["metrics"],
        marker_path=paths["marker"],
        success_payload=success,
    )


def _terminal_output_dir(
    request: TerminalFinalizationRequest,
    artifacts: Mapping[str, str],
) -> Path:
    checkpoint_path = request.checkpoint_path
    relative = PurePosixPath(artifacts["checkpoint_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise TerminalArtifactError("checkpoint relative path is unsafe")
    relative_path = Path(*relative.parts)
    if request.current_output_dir / relative_path == checkpoint_path:
        return request.current_output_dir
    resume_from = request.resume_from
    if resume_from is None:
        raise TerminalArtifactError(
            "historical terminal sidecars require explicit resume_from"
        )
    try:
        normalized_resume = resume_from.resolve(strict=True)
    except OSError as exc:
        raise TerminalArtifactError(
            "explicit resume_from cannot be resolved to the inspected checkpoint"
        ) from exc
    if normalized_resume != checkpoint_path:
        raise TerminalArtifactError(
            "explicit resume_from differs from the inspected terminal checkpoint"
        )
    historical_output = checkpoint_path
    for _part in relative.parts:
        historical_output = historical_output.parent
    if historical_output / relative_path != checkpoint_path:
        raise TerminalArtifactError(
            "terminal checkpoint path does not match its historical run root"
        )
    return historical_output


def prepare_output_dir(path: Path) -> Path:
    """Create and validate one real, absolute run artifact root."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("output_dir must be an absolute Path")
    if path.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("output_dir must be a real directory")
    return path


def artifact_paths(
    output_dir: Path,
    *,
    resolved_name: str = "resolved_recipe.json",
    manifest_name: str = "run_manifest.json",
    metrics_name: str = "metrics.jsonl",
    marker_name: str = "SUCCESS",
) -> dict[str, Path]:
    """Return the exact four terminal sidecar paths under one validated root."""

    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute Path")
    names = {
        "resolved": resolved_name,
        "manifest": manifest_name,
        "metrics": metrics_name,
        "marker": marker_name,
    }
    for label, name in names.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError(f"{label} artifact name must be one filename")
    if len(set(names.values())) != len(names):
        raise ValueError("terminal artifact filenames must be unique")
    return {label: output_dir / name for label, name in names.items()}


def atomic_write(path: Path, data: bytes) -> None:
    """Atomically replace one sidecar and fsync its containing directory."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("atomic artifact path must be absolute")
    if not isinstance(data, bytes):
        raise TypeError("atomic artifact data must be bytes")
    if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("atomic artifact destination is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def canonical_line(payload: Mapping[str, object]) -> bytes:
    """Encode exactly one canonical JSON object followed by one newline."""

    if not isinstance(payload, Mapping):
        raise TypeError("terminal payload must be a mapping")
    return (canonical_json_text(payload) + "\n").encode("utf-8")


def read_exact_json(path: Path) -> dict[str, object]:
    """Read one strict JSON object while rejecting duplicate/non-finite values."""

    regular_file(path)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value {item}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain one JSON object")
    return value


def read_single_json_line(path: Path) -> dict[str, object]:
    """Read one and only one JSON object line."""

    regular_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path.name} must contain exactly one JSON line")
    value = json.loads(
        lines[0],
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value {item}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} line must be a JSON object")
    return value


def regular_file(path: Path) -> None:
    """Reject symlinks and path traversal for an existing artifact file."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("artifact path must be an absolute Path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} must be a regular non-symlink file")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{path.name} path must not traverse symlinks")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one validated regular artifact file."""

    regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("value must be bytes")
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_launch_runtime_audit(launch: LaunchSpec) -> FrozenMapping:
    """Return a canonical audit payload without raw endpoints or local paths."""

    if not isinstance(launch, LaunchSpec):
        raise TypeError("launch must be LaunchSpec")
    bindings = tuple(
        _binding_audit(binding)
        for _artifact_ref, binding in launch.reward_runtime_bindings
    )
    return FrozenMapping(
        {
            "schema_version": 1,
            "reward_runtime_bindings": bindings,
        }
    )


def _binding_audit(binding: RewardRuntimeBindingSpec) -> FrozenMapping:
    payload: dict[str, object] = {
        "artifact_ref": binding.artifact_ref,
        "execution_domain": binding.execution_domain,
        "device": binding.device,
        "dtype": binding.dtype,
    }
    if binding.execution_domain == "remote":
        if binding.endpoint is None:
            raise ValueError("remote reward binding endpoint is missing")
        payload.update(
            {
                "endpoint_identity": _digest_text(
                    "visual_rl.reward_endpoint.v1",
                    binding.endpoint,
                ),
                "trusted_hosts": binding.trusted_hosts,
                "ca_bundle_sha256": _optional_file_digest(binding.ca_bundle),
                "timeout_s": binding.timeout_s,
                "max_response_bytes": binding.max_response_bytes,
            }
        )
    return FrozenMapping(payload)


def _optional_file_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("CA bundle must be an absolute Path")
    if path.is_symlink() or not path.is_file():
        raise ValueError("CA bundle must be a regular non-symlink file")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("CA bundle changed while computing launch audit")
    return hashlib.sha256(data).hexdigest()


def _digest_text(domain: str, value: str) -> str:
    payload = {"domain": domain, "value": value}
    return hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()
