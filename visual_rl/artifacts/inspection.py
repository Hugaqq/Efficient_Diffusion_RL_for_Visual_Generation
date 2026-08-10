"""Read-only inspection for terminal v0.8 run directories.

This module deliberately understands only the v0.8 terminal layout written by
``CoordinatorCheckpointSink``.  It does not fall back to the legacy commit
chain in :mod:`visual_rl.artifacts.status` or :mod:`visual_rl.artifacts.audit`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Literal

from visual_rl.artifacts.checkpoint import AtomicCheckpointManager, CheckpointInspection

__all__ = (
    "AuditReport",
    "InspectionCheck",
    "InspectionError",
    "RunStatus",
    "audit_run",
    "inspect_run",
)


_SUCCESS_KEYS = {
    "schema_version",
    "kind",
    "run_id",
    "committed_steps",
    "checkpoint_relative_path",
    "checkpoint_contract_id",
    "progress_id",
    "state_tree_id",
    "resolved_recipe_sha256",
    "run_manifest_sha256",
    "metrics_sha256",
}
_LATEST_KEYS = {
    "checkpoint_contract_id",
    "path",
    "progress_id",
    "step",
}
_RUN_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "run_id",
    "recipe_id",
    "bound_contract_id",
    "checkpoint_contract_id",
    "update_execution_plan_id",
    "start_optimizer_step",
    "committed_steps",
    "update_count",
    "checkpoint_relative_path",
    "bound_reward_resource_ids",
    "policy_tensor_runtime_spec",
    "resolved_recipe_sha256",
    "metrics_sha256",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")


class InspectionError(RuntimeError):
    """The requested run root itself cannot be inspected safely."""

    def __init__(self, message: str, *, path: Path) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True, slots=True)
class InspectionCheck:
    level: Literal["error", "warning"]
    code: str
    path: Path
    message: str

    def __post_init__(self) -> None:
        if self.level not in {"error", "warning"}:
            raise ValueError("inspection check level must be error or warning")
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("inspection check code must be non-empty")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("inspection check path must be absolute")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("inspection check message must be non-empty")


class _CheckProjection:
    checks: tuple[InspectionCheck, ...]

    @property
    def errors(self) -> tuple[InspectionCheck, ...]:
        return tuple(item for item in self.checks if item.level == "error")

    @property
    def warnings(self) -> tuple[InspectionCheck, ...]:
        return tuple(item for item in self.checks if item.level == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class RunStatus(_CheckProjection):
    output_dir: Path
    run_id: str | None
    committed_steps: int
    authoritative_checkpoint: Path | None
    resumable: bool
    completed: bool
    checks: tuple[InspectionCheck, ...]


@dataclass(frozen=True, slots=True)
class AuditReport(_CheckProjection):
    output_dir: Path
    run_id: str | None
    committed_steps: int
    authoritative_checkpoint: Path | None
    checked_checkpoint_count: int
    checked_artifact_paths: tuple[Path, ...]
    checks: tuple[InspectionCheck, ...]


@dataclass(frozen=True, slots=True)
class _TerminalInspection:
    status: RunStatus
    marker: Mapping[str, object] | None
    checkpoint: CheckpointInspection | None
    latest_path: Path


def inspect_run(path: str | Path) -> RunStatus:
    """Project one terminal v0.8 run without loading training state tensors."""

    root = _run_root(path)
    return _inspect_terminal(root).status


def audit_run(path: str | Path) -> AuditReport:
    """Deep-check one terminal v0.8 run and its authoritative state tree."""

    root = _run_root(path)
    terminal = _inspect_terminal(root)
    status = terminal.status
    checks = list(status.checks)
    checked: list[Path] = []
    marker_path = root / "SUCCESS"
    if terminal.marker is not None:
        checked.append(marker_path)
    if terminal.checkpoint is None or terminal.marker is None:
        return AuditReport(
            output_dir=root,
            run_id=status.run_id,
            committed_steps=status.committed_steps,
            authoritative_checkpoint=status.authoritative_checkpoint,
            checked_checkpoint_count=0,
            checked_artifact_paths=tuple(checked),
            checks=tuple(checks),
        )

    inspection = terminal.checkpoint
    marker = terminal.marker
    checkpoint_path = inspection.committed.path
    checked.extend((terminal.latest_path, checkpoint_path))
    paths = {
        "resolved": root / "resolved_recipe.json",
        "manifest": root / "run_manifest.json",
        "metrics": root / "metrics.jsonl",
    }
    try:
        resolved = _read_json_mapping(paths["resolved"])
        manifest = _read_exact_json(paths["manifest"], _RUN_MANIFEST_KEYS)
        metrics = _read_metrics(paths["metrics"])
        for artifact_path in paths.values():
            checked.append(artifact_path)
        _validate_terminal_payloads(
            marker=marker,
            resolved=resolved,
            manifest=manifest,
            metrics=metrics,
            paths=paths,
            inspection=inspection,
        )
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        checks.append(
            _check(
                "audit.terminal_artifacts",
                root,
                f"terminal artifact verification failed: {type(exc).__name__}: {exc}",
            )
        )
    return AuditReport(
        output_dir=root,
        run_id=status.run_id,
        committed_steps=status.committed_steps,
        authoritative_checkpoint=status.authoritative_checkpoint,
        checked_checkpoint_count=1,
        checked_artifact_paths=tuple(dict.fromkeys(checked)),
        checks=tuple(checks),
    )


def _inspect_terminal(root: Path) -> _TerminalInspection:
    checks: list[InspectionCheck] = []
    marker_path = root / "SUCCESS"
    latest_path = root / "checkpoints" / "latest.json"
    marker: Mapping[str, object] | None = None
    inspection: CheckpointInspection | None = None
    run_id: str | None = None
    committed_steps = 0
    authoritative: Path | None = None
    try:
        marker = _read_exact_json(marker_path, _SUCCESS_KEYS)
        _validate_success_shape(marker)
        run_id = str(marker["run_id"])
        committed_steps = int(marker["committed_steps"])
        checkpoint_relative = _checkpoint_relative(
            marker["checkpoint_relative_path"], committed_steps
        )
        checkpoint_root = _real_directory(root / "checkpoints", "checkpoint root")
        authoritative = root / checkpoint_relative
        if authoritative.parent != checkpoint_root:
            raise ValueError("SUCCESS checkpoint path is outside checkpoints/")
        manager = AtomicCheckpointManager(checkpoint_root)
        inspection = manager.inspect_complete(authoritative)
        _validate_success_checkpoint(marker, inspection)
        latest = _read_exact_json(latest_path, _LATEST_KEYS)
        _validate_latest(latest, inspection)
        for name in ("resolved_recipe.json", "run_manifest.json", "metrics.jsonl"):
            _regular_file(root / name)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        checks.append(
            _check(
                "inspection.terminal_run",
                marker_path,
                f"terminal run is invalid: {type(exc).__name__}: {exc}",
            )
        )
        inspection = None
        authoritative = None

    valid = inspection is not None and not checks
    return _TerminalInspection(
        status=RunStatus(
            output_dir=root,
            run_id=run_id,
            committed_steps=committed_steps,
            authoritative_checkpoint=authoritative,
            resumable=valid,
            completed=valid,
            checks=tuple(checks),
        ),
        marker=marker,
        checkpoint=inspection,
        latest_path=latest_path,
    )


def _validate_terminal_payloads(
    *,
    marker: Mapping[str, object],
    resolved: Mapping[str, object],
    manifest: Mapping[str, object],
    metrics: Mapping[str, object],
    paths: Mapping[str, Path],
    inspection: CheckpointInspection,
) -> None:
    resolved_digest = _sha256_file(paths["resolved"])
    manifest_digest = _sha256_file(paths["manifest"])
    metrics_digest = _sha256_file(paths["metrics"])
    if marker["resolved_recipe_sha256"] != resolved_digest:
        raise ValueError("SUCCESS resolved recipe digest mismatch")
    if marker["run_manifest_sha256"] != manifest_digest:
        raise ValueError("SUCCESS run manifest digest mismatch")
    if marker["metrics_sha256"] != metrics_digest:
        raise ValueError("SUCCESS metrics digest mismatch")
    if manifest["resolved_recipe_sha256"] != resolved_digest:
        raise ValueError("run manifest resolved recipe digest mismatch")
    if manifest["metrics_sha256"] != metrics_digest:
        raise ValueError("run manifest metrics digest mismatch")

    contract = inspection.contract
    committed = inspection.committed.step
    checkpoint_relative = f"checkpoints/step-{committed}"
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "visual_rl_final_run_manifest"
        or manifest["run_id"] != marker["run_id"]
        or manifest["committed_steps"] != committed
        or manifest["checkpoint_relative_path"] != checkpoint_relative
        or manifest["checkpoint_contract_id"] != contract.checkpoint_contract_id
        or manifest["recipe_id"] != contract.recipe_id
        or manifest["bound_contract_id"] != contract.runtime_bound_contract_id
    ):
        raise ValueError("run manifest identity differs from checkpoint/SUCCESS")
    start = manifest["start_optimizer_step"]
    update_count = manifest["update_count"]
    if (
        type(start) is not int
        or start < 0
        or type(update_count) is not int
        or update_count < 1
        or update_count != committed - start
    ):
        raise ValueError("run manifest optimizer step range is invalid")
    _digest("update_execution_plan_id", manifest["update_execution_plan_id"])
    if not isinstance(manifest["bound_reward_resource_ids"], Mapping):
        raise TypeError("bound_reward_resource_ids must be an object")
    if not isinstance(manifest["policy_tensor_runtime_spec"], Mapping):
        raise TypeError("policy_tensor_runtime_spec must be an object")

    recipe_identity = resolved.get("identity")
    expected_recipe_identity = {
        "recipe_id": contract.recipe_id,
        "resolved_fingerprint": contract.resolved_fingerprint,
        "algorithm_materialization_spec_id": (
            contract.algorithm_materialization_spec_id
        ),
        "execution_policy_id": contract.execution_policy_id,
        "reward_plan_id": contract.reward_plan_id,
        "source_content_binding_id": contract.source_content_binding_id,
        "component_artifact_binding_set_id": (
            contract.component_artifact_binding_set_id
        ),
    }
    if (
        resolved.get("schema_version") != 2
        or resolved.get("kind") != "materialized_recipe"
        or not isinstance(recipe_identity, Mapping)
        or any(
            recipe_identity.get(name) != value
            for name, value in expected_recipe_identity.items()
        )
    ):
        raise ValueError("resolved recipe identity differs from checkpoint")
    if metrics.get("schema_version") != 1 or metrics.get("step") != committed - 1:
        raise ValueError("metrics final step differs from checkpoint")
    for name in ("sample_count", "active_transition_count"):
        if type(metrics.get(name)) is not int or metrics[name] <= 0:
            raise ValueError(f"metrics {name} must be a positive integer")
    for name, value in metrics.items():
        if name in {
            "schema_version",
            "step",
            "sample_count",
            "active_transition_count",
        }:
            continue
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"metrics {name} must be a finite float")


def _validate_success_shape(value: Mapping[str, object]) -> None:
    if value["schema_version"] != 1 or value["kind"] != "visual_rl_run_success":
        raise ValueError("SUCCESS schema or kind is invalid")
    _digest("run_id", value["run_id"])
    steps = value["committed_steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("SUCCESS committed_steps must be positive")
    for name in (
        "checkpoint_contract_id",
        "progress_id",
        "state_tree_id",
        "resolved_recipe_sha256",
        "run_manifest_sha256",
        "metrics_sha256",
    ):
        _digest(name, value[name])


def _validate_success_checkpoint(
    marker: Mapping[str, object],
    inspection: CheckpointInspection,
) -> None:
    committed = inspection.committed
    if (
        marker["committed_steps"] != committed.step
        or marker["checkpoint_contract_id"] != committed.checkpoint_contract_id
        or marker["progress_id"] != committed.progress_id
        or marker["state_tree_id"] != committed.state_tree_id
    ):
        raise ValueError("SUCCESS differs from the committed checkpoint receipt")


def _validate_latest(
    latest: Mapping[str, object],
    inspection: CheckpointInspection,
) -> None:
    committed = inspection.committed
    if (
        latest["path"] != committed.path.name
        or latest["step"] != committed.step
        or latest["checkpoint_contract_id"] != committed.checkpoint_contract_id
        or latest["progress_id"] != committed.progress_id
    ):
        raise ValueError("latest.json differs from SUCCESS checkpoint")


def _checkpoint_relative(value: object, committed_steps: int) -> Path:
    if not isinstance(value, str):
        raise TypeError("checkpoint_relative_path must be a string")
    pure = PurePosixPath(value)
    expected = PurePosixPath("checkpoints") / f"step-{committed_steps}"
    if pure != expected:
        raise ValueError("checkpoint_relative_path is not canonical")
    return Path(*pure.parts)


def _read_metrics(path: Path) -> dict[str, object]:
    _regular_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("metrics.jsonl must contain exactly one JSON row")
    return _json_mapping(lines[0], path)


def _read_json_mapping(path: Path) -> dict[str, object]:
    _regular_file(path)
    return _json_mapping(path.read_text(encoding="utf-8"), path)


def _read_exact_json(path: Path, exact_keys: set[str]) -> dict[str, object]:
    value = _read_json_mapping(path)
    if set(value) != exact_keys:
        raise ValueError(f"{path.name} has an invalid exact key set")
    return value


def _json_mapping(text: str, path: Path) -> dict[str, object]:
    def pairs_hook(pairs):
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        text,
        object_pairs_hook=pairs_hook,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value {item}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain one JSON object")
    return value


def _run_root(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise TypeError("run directory must be str or Path")
    requested = Path(os.path.abspath(Path(value).expanduser()))
    try:
        requested.lstat()
    except OSError as exc:
        raise InspectionError("run directory does not exist", path=requested) from exc
    if requested.is_symlink() or not requested.is_dir():
        raise InspectionError("run directory must be a real directory", path=requested)
    return requested.resolve(strict=True)


def _real_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} path must not traverse symlinks")
    return resolved


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} must be a regular non-symlink file")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{path.name} path must not traverse symlinks")


def _sha256_file(path: Path) -> str:
    _regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _check(code: str, path: Path, message: str) -> InspectionCheck:
    return InspectionCheck("error", code, path, message)
