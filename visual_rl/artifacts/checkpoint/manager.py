"""Atomic complete-marker checkpoint directories for the v0.8 mainline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.artifacts.checkpoint.protocol import (
    CHECKPOINT_CONTRACT_SCHEMA_VERSION,
    CHECKPOINT_PROGRESS_SCHEMA_VERSION,
    CheckpointContract,
    CheckpointProgress,
    assert_compatible_contract,
)

if TYPE_CHECKING:
    from visual_rl.models.state.projection import ModelStateProjection

__all__ = (
    "AtomicCheckpointManager",
    "CheckpointInspection",
    "CommittedCheckpoint",
)

_CONTRACT_FILE = "checkpoint_contract.json"
_COMPLETE_FILE = "complete.json"
_LATEST_FILE = "latest.json"
_PROGRESS_FILE = "progress.json"
_STATE_TREE_FILE = "state_tree.json"
_RESERVED_FILES = frozenset(
    {_CONTRACT_FILE, _COMPLETE_FILE, _PROGRESS_FILE, _STATE_TREE_FILE}
)


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class CommittedCheckpoint:
    path: Path
    step: int
    checkpoint_contract_id: str
    progress: CheckpointProgress
    state_tree_id: str
    progress_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("committed checkpoint path must be a Path")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("committed checkpoint step must be non-negative")
        if not isinstance(self.progress, CheckpointProgress):
            raise TypeError("committed checkpoint progress must be CheckpointProgress")
        if self.progress.global_step != self.step:
            raise ValueError("committed checkpoint step disagrees with progress")
        object.__setattr__(self, "progress_id", self.progress.progress_id)

    def __repr__(self) -> str:
        return (
            f"CommittedCheckpoint(path={self.path!r}, step={self.step!r}, "
            f"checkpoint_contract_id={self.checkpoint_contract_id!r}, "
            f"progress_id={self.progress_id!r})"
        )


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """Fully validated checkpoint receipt and its actual durable metadata."""

    committed: CommittedCheckpoint
    contract: CheckpointContract
    progress: CheckpointProgress

    def __post_init__(self) -> None:
        if not isinstance(self.committed, CommittedCheckpoint):
            raise TypeError("committed must be a CommittedCheckpoint")
        if not isinstance(self.contract, CheckpointContract):
            raise TypeError("contract must be a CheckpointContract")
        if not isinstance(self.progress, CheckpointProgress):
            raise TypeError("progress must be a CheckpointProgress")
        if self.committed.checkpoint_contract_id != (
            self.contract.checkpoint_contract_id
        ):
            raise ValueError("inspection contract disagrees with committed receipt")
        if self.committed.progress is not self.progress:
            raise ValueError(
                "inspection progress must be the committed progress object"
            )
        if self.committed.progress_id != self.progress.progress_id:
            raise ValueError("inspection progress identity disagrees")
        if self.committed.step != self.progress.global_step:
            raise ValueError("inspection step disagrees with progress")
        if self.progress.execution_transform_plan_id != (
            self.contract.execution_transform_plan_id
        ):
            raise ValueError("inspection progress transform plan disagrees")

    @property
    def checkpoint(self) -> CommittedCheckpoint:
        """Compatibility-friendly name for the complete committed receipt."""

        return self.committed


class AtomicCheckpointManager:
    """Commit ``step-N.tmp-UUID`` to ``step-N`` before updating latest.json."""

    def __init__(self, root: str | Path) -> None:
        if not isinstance(root, (str, Path)) or isinstance(root, bool):
            raise TypeError("checkpoint root must be str or Path")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("checkpoint root must be a real directory")

    def commit(
        self,
        step: int,
        contract: CheckpointContract,
        writer: Callable[[Path], None],
        *,
        progress: CheckpointProgress,
        fault_injector: Callable[[str], None] | None = None,
    ) -> CommittedCheckpoint:
        if type(step) is not int or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        if not isinstance(contract, CheckpointContract):
            raise TypeError("contract must be a CheckpointContract")
        if not callable(writer):
            raise TypeError("writer must be callable")
        if not isinstance(progress, CheckpointProgress):
            raise TypeError("progress must be CheckpointProgress")
        if progress.global_step != step:
            raise ValueError("checkpoint step must equal progress.global_step")
        if progress.execution_transform_plan_id != contract.execution_transform_plan_id:
            raise ValueError(
                "checkpoint progress transform plan does not match contract"
            )
        final = self.root / f"step-{step}"
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"checkpoint already exists: {final}")
        temporary = self.root / f"step-{step}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o700)
        renamed = False
        try:
            writer(temporary)
            _fault(fault_injector, "after_writer")
            state_tree = _build_state_tree(temporary)
            state_tree_id = _payload_digest(state_tree)
            _atomic_json(
                temporary / _STATE_TREE_FILE,
                {"state_tree_id": state_tree_id, "tree": state_tree},
            )
            _atomic_json(
                temporary / _CONTRACT_FILE,
                {
                    "checkpoint_contract_id": contract.checkpoint_contract_id,
                    "contract": contract.to_payload(),
                },
            )
            _atomic_json(
                temporary / _PROGRESS_FILE,
                {
                    "progress_id": progress.progress_id,
                    "progress": progress.to_payload(),
                },
            )
            _atomic_json(
                temporary / _COMPLETE_FILE,
                {
                    "checkpoint_contract_id": contract.checkpoint_contract_id,
                    "kind": "visual_rl_checkpoint_complete",
                    "progress_id": progress.progress_id,
                    "state_tree_id": state_tree_id,
                    "step": step,
                },
            )
            _fault(fault_injector, "after_complete_marker")
            _fsync_directory(temporary)
            os.replace(temporary, final)
            renamed = True
            _fsync_directory(self.root)
            _fault(fault_injector, "after_checkpoint_rename")
            _atomic_json(
                self.root / _LATEST_FILE,
                {
                    "checkpoint_contract_id": contract.checkpoint_contract_id,
                    "path": final.name,
                    "progress_id": progress.progress_id,
                    "step": step,
                },
            )
            _fault(fault_injector, "after_latest")
        except BaseException:
            if not renamed and temporary.exists() and not temporary.is_symlink():
                shutil.rmtree(temporary)
            raise
        return CommittedCheckpoint(
            final,
            step,
            contract.checkpoint_contract_id,
            progress,
            state_tree_id,
        )

    def latest_complete(
        self,
        *,
        expected_contract: CheckpointContract | None = None,
        explicit_path: str | Path | None = None,
    ) -> CommittedCheckpoint | None:
        """Resolve explicit path first, else newest complete committed directory."""

        if explicit_path is not None:
            return self.inspect_complete(
                explicit_path,
                expected_contract=expected_contract,
            ).committed

        candidates: list[CommittedCheckpoint] = []
        for path in self.root.glob("step-*"):
            if ".tmp-" in path.name or not path.is_dir() or path.is_symlink():
                continue
            try:
                candidate = self.inspect_complete(
                    path,
                    expected_contract=expected_contract,
                ).committed
            except (OSError, ValueError, TypeError):
                continue
            candidates.append(candidate)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.step)

    def inspect_complete(
        self,
        explicit_path: str | Path,
        *,
        expected_contract: CheckpointContract | None = None,
    ) -> CheckpointInspection:
        """Inspect one explicit complete directory and return its actual metadata."""

        if expected_contract is not None and not isinstance(
            expected_contract,
            CheckpointContract,
        ):
            raise TypeError("expected_contract must be CheckpointContract or None")
        path = _direct_checkpoint_path(self.root, explicit_path)
        complete = _read_exact_json(
            path / _COMPLETE_FILE,
            {
                "checkpoint_contract_id",
                "kind",
                "progress_id",
                "state_tree_id",
                "step",
            },
        )
        if complete["kind"] != "visual_rl_checkpoint_complete":
            raise ValueError("checkpoint complete marker kind is invalid")
        step = complete["step"]
        if type(step) is not int or step < 0 or path.name != f"step-{step}":
            raise ValueError("checkpoint step/path identity is invalid")
        envelope = _read_exact_json(
            path / _CONTRACT_FILE,
            {"checkpoint_contract_id", "contract"},
        )
        if envelope["checkpoint_contract_id"] != complete["checkpoint_contract_id"]:
            raise ValueError("checkpoint contract identities disagree")
        found = _contract_from_payload(envelope["contract"])
        if found.checkpoint_contract_id != envelope["checkpoint_contract_id"]:
            raise ValueError("checkpoint contract payload hash is invalid")
        if expected_contract is not None:
            assert_compatible_contract(expected_contract, found)
        progress_envelope = _read_exact_json(
            path / _PROGRESS_FILE,
            {"progress_id", "progress"},
        )
        progress = _progress_from_payload(progress_envelope["progress"])
        if progress.progress_id != progress_envelope["progress_id"]:
            raise ValueError("checkpoint progress payload hash is invalid")
        if progress.progress_id != complete["progress_id"]:
            raise ValueError("checkpoint progress identities disagree")
        if progress.global_step != step:
            raise ValueError("checkpoint progress step disagrees with marker")
        if progress.execution_transform_plan_id != found.execution_transform_plan_id:
            raise ValueError("checkpoint progress transform plan disagrees")
        tree_envelope = _read_exact_json(
            path / _STATE_TREE_FILE,
            {"state_tree_id", "tree"},
        )
        tree = tree_envelope["tree"]
        if not isinstance(tree, Mapping):
            raise TypeError("checkpoint state tree must be an object")
        state_tree_id = _payload_digest(tree)
        if state_tree_id != tree_envelope["state_tree_id"]:
            raise ValueError("checkpoint state tree payload hash is invalid")
        if state_tree_id != complete["state_tree_id"]:
            raise ValueError("checkpoint state tree identities disagree")
        _verify_state_tree(path, tree)
        committed = CommittedCheckpoint(
            path,
            step,
            found.checkpoint_contract_id,
            progress,
            state_tree_id,
        )
        return CheckpointInspection(
            committed=committed,
            contract=found,
            progress=progress,
        )


def _contract_from_payload(value: object) -> CheckpointContract:
    from visual_rl.artifacts.checkpoint.protocol import (
        ComponentContractRef,
        OptimizerGroupContract,
        ParameterContract,
    )

    if not isinstance(value, Mapping):
        raise TypeError("checkpoint contract payload must be an object")
    expected = set(CheckpointContract.__dataclass_fields__) | {"schema_version"}
    if set(value) != expected:
        raise ValueError("checkpoint contract payload has an invalid exact key set")
    if value["schema_version"] != CHECKPOINT_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint contract schema version is unsupported; "
            f"expected {CHECKPOINT_CONTRACT_SCHEMA_VERSION}"
        )
    values = dict(value)
    values.pop("schema_version")
    values["components"] = tuple(
        ComponentContractRef(**item) for item in values["components"]
    )
    values["model_state_projection"] = _model_state_projection_from_payload(
        values["model_state_projection"]
    )
    from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence

    values["model_execution_numerics"] = ModelExecutionNumericsEvidence.from_payload(
        values["model_execution_numerics"]
    )
    values["trainable_parameters"] = tuple(
        ParameterContract(
            name=item["name"], shape=tuple(item["shape"]), dtype=item["dtype"]
        )
        for item in values["trainable_parameters"]
    )
    values["optimizer_groups"] = tuple(
        OptimizerGroupContract(
            group_id=item["group_id"],
            parameter_names=tuple(item["parameter_names"]),
            hyperparameters_id=item["hyperparameters_id"],
        )
        for item in values["optimizer_groups"]
    )
    values["state_schema_versions"] = tuple(
        (item["name"], item["version"]) for item in values["state_schema_versions"]
    )
    values["execution_transform_chain"] = tuple(
        (item["transform_id"], item["contract_id"])
        for item in values["execution_transform_chain"]
    )
    return CheckpointContract(**values)


def _model_state_projection_from_payload(value: object) -> ModelStateProjection:
    from visual_rl.models.state.projection import (
        MODEL_STATE_PROJECTION_SCHEMA_VERSION,
        ModelComponentStateMembership,
        ModelParameterStateMembership,
        ModelStateProjection,
    )

    if not isinstance(value, Mapping):
        raise TypeError("model state projection payload must be an object")
    expected = set(ModelStateProjection.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("model state projection payload has an invalid exact key set")
    if value["schema_version"] != MODEL_STATE_PROJECTION_SCHEMA_VERSION:
        raise ValueError("model state projection schema version is unsupported")
    values = dict(value)
    component_membership = values["component_membership"]
    parameter_membership = values["parameter_membership"]
    if not isinstance(component_membership, list) or not isinstance(
        parameter_membership, list
    ):
        raise TypeError("model state projection memberships must be lists")
    component_keys = set(ModelComponentStateMembership.__dataclass_fields__)
    parameter_keys = set(ModelParameterStateMembership.__dataclass_fields__)
    if any(
        not isinstance(item, Mapping) or set(item) != component_keys
        for item in component_membership
    ):
        raise ValueError(
            "model component state membership has an invalid exact key set"
        )
    if any(
        not isinstance(item, Mapping) or set(item) != parameter_keys
        for item in parameter_membership
    ):
        raise ValueError(
            "model parameter state membership has an invalid exact key set"
        )
    values["component_membership"] = tuple(
        ModelComponentStateMembership(
            name=item["name"],
            roles=tuple(item["roles"]),
            managed_residency=item["managed_residency"],
        )
        for item in component_membership
    )
    values["parameter_membership"] = tuple(
        ModelParameterStateMembership(**item) for item in parameter_membership
    )
    for name in (
        "standalone_saved_component_names",
        "standalone_parameter_names",
        "artifact_rehydrated_component_names",
    ):
        values[name] = tuple(values[name])
    return ModelStateProjection(**values)


def _progress_from_payload(value: object) -> CheckpointProgress:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint progress payload must be an object")
    expected = set(CheckpointProgress.__dataclass_fields__) | {"schema_version"}
    if set(value) != expected:
        raise ValueError("checkpoint progress payload has an invalid exact key set")
    if value["schema_version"] != CHECKPOINT_PROGRESS_SCHEMA_VERSION:
        raise ValueError("checkpoint progress schema version is unsupported")
    values = dict(value)
    values.pop("schema_version")
    values["active_reward_ids"] = tuple(values["active_reward_ids"])
    values["source_cursors"] = tuple(
        (item["source_id"], item["cursor"]) for item in values["source_cursors"]
    )
    values["dynamics_selection_policy"] = (
        DynamicsSelectionPolicyState.from_checkpoint_payload(
            values["dynamics_selection_policy"]
        )
    )
    return CheckpointProgress(**values)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:  # noqa: BLE001
        # Atomic publish must remove its temporary file even on cancellation.
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _direct_checkpoint_path(root: Path, value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise TypeError("explicit checkpoint path must be str or Path")
    expanded = Path(value).expanduser()
    if ".." in expanded.parts:
        raise ValueError("explicit checkpoint path must not contain '..'")
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    path = Path(os.path.abspath(absolute))
    if path.parent != root:
        raise ValueError("checkpoint must be a direct real directory under root")
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint must be a direct real directory under root")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("checkpoint directory cannot be resolved safely") from exc
    if resolved != path:
        raise ValueError("checkpoint directory path must not traverse symlinks")
    return path


def _read_exact_json(path: Path, exact_keys: set[str]) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} must be a regular non-symlink file")

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
    if not isinstance(value, dict) or set(value) != exact_keys:
        raise ValueError(f"{path.name} has an invalid exact key set")
    return value


def _payload_digest(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_state_tree(root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("checkpoint writer must not create symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("checkpoint writer created a non-regular file")
        relative = path.relative_to(root).as_posix()
        if relative in _RESERVED_FILES:
            raise ValueError(
                f"checkpoint writer must not create reserved file {relative!r}"
            )
        data = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if not entries:
        raise ValueError("checkpoint writer produced no training state files")
    return {"schema_version": 1, "files": entries}


def _verify_state_tree(root: Path, value: Mapping[str, object]) -> None:
    if set(value) != {"schema_version", "files"} or value["schema_version"] != 1:
        raise ValueError("checkpoint state tree schema is invalid")
    raw_files = value["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("checkpoint state tree must contain files")
    expected_paths: list[str] = []
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size"}:
            raise ValueError("checkpoint state tree entry is invalid")
        relative = item["path"]
        digest = item["sha256"]
        size = item["size"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in _RESERVED_FILES
        ):
            raise ValueError("checkpoint state path is invalid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("checkpoint state digest is invalid")
        if type(size) is not int or size < 0:
            raise ValueError("checkpoint state size is invalid")
        expected_paths.append(relative)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint state file is missing or not regular")
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("checkpoint state file digest mismatch")
    if expected_paths != sorted(set(expected_paths)):
        raise ValueError("checkpoint state paths must be sorted and unique")
    actual_paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.relative_to(root).as_posix() not in _RESERVED_FILES
    )
    if tuple(expected_paths) != actual_paths:
        raise ValueError("checkpoint state tree does not cover exact state files")


def _fault(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
