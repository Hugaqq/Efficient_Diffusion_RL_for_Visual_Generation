"""VisualRL's sole public orchestration path: the high-level Python API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import threading
from typing import Any

import yaml

from visual_rl.api_types import AuditReport, RunResult, RunStatus, ValidationReport
from visual_rl.configs.resolver import resolve_config
from visual_rl.configs.schema import VisualRLConfig
from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    ValidatedRuntimeEnv,
    ValidationCheck,
)
from visual_rl.errors import (
    ArtifactError,
    ComponentError,
    ConfigError,
    RunError,
    ValidationError,
    VisualRLError,
)

__all__ = ("audit_run", "inspect_run", "load")


class _DuplicateKeyError(yaml.YAMLError):
    def __init__(self, key: object, line: int, column: int) -> None:
        super().__init__(
            f"duplicate mapping key {key!r} at line {line}, column {column}"
        )
        self.key = key


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise _DuplicateKeyError(
                key,
                key_node.start_mark.line + 1,
                key_node.start_mark.column + 1,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _Experiment:
    """Internal handle created only by :func:`load`."""

    __slots__ = (
        "_config",
        "_context",
        "_raw_snapshot",
        "_run_claimed",
        "_run_lock",
        "_runtime_env",
        "_validation_report",
    )

    def __init__(
        self,
        raw_snapshot: FrozenMapping,
        context: ResolutionContext,
        *,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _EXPERIMENT_TOKEN:
            raise TypeError("Experiment handles can only be created by visual_rl.load()")
        self._raw_snapshot = raw_snapshot
        self._context = context
        self._config: VisualRLConfig | None = None
        self._validation_report: ValidationReport | None = None
        self._runtime_env: ValidatedRuntimeEnv | None = None
        self._run_claimed = False
        self._run_lock = threading.Lock()

    def resolve(self) -> VisualRLConfig:
        if self._config is None:
            self._config = resolve_config(self._raw_snapshot, self._context)
        return self._config

    def validate(self) -> ValidationReport:
        if self._validation_report is not None:
            return self._validation_report
        try:
            config = self.resolve()
        except (ConfigError, ComponentError) as exc:
            report = ValidationReport(
                checks=(_resolution_check(exc, self._context.config_path),),
                runtime_rank=None,
                runtime_world_size=None,
            )
            self._validation_report = report
            self._runtime_env = None
            return report

        # Delayed import keeps import/load/resolve torch-free. Preflight itself
        # probes training dependencies in a bounded child process.
        from visual_rl.preflight import run_preflight

        report, runtime_env = run_preflight(
            config,
            config_dir=self._context.config_dir,
            phase="validate",
        )
        if not isinstance(report, ValidationReport):
            raise TypeError("run_preflight() must return a ValidationReport")
        if runtime_env is not None and not isinstance(
            runtime_env, ValidatedRuntimeEnv
        ):
            raise TypeError(
                "run_preflight() runtime snapshot must be ValidatedRuntimeEnv or None"
            )
        self._validation_report = report
        self._runtime_env = runtime_env
        return report

    def run(self) -> RunResult:
        # The claim is deliberately the first state-changing operation. Every
        # first run attempt, including validation failure, consumes the handle.
        with self._run_lock:
            if self._run_claimed:
                raise RunError("This experiment handle has already attempted run()")
            self._run_claimed = True

        report = self.validate()
        if self._config is None:
            raise ValidationError(
                "Experiment configuration did not resolve",
                checks=report.checks,
            )
        config = self._config

        from visual_rl.preflight import run_preflight

        run_report, runtime_env = run_preflight(
            config,
            config_dir=self._context.config_dir,
            phase="run",
            cached_report=report,
            cached_env=self._runtime_env,
        )
        self._validation_report = run_report
        if not run_report.ok or runtime_env is None:
            self._runtime_env = None
            raise ValidationError(
                "Experiment failed its run-phase validation",
                checks=run_report.checks,
            )
        self._runtime_env = runtime_env

        # Heavy training imports occur only after the run-once claim and the
        # complete run-phase Preflight have succeeded.
        from visual_rl.runner import ExperimentRunner

        try:
            result = ExperimentRunner(config, runtime_env).run()
        except VisualRLError:
            raise
        except BaseException as exc:
            raise RunError("Experiment execution failed") from exc
        if not isinstance(result, RunResult):
            raise RunError("ExperimentRunner.run() must return RunResult directly")
        return result


_EXPERIMENT_TOKEN = object()


def load(path: str | Path) -> _Experiment:
    """Read and freeze exactly one complete YAML file."""

    config_path = _absolute_input_path(path)
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as exc:
        raise ConfigError(
            f"Cannot read configuration file: {config_path}",
            path=str(config_path),
        ) from exc
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"Configuration file must be UTF-8: {config_path}",
            path=str(config_path),
        ) from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except _DuplicateKeyError as exc:
        raise ConfigError(
            f"Invalid YAML in {config_path}: {exc}",
            key=str(exc.key),
            path=str(config_path),
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in {config_path}: {exc}",
            path=str(config_path),
        ) from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(
            "Configuration YAML root must be a mapping",
            key="<root>",
            path=str(config_path),
        )
    try:
        snapshot = FrozenMapping(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Configuration must contain plain YAML values: {exc}",
            path=str(config_path),
        ) from exc
    context = ResolutionContext(
        config_path=config_path,
        config_dir=config_path.parent,
    )
    return _Experiment(snapshot, context, _factory_token=_EXPERIMENT_TOKEN)


def inspect_run(path: str | Path) -> RunStatus:
    """Read one run's authoritative status without loading training code."""

    root = _absolute_output_path(path)
    from visual_rl.artifacts.status import inspect_run_status

    try:
        value = inspect_run_status(root)
        if isinstance(value, RunStatus):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("inspect_run_status() must return a mapping")
        return RunStatus(
            output_dir=root,
            run_id=_optional_string(value.get("run_id")),
            committed_steps=_status_steps(value),
            authoritative_checkpoint=_optional_child_path(
                root, value.get("authoritative_checkpoint")
            ),
            resumable=bool(value.get("resumable", False)),
            pending_transaction_count=_nonnegative_int(
                value.get("pending_transaction_count", 0),
                field="pending_transaction_count",
            ),
            checks=_checks_from_projection(value, prefix="status"),
        )
    except ArtifactError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactError(
            f"Cannot inspect run directory: {root}",
            path=str(root),
        ) from exc


def audit_run(path: str | Path) -> AuditReport:
    """Deep-audit one authoritative run without loading training code."""

    root = _absolute_output_path(path)
    from visual_rl.artifacts.audit import audit_run_artifacts

    try:
        value = audit_run_artifacts(root)
        if isinstance(value, AuditReport):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("audit_run_artifacts() must return a mapping")
        raw_paths = value.get("checked_artifact_paths", ())
        if not isinstance(raw_paths, (list, tuple)):
            raise TypeError("checked_artifact_paths must be a sequence")
        return AuditReport(
            output_dir=root,
            run_id=_optional_string(value.get("run_id")),
            committed_steps=_audit_steps(value),
            checked_commit_count=_nonnegative_int(
                value.get("checked_commit_count", value.get("commit_markers", 0)),
                field="checked_commit_count",
            ),
            checked_artifact_paths=tuple(
                _child_path(root, item) for item in raw_paths
            ),
            checks=_checks_from_projection(value, prefix="audit"),
        )
    except ArtifactError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactError(
            f"Cannot audit run directory: {root}",
            path=str(root),
        ) from exc


def _absolute_input_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise TypeError("configuration path must be str or Path")
    return Path(value).expanduser().resolve(strict=False)


def _absolute_output_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise TypeError("run directory must be str or Path")
    return Path(value).expanduser().resolve(strict=False)


def _resolution_check(
    exc: ConfigError | ComponentError,
    config_path: Path,
) -> ValidationCheck:
    if isinstance(exc, ComponentError):
        code = "component.selection"
        path = ".".join(
            part for part in (exc.kind, exc.name) if isinstance(part, str)
        )
    else:
        code = "config.resolve"
        path = exc.key or str(config_path)
    return ValidationCheck(
        level="error",
        code=code,
        path=path or str(config_path),
        message=str(exc),
        volatile=False,
    )


def _checks_from_projection(
    value: Mapping[str, Any],
    *,
    prefix: str,
) -> tuple[ValidationCheck, ...]:
    checks = value.get("checks")
    if isinstance(checks, tuple) and all(
        isinstance(item, ValidationCheck) for item in checks
    ):
        return checks
    projected: list[ValidationCheck] = []
    for level, key in (("error", "errors"), ("warning", "warnings")):
        raw = value.get(key, ())
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)):
            raise TypeError(f"{key} must be a sequence")
        projected.extend(
            ValidationCheck(
                level=level,
                code=f"{prefix}.{level}",
                path=str(value.get("run_dir", value.get("output_dir", ""))),
                message=str(message),
                volatile=False,
            )
            for message in raw
        )
    return tuple(projected)


def _status_steps(value: Mapping[str, Any]) -> int:
    raw = value.get(
        "committed_steps",
        value.get("authoritative_completed_steps", value.get("completed_steps", 0)),
    )
    return _nonnegative_int(raw, field="committed_steps")


def _audit_steps(value: Mapping[str, Any]) -> int:
    if "committed_steps" in value:
        return _nonnegative_int(value["committed_steps"], field="committed_steps")
    steps = value.get("steps", ())
    if not isinstance(steps, (list, tuple)):
        raise TypeError("steps must be a sequence")
    if not steps:
        return 0
    return _nonnegative_int(max(steps) + 1, field="committed_steps")


def _nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("run_id must be None or a non-empty string")
    return value


def _optional_child_path(root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    return _child_path(root, value)


def _child_path(root: Path, value: Any) -> Path:
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise TypeError("artifact path must be str or Path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)
