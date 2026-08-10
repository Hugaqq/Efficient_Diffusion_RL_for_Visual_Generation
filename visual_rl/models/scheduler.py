"""Load-after scheduler artifact ABI owned by the model domain.

The blueprint snapshots scheduler class/config identity and can reconstruct an
uninitialized fresh cursor.  Diffusion step count, dynamic-shift conditioning,
and ``set_timesteps`` remain owned by the Dynamics binder.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from visual_rl.core.contracts.model import LatentLayout

__all__ = (
    "SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA",
    "ModelScheduleContext",
    "SchedulerArtifactBlueprint",
)

SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA = "diffusers.scheduler-artifact.v1"


def _identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _json_value(value: object, *, path: str) -> object:
    """Own a scheduler config as a strict, canonical JSON value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite floats")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"{path} scheduler config keys must be non-empty strings"
                )
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains non-JSON scheduler config value {type(value).__name__}"
    )


def _canonical_scheduler_config(config: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize Diffusers config without erasing semantic sequence order."""

    canonical = _json_value(config, path="scheduler.config")
    assert isinstance(canonical, dict)
    default_values = canonical.get("_use_default_values")
    if default_values is not None:
        if not isinstance(default_values, list) or not all(
            isinstance(item, str) for item in default_values
        ):
            raise ValueError(
                "scheduler.config._use_default_values must be a sequence of strings"
            )
        # Diffusers derives this bookkeeping field from a set.  Its iteration
        # order is not scheduler semantics; all other sequences keep order.
        canonical["_use_default_values"] = sorted(default_values)
    return canonical


@runtime_checkable
class ModelScheduleContext(Protocol):
    """Structural view of model-owned geometry consumed by a Dynamics binder."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def layout(self) -> LatentLayout: ...

    @property
    def axis_semantics(self) -> tuple[str, ...]: ...

    @property
    def device(self) -> Any: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def spatial_stride(self) -> tuple[int, int] | None: ...

    @property
    def temporal_stride(self) -> int | None: ...

    @property
    def scheduler_patch_size(self) -> int | None: ...


@dataclass(frozen=True, slots=True, eq=False)
class SchedulerArtifactBlueprint:
    """Immutable class/config snapshot for one loaded scheduler artifact."""

    _scheduler_type: type[Any] = field(repr=False)
    _config_json: str = field(repr=False)
    source_scheduler_identity: str
    schema_id: str = SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA
    scheduler_class_path: str = field(init=False)
    scheduler_config_digest: str = field(init=False)
    blueprint_identity: str = field(init=False)
    artifact_identity: str = field(init=False)
    scheduler_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self._scheduler_type, type):
            raise TypeError("scheduler_type must be a class")
        if not callable(getattr(self._scheduler_type, "from_config", None)):
            raise TypeError("scheduler type must expose from_config()")
        if not isinstance(self._config_json, str):
            raise TypeError("scheduler config JSON must be a string")
        try:
            config = json.loads(self._config_json)
        except (TypeError, ValueError):
            raise ValueError("scheduler config must be valid JSON") from None
        if not isinstance(config, dict):
            raise TypeError("scheduler config must encode a mapping")
        source_identity = _non_empty(
            "source_scheduler_identity",
            self.source_scheduler_identity,
        )
        if self.schema_id != SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA:
            raise ValueError("unsupported scheduler artifact blueprint schema")
        scheduler_type = self._scheduler_type
        class_path = f"{scheduler_type.__module__}.{scheduler_type.__qualname__}"
        config_digest = _identity(config)
        blueprint_identity = _identity(
            {
                "schema_id": self.schema_id,
                "scheduler_class_path": class_path,
                "scheduler_config_digest": config_digest,
                "source_scheduler_identity": source_identity,
            }
        )
        object.__setattr__(self, "scheduler_class_path", class_path)
        object.__setattr__(self, "scheduler_config_digest", config_digest)
        object.__setattr__(self, "blueprint_identity", blueprint_identity)
        object.__setattr__(
            self,
            "artifact_identity",
            f"scheduler-artifact.v1:{blueprint_identity}",
        )
        object.__setattr__(
            self,
            "scheduler_identity",
            f"scheduler-blueprint.v1:{blueprint_identity}",
        )

    @classmethod
    def from_scheduler(cls, scheduler: object) -> SchedulerArtifactBlueprint:
        """Snapshot class/config only; the live scheduler is never retained."""

        if scheduler is None:
            raise TypeError("scheduler must not be None")
        scheduler_type = type(scheduler)
        if not callable(getattr(scheduler_type, "from_config", None)):
            raise TypeError("scheduler type must expose from_config()")
        if not callable(getattr(scheduler, "set_timesteps", None)):
            raise TypeError("scheduler must expose set_timesteps()")
        config = getattr(scheduler, "config", None)
        if not isinstance(config, Mapping):
            raise TypeError("scheduler.config must be a mapping")
        canonical = _canonical_scheduler_config(config)
        config_json = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        explicit = getattr(scheduler, "scheduler_identity", None)
        source_identity = (
            explicit
            if isinstance(explicit, str) and explicit
            else f"{scheduler_type.__module__}.{scheduler_type.__qualname__}"
        )
        return cls(
            _scheduler_type=scheduler_type,
            _config_json=config_json,
            source_scheduler_identity=source_identity,
        )

    @property
    def scheduler_type(self) -> type[Any]:
        return self._scheduler_type

    def config_payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible copy of the frozen config."""

        result = json.loads(self._config_json)
        assert isinstance(result, dict)
        return result

    def instantiate_scheduler(self) -> object:
        """Reconstruct a fresh cursor without selecting a diffusion schedule."""

        scheduler = self._scheduler_type.from_config(self.config_payload())
        if not isinstance(scheduler, self._scheduler_type):
            raise TypeError("scheduler from_config() returned an incompatible type")
        if not callable(getattr(scheduler, "set_timesteps", None)):
            raise TypeError("fresh scheduler must expose set_timesteps()")
        return scheduler

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SchedulerArtifactBlueprint)
            and self.blueprint_identity == other.blueprint_identity
        )

    def __hash__(self) -> int:
        return hash(self.blueprint_identity)
