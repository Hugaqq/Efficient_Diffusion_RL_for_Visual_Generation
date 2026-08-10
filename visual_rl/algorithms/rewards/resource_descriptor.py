"""Import-safe declarations for physical reward resource semantics.

These values describe what may be bound later.  They deliberately reject
actual device, endpoint, path, rank, and worker facts so declaration identity
remains portable across launches.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict

__all__ = (
    "RewardResourceDescriptor",
    "RewardRuntimePolicy",
)

_BUILTIN_FACTORY_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_PLUGIN_FACTORY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_BUILTIN_FACTORY_IDS = frozenset(
    {
        "mock",
        "prompt_color",
        "prompt_color_guarded",
        "prompt_color_margin",
        "reward_3d",
        "reward_general",
    }
)
_RESOURCE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
_FACTORY_CONFIG_RUNTIME_KEYS = frozenset(
    {
        "absolute_path",
        "cache_dir",
        "device",
        "device_id",
        "dtype",
        "endpoint",
        "hostname",
        "local_rank",
        "output_dir",
        "path",
        "rank",
        "run_dir",
        "runtime_context",
        "url",
        "worker",
        "worker_domain",
        "world_size",
    }
)


def _resource_token(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _RESOURCE_TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a canonical resource token")
    return value


def _canonical_set(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{field_name} must be a non-empty sequence")
    resolved = tuple(
        _resource_token(item, field_name=f"{field_name} item") for item in value
    )
    if resolved != tuple(sorted(set(resolved))):
        raise ValueError(f"{field_name} must be sorted and unique")
    return resolved


def _validate_semantic_factory_config(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{location} keys must be non-empty strings")
            if key.lower() in _FACTORY_CONFIG_RUNTIME_KEYS:
                raise ValueError(
                    f"{location}.{key} is an actual runtime/location fact, not "
                    "semantic factory config"
                )
            _validate_semantic_factory_config(
                item,
                location=f"{location}.{key}",
            )
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_semantic_factory_config(
                item,
                location=f"{location}[{index}]",
            )
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} must not contain non-finite numbers")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(f"{location} must contain only JSON-safe values")


@dataclass(frozen=True, slots=True)
class RewardRuntimePolicy:
    """Semantic constraints on a later runtime bind, never actual facts."""

    allowed_devices: tuple[str, ...]
    allowed_dtypes: tuple[str, ...]
    allowed_worker_domains: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "allowed_devices",
            "allowed_dtypes",
            "allowed_worker_domains",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_set(getattr(self, name), field_name=name),
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RewardRuntimePolicy:
        if not isinstance(values, Mapping):
            raise TypeError("allowed_runtime_policy must be a mapping")
        expected = {
            "allowed_devices",
            "allowed_dtypes",
            "allowed_worker_domains",
        }
        if set(values) != expected:
            raise ValueError(
                "allowed_runtime_policy must contain exactly "
                "allowed_devices, allowed_dtypes, and allowed_worker_domains"
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RewardResourceDescriptor:
    """Declared physical reward factory semantics before artifact binding."""

    schema_version: int
    factory_class: str
    artifact_ref: str
    protocol: str
    protocol_version: str
    semantic_factory_config: FrozenMapping
    allowed_runtime_policy: RewardRuntimePolicy

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("reward resource schema_version must equal 1")
        if not isinstance(self.factory_class, str) or not (
            self.factory_class in _BUILTIN_FACTORY_IDS
            or _PLUGIN_FACTORY_RE.fullmatch(self.factory_class)
        ):
            raise ValueError(
                "factory_class must be an explicit builtin id or module:Class"
            )
        if not isinstance(self.artifact_ref, str) or not _BUILTIN_FACTORY_RE.fullmatch(
            self.artifact_ref
        ):
            raise ValueError("artifact_ref must be a canonical artifact id")
        _resource_token(self.protocol, field_name="protocol")
        _resource_token(self.protocol_version, field_name="protocol_version")
        if not isinstance(self.semantic_factory_config, FrozenMapping):
            raise TypeError("semantic_factory_config must be a FrozenMapping")
        _validate_semantic_factory_config(
            self.semantic_factory_config,
            location="semantic_factory_config",
        )
        if not isinstance(self.allowed_runtime_policy, RewardRuntimePolicy):
            raise TypeError("allowed_runtime_policy must be a RewardRuntimePolicy")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> RewardResourceDescriptor:
        if not isinstance(values, Mapping):
            raise TypeError("reward resource descriptor must be a mapping")
        expected = {
            "schema_version",
            "factory_class",
            "artifact_ref",
            "protocol",
            "protocol_version",
            "semantic_factory_config",
            "allowed_runtime_policy",
        }
        if set(values) != expected:
            raise ValueError(
                "reward resource descriptor has an invalid exact key set: "
                f"missing={sorted(expected - set(values))}, "
                f"unknown={sorted(set(values) - expected)}"
            )
        semantic = values["semantic_factory_config"]
        if not isinstance(semantic, Mapping):
            raise TypeError("semantic_factory_config must be a mapping")
        _validate_semantic_factory_config(
            semantic,
            location="semantic_factory_config",
        )
        return cls(
            schema_version=values["schema_version"],
            factory_class=values["factory_class"],
            artifact_ref=values["artifact_ref"],
            protocol=values["protocol"],
            protocol_version=values["protocol_version"],
            semantic_factory_config=FrozenMapping(semantic),
            allowed_runtime_policy=RewardRuntimePolicy.from_mapping(
                values["allowed_runtime_policy"]
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "factory_class": self.factory_class,
            "artifact_ref": self.artifact_ref,
            "protocol": self.protocol,
            "protocol_version": self.protocol_version,
            "semantic_factory_config": to_plain_dict(self.semantic_factory_config),
            "allowed_runtime_policy": {
                "allowed_devices": list(self.allowed_runtime_policy.allowed_devices),
                "allowed_dtypes": list(self.allowed_runtime_policy.allowed_dtypes),
                "allowed_worker_domains": list(
                    self.allowed_runtime_policy.allowed_worker_domains
                ),
            },
        }
