"""Import-safe reward capability and routing contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Literal

from visual_rl.core.contracts.model import MediaKind
from visual_rl.core.identity import canonical_identity
from visual_rl.core.immutable import FrozenMapping

__all__ = (
    "LogicalRewardSpec",
    "RewardContract",
    "RewardGranularity",
    "RewardPlanSpec",
    "RewardResourceSpec",
    "RewardRoute",
    "RewardRouteBinding",
    "RewardRouteSpec",
    "RewardRoutingContract",
)


class RewardGranularity(str, Enum):
    POINTWISE = "pointwise"
    GROUPWISE = "groupwise"


@dataclass(frozen=True)
class RewardContract:
    accepted_media: tuple[MediaKind, ...]
    required_payload_type: str | None
    granularity: RewardGranularity
    output_rank: int
    frame_aggregation: str | None

    def __post_init__(self) -> None:
        _unique("accepted_media", self.accepted_media)
        if not self.accepted_media:
            raise ValueError("accepted_media must not be empty")
        if any(not isinstance(item, MediaKind) for item in self.accepted_media):
            raise TypeError("accepted_media must contain MediaKind values")
        if self.required_payload_type is not None:
            _text("required_payload_type", self.required_payload_type)
        if not isinstance(self.granularity, RewardGranularity):
            raise TypeError("granularity must be a RewardGranularity")
        if type(self.output_rank) is not int or self.output_rank < 0:
            raise ValueError("reward output_rank must be non-negative")
        if self.frame_aggregation is not None:
            _text("frame_aggregation", self.frame_aggregation)


@dataclass(frozen=True)
class RewardRoute:
    source_id: str
    phase: str
    reward_id: str
    weight: float
    payload_type: str | None

    def __post_init__(self) -> None:
        for name in ("source_id", "phase", "reward_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError("reward weight must be numeric")


@dataclass(frozen=True)
class RewardRoutingContract:
    routes: tuple[RewardRoute, ...]

    def __post_init__(self) -> None:
        keys = tuple(
            (item.source_id, item.phase, item.reward_id) for item in self.routes
        )
        if len(keys) != len(set(keys)):
            raise ValueError("reward routes must be unique by source/phase/reward")


@dataclass(frozen=True, slots=True)
class RewardResourceSpec:
    """Physical resource description without endpoint or live runtime state."""

    descriptor: FrozenMapping
    artifact_identity: FrozenMapping | None = None
    resource_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, FrozenMapping):
            raise TypeError("descriptor must be a FrozenMapping")
        _text("descriptor.artifact_ref", self.descriptor.get("artifact_ref"))
        if self.artifact_identity is not None and not isinstance(
            self.artifact_identity,
            FrozenMapping,
        ):
            raise TypeError("artifact_identity must be a FrozenMapping or None")
        _reject_runtime_resource_keys(self.descriptor)
        if self.artifact_identity is not None:
            if not self.artifact_identity:
                raise ValueError("materialized artifact_identity must not be empty")
            _reject_runtime_resource_keys(self.artifact_identity)
        object.__setattr__(
            self,
            "resource_identity",
            canonical_identity(
                "reward-resource-spec.v1",
                {
                    "descriptor": self.descriptor,
                    "artifact_identity": self.artifact_identity,
                },
            ),
        )

    @property
    def provisional(self) -> bool:
        return self.artifact_identity is None

    @property
    def materialized(self) -> bool:
        return self.artifact_identity is not None

    @property
    def state(self) -> Literal["provisional", "materialized"]:
        return "provisional" if self.provisional else "materialized"

    @property
    def artifact_ref(self) -> str:
        return _text("descriptor.artifact_ref", self.descriptor["artifact_ref"])

    def to_payload(self) -> dict[str, object]:
        return {
            "state": self.state,
            "resource_identity": self.resource_identity,
            "descriptor": dict(self.descriptor),
            "artifact_identity": (
                None if self.artifact_identity is None else dict(self.artifact_identity)
            ),
        }


@dataclass(frozen=True, slots=True)
class LogicalRewardSpec:
    """One logical reward bound to a shareable physical resource identity."""

    logical_reward_id: str
    component_declaration_id: str
    resource_identity: str
    contract: RewardContract

    def __post_init__(self) -> None:
        for name in (
            "logical_reward_id",
            "resource_identity",
        ):
            _text(name, getattr(self, name))
        if (
            not isinstance(self.component_declaration_id, str)
            or re.fullmatch(
                r"component-declaration\.v1:[0-9a-f]{64}",
                self.component_declaration_id,
            )
            is None
        ):
            raise ValueError(
                "component_declaration_id must be a component-declaration.v1 identity"
            )
        if not isinstance(self.contract, RewardContract):
            raise TypeError("contract must be a RewardContract")
        if self.contract.output_rank != 1:
            raise ValueError("logical reward contract must produce rank-one [B]")

    def to_payload(self) -> dict[str, object]:
        return {
            "logical_reward_id": self.logical_reward_id,
            "component_declaration_id": self.component_declaration_id,
            "resource_identity": self.resource_identity,
            "contract": self.contract,
        }


@dataclass(frozen=True, slots=True)
class RewardRouteBinding:
    logical_reward_id: str
    weight: float

    def __post_init__(self) -> None:
        _text("logical_reward_id", self.logical_reward_id)
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
        ):
            raise ValueError("reward route weight must be finite numeric")
        object.__setattr__(self, "weight", float(self.weight))

    def to_payload(self) -> dict[str, object]:
        return {
            "logical_reward_id": self.logical_reward_id,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class RewardRouteSpec:
    source_id: str
    phase_id: str
    rewards: tuple[RewardRouteBinding, ...]

    def __post_init__(self) -> None:
        _text("source_id", self.source_id)
        _text("phase_id", self.phase_id)
        if type(self.rewards) is not tuple or not self.rewards:
            raise ValueError("route rewards must be a non-empty tuple")
        if any(not isinstance(item, RewardRouteBinding) for item in self.rewards):
            raise TypeError("rewards must contain RewardRouteBinding values")
        ordered = tuple(sorted(self.rewards, key=lambda item: item.logical_reward_id))
        logical_ids = tuple(item.logical_reward_id for item in ordered)
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("logical reward ids must be unique within a route")
        if not any(item.weight != 0.0 for item in ordered):
            raise ValueError("a route must contain a non-zero reward weight")
        object.__setattr__(self, "rewards", ordered)

    def to_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "phase_id": self.phase_id,
            "rewards": tuple(item.to_payload() for item in self.rewards),
        }

    @property
    def logical_reward_ids(self) -> tuple[str, ...]:
        return tuple(item.logical_reward_id for item in self.rewards)

    def binding(self, logical_reward_id: str) -> RewardRouteBinding:
        _text("logical_reward_id", logical_reward_id)
        for binding in self.rewards:
            if binding.logical_reward_id == logical_reward_id:
                return binding
        raise KeyError(f"reward {logical_reward_id!r} is inactive for this route")


@dataclass(frozen=True, slots=True)
class RewardPlanSpec:
    """Compiler projection consumed by reward planning and runtime binding."""

    resources: tuple[RewardResourceSpec, ...]
    logical_rewards: tuple[LogicalRewardSpec, ...]
    routes: tuple[RewardRouteSpec, ...]

    def __post_init__(self) -> None:
        if type(self.resources) is not tuple or not self.resources:
            raise ValueError("reward plan resources must be a non-empty tuple")
        if any(not isinstance(item, RewardResourceSpec) for item in self.resources):
            raise TypeError("resources must contain RewardResourceSpec values")
        resources = tuple(
            sorted(self.resources, key=lambda item: item.resource_identity)
        )
        resource_ids = tuple(item.resource_identity for item in resources)
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("reward resource identities must be unique")
        if type(self.logical_rewards) is not tuple or not self.logical_rewards:
            raise ValueError("logical_rewards must be a non-empty tuple")
        if any(
            not isinstance(item, LogicalRewardSpec) for item in self.logical_rewards
        ):
            raise TypeError("logical_rewards must contain LogicalRewardSpec values")
        logical_rewards = tuple(
            sorted(self.logical_rewards, key=lambda item: item.logical_reward_id)
        )
        logical_ids = tuple(item.logical_reward_id for item in logical_rewards)
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("logical reward ids must be unique")
        unknown_resources = tuple(
            sorted(
                {
                    item.resource_identity
                    for item in logical_rewards
                    if item.resource_identity not in resource_ids
                }
            )
        )
        unused_resources = tuple(
            sorted(
                set(resource_ids) - {item.resource_identity for item in logical_rewards}
            )
        )
        if unknown_resources or unused_resources:
            raise ValueError(
                "logical rewards/resources differ: "
                f"unknown={list(unknown_resources)}, "
                f"unused={list(unused_resources)}"
            )
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError("routes must be a non-empty tuple")
        if any(not isinstance(item, RewardRouteSpec) for item in self.routes):
            raise TypeError("routes must contain RewardRouteSpec values")
        routes = tuple(
            sorted(self.routes, key=lambda item: (item.source_id, item.phase_id))
        )
        route_keys = tuple((item.source_id, item.phase_id) for item in routes)
        if len(route_keys) != len(set(route_keys)):
            raise ValueError("reward routes must be unique by source/phase")
        routed_ids = {
            binding.logical_reward_id for route in routes for binding in route.rewards
        }
        unknown_logical = tuple(sorted(routed_ids - set(logical_ids)))
        missing_logical = tuple(sorted(set(logical_ids) - routed_ids))
        if unknown_logical or missing_logical:
            raise ValueError(
                "reward routes/logical rewards differ: "
                f"unknown={list(unknown_logical)}, missing={list(missing_logical)}"
            )
        if len({item.provisional for item in resources}) != 1:
            raise ValueError("reward plan cannot mix provisional and bound resources")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "logical_rewards", logical_rewards)
        object.__setattr__(self, "routes", routes)

    @property
    def state(self) -> Literal["provisional", "materialized"]:
        return "provisional" if self.resources[0].provisional else "materialized"

    @property
    def provisional(self) -> bool:
        return self.state == "provisional"

    @property
    def materialized(self) -> bool:
        return self.state == "materialized"

    @property
    def resource_identities(self) -> tuple[str, ...]:
        """Return physical resource identities in canonical plan order."""

        return tuple(item.resource_identity for item in self.resources)

    @property
    def logical_reward_ids(self) -> tuple[str, ...]:
        """Return logical reward ids in canonical plan order."""

        return tuple(item.logical_reward_id for item in self.logical_rewards)

    def resource(self, resource_identity: str) -> RewardResourceSpec:
        _text("resource_identity", resource_identity)
        for resource in self.resources:
            if resource.resource_identity == resource_identity:
                return resource
        raise KeyError(f"unknown reward resource identity {resource_identity!r}")

    def logical_reward(self, logical_reward_id: str) -> LogicalRewardSpec:
        _text("logical_reward_id", logical_reward_id)
        for logical in self.logical_rewards:
            if logical.logical_reward_id == logical_reward_id:
                return logical
        raise KeyError(f"unknown logical reward {logical_reward_id!r}")

    def route_for(self, *, source_id: str, phase_id: str) -> RewardRouteSpec:
        _text("source_id", source_id)
        _text("phase_id", phase_id)
        for route in self.routes:
            if route.source_id == source_id and route.phase_id == phase_id:
                return route
        raise KeyError(f"unknown reward route {(source_id, phase_id)!r}")

    def bind_artifacts(
        self,
        artifact_identities: Mapping[str, FrozenMapping]
        | tuple[tuple[str, FrozenMapping], ...],
    ) -> RewardPlanSpec:
        """Return a materialized copy after exact, path-free artifact binding."""

        if self.materialized:
            raise ValueError("reward plan artifacts are already bound")
        bindings = _artifact_identity_bindings(artifact_identities)
        expected_refs = {resource.artifact_ref for resource in self.resources}
        observed_refs = set(bindings)
        missing = tuple(sorted(expected_refs - observed_refs))
        extra = tuple(sorted(observed_refs - expected_refs))
        if missing or extra:
            raise ValueError(
                "reward artifact identities must exactly cover descriptor refs: "
                f"missing={list(missing)}, extra={list(extra)}"
            )

        remapped_resource_ids: dict[str, str] = {}
        resources = []
        for resource in self.resources:
            materialized = RewardResourceSpec(
                descriptor=resource.descriptor,
                artifact_identity=bindings[resource.artifact_ref],
            )
            remapped_resource_ids[resource.resource_identity] = (
                materialized.resource_identity
            )
            resources.append(materialized)
        logical_rewards = tuple(
            replace(
                logical_reward,
                resource_identity=remapped_resource_ids[
                    logical_reward.resource_identity
                ],
            )
            for logical_reward in self.logical_rewards
        )
        return RewardPlanSpec(
            resources=tuple(resources),
            logical_rewards=logical_rewards,
            routes=self.routes,
        )

    @property
    def plan_id(self) -> str:
        return canonical_identity("reward-plan-spec.v1", self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "state": self.state,
            "resources": tuple(item.to_payload() for item in self.resources),
            "logical_rewards": tuple(
                item.to_payload() for item in self.logical_rewards
            ),
            "routes": tuple(item.to_payload() for item in self.routes),
        }


def _unique(name: str, values: tuple[object, ...]) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _reject_runtime_resource_keys(value: object) -> None:
    forbidden = {
        "absolute_path",
        "cache_dir",
        "client",
        "cwd",
        "device",
        "device_id",
        "endpoint",
        "group_rank",
        "group_world_size",
        "handle",
        "hostname",
        "launch_id",
        "local_process_index",
        "local_rank",
        "local_world_size",
        "logging_dir",
        "master_addr",
        "master_port",
        "node_rank",
        "num_processes",
        "output_dir",
        "path",
        "pid",
        "process_index",
        "rank",
        "run_dir",
        "runtime_context",
        "session",
        "source_path",
        "url",
        "worker",
        "worker_domain",
        "workdir",
        "working_directory",
        "world_size",
    }
    if isinstance(value, FrozenMapping):
        for key, item in value.items():
            normalized = key.lower()
            if (
                normalized in forbidden
                or normalized.endswith(("_endpoint", "_location", "_path", "_url"))
                or normalized.startswith(("device_", "endpoint_"))
            ):
                raise ValueError(
                    f"reward resource plan cannot contain runtime key {key!r}"
                )
            _reject_runtime_resource_keys(item)
    elif isinstance(value, tuple):
        for item in value:
            _reject_runtime_resource_keys(item)
    elif isinstance(value, Path):
        raise TypeError("reward resource plan cannot contain filesystem Path values")


def _artifact_identity_bindings(
    artifact_identities: Mapping[str, FrozenMapping]
    | tuple[tuple[str, FrozenMapping], ...],
) -> dict[str, FrozenMapping]:
    if isinstance(artifact_identities, Mapping):
        pairs = tuple(artifact_identities.items())
    elif type(artifact_identities) is tuple:
        pairs = artifact_identities
    else:
        raise TypeError(
            "artifact_identities must be a mapping or tuple of (artifact_ref, identity)"
        )

    bindings: dict[str, FrozenMapping] = {}
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError(
                "artifact identity bindings must be (artifact_ref, identity) pairs"
            )
        artifact_ref, identity = pair
        _text("artifact_ref", artifact_ref)
        if artifact_ref in bindings:
            raise ValueError(f"duplicate reward artifact identity {artifact_ref!r}")
        if not isinstance(identity, FrozenMapping):
            raise TypeError("each reward artifact identity must be a FrozenMapping")
        if not identity:
            raise ValueError("each reward artifact identity must be non-empty")
        _reject_runtime_resource_keys(identity)
        bindings[artifact_ref] = identity
    return bindings
