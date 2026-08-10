"""Import-safe ownership descriptors for coarse post-training algorithms.

The blueprint records the four internal roles owned by an algorithm.  It is a
pure value: there are no live components, factories, models, or runtime ports
in this module.  A composition compiler may later resolve these declarations
to concrete components without asking a runtime ``AlgorithmModule`` to make
the same choices again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from visual_rl.core.contracts import (
    AlgorithmComponentResolution,
    AlgorithmComponentRole,
)
from visual_rl.core.identity import canonical_identity
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict

__all__ = (
    "AlgorithmBlueprint",
    "AlgorithmSlotBlueprint",
)

_CANONICAL_ROLES = (
    AlgorithmComponentRole.TRAINER,
    AlgorithmComponentRole.DYNAMICS,
    AlgorithmComponentRole.ROLLOUT,
    AlgorithmComponentRole.CREDIT,
)
_EXPECTED_RESOLUTION = {
    AlgorithmComponentRole.TRAINER: (AlgorithmComponentResolution.ALGORITHM_DEFAULT),
    AlgorithmComponentRole.DYNAMICS: AlgorithmComponentResolution.MODEL_BOUND,
    AlgorithmComponentRole.ROLLOUT: (AlgorithmComponentResolution.ALGORITHM_DEFAULT),
    AlgorithmComponentRole.CREDIT: (AlgorithmComponentResolution.ALGORITHM_DEFAULT),
}
_STAGE_ORDER = (
    "prelude",
    "rollout",
    "reward",
    "advantage",
    "credit",
    "optimize",
)


@dataclass(frozen=True, slots=True)
class AlgorithmSlotBlueprint:
    """One algorithm-owned role and its canonical semantic parameters.

    ``component_id`` is intentionally absent for a model-bound Dynamics role:
    the algorithm specifies the implementation family and semantic parameters,
    while the compiler selects the concrete implementation compatible with the
    model scheduler ABI.
    """

    role: AlgorithmComponentRole
    implementation_family: str
    resolution: AlgorithmComponentResolution
    component_id: str | None
    params: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.role, AlgorithmComponentRole):
            raise TypeError("role must be an AlgorithmComponentRole")
        if self.role not in _CANONICAL_ROLES:
            raise ValueError("algorithm blueprint does not own conditioner slots")
        if (
            not isinstance(self.implementation_family, str)
            or not self.implementation_family
            or self.implementation_family.strip() != self.implementation_family
        ):
            raise ValueError("implementation_family must be canonical text")
        if not isinstance(self.resolution, AlgorithmComponentResolution):
            raise TypeError("resolution must be an AlgorithmComponentResolution")
        if self.resolution is not _EXPECTED_RESOLUTION[self.role]:
            raise ValueError("slot resolution violates algorithm ownership")
        if self.resolution is AlgorithmComponentResolution.ALGORITHM_DEFAULT:
            if (
                not isinstance(self.component_id, str)
                or not self.component_id
                or self.component_id.strip() != self.component_id
            ):
                raise ValueError(
                    "algorithm-default slots require a canonical component_id"
                )
        elif self.component_id is not None:
            raise ValueError("model-bound slots cannot name a concrete component")
        if not isinstance(self.params, FrozenMapping):
            object.__setattr__(self, "params", FrozenMapping(self.params))

    def to_payload(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "implementation_family": self.implementation_family,
            "resolution": self.resolution.value,
            "component_id": self.component_id,
            "params": to_plain_dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class AlgorithmBlueprint:
    """Canonical internal shape emitted by one frozen algorithm config."""

    algorithm_component_id: str
    slots: tuple[AlgorithmSlotBlueprint, ...]
    objective_identity: str
    beta: float
    stage_order: tuple[str, ...] = _STAGE_ORDER
    blueprint_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("algorithm_component_id", "objective_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be canonical text")
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not math.isfinite(float(self.beta))
            or float(self.beta) < 0.0
        ):
            raise ValueError("algorithm blueprint beta must be finite and non-negative")
        beta = float(self.beta)
        object.__setattr__(self, "beta", 0.0 if beta == 0.0 else beta)
        if type(self.slots) is not tuple or any(
            not isinstance(item, AlgorithmSlotBlueprint) for item in self.slots
        ):
            raise TypeError("slots must contain AlgorithmSlotBlueprint values")
        roles = tuple(item.role for item in self.slots)
        if roles != _CANONICAL_ROLES:
            raise ValueError(
                "algorithm blueprint slots must use canonical trainer/dynamics/"
                "rollout/credit order"
            )
        if self.stage_order != _STAGE_ORDER:
            raise ValueError("algorithm blueprint must use canonical stage order")
        object.__setattr__(
            self,
            "blueprint_id",
            canonical_identity("algorithm-blueprint.v1", self),
        )

    def slot(
        self,
        role: AlgorithmComponentRole | str,
    ) -> AlgorithmSlotBlueprint:
        try:
            requested = AlgorithmComponentRole(role)
        except (TypeError, ValueError):
            raise KeyError(role) from None
        for item in self.slots:
            if item.role is requested:
                return item
        raise KeyError(role)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "algorithm_component_id": self.algorithm_component_id,
            "slots": [item.to_payload() for item in self.slots],
            "objective_identity": self.objective_identity,
            "beta": self.beta,
            "stage_order": list(self.stage_order),
        }
