"""Immutable key-derived Dynamics selection policy.

The training hot path derives every rollout and selection RNG stream from a
canonical iteration identity.  Checkpoints therefore persist this policy, not
the cursor of a mutable generator that the hot path never consumes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from visual_rl.algorithms.dynamics.interface import DynamicsContractError

__all__ = (
    "DYNAMICS_SELECTION_SEED_DERIVATION_SCHEMA",
    "DYNAMICS_SELECTION_SEED_DERIVATION_VERSION",
    "DynamicsSelectionPolicyState",
)

DYNAMICS_SELECTION_SEED_DERIVATION_SCHEMA = (
    "visual-rl.iteration-keyed-dynamics-selection"
)
DYNAMICS_SELECTION_SEED_DERIVATION_VERSION = 1
_POLICY_STATE_SCHEMA_VERSION = 2
_DEFAULT_SELECTION_CONTRACT_IDENTITY = "visual-rl.rollout-selection.default.v1"


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class DynamicsSelectionPolicyState:
    """Serializable identity of iteration-keyed RNG selection semantics."""

    base_seed: int
    selection_contract_identity: str = _DEFAULT_SELECTION_CONTRACT_IDENTITY
    seed_derivation_schema: str = DYNAMICS_SELECTION_SEED_DERIVATION_SCHEMA
    seed_derivation_version: int = DYNAMICS_SELECTION_SEED_DERIVATION_VERSION
    policy_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.base_seed) is not int or self.base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        _canonical_text("selection_contract_identity", self.selection_contract_identity)
        _canonical_text("seed_derivation_schema", self.seed_derivation_schema)
        if (
            type(self.seed_derivation_version) is not int
            or self.seed_derivation_version < 1
        ):
            raise ValueError("seed_derivation_version must be a positive integer")
        object.__setattr__(
            self,
            "policy_identity",
            _identity(self._identity_payload()),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": _POLICY_STATE_SCHEMA_VERSION,
            "base_seed": self.base_seed,
            "selection_contract_identity": self.selection_contract_identity,
            "seed_derivation_schema": self.seed_derivation_schema,
            "seed_derivation_version": self.seed_derivation_version,
        }

    def derive_stream_seed(self, *, rollout_identity: str, stream: str) -> int:
        """Derive one independent 63-bit seed without advancing mutable state."""

        rollout_identity = _canonical_text("rollout_identity", rollout_identity)
        stream = _canonical_text("stream", stream)
        digest = hashlib.sha256(
            (
                f"{self.seed_derivation_schema}\0"
                f"{self.seed_derivation_version}\0"
                f"{self.base_seed}\0{rollout_identity}\0{stream}"
            ).encode()
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)

    def to_checkpoint_payload(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload["policy_identity"] = self.policy_identity
        return payload

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: object,
    ) -> DynamicsSelectionPolicyState:
        if not isinstance(payload, Mapping):
            raise TypeError("Dynamics selection policy payload must be a mapping")
        expected = {
            "schema_version",
            "base_seed",
            "selection_contract_identity",
            "seed_derivation_schema",
            "seed_derivation_version",
            "policy_identity",
        }
        if set(payload) != expected:
            raise DynamicsContractError(
                "Dynamics selection policy payload has invalid fields"
            )
        if payload["schema_version"] != _POLICY_STATE_SCHEMA_VERSION:
            raise DynamicsContractError(
                "Dynamics selection policy schema version is unsupported"
            )
        result = cls(
            base_seed=payload["base_seed"],
            selection_contract_identity=payload["selection_contract_identity"],
            seed_derivation_schema=payload["seed_derivation_schema"],
            seed_derivation_version=payload["seed_derivation_version"],
        )
        if payload["policy_identity"] != result.policy_identity:
            raise DynamicsContractError("Dynamics selection policy identity mismatch")
        return result

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicsSelectionPolicyState)
            and self.policy_identity == other.policy_identity
        )

    def __hash__(self) -> int:
        return hash(self.policy_identity)
