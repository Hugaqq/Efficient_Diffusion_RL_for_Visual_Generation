"""Iteration-scoped composition of policy, dynamics, RNG, and conditioner ports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from visual_rl.algorithms.conditioning.interface import LatentConditioner, LatentSpec
from visual_rl.algorithms.dynamics.interface import Dynamics
from visual_rl.algorithms.dynamics.replay import (
    DynamicsReplayBinding,
    DynamicsReplayRequest,
    FlowMatchScheduleConditioning,
)
from visual_rl.algorithms.dynamics.selection import (
    DYNAMICS_SELECTION_SEED_DERIVATION_SCHEMA,
    DYNAMICS_SELECTION_SEED_DERIVATION_VERSION,
    DynamicsSelectionPolicyState,
)
from visual_rl.algorithms.rollout.interface import RolloutRequest
from visual_rl.algorithms.trainer.interface import IterationIdentity
from visual_rl.core.contracts import LikelihoodSemantics
from visual_rl.core.contracts.runtime import PolicyRuntimePort
from visual_rl.data.samples import StackedSampleBatch

__all__ = (
    "DynamicsForRolloutFactory",
    "IterationRolloutRequestFactory",
    "RolloutRequestFactoryError",
)


class RolloutRequestFactoryError(ValueError):
    """Raised when iteration identity or composed runtime ports do not align."""


@runtime_checkable
class DynamicsForRolloutFactory(Protocol):
    """Narrow training-side port for one fresh iteration Dynamics instance."""

    def schedule_conditioning(
        self,
        context: object,
    ) -> FlowMatchScheduleConditioning | None: ...

    def create(self, request: DynamicsReplayRequest) -> Dynamics: ...


def _iteration_payload(identity: IterationIdentity) -> dict[str, object]:
    return {
        "optimizer_step": identity.optimizer_step,
        "source_id": identity.source_id,
        "phase_id": identity.phase_id,
        "row_identities": list(identity.row_identities),
        "group_ids": list(identity.group_ids),
        "member_ids": list(identity.member_ids),
    }


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_batch_identity(
    samples: StackedSampleBatch,
    identity: IterationIdentity,
) -> None:
    observed = (
        tuple(row.identity for row in samples.rows),
        tuple(row.group_id for row in samples.rows),
        tuple(row.member_id for row in samples.rows),
    )
    expected = (
        identity.row_identities,
        identity.group_ids,
        identity.member_ids,
    )
    if observed != expected:
        raise RolloutRequestFactoryError(
            "sample rows do not match the canonical iteration identity"
        )
    if any(
        row.phase != identity.phase_id or row.optimizer_step != identity.optimizer_step
        for row in samples.rows
    ):
        raise RolloutRequestFactoryError(
            "sample phase/step does not match the canonical iteration identity"
        )


@dataclass(frozen=True, slots=True)
class IterationRolloutRequestFactory:
    """Callable RolloutStage port with no mutable per-iteration state."""

    adapter: PolicyRuntimePort
    dynamics_factory: DynamicsForRolloutFactory
    num_steps: int
    likelihood_semantics: LikelihoodSemantics
    base_seed: int
    device: Any
    dtype: Any
    conditioner: LatentConditioner | None = None
    selection_contract_identity: str = "visual-rl.rollout-selection.default.v1"
    seed_derivation_schema: str = DYNAMICS_SELECTION_SEED_DERIVATION_SCHEMA
    seed_derivation_version: int = DYNAMICS_SELECTION_SEED_DERIVATION_VERSION
    _selection_policy: DynamicsSelectionPolicyState = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        import torch

        required_policy_methods = (
            "latent_spec_for_batch",
            "model_schedule_context",
            "encode",
            "prepare_latents",
            "predict",
            "decode",
        )
        missing = tuple(
            name
            for name in required_policy_methods
            if not callable(getattr(self.adapter, name, None))
        )
        if missing:
            raise TypeError(
                "adapter must implement PolicyRuntimePort operations: "
                + ", ".join(missing)
            )
        if not isinstance(self.dynamics_factory, DynamicsForRolloutFactory):
            raise TypeError("dynamics_factory must implement create(request)")
        if type(self.num_steps) is not int or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        try:
            semantics = LikelihoodSemantics(self.likelihood_semantics)
        except (TypeError, ValueError):
            raise ValueError("invalid likelihood semantics") from None
        policy = DynamicsSelectionPolicyState(
            base_seed=self.base_seed,
            selection_contract_identity=self.selection_contract_identity,
            seed_derivation_schema=self.seed_derivation_schema,
            seed_derivation_version=self.seed_derivation_version,
        )
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError):
            raise TypeError("device must be torch.device-compatible") from None
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        if self.conditioner is not None and not isinstance(
            self.conditioner, LatentConditioner
        ):
            raise TypeError("conditioner must be a LatentConditioner")
        object.__setattr__(self, "likelihood_semantics", semantics)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "_selection_policy", policy)

    @property
    def dynamics_selection_policy(self) -> DynamicsSelectionPolicyState:
        return self._selection_policy

    @property
    def dynamics_selection_policy_identity(self) -> str:
        return self._selection_policy.policy_identity

    def __call__(
        self,
        samples: StackedSampleBatch,
        identity: IterationIdentity,
    ) -> RolloutRequest:
        import torch

        if not isinstance(samples, StackedSampleBatch):
            raise TypeError("samples must be a StackedSampleBatch")
        if not isinstance(identity, IterationIdentity):
            raise TypeError("identity must be an IterationIdentity")
        samples.validate()
        _validate_batch_identity(samples, identity)

        latent_spec = self.adapter.latent_spec_for_batch(
            samples,
            device=self.device,
            dtype=self.dtype,
        )
        model_schedule_context = self.adapter.model_schedule_context(latent_spec)
        schedule_conditioning = self.dynamics_factory.schedule_conditioning(
            model_schedule_context
        )

        identity_payload = {
            "schema_version": 1,
            "dynamics_selection_policy_identity": (
                self.dynamics_selection_policy_identity
            ),
            "iteration": _iteration_payload(identity),
        }
        rollout_identity = f"iteration-rollout.v1:{_digest(identity_payload)}"
        replay_request = DynamicsReplayRequest(
            rollout_identity=rollout_identity,
            num_steps=self.num_steps,
            schedule_conditioning=schedule_conditioning,
        )
        dynamics = self.dynamics_factory.create(replay_request)
        if not isinstance(dynamics, Dynamics):
            raise TypeError("Dynamics factory returned a non-Dynamics value")
        binding = getattr(dynamics, "replay_binding", None)
        if not isinstance(binding, DynamicsReplayBinding):
            raise RolloutRequestFactoryError(
                "per-rollout Dynamics must expose its explicit replay binding"
            )
        if binding.request.request_identity != replay_request.request_identity:
            raise RolloutRequestFactoryError(
                "Dynamics replay binding does not match the iteration request identity"
            )

        conditioner_spec = None
        if self.conditioner is not None:
            conditioner_spec = self.conditioner.bind_model_geometry(
                model_schedule_context
            )
            if not isinstance(conditioner_spec, LatentSpec):
                raise TypeError(
                    "conditioner geometry binding must return conditioning.LatentSpec"
                )

        rollout_seed = self._selection_policy.derive_stream_seed(
            rollout_identity=rollout_identity,
            stream="rollout",
        )
        selection_seed = self._selection_policy.derive_stream_seed(
            rollout_identity=rollout_identity,
            stream="selection",
        )
        generator = torch.Generator(device=self.device).manual_seed(rollout_seed)
        selection_generator = torch.Generator(device=self.device).manual_seed(
            selection_seed
        )
        return RolloutRequest(
            adapter=self.adapter,
            dynamics=dynamics,
            samples=samples,
            latent_spec=latent_spec,
            generator=generator,
            selection_generator=selection_generator,
            likelihood_semantics=self.likelihood_semantics,
            conditioner=self.conditioner,
            conditioner_latent_spec=conditioner_spec,
            dynamics_replay_binding=binding,
            selection_contract_identity=self.selection_contract_identity,
        )
