"""Canonical stochastic-transition interfaces for diffusion/flow dynamics.

Concrete dynamics implement model-independent schedule/noise operations and the
single authoritative :meth:`Dynamics.transition_mean_std`.  Sampling and
arbitrary-action replay are final template methods so the two paths cannot
silently drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, final

from visual_rl.core.contracts import LikelihoodSemantics


class DynamicsContractError(ValueError):
    """Raised when a Dynamics implementation violates its numerical contract."""


class DynamicsComponent(ABC):
    """Registry-independent runtime boundary for one bound Dynamics factory.

    Static declarations live in :mod:`visual_rl.algorithms.dynamics.config`.
    Runtime loading checks this domain-owned interface after environment and
    artifact gates; the Dynamics domain therefore does not depend on the
    legacy registry interface hierarchy.
    """

    INTERFACE_VERSION = "1.0"

    @classmethod
    @abstractmethod
    def describe(cls, config: object) -> object:
        """Return the exact declared contract implemented by ``config``."""

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> DynamicsComponent:
        """Construct a run-level factory from one declared frozen config."""

        raise NotImplementedError


def _strings(name: str, value: object, batch_size: int) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != batch_size:
        raise DynamicsContractError(f"{name} must contain B entries")
    if any(not isinstance(item, str) or not item for item in value):
        raise DynamicsContractError(f"{name} values must be non-empty strings")
    return value


def _finite(name: str, value: Any) -> None:
    import torch

    if not bool(torch.isfinite(value).all()):
        raise DynamicsContractError(f"{name} must be finite")


def _detached(name: str, value: Any) -> None:
    if value.requires_grad or value.grad_fn is not None:
        raise DynamicsContractError(f"{name} must be detached")


@dataclass(frozen=True)
class TransitionInput:
    """One batched transition request; tensors use explicit ``[B,...]`` shape."""

    x_t: Any
    model_prediction: Any
    t: Any
    t_next: Any
    mask: Any
    transition_index: Any
    condition_identity: tuple[str, ...]
    guidance_identity: tuple[str, ...]
    storage_dtype_identity: tuple[str, ...]
    quantization_identity: tuple[str, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.x_t.shape[0])

    def validate(self) -> None:
        import torch

        if not isinstance(self.x_t, torch.Tensor):
            raise TypeError("x_t must be a torch.Tensor")
        if not isinstance(self.model_prediction, torch.Tensor):
            raise TypeError("model_prediction must be a torch.Tensor")
        if self.x_t.ndim < 2 or self.x_t.shape[0] < 1:
            raise DynamicsContractError("x_t must have shape [B,...]")
        if tuple(self.model_prediction.shape) != tuple(self.x_t.shape):
            raise DynamicsContractError("model_prediction must match x_t shape")
        if (
            not self.x_t.is_floating_point()
            or not self.model_prediction.is_floating_point()
        ):
            raise TypeError("x_t and model_prediction must be floating point")
        if self.model_prediction.device != self.x_t.device:
            raise DynamicsContractError("model_prediction and x_t devices must match")
        _detached("x_t", self.x_t)
        _finite("x_t", self.x_t)
        _finite("model_prediction", self.model_prediction)

        batch_size = self.batch_size
        for name in ("t", "t_next"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(value.shape) != (batch_size,):
                raise DynamicsContractError(f"{name} must have shape [B]")
            if value.device != self.x_t.device:
                raise DynamicsContractError(f"{name} must be on the x_t device")
            _detached(name, value)
            _finite(name, value)
        if self.t.dtype != self.t_next.dtype:
            raise DynamicsContractError("t and t_next must use the same dtype")

        if not isinstance(self.mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
        if tuple(self.mask.shape) != (batch_size,) or self.mask.dtype != torch.bool:
            raise DynamicsContractError("mask must be bool [B]")
        if self.mask.device != self.x_t.device:
            raise DynamicsContractError("mask must be on the x_t device")
        _detached("mask", self.mask)
        if not bool(self.mask.any()):
            raise DynamicsContractError("at least one transition row must be active")

        if not isinstance(self.transition_index, torch.Tensor):
            raise TypeError("transition_index must be a torch.Tensor")
        if (
            tuple(self.transition_index.shape) != (batch_size,)
            or self.transition_index.dtype != torch.int64
        ):
            raise DynamicsContractError("transition_index must be int64 [B]")
        if self.transition_index.device != self.x_t.device:
            raise DynamicsContractError("transition_index must be on the x_t device")
        _detached("transition_index", self.transition_index)
        if bool((self.transition_index < 0).any()):
            raise DynamicsContractError("transition_index must be non-negative")

        _strings("condition_identity", self.condition_identity, batch_size)
        _strings("guidance_identity", self.guidance_identity, batch_size)
        storage = _strings(
            "storage_dtype_identity",
            self.storage_dtype_identity,
            batch_size,
        )
        _strings("quantization_identity", self.quantization_identity, batch_size)
        if any(item != str(self.x_t.dtype) for item in storage):
            raise DynamicsContractError(
                "storage_dtype_identity must match the stored x_t dtype"
            )


@dataclass(frozen=True)
class TransitionMeanStd:
    """Authoritative parameters shared by sampling and action replay."""

    mean: Any
    std: Any
    dt: Any

    def validate_against(self, transition: TransitionInput) -> None:
        import torch

        transition.validate()
        if not isinstance(self.mean, torch.Tensor):
            raise TypeError("transition mean must be a torch.Tensor")
        if not isinstance(self.std, torch.Tensor):
            raise TypeError("transition std must be a torch.Tensor")
        if not isinstance(self.dt, torch.Tensor):
            raise TypeError("transition dt must be a torch.Tensor")
        if tuple(self.mean.shape) != tuple(transition.x_t.shape):
            raise DynamicsContractError("transition mean must match x_t shape")
        if not self.mean.is_floating_point() or not self.std.is_floating_point():
            raise TypeError("transition mean/std must be floating point")
        if (
            self.mean.device != transition.x_t.device
            or self.std.device != transition.x_t.device
        ):
            raise DynamicsContractError("transition mean/std device mismatch")
        if (
            self.mean.dtype != transition.x_t.dtype
            or self.std.dtype != transition.x_t.dtype
        ):
            raise DynamicsContractError("transition mean/std dtype must match x_t")

        expected = tuple(transition.x_t.shape)
        actual = tuple(self.std.shape)
        if len(actual) != len(expected) or actual[0] != expected[0]:
            raise DynamicsContractError(
                "transition std must have one broadcastable row per sample"
            )
        if any(
            value not in {1, target} for value, target in zip(actual[1:], expected[1:])
        ):
            raise DynamicsContractError("transition std is not broadcastable to x_t")
        if tuple(self.dt.shape) != (transition.batch_size,):
            raise DynamicsContractError("transition dt must have shape [B]")
        if self.dt.device != transition.x_t.device or not self.dt.is_floating_point():
            raise DynamicsContractError(
                "transition dt must be floating [B] on x_t device"
            )

        _finite("transition mean", self.mean)
        _finite("transition std", self.std)
        _finite("transition dt", self.dt)
        active = transition.mask
        if bool((self.dt.masked_select(active) == 0).any()):
            raise DynamicsContractError("active transition dt must be non-zero")
        if bool((self.std <= 0).any()):
            raise DynamicsContractError(
                "stochastic transition std must be strictly positive; "
                "ODE/zero-variance transitions cannot provide policy log-prob"
            )


@dataclass(frozen=True, slots=True)
class DeterministicTransitionOutput:
    """Dynamics-owned result of one explicit deterministic ODE transition."""

    next_state: Any
    dt: Any

    def validate_against(self, transition: TransitionInput) -> None:
        import torch

        transition.validate()
        if not isinstance(self.next_state, torch.Tensor):
            raise TypeError("deterministic next_state must be a torch.Tensor")
        if tuple(self.next_state.shape) != tuple(transition.x_t.shape):
            raise DynamicsContractError("deterministic next_state must match x_t shape")
        if self.next_state.device != transition.x_t.device:
            raise DynamicsContractError("deterministic next_state device mismatch")
        if self.next_state.dtype != transition.x_t.dtype:
            raise DynamicsContractError("deterministic next_state dtype mismatch")
        if not self.next_state.is_floating_point():
            raise TypeError("deterministic next_state must be floating point")
        _detached("deterministic next_state", self.next_state)
        _finite("deterministic next_state", self.next_state)

        if not isinstance(self.dt, torch.Tensor):
            raise TypeError("deterministic dt must be a torch.Tensor")
        if tuple(self.dt.shape) != (transition.batch_size,):
            raise DynamicsContractError("deterministic dt must have shape [B]")
        if self.dt.device != transition.x_t.device or not self.dt.is_floating_point():
            raise DynamicsContractError(
                "deterministic dt must be floating [B] on x_t device"
            )
        _detached("deterministic dt", self.dt)
        _finite("deterministic dt", self.dt)
        if bool((self.dt.masked_select(transition.mask) == 0).any()):
            raise DynamicsContractError("active deterministic dt must be non-zero")


@dataclass(frozen=True, slots=True)
class TransitionEvaluation:
    """One authoritative mean/std evaluation and its action log-probability."""

    stats: TransitionMeanStd
    log_prob: Any

    def validate_against(self, transition: TransitionInput) -> None:
        import torch

        if not isinstance(self.stats, TransitionMeanStd):
            raise TypeError("evaluation stats must be TransitionMeanStd")
        self.stats.validate_against(transition)
        if not isinstance(self.log_prob, torch.Tensor):
            raise TypeError("evaluation log_prob must be a torch.Tensor")
        if tuple(self.log_prob.shape) != (transition.batch_size,):
            raise DynamicsContractError("evaluation log_prob must have shape [B]")
        if not self.log_prob.is_floating_point():
            raise TypeError("evaluation log_prob must be floating point")
        if self.log_prob.device != transition.x_t.device:
            raise DynamicsContractError("evaluation log_prob device mismatch")
        _finite(
            "active evaluation log_prob",
            self.log_prob.masked_select(transition.mask),
        )
        if bool((self.log_prob.masked_select(~transition.mask) != 0).any()):
            raise DynamicsContractError(
                "inactive evaluation log_prob values must be zero"
            )


@dataclass(frozen=True, slots=True)
class TransitionPolicyMetadata:
    """Optional per-row statistics consumed by a registered CreditStrategy.

    Dynamics owns these values because they are derived from the transition
    equation.  Keeping them here prevents Trainer code from reimplementing
    scheduler mathematics or switching on a recipe/model name.
    """

    transition_std_dev: Any | None = None
    rectification_coefficient: Any | None = None

    def validate_against(self, transition: TransitionInput) -> None:
        import torch

        if not isinstance(transition, TransitionInput):
            raise TypeError("transition must be a TransitionInput")
        for name in ("transition_std_dev", "rectification_coefficient"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor or None")
            if tuple(value.shape) != (transition.batch_size,):
                raise DynamicsContractError(f"{name} must have shape [B]")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if value.device != transition.x_t.device:
                raise DynamicsContractError(f"{name} device mismatch")
            if value.requires_grad or value.grad_fn is not None:
                raise DynamicsContractError(f"{name} must be detached")
            _finite(name, value)
            if bool((value <= 0).any()):
                raise DynamicsContractError(f"{name} must be strictly positive")

    def slice(self, indices: Any) -> TransitionPolicyMetadata:
        """Select replay rows without exposing concrete Dynamics formulas."""

        import torch

        values = tuple(
            value
            for value in (
                self.transition_std_dev,
                self.rectification_coefficient,
            )
            if value is not None
        )
        if not values:
            return self
        first = values[0]
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 1 or indices.dtype not in {torch.int32, torch.int64}:
                raise TypeError("metadata indices must be a 1-D integer tensor")
            resolved = indices.to(device=first.device, dtype=torch.int64)
        elif isinstance(indices, (tuple, list)):
            if not indices or any(type(item) is not int for item in indices):
                raise TypeError("metadata indices must contain integers")
            resolved = torch.tensor(indices, dtype=torch.int64, device=first.device)
        else:
            raise TypeError("metadata indices must be a sequence or tensor")
        if resolved.numel() < 1:
            raise ValueError("metadata slice must not be empty")
        batch_size = int(first.shape[0])
        if bool((resolved < 0).any()) or bool((resolved >= batch_size).any()):
            raise IndexError("metadata index is out of range")

        def select(value: Any | None) -> Any | None:
            if value is None:
                return None
            return value.index_select(0, resolved.to(device=value.device))

        return TransitionPolicyMetadata(
            transition_std_dev=select(self.transition_std_dev),
            rectification_coefficient=select(self.rectification_coefficient),
        )

    def to(self, device: Any, dtype: Any = None) -> TransitionPolicyMetadata:
        """Move detached metadata with its replay record."""

        def move(value: Any | None) -> Any | None:
            if value is None:
                return None
            target_dtype = dtype if dtype is not None else value.dtype
            return value.to(device=device, dtype=target_dtype).detach()

        return TransitionPolicyMetadata(
            transition_std_dev=move(self.transition_std_dev),
            rectification_coefficient=move(self.rectification_coefficient),
        )


@dataclass(frozen=True)
class TransitionOutput:
    """Dynamics-owned result of one stochastic transition."""

    sampled_next: Any
    mean: Any
    std: Any
    dt: Any
    log_prob: Any
    mask: Any

    def validate_against(self, transition: TransitionInput) -> None:
        import torch

        stats = TransitionMeanStd(self.mean, self.std, self.dt)
        stats.validate_against(transition)
        if not isinstance(self.sampled_next, torch.Tensor):
            raise TypeError("sampled_next must be a torch.Tensor")
        if tuple(self.sampled_next.shape) != tuple(transition.x_t.shape):
            raise DynamicsContractError("sampled_next must match x_t shape")
        if self.sampled_next.device != transition.x_t.device:
            raise DynamicsContractError("sampled_next device mismatch")
        if self.sampled_next.dtype != transition.x_t.dtype:
            raise DynamicsContractError("sampled_next dtype mismatch")
        _finite("sampled_next", self.sampled_next)
        if not isinstance(self.log_prob, torch.Tensor):
            raise TypeError("log_prob must be a torch.Tensor")
        if tuple(self.log_prob.shape) != (transition.batch_size,):
            raise DynamicsContractError("log_prob must have shape [B]")
        if not self.log_prob.is_floating_point():
            raise TypeError("log_prob must be floating point")
        if self.log_prob.device != transition.x_t.device:
            raise DynamicsContractError("log_prob device mismatch")
        _finite("active log_prob", self.log_prob.masked_select(transition.mask))
        if not isinstance(self.mask, torch.Tensor) or not torch.equal(
            self.mask,
            transition.mask,
        ):
            raise DynamicsContractError("TransitionOutput.mask must echo input mask")


@dataclass(frozen=True)
class TransitionRecord:
    """Detached replay record containing action and post-hook next state."""

    x_t: Any
    sampled_action: Any
    conditioned_next: Any
    t: Any
    t_next: Any
    mean: Any
    std: Any
    dt: Any
    old_log_prob: Any
    mask: Any
    transition_index: Any
    likelihood_semantics: LikelihoodSemantics
    condition_identity: tuple[str, ...]
    guidance_identity: tuple[str, ...]
    storage_dtype_identity: tuple[str, ...]
    quantization_identity: tuple[str, ...]
    policy_metadata: TransitionPolicyMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.likelihood_semantics, LikelihoodSemantics):
            try:
                object.__setattr__(
                    self,
                    "likelihood_semantics",
                    LikelihoodSemantics(self.likelihood_semantics),
                )
            except (TypeError, ValueError):
                raise DynamicsContractError("invalid likelihood semantics") from None
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.x_t.shape[0])

    @property
    def scoring_target(self) -> Any:
        if self.likelihood_semantics is LikelihoodSemantics.EXACT_ENV_ACTION:
            return self.sampled_action
        return self.conditioned_next

    def validate(self) -> None:
        import torch

        transition = TransitionInput(
            x_t=self.x_t,
            model_prediction=self.x_t,
            t=self.t,
            t_next=self.t_next,
            mask=self.mask,
            transition_index=self.transition_index,
            condition_identity=self.condition_identity,
            guidance_identity=self.guidance_identity,
            storage_dtype_identity=self.storage_dtype_identity,
            quantization_identity=self.quantization_identity,
        )
        stats = TransitionMeanStd(self.mean, self.std, self.dt)
        stats.validate_against(transition)
        if not isinstance(self.policy_metadata, TransitionPolicyMetadata):
            raise TypeError("policy_metadata must be TransitionPolicyMetadata")
        self.policy_metadata.validate_against(transition)
        for name in ("sampled_action", "conditioned_next"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(value.shape) != tuple(self.x_t.shape):
                raise DynamicsContractError(f"{name} must match x_t shape")
            if value.dtype != self.x_t.dtype or value.device != self.x_t.device:
                raise DynamicsContractError(f"{name} dtype/device mismatch")
            _detached(name, value)
            _finite(name, value)
        if not isinstance(self.old_log_prob, torch.Tensor):
            raise TypeError("old_log_prob must be a torch.Tensor")
        if tuple(self.old_log_prob.shape) != (self.batch_size,):
            raise DynamicsContractError("old_log_prob must have shape [B]")
        if not self.old_log_prob.is_floating_point():
            raise TypeError("old_log_prob must be floating point")
        if self.old_log_prob.device != self.x_t.device:
            raise DynamicsContractError("old_log_prob device mismatch")
        _detached("old_log_prob", self.old_log_prob)
        _finite("active old_log_prob", self.old_log_prob.masked_select(self.mask))
        for name in (
            "x_t",
            "t",
            "t_next",
            "mean",
            "std",
            "dt",
            "mask",
            "transition_index",
        ):
            _detached(name, getattr(self, name))

    def slice(self, indices: Any) -> TransitionRecord:
        import torch

        if isinstance(indices, torch.Tensor):
            if indices.ndim != 1 or indices.dtype not in {torch.int32, torch.int64}:
                raise TypeError("record indices must be a 1-D integer tensor")
            resolved = [int(item) for item in indices.to(device="cpu").tolist()]
        elif isinstance(indices, (tuple, list)):
            if any(type(item) is not int for item in indices):
                raise TypeError("record indices must contain integers")
            resolved = list(indices)
        else:
            raise TypeError("record indices must be a sequence or tensor")
        if not resolved:
            raise ValueError("record slice must not be empty")
        if any(not 0 <= item < self.batch_size for item in resolved):
            raise IndexError("record index is out of range")

        cache: dict[object, Any] = {}

        def select(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                index = cache.get(value.device)
                if index is None:
                    index = torch.tensor(
                        resolved, dtype=torch.long, device=value.device
                    )
                    cache[value.device] = index
                return value.index_select(0, index)
            return tuple(value[item] for item in resolved)

        return replace(
            self,
            x_t=select(self.x_t),
            sampled_action=select(self.sampled_action),
            conditioned_next=select(self.conditioned_next),
            t=select(self.t),
            t_next=select(self.t_next),
            mean=select(self.mean),
            std=select(self.std),
            dt=select(self.dt),
            old_log_prob=select(self.old_log_prob),
            mask=select(self.mask),
            transition_index=select(self.transition_index),
            condition_identity=select(self.condition_identity),
            guidance_identity=select(self.guidance_identity),
            storage_dtype_identity=select(self.storage_dtype_identity),
            quantization_identity=select(self.quantization_identity),
            policy_metadata=self.policy_metadata.slice(resolved),
        )

    def to(self, device: Any, dtype: Any = None) -> TransitionRecord:
        import torch

        def move(value: Any, *, cast: bool) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            target_dtype = (
                dtype
                if cast
                and dtype is not None
                and (value.is_floating_point() or value.is_complex())
                else None
            )
            return value.to(device=device, dtype=target_dtype)

        x_t = move(self.x_t, cast=True)
        return replace(
            self,
            x_t=x_t,
            sampled_action=move(self.sampled_action, cast=True),
            conditioned_next=move(self.conditioned_next, cast=True),
            t=move(self.t, cast=False),
            t_next=move(self.t_next, cast=False),
            mean=move(self.mean, cast=True),
            std=move(self.std, cast=True),
            dt=move(self.dt, cast=True),
            old_log_prob=move(self.old_log_prob, cast=True),
            mask=move(self.mask, cast=False),
            transition_index=move(self.transition_index, cast=False),
            storage_dtype_identity=tuple(
                str(x_t.dtype) for _ in range(self.batch_size)
            ),
            policy_metadata=self.policy_metadata.to(device, dtype),
        )

    def detach(self) -> TransitionRecord:
        def detached(value: Any) -> Any:
            return value.detach() if hasattr(value, "detach") else value

        return replace(
            self,
            x_t=detached(self.x_t),
            sampled_action=detached(self.sampled_action),
            conditioned_next=detached(self.conditioned_next),
            t=detached(self.t),
            t_next=detached(self.t_next),
            mean=detached(self.mean),
            std=detached(self.std),
            dt=detached(self.dt),
            old_log_prob=detached(self.old_log_prob),
            mask=detached(self.mask),
            transition_index=detached(self.transition_index),
            policy_metadata=TransitionPolicyMetadata(
                transition_std_dev=(
                    None
                    if self.policy_metadata.transition_std_dev is None
                    else self.policy_metadata.transition_std_dev.detach()
                ),
                rectification_coefficient=(
                    None
                    if self.policy_metadata.rectification_coefficient is None
                    else self.policy_metadata.rectification_coefficient.detach()
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class TransitionSchedule:
    """Explicit current/next timestep pairs, including the terminal value."""

    timesteps: Any
    next_timesteps: Any

    def __post_init__(self) -> None:
        import torch

        for name in ("timesteps", "next_timesteps"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 1 or value.numel() < 1:
                raise DynamicsContractError(f"{name} must be non-empty 1-D")
            if value.dtype == torch.bool or value.is_complex():
                raise TypeError(f"{name} must use a real numeric dtype")
            if value.requires_grad or value.grad_fn is not None:
                raise DynamicsContractError(f"{name} must be detached")
            if not bool(torch.isfinite(value).all()):
                raise DynamicsContractError(f"{name} must be finite")
        if self.timesteps.shape != self.next_timesteps.shape:
            raise DynamicsContractError(
                "timesteps and next_timesteps must have the same shape"
            )
        if self.timesteps.dtype != self.next_timesteps.dtype:
            raise DynamicsContractError(
                "timesteps and next_timesteps must preserve one dtype"
            )
        if self.timesteps.device != self.next_timesteps.device:
            raise DynamicsContractError(
                "timesteps and next_timesteps must share one device"
            )
        if self.timesteps.numel() > 1 and not bool(
            torch.equal(self.next_timesteps[:-1], self.timesteps[1:])
        ):
            raise DynamicsContractError(
                "each next timestep must equal the following current timestep"
            )

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.numel())


class Dynamics(ABC):
    """Base class with one mean/std implementation for sample and replay."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        forbidden = {
            "deterministic_ode_step",
            "sample_transition",
            "transition_log_prob",
        }.intersection(cls.__dict__)
        if forbidden:
            raise TypeError(
                "Dynamics subclasses must not override template methods: "
                f"{sorted(forbidden)}"
            )

    @abstractmethod
    def timesteps(self, *, num_steps: int, device: Any) -> Any:
        """Return the explicit schedule used by rollout."""

        raise NotImplementedError

    @property
    def dynamics_config_identity(self) -> str:
        """Stable identity of the equation/configuration, excluding a cursor.

        Third-party Dynamics may override this when constructor parameters
        affect the transition equation.  The default remains useful for
        stateless analytical Dynamics used in tests and plugins.
        """

        dynamics_type = type(self)
        return f"{dynamics_type.__module__}.{dynamics_type.__qualname__}"

    @property
    def scheduler_identity(self) -> str:
        """Stable scheduler family/revision identity for schedule snapshots."""

        return self.dynamics_config_identity

    def schedule_sigmas(self, *, num_steps: int, device: Any) -> Any | None:
        """Return ordered sigma boundaries when sigma is the transition clock.

        Generic Dynamics may return ``None``; their snapshot ``dt`` is then
        derived from explicit current/next timesteps.  Flow-SDE Dynamics
        override this hook so replay captures the actual sigma-space ``dt``.
        """

        del num_steps, device
        return None

    def terminal_timestep(self, *, device: Any) -> Any:
        """Return the scheduler-defined scalar after the final transition.

        Concrete rollout-capable Dynamics must implement this method.  A
        default zero would silently encode one scheduler family's convention
        and make third-party schedules impossible to validate.
        """

        del device
        raise NotImplementedError(
            "Dynamics must expose its scheduler-defined terminal timestep"
        )

    @final
    def transition_schedule(
        self,
        *,
        num_steps: int,
        device: Any,
    ) -> TransitionSchedule:
        """Return explicit ``(t, t_next)`` pairs without rollout assumptions."""

        import torch

        timesteps = self.timesteps(num_steps=num_steps, device=device)
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps() must return a torch.Tensor")
        if timesteps.ndim != 1 or timesteps.numel() != num_steps:
            raise DynamicsContractError(
                "timesteps() must return exactly num_steps values"
            )
        terminal = self.terminal_timestep(device=device)
        if not isinstance(terminal, torch.Tensor) or terminal.numel() != 1:
            raise TypeError("terminal_timestep() must return a scalar tensor")
        terminal = terminal.reshape(()).to(
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        next_timesteps = torch.cat((timesteps[1:], terminal[None]), dim=0)
        return TransitionSchedule(
            timesteps=timesteps.detach(),
            next_timesteps=next_timesteps.detach(),
        )

    @abstractmethod
    def add_noise(self, clean: Any, noise: Any, timestep: Any) -> Any:
        """Construct a noisy state without hidden schedule position."""

        raise NotImplementedError

    @abstractmethod
    def transition_mean_std(
        self,
        transition: TransitionInput,
    ) -> TransitionMeanStd:
        """Compute the sole authoritative stochastic transition parameters."""

        raise NotImplementedError

    def policy_metadata(
        self,
        transition: TransitionInput,
        stats: TransitionMeanStd,
    ) -> TransitionPolicyMetadata:
        """Return equation-owned statistics required by a credit strategy.

        The default is intentionally empty.  Concrete Dynamics implementations
        expose only values that are mathematically part of their transition;
        an incompatible CreditStrategy is rejected by its typed input checks.
        """

        stats.validate_against(transition)
        return TransitionPolicyMetadata()

    def _checked_mean_std(
        self,
        transition: TransitionInput,
    ) -> TransitionMeanStd:
        transition.validate()
        result = self.transition_mean_std(transition)
        if not isinstance(result, TransitionMeanStd):
            raise TypeError("transition_mean_std() must return TransitionMeanStd")
        result.validate_against(transition)
        return result

    def _deterministic_ode_step(
        self,
        transition: TransitionInput,
    ) -> DeterministicTransitionOutput:
        """Implement the model/scheduler-specific ODE equation or fail closed."""

        del transition
        raise NotImplementedError(
            "Dynamics does not implement an explicit deterministic ODE transition"
        )

    @final
    def deterministic_ode_step(
        self,
        transition: TransitionInput,
    ) -> DeterministicTransitionOutput:
        """Advance one ODE step without substituting the stochastic SDE mean."""

        transition.validate()
        result = self._deterministic_ode_step(transition)
        if not isinstance(result, DeterministicTransitionOutput):
            raise TypeError(
                "_deterministic_ode_step() must return DeterministicTransitionOutput"
            )
        result.validate_against(transition)
        return result

    @final
    def sample_transition(
        self,
        transition: TransitionInput,
        *,
        generator: Any,
    ) -> TransitionOutput:
        """Sample from the same mean/std function used during replay."""

        import torch

        if not isinstance(generator, torch.Generator):
            raise TypeError("sample_transition requires an explicit torch.Generator")
        stats = self._checked_mean_std(transition)
        sampled = self._sample_from_mean_std(
            transition,
            stats,
            generator=generator,
        )
        if not isinstance(sampled, torch.Tensor):
            raise TypeError("_sample_from_mean_std() must return a torch.Tensor")
        log_prob = self._log_prob(
            transition,
            sampled,
            stats,
        )
        output = TransitionOutput(
            sampled_next=sampled,
            mean=stats.mean,
            std=stats.std,
            dt=stats.dt,
            log_prob=log_prob,
            mask=transition.mask,
        )
        output.validate_against(transition)
        return output

    def _sample_from_mean_std(
        self,
        transition: TransitionInput,
        stats: TransitionMeanStd,
        *,
        generator: Any,
    ) -> Any:
        """Protected sampling hook; the public template remains final."""

        import torch

        noise = torch.randn(
            tuple(transition.x_t.shape),
            generator=generator,
            device=transition.x_t.device,
            dtype=transition.x_t.dtype,
        )
        return stats.mean + stats.std * noise

    def _log_prob_epsilon(self) -> float:
        """Compatibility epsilon used by a revision-pinned transition kernel."""

        return 0.0

    @final
    def evaluate_transition(
        self,
        transition: TransitionInput,
        action_latent: Any,
    ) -> TransitionEvaluation:
        """Evaluate one stored action without recomputing transition statistics."""

        stats = self._checked_mean_std(transition)
        result = TransitionEvaluation(
            stats=stats,
            log_prob=self._log_prob(transition, action_latent, stats),
        )
        result.validate_against(transition)
        return result

    @final
    def transition_log_prob(
        self,
        transition: TransitionInput,
        action_latent: Any,
    ) -> Any:
        """Score an arbitrary stored action through the authoritative mean/std."""

        return self.evaluate_transition(transition, action_latent).log_prob

    def _log_prob(
        self,
        transition: TransitionInput,
        action_latent: Any,
        stats: TransitionMeanStd,
    ) -> Any:
        import math

        import torch

        if not isinstance(action_latent, torch.Tensor):
            raise TypeError("action_latent must be a torch.Tensor")
        if tuple(action_latent.shape) != tuple(transition.x_t.shape):
            raise DynamicsContractError("action_latent must match x_t shape")
        if action_latent.device != transition.x_t.device:
            raise DynamicsContractError("action_latent device mismatch")
        if action_latent.dtype != transition.x_t.dtype:
            raise DynamicsContractError("action_latent dtype mismatch")
        _finite("action_latent", action_latent)
        epsilon = self._log_prob_epsilon()
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) < 0
        ):
            raise DynamicsContractError(
                "log-prob epsilon must be a finite non-negative number"
            )
        epsilon = float(epsilon)
        value = (
            -(action_latent.detach() - stats.mean).square()
            / (2.0 * (stats.std.square() + epsilon))
            - torch.log(stats.std + epsilon)
            - 0.5
            * torch.log(
                torch.as_tensor(
                    2.0 * math.pi,
                    device=transition.x_t.device,
                    dtype=transition.x_t.dtype,
                )
            )
        )
        value = value.mean(dim=tuple(range(1, value.ndim)))
        value = torch.where(transition.mask, value, torch.zeros_like(value))
        _finite("active transition log_prob", value.masked_select(transition.mask))
        return value

    @final
    def make_record(
        self,
        transition: TransitionInput,
        output: TransitionOutput,
        *,
        conditioned_next: Any,
        likelihood_semantics: LikelihoodSemantics,
    ) -> TransitionRecord:
        """Freeze a rollout result after an optional Conditioner hook."""

        import torch

        output.validate_against(transition)
        if not isinstance(conditioned_next, torch.Tensor):
            raise TypeError("conditioned_next must be a torch.Tensor")
        if tuple(conditioned_next.shape) != tuple(transition.x_t.shape):
            raise DynamicsContractError("conditioned_next must match x_t shape")
        if (
            conditioned_next.dtype != transition.x_t.dtype
            or conditioned_next.device != transition.x_t.device
        ):
            raise DynamicsContractError("conditioned_next dtype/device mismatch")
        _finite("conditioned_next", conditioned_next)
        try:
            semantics = LikelihoodSemantics(likelihood_semantics)
        except (TypeError, ValueError):
            raise DynamicsContractError("invalid likelihood semantics") from None
        old_log_prob = output.log_prob
        if semantics is LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE:
            old_log_prob = self._log_prob(
                transition,
                conditioned_next,
                TransitionMeanStd(output.mean, output.std, output.dt),
            )
        record = TransitionRecord(
            x_t=transition.x_t.detach(),
            sampled_action=output.sampled_next.detach(),
            conditioned_next=conditioned_next.detach(),
            t=transition.t.detach(),
            t_next=transition.t_next.detach(),
            mean=output.mean.detach(),
            std=output.std.detach(),
            dt=output.dt.detach(),
            old_log_prob=old_log_prob.detach(),
            mask=transition.mask.detach(),
            transition_index=transition.transition_index.detach(),
            likelihood_semantics=semantics,
            condition_identity=transition.condition_identity,
            guidance_identity=transition.guidance_identity,
            storage_dtype_identity=transition.storage_dtype_identity,
            quantization_identity=transition.quantization_identity,
            policy_metadata=self.policy_metadata(
                transition,
                TransitionMeanStd(output.mean, output.std, output.dt),
            ),
        )
        record.validate()
        return record


__all__ = [
    "DeterministicTransitionOutput",
    "Dynamics",
    "DynamicsComponent",
    "DynamicsContractError",
    "TransitionEvaluation",
    "TransitionInput",
    "TransitionMeanStd",
    "TransitionOutput",
    "TransitionPolicyMetadata",
    "TransitionRecord",
    "TransitionSchedule",
]
