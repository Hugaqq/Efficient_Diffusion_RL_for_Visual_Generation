"""Internal helpers for immutable flow-matching scheduler replay state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from visual_rl.algorithms.dynamics.interface import DynamicsContractError


@dataclass(frozen=True)
class _FrozenFlowSchedule:
    """Owned CPU copy of one explicit inference schedule."""

    _timesteps: Any = field(repr=False)
    _sigmas: Any = field(repr=False)
    _terminal_timestep: Any = field(repr=False)
    scheduler_identity: str
    replay_state_identity: str
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        timesteps: Any,
        sigmas: Any,
        *,
        label: str,
        scheduler_identity: str,
        terminal_timestep: Any = None,
        schema_version: int = 1,
    ) -> _FrozenFlowSchedule:
        import torch

        if not isinstance(scheduler_identity, str) or not scheduler_identity:
            raise DynamicsContractError(
                f"{label} scheduler_identity must be a non-empty string"
            )
        if type(schema_version) is not int or schema_version != 1:
            raise DynamicsContractError(f"{label} schedule schema_version must be 1")
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError(f"{label} timesteps must be a torch.Tensor")
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(f"{label} sigmas must be a torch.Tensor")
        if timesteps.ndim != 1 or timesteps.numel() < 1:
            raise DynamicsContractError(f"{label} timesteps must be non-empty 1-D")
        if sigmas.ndim != 1 or sigmas.numel() != timesteps.numel() + 1:
            raise DynamicsContractError(
                f"{label} sigmas must be 1-D with len(timesteps) + 1 entries"
            )
        if timesteps.dtype == torch.bool or timesteps.is_complex():
            raise TypeError(f"{label} timesteps must use a real numeric dtype")
        if not sigmas.is_floating_point():
            raise TypeError(f"{label} sigmas must be floating point")
        if not bool(torch.isfinite(timesteps).all()) or not bool(
            torch.isfinite(sigmas).all()
        ):
            raise DynamicsContractError(f"{label} scheduler state must be finite")
        if torch.unique(timesteps).numel() != timesteps.numel():
            raise DynamicsContractError(f"{label} timesteps must be unique")
        if bool((sigmas < 0).any()):
            raise DynamicsContractError(f"{label} sigmas must be non-negative")
        if not bool((sigmas[:-1] > sigmas[1:]).all()):
            raise DynamicsContractError(
                f"{label} sigmas must be strictly descending so every dt is negative"
            )

        owned_timesteps = timesteps.detach().to(device="cpu").contiguous().clone()
        owned_sigmas = sigmas.detach().to(device="cpu").contiguous().clone()
        if terminal_timestep is None:
            terminal = torch.zeros((), dtype=owned_timesteps.dtype)
        else:
            if not isinstance(terminal_timestep, torch.Tensor):
                raise TypeError(
                    f"{label} terminal_timestep must be a scalar torch.Tensor"
                )
            if terminal_timestep.numel() != 1:
                raise DynamicsContractError(
                    f"{label} terminal_timestep must contain one value"
                )
            if terminal_timestep.dtype != owned_timesteps.dtype:
                raise DynamicsContractError(
                    f"{label} terminal_timestep must preserve timestep dtype"
                )
            if not bool(torch.isfinite(terminal_timestep).all()):
                raise DynamicsContractError(f"{label} terminal_timestep must be finite")
            terminal = (
                terminal_timestep.detach()
                .to(device="cpu")
                .reshape(())
                .contiguous()
                .clone()
            )
        replay_state_identity = _flow_schedule_identity(
            timesteps=owned_timesteps,
            sigmas=owned_sigmas,
            terminal_timestep=terminal,
            scheduler_identity=scheduler_identity,
            schema_version=schema_version,
        )
        return cls(
            _timesteps=owned_timesteps,
            _sigmas=owned_sigmas,
            _terminal_timestep=terminal,
            scheduler_identity=scheduler_identity,
            schema_version=schema_version,
            replay_state_identity=replay_state_identity,
        )

    @property
    def num_steps(self) -> int:
        return int(self._timesteps.numel())

    @property
    def timesteps(self) -> Any:
        return self._timesteps.clone()

    @property
    def sigmas(self) -> Any:
        return self._sigmas.clone()

    @property
    def terminal_timestep(self) -> Any:
        return self._terminal_timestep.clone()


def _tensor_identity(value: Any) -> dict[str, object]:
    import torch

    owned = value.detach().to(device="cpu").contiguous()
    raw = owned.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(owned.dtype),
        "shape": list(owned.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _flow_schedule_identity(
    *,
    timesteps: Any,
    sigmas: Any,
    terminal_timestep: Any,
    scheduler_identity: str,
    schema_version: int,
) -> str:
    payload = {
        "schema_version": schema_version,
        "scheduler_identity": scheduler_identity,
        "timesteps": _tensor_identity(timesteps),
        "sigmas": _tensor_identity(sigmas),
        "terminal_timestep": _tensor_identity(terminal_timestep),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scheduler_identity(scheduler: object) -> str:
    explicit = getattr(scheduler, "scheduler_identity", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    scheduler_type = type(scheduler)
    return f"{scheduler_type.__module__}.{scheduler_type.__qualname__}"


def _schedule_from_scheduler(
    scheduler: object,
    *,
    label: str,
    expected_steps: int | None,
    scheduler_identity: str | None = None,
) -> _FrozenFlowSchedule:
    import torch

    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(timesteps, torch.Tensor):
        raise TypeError(f"{label} scheduler.timesteps must be a torch.Tensor")
    if not isinstance(sigmas, torch.Tensor):
        try:
            sigmas = torch.as_tensor(sigmas)
        except (TypeError, ValueError, RuntimeError):
            raise TypeError(
                f"{label} scheduler.sigmas must be tensor-convertible"
            ) from None
    if expected_steps is not None:
        if type(expected_steps) is not int or expected_steps < 1:
            raise ValueError("expected_steps must be a positive integer")
        if timesteps.numel() != expected_steps:
            raise DynamicsContractError(
                f"{label} scheduler must expose {expected_steps} timesteps"
            )
    return _FrozenFlowSchedule.create(
        timesteps.reshape(-1),
        sigmas.reshape(-1),
        label=label,
        scheduler_identity=(
            _scheduler_identity(scheduler)
            if scheduler_identity is None
            else scheduler_identity
        ),
    )


def _schedule_payload(
    schedule: _FrozenFlowSchedule,
    *,
    batch_size: int,
    device: Any,
    timesteps_key: str,
    sigmas_key: str,
) -> dict[str, Any]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    timesteps = schedule._timesteps.to(device=device)
    sigmas = schedule._sigmas.to(device=device)
    return {
        timesteps_key: timesteps[None, :].expand(batch_size, -1).clone(),
        sigmas_key: sigmas[None, :].expand(batch_size, -1).clone(),
    }


def _schedule_from_payload(
    payload: Mapping[str, Any],
    *,
    batch_size: int,
    label: str,
    timesteps_key: str,
    sigmas_key: str,
    scheduler_identity: str,
) -> _FrozenFlowSchedule:
    import torch

    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} recompute payload must be a mapping")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    has_timesteps = timesteps_key in payload
    has_sigmas = sigmas_key in payload
    if has_timesteps != has_sigmas or not has_timesteps:
        raise DynamicsContractError(
            f"{label} recompute payload has an incomplete scheduler state"
        )
    timesteps = payload[timesteps_key]
    sigmas = payload[sigmas_key]
    if (
        not isinstance(timesteps, torch.Tensor)
        or not isinstance(sigmas, torch.Tensor)
        or timesteps.ndim != 2
        or sigmas.ndim != 2
        or timesteps.shape[0] != batch_size
        or sigmas.shape[0] != batch_size
        or timesteps.shape[1] < 1
        or sigmas.shape[1] != timesteps.shape[1] + 1
    ):
        raise DynamicsContractError(
            f"{label} recompute scheduler state has invalid shape"
        )
    if not torch.equal(
        timesteps, timesteps[:1].expand_as(timesteps)
    ) or not torch.equal(
        sigmas,
        sigmas[:1].expand_as(sigmas),
    ):
        raise DynamicsContractError(
            f"{label} recompute rows must share one scheduler state"
        )
    return _FrozenFlowSchedule.create(
        timesteps[0],
        sigmas[0],
        label=label,
        scheduler_identity=scheduler_identity,
    )


def _validate_num_steps(
    schedule: _FrozenFlowSchedule,
    *,
    num_steps: int,
    label: str,
) -> None:
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if num_steps != schedule.num_steps:
        raise DynamicsContractError(
            f"{label} replay state has {schedule.num_steps} steps, not {num_steps}"
        )


def _transition_sigma_pair(
    schedule: _FrozenFlowSchedule,
    transition: Any,
    *,
    label: str,
) -> tuple[Any, Any, Any]:
    """Resolve and validate the explicit current/next schedule position."""

    import torch

    if transition.t.dtype != schedule._timesteps.dtype:
        raise DynamicsContractError(
            f"{label} transition timesteps must preserve replay-state dtype"
        )
    indices = transition.transition_index
    if bool((indices >= schedule.num_steps).any()):
        raise DynamicsContractError(f"{label} transition_index is outside the schedule")
    schedule_timesteps = schedule._timesteps.to(device=transition.x_t.device)
    expected_t = schedule_timesteps.index_select(0, indices)
    if not torch.equal(transition.t, expected_t):
        raise DynamicsContractError(
            f"{label} t does not match transition_index in replay state"
        )
    next_indices = torch.clamp(indices + 1, max=schedule.num_steps - 1)
    expected_next = schedule_timesteps.index_select(0, next_indices)
    is_terminal = indices == schedule.num_steps - 1
    terminal = schedule._terminal_timestep.to(device=transition.x_t.device)
    expected_next = torch.where(
        is_terminal, terminal.expand_as(expected_next), expected_next
    )
    if not torch.equal(transition.t_next, expected_next):
        raise DynamicsContractError(
            f"{label} t_next does not match transition_index in replay state"
        )

    sigmas = schedule._sigmas.to(
        device=transition.x_t.device,
        dtype=transition.x_t.dtype,
    )
    sigma = sigmas.index_select(0, indices)
    sigma_next = sigmas.index_select(0, indices + 1)
    dt = sigma_next - sigma
    if not bool(torch.isfinite(sigma).all()) or not bool(
        torch.isfinite(sigma_next).all()
    ):
        raise DynamicsContractError(f"{label} transition sigmas must be finite")
    if bool((sigma <= 0).any()):
        raise DynamicsContractError(f"{label} current sigma must be strictly positive")
    if bool((dt >= 0).any()):
        raise DynamicsContractError(f"{label} transition dt must be negative")
    return sigma, sigma_next, dt


def _add_flow_noise(
    schedule: _FrozenFlowSchedule,
    clean: Any,
    noise: Any,
    timestep: Any,
    *,
    label: str,
) -> Any:
    import torch

    if not isinstance(clean, torch.Tensor) or not isinstance(noise, torch.Tensor):
        raise TypeError(f"{label} clean and noise must be torch.Tensor values")
    if tuple(clean.shape) != tuple(noise.shape) or clean.ndim < 2:
        raise DynamicsContractError(f"{label} clean and noise must share shape [B,...]")
    if clean.device != noise.device or clean.dtype != noise.dtype:
        raise DynamicsContractError(f"{label} clean/noise dtype and device must match")
    if not clean.is_floating_point():
        raise TypeError(f"{label} clean/noise must be floating point")
    if not isinstance(timestep, torch.Tensor):
        raise TypeError(f"{label} timestep must be a torch.Tensor")
    if timestep.device != clean.device:
        raise DynamicsContractError(f"{label} timestep must be on the latent device")
    if timestep.dtype != schedule._timesteps.dtype:
        raise DynamicsContractError(
            f"{label} timestep must preserve replay-state dtype"
        )
    values = timestep.reshape(-1)
    if values.numel() == 1:
        values = values.expand(clean.shape[0])
    if tuple(values.shape) != (clean.shape[0],):
        raise DynamicsContractError(f"{label} timestep must be scalar or [B]")
    candidates = schedule._timesteps.to(device=clean.device)
    matches = values[:, None] == candidates[None, :]
    counts = matches.sum(dim=1)
    if not bool((counts == 1).all()):
        raise DynamicsContractError(f"{label} timestep is absent from replay state")
    indices = matches.to(dtype=torch.int64).argmax(dim=1)
    sigma = schedule._sigmas.to(device=clean.device, dtype=clean.dtype).index_select(
        0,
        indices,
    )
    shape = (clean.shape[0], *([1] * (clean.ndim - 1)))
    sigma = sigma.reshape(shape)
    return (1 - sigma) * clean + sigma * noise


@dataclass(frozen=True)
class _SchedulerConfig:
    stochastic_sampling: bool


class _SchedulerView:
    """Minimal scheduler protocol consumed by the frozen legacy kernels."""

    def __init__(
        self,
        schedule: _FrozenFlowSchedule,
        *,
        device: Any,
        stochastic_sampling: bool,
    ) -> None:
        self.timesteps = schedule._timesteps.to(device=device)
        self.sigmas = schedule._sigmas
        self.config = _SchedulerConfig(stochastic_sampling=stochastic_sampling)

    def index_for_timestep(self, timestep: Any) -> int:
        import torch

        value = torch.as_tensor(timestep, device=self.timesteps.device)
        if value.numel() != 1 or value.dtype != self.timesteps.dtype:
            raise DynamicsContractError(
                "scheduler lookup must preserve one exact timestep value"
            )
        matches = (self.timesteps == value.reshape(())).nonzero().reshape(-1)
        if matches.numel() != 1:
            raise DynamicsContractError("scheduler timestep lookup is not unique")
        return int(matches.item())


__all__ = [
    "_FrozenFlowSchedule",
    "_SchedulerView",
    "_add_flow_noise",
    "_schedule_from_payload",
    "_schedule_from_scheduler",
    "_schedule_payload",
    "_transition_sigma_pair",
    "_validate_num_steps",
]
