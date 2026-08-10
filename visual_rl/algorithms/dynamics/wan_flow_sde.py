"""Wan Flash/World-R1 flow-SDE Dynamics with typed schedule replay."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from visual_rl.algorithms.dynamics.config import (
    WanFlowSDEConfig,
    WanFlowSDEProfile,
)
from visual_rl.algorithms.dynamics.flow_schedule import (
    _add_flow_noise,
    _FrozenFlowSchedule,
    _schedule_from_payload,
    _schedule_from_scheduler,
    _schedule_payload,
    _SchedulerView,
    _transition_sigma_pair,
    _validate_num_steps,
)
from visual_rl.algorithms.dynamics.interface import (
    DeterministicTransitionOutput,
    Dynamics,
    DynamicsContractError,
    TransitionInput,
    TransitionMeanStd,
    TransitionPolicyMetadata,
)
from visual_rl.algorithms.dynamics.replay import (
    DynamicsInstanceFactory,
    DynamicsReplayBinding,
    DynamicsReplayRequest,
    DynamicsReplayStateFactory,
    FlowMatchScheduleConditioning,
    materialize_scheduler_from_blueprint,
)
from visual_rl.algorithms.dynamics.transition import (
    wan_sde_step_with_logprob,
)
from visual_rl.core.contracts import LatentLayout
from visual_rl.models.scheduler import ModelScheduleContext, SchedulerArtifactBlueprint

WAN_SCHEDULER_TIMESTEPS_KEY = "wan_scheduler_timesteps"
WAN_SCHEDULER_SIGMAS_KEY = "wan_scheduler_sigmas"
WAN_STOCHASTIC_SAMPLING_KEY = "wan_scheduler_stochastic_sampling"


@dataclass(frozen=True, init=False)
class WanScheduleReplayState:
    """Immutable Wan schedule plus the sampling mode that affects its mean."""

    _schedule: _FrozenFlowSchedule = field(repr=False)
    stochastic_sampling: bool

    def __init__(
        self,
        timesteps: Any,
        sigmas: Any,
        *,
        stochastic_sampling: bool,
        scheduler_identity: str = "manual.wan-flow-sde",
        terminal_timestep: Any = None,
        schema_version: int = 1,
    ) -> None:
        if type(stochastic_sampling) is not bool:
            raise TypeError("Wan stochastic_sampling must be bool")
        schedule = _FrozenFlowSchedule.create(
            timesteps,
            sigmas,
            label="Wan",
            scheduler_identity=scheduler_identity,
            terminal_timestep=terminal_timestep,
            schema_version=schema_version,
        )
        object.__setattr__(self, "_schedule", schedule)
        object.__setattr__(self, "stochastic_sampling", stochastic_sampling)

    @classmethod
    def _from_frozen(
        cls,
        schedule: _FrozenFlowSchedule,
        *,
        stochastic_sampling: bool,
    ) -> WanScheduleReplayState:
        result = object.__new__(cls)
        object.__setattr__(result, "_schedule", schedule)
        object.__setattr__(result, "stochastic_sampling", stochastic_sampling)
        return result

    @classmethod
    def from_scheduler(
        cls,
        scheduler: object,
        *,
        expected_steps: int | None = None,
        scheduler_identity: str | None = None,
        stochastic_sampling: bool | None = None,
    ) -> WanScheduleReplayState:
        """Capture schedule values while allowing Dynamics-owned sampling mode."""

        schedule = _schedule_from_scheduler(
            scheduler,
            label="Wan",
            expected_steps=expected_steps,
            scheduler_identity=scheduler_identity,
        )
        if stochastic_sampling is None:
            config = getattr(scheduler, "config", object())
            stochastic_sampling = bool(
                getattr(config, "stochastic_sampling", False)
            )
        elif type(stochastic_sampling) is not bool:
            raise TypeError("Wan stochastic_sampling override must be bool")
        return cls._from_frozen(
            schedule,
            stochastic_sampling=stochastic_sampling,
        )

    @classmethod
    def from_recompute_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        batch_size: int,
        scheduler_identity: str = "recompute-payload.wan-flow-sde",
    ) -> WanScheduleReplayState:
        import torch

        schedule = _schedule_from_payload(
            payload,
            batch_size=batch_size,
            label="Wan",
            timesteps_key=WAN_SCHEDULER_TIMESTEPS_KEY,
            sigmas_key=WAN_SCHEDULER_SIGMAS_KEY,
            scheduler_identity=scheduler_identity,
        )
        if WAN_STOCHASTIC_SAMPLING_KEY not in payload:
            raise DynamicsContractError(
                "Wan recompute payload has an incomplete scheduler state"
            )
        values = payload[WAN_STOCHASTIC_SAMPLING_KEY]
        if (
            not isinstance(values, torch.Tensor)
            or values.dtype != torch.bool
            or tuple(values.shape) != (batch_size,)
        ):
            raise DynamicsContractError(
                "Wan stochastic sampling replay state must be bool [B]"
            )
        if not torch.equal(values, values[:1].expand_as(values)):
            raise DynamicsContractError(
                "Wan recompute rows must share one scheduler state"
            )
        return cls._from_frozen(
            schedule,
            stochastic_sampling=bool(values[0].item()),
        )

    def to_recompute_payload(
        self,
        *,
        batch_size: int,
        device: Any,
    ) -> dict[str, Any]:
        import torch

        payload = _schedule_payload(
            self._schedule,
            batch_size=batch_size,
            device=device,
            timesteps_key=WAN_SCHEDULER_TIMESTEPS_KEY,
            sigmas_key=WAN_SCHEDULER_SIGMAS_KEY,
        )
        payload[WAN_STOCHASTIC_SAMPLING_KEY] = torch.full(
            (batch_size,),
            self.stochastic_sampling,
            dtype=torch.bool,
            device=device,
        )
        return payload

    @property
    def timesteps(self) -> Any:
        return self._schedule.timesteps

    @property
    def sigmas(self) -> Any:
        return self._schedule.sigmas

    @property
    def terminal_timestep(self) -> Any:
        return self._schedule.terminal_timestep

    @property
    def scheduler_identity(self) -> str:
        return self._schedule.scheduler_identity

    @property
    def schema_version(self) -> int:
        return self._schedule.schema_version

    @property
    def num_steps(self) -> int:
        return self._schedule.num_steps

    @property
    def replay_state_identity(self) -> str:
        digest = hashlib.sha256(
            (
                self._schedule.replay_state_identity
                + f":stochastic_sampling={self.stochastic_sampling}"
            ).encode("ascii")
        ).hexdigest()
        return digest


@dataclass(frozen=True, slots=True)
class WanDynamicsReplayStateFactory:
    """Build a fresh Wan schedule state from a frozen scheduler blueprint."""

    scheduler_blueprint: SchedulerArtifactBlueprint
    stochastic_sampling: bool = True
    factory_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler_blueprint, SchedulerArtifactBlueprint):
            raise TypeError("scheduler_blueprint must be SchedulerArtifactBlueprint")
        if type(self.stochastic_sampling) is not bool:
            raise TypeError("stochastic_sampling must be bool")
        digest = hashlib.sha256(
            (
                "wan-replay-state-factory.v1:"
                + self.scheduler_blueprint.blueprint_identity
                + f":stochastic_sampling={self.stochastic_sampling}"
            ).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "factory_identity", f"wan-replay.v1:{digest}")

    @classmethod
    def from_scheduler(
        cls,
        scheduler: object,
        *,
        stochastic_sampling: bool = True,
    ) -> WanDynamicsReplayStateFactory:
        return cls(
            SchedulerArtifactBlueprint.from_scheduler(scheduler),
            stochastic_sampling=stochastic_sampling,
        )

    @property
    def replay_state_type(self) -> type[WanScheduleReplayState]:
        return WanScheduleReplayState

    def create(
        self,
        request: DynamicsReplayRequest,
    ) -> DynamicsReplayBinding[WanScheduleReplayState]:
        if not isinstance(request, DynamicsReplayRequest):
            raise TypeError("request must be a DynamicsReplayRequest")
        scheduler = materialize_scheduler_from_blueprint(
            self.scheduler_blueprint,
            request,
        )
        state = WanScheduleReplayState.from_scheduler(
            scheduler,
            expected_steps=request.num_steps,
            scheduler_identity=self.scheduler_blueprint.scheduler_identity,
            stochastic_sampling=self.stochastic_sampling,
        )
        return DynamicsReplayBinding(
            request=request,
            factory_identity=self.factory_identity,
            replay_state=state,
        )


class WanFlowSDEDynamics(Dynamics):
    """Frozen Wan transition behind one explicitly declared typed profile."""

    def __init__(
        self,
        replay_state: WanScheduleReplayState,
        *,
        config: WanFlowSDEConfig,
        replay_binding: DynamicsReplayBinding[WanScheduleReplayState] | None = None,
    ) -> None:
        if not isinstance(replay_state, WanScheduleReplayState):
            raise TypeError("replay_state must be WanScheduleReplayState")
        if not isinstance(config, WanFlowSDEConfig):
            raise TypeError("config must be a WanFlowSDEConfig")
        if replay_state.stochastic_sampling is not config.stochastic_sampling:
            raise DynamicsContractError(
                "Wan replay sampling mode does not match Dynamics config"
            )
        if replay_binding is not None:
            if not isinstance(replay_binding, DynamicsReplayBinding):
                raise TypeError("replay_binding must be a DynamicsReplayBinding")
            if replay_binding.replay_state is not replay_state:
                raise DynamicsContractError(
                    "Wan replay binding must own the bound replay state"
                )
        self.replay_state = replay_state
        self.replay_binding = replay_binding
        self.config = config
        self.profile = config.profile

    @property
    def dynamics_config_identity(self) -> str:
        return (
            f"wan-flow-sde.v2:profile={self.profile.value}:"
            f"likelihood={self.config.likelihood_semantics.value}:"
            f"replay={self.config.replay_target.value}:"
            f"stochastic_sampling={self.config.stochastic_sampling}"
        )

    @property
    def scheduler_identity(self) -> str:
        return self.replay_state.scheduler_identity

    def schedule_sigmas(self, *, num_steps: int, device: Any) -> Any:
        _validate_num_steps(
            self.replay_state._schedule,
            num_steps=num_steps,
            label="Wan",
        )
        return self.replay_state._schedule._sigmas.to(device=device).clone()

    @property
    def scheduler_is_deterministic(self) -> bool:
        return (
            self.profile is not WanFlowSDEProfile.FLASH
            and not self.replay_state.stochastic_sampling
        )

    def timesteps(self, *, num_steps: int, device: Any) -> Any:
        _validate_num_steps(
            self.replay_state._schedule,
            num_steps=num_steps,
            label="Wan",
        )
        return self.replay_state._schedule._timesteps.to(device=device).clone()

    def terminal_timestep(self, *, device: Any) -> Any:
        return self.replay_state.terminal_timestep.to(device=device)

    def add_noise(self, clean: Any, noise: Any, timestep: Any) -> Any:
        return _add_flow_noise(
            self.replay_state._schedule,
            clean,
            noise,
            timestep,
            label="Wan",
        )

    def transition_mean_std(
        self,
        transition: TransitionInput,
    ) -> TransitionMeanStd:
        """Delegate the frozen Wan equation and expose its authoritative stats."""

        import torch

        transition.validate()
        if transition.x_t.dtype != torch.float32:
            raise DynamicsContractError(
                "Wan parity Dynamics requires FP32 transition latents"
            )
        sigma, _sigma_next, dt = _transition_sigma_pair(
            self.replay_state._schedule,
            transition,
            label="Wan",
        )
        sigmas = self.replay_state._schedule._sigmas.to(
            device=transition.x_t.device,
            dtype=transition.x_t.dtype,
        )
        sigma_max_index = 1 if self.profile is WanFlowSDEProfile.FLASH else 0
        if sigmas.numel() <= sigma_max_index:
            raise DynamicsContractError("Wan scheduler does not expose enough sigmas")
        sigma_max = sigmas[sigma_max_index]
        sigma_min = sigmas[-1]
        diffusion_scale = sigma_min + (sigma_max - sigma_min) * sigma
        if not bool(torch.isfinite(diffusion_scale).all()) or bool(
            (diffusion_scale <= 0).any()
        ):
            raise DynamicsContractError(
                "Wan diffusion scale must be finite and strictly positive"
            )

        scheduler = _SchedulerView(
            self.replay_state._schedule,
            device=transition.x_t.device,
            stochastic_sampling=self.replay_state.stochastic_sampling,
        )
        _unused_sample, _unused_log_prob, mean, std = wan_sde_step_with_logprob(
            scheduler,
            transition.model_prediction,
            transition.t,
            transition.x_t,
            profile=self.profile,
            prev_sample=transition.x_t,
        )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise DynamicsContractError("Wan transition mean/std must be finite")
        if bool((std <= 0).any()) or bool((std.square() == 0).any()):
            raise DynamicsContractError(
                "Wan stochastic transition std must be strictly positive "
                "with representable variance"
            )
        return TransitionMeanStd(mean=mean, std=std, dt=dt)

    def _deterministic_ode_step(
        self,
        transition: TransitionInput,
    ) -> DeterministicTransitionOutput:
        """Advance one deterministic Euler flow step on the frozen Wan schedule."""

        transition.validate()
        _sigma, _sigma_next, dt = _transition_sigma_pair(
            self.replay_state._schedule,
            transition,
            label="Wan",
        )
        broadcast = (transition.batch_size, *([1] * (transition.x_t.ndim - 1)))
        next_state = (
            transition.x_t + dt.reshape(broadcast) * transition.model_prediction
        )
        return DeterministicTransitionOutput(
            next_state=next_state.detach(),
            dt=dt.detach(),
        )

    def policy_metadata(
        self,
        transition: TransitionInput,
        stats: TransitionMeanStd,
    ) -> TransitionPolicyMetadata:
        """Expose Flash rectification without leaking its formula into Trainer."""

        import torch

        stats.validate_against(transition)
        if self.profile is not WanFlowSDEProfile.FLASH:
            return TransitionPolicyMetadata()
        sigma, _sigma_next, _dt = _transition_sigma_pair(
            self.replay_state._schedule,
            transition,
            label="Wan",
        )
        flat_std = stats.std.detach().reshape(transition.batch_size, -1)
        transition_std = flat_std[:, 0]
        if not bool((flat_std == transition_std[:, None]).all()):
            raise DynamicsContractError(
                "Wan transition std must be row-scalar for Flash credit"
            )
        sigmas = self.replay_state._schedule._sigmas.to(
            device=transition.x_t.device,
            dtype=transition.x_t.dtype,
        )
        if sigmas.numel() < 2:
            raise DynamicsContractError(
                "Wan Flash scheduler must expose at least two sigmas"
            )
        sigma_max = sigmas[1]
        sigma_min = sigmas[-1]
        diffusion_scale = sigma_min + (sigma_max - sigma_min) * sigma
        sqrt_negative_dt = torch.sqrt(-stats.dt.detach())
        sigma = sigma.detach().reshape(transition.batch_size, -1)[:, 0]
        diffusion_scale = diffusion_scale.detach().reshape(
            transition.batch_size,
            -1,
        )[:, 0]
        coefficient = 1.0 / (
            sqrt_negative_dt / diffusion_scale
            + diffusion_scale * sqrt_negative_dt * (1.0 - sigma) / (2.0 * sigma)
        )
        if not bool(torch.isfinite(coefficient).all()) or bool(
            (coefficient <= 0).any()
        ):
            raise DynamicsContractError(
                "Wan Flash rectification coefficient must be finite and positive"
            )
        return TransitionPolicyMetadata(rectification_coefficient=coefficient.detach())

    def _sample_from_mean_std(
        self,
        transition: TransitionInput,
        stats: TransitionMeanStd,
        *,
        generator: Any,
    ) -> Any:
        if self.scheduler_is_deterministic:
            return stats.mean
        return super()._sample_from_mean_std(
            transition,
            stats,
            generator=generator,
        )

    def _log_prob_epsilon(self) -> float:
        return 0.0 if self.profile is WanFlowSDEProfile.FLASH else 1.0e-12


class RegisteredWanFlowSDE(DynamicsInstanceFactory[WanScheduleReplayState]):
    """Runtime Wan factory consuming the declaration provider's exact config."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.dynamics.config:WanFlowSDEConfig"

    def __init__(self, config: WanFlowSDEConfig) -> None:
        if not isinstance(config, WanFlowSDEConfig):
            raise TypeError("config must be WanFlowSDEConfig")
        self.config = config

    @classmethod
    def describe(cls, config: object) -> object:
        if not isinstance(config, WanFlowSDEConfig):
            raise TypeError("config must be WanFlowSDEConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RegisteredWanFlowSDE:
        if not isinstance(config, WanFlowSDEConfig):
            raise TypeError("config must be WanFlowSDEConfig")
        if not isinstance(runtime_context, Mapping):
            raise TypeError("runtime_context must be a mapping")
        return cls(config)

    @property
    def replay_state_type(self) -> type[WanScheduleReplayState]:
        return WanScheduleReplayState

    @property
    def dynamics_binding_family(self) -> str:
        return "wan.flow-sde.v1"

    @property
    def replay_state_schema_id(self) -> str:
        return "wan.schedule-replay.v1"

    def bind_replay_state_factory(
        self,
        blueprint: SchedulerArtifactBlueprint,
    ) -> DynamicsReplayStateFactory[WanScheduleReplayState]:
        if not isinstance(blueprint, SchedulerArtifactBlueprint):
            raise TypeError("blueprint must be a SchedulerArtifactBlueprint")
        return WanDynamicsReplayStateFactory(
            blueprint,
            stochastic_sampling=self.config.stochastic_sampling,
        )

    def schedule_conditioning(
        self,
        blueprint: SchedulerArtifactBlueprint,
        context: ModelScheduleContext,
    ) -> FlowMatchScheduleConditioning | None:
        if not isinstance(blueprint, SchedulerArtifactBlueprint):
            raise TypeError("blueprint must be a SchedulerArtifactBlueprint")
        _validate_schedule_context(context)
        return None

    def create(
        self,
        binding: DynamicsReplayBinding[WanScheduleReplayState],
    ) -> WanFlowSDEDynamics:
        if not isinstance(binding, DynamicsReplayBinding):
            raise TypeError("binding must be a DynamicsReplayBinding")
        if type(binding.replay_state) is not WanScheduleReplayState:
            raise TypeError("wan-flow-sde requires WanScheduleReplayState")
        return WanFlowSDEDynamics(
            binding.replay_state,
            config=self.config,
            replay_binding=binding,
        )


def _validate_schedule_context(context: ModelScheduleContext) -> None:
    if not isinstance(context, ModelScheduleContext):
        raise TypeError("context must implement ModelScheduleContext")
    if context.layout is not LatentLayout.BCTHW:
        raise TypeError("Wan flow-sde requires bcthw model schedule context")
    expected_axes = ("batch", "channel", "time", "height", "width")
    if context.axis_semantics != expected_axes or len(context.shape) != 5:
        raise TypeError("Wan flow-sde received incompatible latent axis semantics")


__all__ = [
    "WAN_SCHEDULER_SIGMAS_KEY",
    "WAN_SCHEDULER_TIMESTEPS_KEY",
    "WAN_STOCHASTIC_SAMPLING_KEY",
    "RegisteredWanFlowSDE",
    "WanDynamicsReplayStateFactory",
    "WanFlowSDEDynamics",
    "WanScheduleReplayState",
]
