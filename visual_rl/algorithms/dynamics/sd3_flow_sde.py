"""SD3 flow-SDE Dynamics wrapper with typed scheduler replay state."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from visual_rl.algorithms.dynamics.config import FlowSDEConfig
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
    scheduler_dynamic_shift_config,
)
from visual_rl.algorithms.dynamics.transition import (
    sd3_sde_step_with_logprob,
)
from visual_rl.core.contracts import LatentLayout
from visual_rl.models.scheduler import ModelScheduleContext, SchedulerArtifactBlueprint

SD3_SCHEDULER_TIMESTEPS_KEY = "sd3_scheduler_timesteps"
SD3_SCHEDULER_SIGMAS_KEY = "sd3_scheduler_sigmas"


@dataclass(frozen=True, init=False)
class SD3ScheduleReplayState:
    """Immutable SD3 schedule sufficient for exact policy recomputation."""

    _schedule: _FrozenFlowSchedule = field(repr=False)

    def __init__(
        self,
        timesteps: Any,
        sigmas: Any,
        *,
        scheduler_identity: str = "manual.sd3-flow-sde",
        terminal_timestep: Any = None,
        schema_version: int = 1,
    ) -> None:
        schedule = _FrozenFlowSchedule.create(
            timesteps,
            sigmas,
            label="SD3",
            scheduler_identity=scheduler_identity,
            terminal_timestep=terminal_timestep,
            schema_version=schema_version,
        )
        object.__setattr__(self, "_schedule", schedule)

    @classmethod
    def _from_frozen(
        cls,
        schedule: _FrozenFlowSchedule,
    ) -> SD3ScheduleReplayState:
        result = object.__new__(cls)
        object.__setattr__(result, "_schedule", schedule)
        return result

    @classmethod
    def from_scheduler(
        cls,
        scheduler: object,
        *,
        expected_steps: int | None = None,
        scheduler_identity: str | None = None,
    ) -> SD3ScheduleReplayState:
        """Capture an owned schedule without normalizing timestep dtype."""

        return cls._from_frozen(
            _schedule_from_scheduler(
                scheduler,
                label="SD3",
                expected_steps=expected_steps,
                scheduler_identity=scheduler_identity,
            )
        )

    @classmethod
    def from_recompute_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        batch_size: int,
        scheduler_identity: str = "recompute-payload.sd3-flow-sde",
    ) -> SD3ScheduleReplayState:
        """Restore the same row-shared state used by the legacy SD3 adapter."""

        return cls._from_frozen(
            _schedule_from_payload(
                payload,
                batch_size=batch_size,
                label="SD3",
                timesteps_key=SD3_SCHEDULER_TIMESTEPS_KEY,
                sigmas_key=SD3_SCHEDULER_SIGMAS_KEY,
                scheduler_identity=scheduler_identity,
            )
        )

    def to_recompute_payload(
        self,
        *,
        batch_size: int,
        device: Any,
    ) -> dict[str, Any]:
        """Emit the legacy-compatible two-tensor SD3 scheduler payload."""

        return _schedule_payload(
            self._schedule,
            batch_size=batch_size,
            device=device,
            timesteps_key=SD3_SCHEDULER_TIMESTEPS_KEY,
            sigmas_key=SD3_SCHEDULER_SIGMAS_KEY,
        )

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
        return self._schedule.replay_state_identity


@dataclass(frozen=True, slots=True)
class SD3DynamicsReplayStateFactory:
    """Build a fresh SD3 schedule state from a frozen scheduler blueprint."""

    scheduler_blueprint: SchedulerArtifactBlueprint
    factory_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler_blueprint, SchedulerArtifactBlueprint):
            raise TypeError("scheduler_blueprint must be SchedulerArtifactBlueprint")
        digest = hashlib.sha256(
            (
                "sd3-replay-state-factory.v1:"
                + self.scheduler_blueprint.blueprint_identity
            ).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "factory_identity", f"sd3-replay.v1:{digest}")

    @classmethod
    def from_scheduler(cls, scheduler: object) -> SD3DynamicsReplayStateFactory:
        return cls(SchedulerArtifactBlueprint.from_scheduler(scheduler))

    @property
    def replay_state_type(self) -> type[SD3ScheduleReplayState]:
        return SD3ScheduleReplayState

    @property
    def requires_schedule_conditioning(self) -> bool:
        return scheduler_dynamic_shift_config(self.scheduler_blueprint) is not None

    def schedule_conditioning(
        self,
        *,
        latent_height: int,
        latent_width: int,
        patch_size: int,
    ) -> FlowMatchScheduleConditioning:
        """Bind SD3 latent geometry to the frozen scheduler capability."""

        return FlowMatchScheduleConditioning.from_latent_geometry(
            latent_height=latent_height,
            latent_width=latent_width,
            patch_size=patch_size,
            dynamic_shift=scheduler_dynamic_shift_config(self.scheduler_blueprint),
        )

    def create(
        self,
        request: DynamicsReplayRequest,
    ) -> DynamicsReplayBinding[SD3ScheduleReplayState]:
        if not isinstance(request, DynamicsReplayRequest):
            raise TypeError("request must be a DynamicsReplayRequest")
        scheduler = materialize_scheduler_from_blueprint(
            self.scheduler_blueprint,
            request,
        )
        state = SD3ScheduleReplayState.from_scheduler(
            scheduler,
            expected_steps=request.num_steps,
            scheduler_identity=self.scheduler_blueprint.scheduler_identity,
        )
        return DynamicsReplayBinding(
            request=request,
            factory_identity=self.factory_identity,
            replay_state=state,
        )


class SD3FlowSDEDynamics(Dynamics):
    """Revision-pinned SD3 SDE exposed through the common Dynamics template."""

    def __init__(
        self,
        replay_state: SD3ScheduleReplayState,
        *,
        config: FlowSDEConfig | None = None,
        replay_binding: DynamicsReplayBinding[SD3ScheduleReplayState] | None = None,
    ) -> None:
        if not isinstance(replay_state, SD3ScheduleReplayState):
            raise TypeError("replay_state must be SD3ScheduleReplayState")
        resolved_config = FlowSDEConfig() if config is None else config
        if not isinstance(resolved_config, FlowSDEConfig):
            raise TypeError("config must be a FlowSDEConfig")
        if replay_binding is not None:
            if not isinstance(replay_binding, DynamicsReplayBinding):
                raise TypeError("replay_binding must be a DynamicsReplayBinding")
            if replay_binding.replay_state is not replay_state:
                raise DynamicsContractError(
                    "SD3 replay binding must own the bound replay state"
                )
        self.replay_state = replay_state
        self.replay_binding = replay_binding
        self.config = resolved_config
        self.noise_level = resolved_config.noise_level

    @property
    def dynamics_config_identity(self) -> str:
        base = f"sd3-flow-sde.v1:noise_level={self.noise_level!r}"
        if self.replay_binding is None:
            return base
        conditioning = self.replay_binding.request.schedule_conditioning
        if conditioning is None:
            return base
        return f"{base}:schedule_conditioning={conditioning.conditioning_identity}"

    @property
    def scheduler_identity(self) -> str:
        return self.replay_state.scheduler_identity

    def schedule_sigmas(self, *, num_steps: int, device: Any) -> Any:
        _validate_num_steps(
            self.replay_state._schedule,
            num_steps=num_steps,
            label="SD3",
        )
        return self.replay_state._schedule._sigmas.to(device=device).clone()

    def timesteps(self, *, num_steps: int, device: Any) -> Any:
        _validate_num_steps(
            self.replay_state._schedule,
            num_steps=num_steps,
            label="SD3",
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
            label="SD3",
        )

    def transition_mean_std(
        self,
        transition: TransitionInput,
    ) -> TransitionMeanStd:
        """Delegate the frozen SD3 equation and expose its authoritative stats."""

        import torch

        transition.validate()
        if transition.x_t.dtype != torch.float32:
            raise DynamicsContractError(
                "SD3 parity Dynamics requires FP32 transition latents"
            )
        sigma, _sigma_next, dt = _transition_sigma_pair(
            self.replay_state._schedule,
            transition,
            label="SD3",
        )
        all_sigmas = self.replay_state._schedule._sigmas.to(
            device=transition.x_t.device,
            dtype=transition.x_t.dtype,
        )
        if all_sigmas.numel() < 2:
            raise DynamicsContractError("SD3 scheduler must expose at least two sigmas")
        sigma_max = all_sigmas[1]
        denominator = 1 - torch.where(sigma == 1, sigma_max, sigma)
        if not bool(torch.isfinite(denominator).all()) or bool(
            (denominator <= 0).any()
        ):
            raise DynamicsContractError(
                "SD3 diffusion denominator must be finite and positive"
            )

        scheduler = _SchedulerView(
            self.replay_state._schedule,
            device=transition.x_t.device,
            stochastic_sampling=True,
        )
        _unused_sample, _unused_log_prob, mean, std = sd3_sde_step_with_logprob(
            scheduler,
            transition.model_prediction,
            transition.t,
            transition.x_t,
            prev_sample=transition.x_t,
            noise_level=self.noise_level,
        )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise DynamicsContractError("SD3 transition mean/std must be finite")
        if bool((std <= 0).any()) or bool((std.square() == 0).any()):
            raise DynamicsContractError(
                "SD3 stochastic transition std must be strictly positive "
                "with representable variance"
            )
        return TransitionMeanStd(mean=mean, std=std, dt=dt)

    def _deterministic_ode_step(
        self,
        transition: TransitionInput,
    ) -> DeterministicTransitionOutput:
        """Match the upstream TempFlow deterministic Euler flow update."""

        transition.validate()
        _sigma, _sigma_next, dt = _transition_sigma_pair(
            self.replay_state._schedule,
            transition,
            label="SD3",
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
        """Expose the scalar SDE standard deviation used by TempFlow credit."""

        stats.validate_against(transition)
        flat = stats.std.detach().reshape(transition.batch_size, -1)
        first = flat[:, 0]
        if not bool((flat == first[:, None]).all()):
            raise DynamicsContractError(
                "SD3 transition std must be row-scalar for TempFlow credit"
            )
        return TransitionPolicyMetadata(transition_std_dev=first.clone())


class RegisteredSD3FlowSDE(DynamicsInstanceFactory[SD3ScheduleReplayState]):
    """Runtime SD3 factory bound from the same config used by declaration."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.dynamics.config:FlowSDEConfig"

    def __init__(self, config: FlowSDEConfig) -> None:
        if not isinstance(config, FlowSDEConfig):
            raise TypeError("config must be FlowSDEConfig")
        self.config = config

    @classmethod
    def describe(cls, config: object) -> object:
        if not isinstance(config, FlowSDEConfig):
            raise TypeError("config must be FlowSDEConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RegisteredSD3FlowSDE:
        if not isinstance(config, FlowSDEConfig):
            raise TypeError("config must be FlowSDEConfig")
        if not isinstance(runtime_context, Mapping):
            raise TypeError("runtime_context must be a mapping")
        return cls(config)

    @property
    def replay_state_type(self) -> type[SD3ScheduleReplayState]:
        return SD3ScheduleReplayState

    @property
    def dynamics_binding_family(self) -> str:
        return "sd3.flow-sde.v1"

    @property
    def replay_state_schema_id(self) -> str:
        return "sd3.schedule-replay.v1"

    def bind_replay_state_factory(
        self,
        blueprint: SchedulerArtifactBlueprint,
    ) -> DynamicsReplayStateFactory[SD3ScheduleReplayState]:
        if not isinstance(blueprint, SchedulerArtifactBlueprint):
            raise TypeError("blueprint must be a SchedulerArtifactBlueprint")
        return SD3DynamicsReplayStateFactory(blueprint)

    def schedule_conditioning(
        self,
        blueprint: SchedulerArtifactBlueprint,
        context: ModelScheduleContext,
    ) -> FlowMatchScheduleConditioning | None:
        if not isinstance(blueprint, SchedulerArtifactBlueprint):
            raise TypeError("blueprint must be a SchedulerArtifactBlueprint")
        _validate_schedule_context(context)
        dynamic_shift = scheduler_dynamic_shift_config(blueprint)
        if dynamic_shift is None:
            return None
        patch_size = context.scheduler_patch_size
        if type(patch_size) is not int or patch_size < 1:
            raise TypeError("SD3 dynamic-shift binding requires scheduler_patch_size")
        return FlowMatchScheduleConditioning.from_latent_geometry(
            latent_height=context.shape[2],
            latent_width=context.shape[3],
            patch_size=patch_size,
            dynamic_shift=dynamic_shift,
        )

    def create(
        self,
        binding: DynamicsReplayBinding[SD3ScheduleReplayState],
    ) -> SD3FlowSDEDynamics:
        if not isinstance(binding, DynamicsReplayBinding):
            raise TypeError("binding must be a DynamicsReplayBinding")
        if type(binding.replay_state) is not SD3ScheduleReplayState:
            raise TypeError("flow-sde requires SD3ScheduleReplayState")
        return SD3FlowSDEDynamics(
            binding.replay_state,
            config=self.config,
            replay_binding=binding,
        )


def _validate_schedule_context(context: ModelScheduleContext) -> None:
    if not isinstance(context, ModelScheduleContext):
        raise TypeError("context must implement ModelScheduleContext")
    if context.layout is not LatentLayout.BCHW:
        raise TypeError("SD3 flow-sde requires bchw model schedule context")
    expected_axes = ("batch", "channel", "height", "width")
    if context.axis_semantics != expected_axes or len(context.shape) != 4:
        raise TypeError("SD3 flow-sde received incompatible latent axis semantics")


__all__ = [
    "SD3_SCHEDULER_SIGMAS_KEY",
    "SD3_SCHEDULER_TIMESTEPS_KEY",
    "RegisteredSD3FlowSDE",
    "SD3DynamicsReplayStateFactory",
    "SD3FlowSDEDynamics",
    "SD3ScheduleReplayState",
]
