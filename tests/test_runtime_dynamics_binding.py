"""Focused runtime scheduler-artifact to Dynamics binding contracts."""

from __future__ import annotations

import pytest
import torch

from visual_rl.algorithms.dynamics.config import FlowSDEConfig
from visual_rl.algorithms.dynamics.replay import (
    DynamicsReplayRequest,
)
from visual_rl.algorithms.dynamics.sd3_flow_sde import (
    RegisteredSD3FlowSDE,
    SD3DynamicsReplayStateFactory,
)
from visual_rl.algorithms.dynamics.wan_flow_sde import (
    WanDynamicsReplayStateFactory,
)
from visual_rl.core.contracts import LatentLayout
from visual_rl.models import ModelLatentSpec, SchedulerArtifactBlueprint
from visual_rl.runtime import AlgorithmRuntimeBindingError, PerRolloutDynamicsFactory


class _Config(dict):
    def __getattr__(self, name: str):
        return self[name]


class _Scheduler:
    def __init__(self, config=None) -> None:
        values = {
            "stochastic_sampling": True,
            "use_dynamic_shifting": True,
            "base_image_seq_len": 4,
            "max_image_seq_len": 64,
            "base_shift": 0.5,
            "max_shift": 1.0,
        }
        values.update({} if config is None else config)
        self.config = _Config(values)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps, device, mu=None) -> None:
        if mu is None:
            raise AssertionError("dynamic scheduler requires bound mu")
        self.timesteps = torch.linspace(
            900.0,
            100.0,
            num_inference_steps,
            dtype=torch.float32,
            device=device,
        )
        self.sigmas = torch.linspace(
            1.0,
            0.1,
            num_inference_steps + 1,
            dtype=torch.float32,
            device=device,
        )


def _blueprint() -> SchedulerArtifactBlueprint:
    return SchedulerArtifactBlueprint.from_scheduler(_Scheduler())


def _sd3_component() -> RegisteredSD3FlowSDE:
    return RegisteredSD3FlowSDE.from_config(
        FlowSDEConfig(),
        runtime_context={},
    )


def _context() -> ModelLatentSpec:
    return ModelLatentSpec(
        shape=(2, 4, 8, 8),
        layout=LatentLayout.BCHW,
        axis_semantics=("batch", "channel", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(8, 8),
        scheduler_patch_size=2,
    )


def _runtime_factory() -> PerRolloutDynamicsFactory:
    component = _sd3_component()
    blueprint = _blueprint()
    replay_factory = component.bind_replay_state_factory(blueprint)
    return PerRolloutDynamicsFactory(
        component=component,
        scheduler_blueprint=blueprint,
        replay_state_factory=replay_factory,
        dynamics_binding_family=component.dynamics_binding_family,
        replay_state_schema_id=component.replay_state_schema_id,
    )


def test_runtime_binding_owns_conditioning_factory_and_evidence() -> None:
    factory = _runtime_factory()
    conditioning = factory.schedule_conditioning(_context())
    assert conditioning is not None
    dynamics = factory.create(
        DynamicsReplayRequest(
            rollout_identity="typed-runtime-binding",
            num_steps=3,
            schedule_conditioning=conditioning,
        )
    )

    assert type(dynamics.replay_binding.replay_state) is (
        factory.component.replay_state_type
    )
    evidence = factory.binding_evidence
    assert evidence.scheduler_blueprint_identity == (
        factory.scheduler_blueprint.blueprint_identity
    )
    assert evidence.replay_state_factory_identity == (
        factory.replay_state_factory.factory_identity
    )
    assert evidence.replay_state_type_path.endswith(":SD3ScheduleReplayState")
    assert len(evidence.binding_identity) == 64


def test_runtime_binding_rejects_wrong_blueprint_family_and_state_type() -> None:
    component = _sd3_component()
    blueprint = _blueprint()
    replay_factory = SD3DynamicsReplayStateFactory(blueprint)

    with pytest.raises(TypeError, match="SchedulerArtifactBlueprint"):
        PerRolloutDynamicsFactory(
            component=component,
            scheduler_blueprint=object(),  # type: ignore[arg-type]
            replay_state_factory=replay_factory,
            dynamics_binding_family=component.dynamics_binding_family,
            replay_state_schema_id=component.replay_state_schema_id,
        )

    with pytest.raises(AlgorithmRuntimeBindingError, match="binding family"):
        PerRolloutDynamicsFactory(
            component=component,
            scheduler_blueprint=blueprint,
            replay_state_factory=replay_factory,
            dynamics_binding_family="wan.flow-sde.v1",
            replay_state_schema_id=component.replay_state_schema_id,
        )

    with pytest.raises(AlgorithmRuntimeBindingError, match="incompatible"):
        PerRolloutDynamicsFactory(
            component=component,
            scheduler_blueprint=blueprint,
            replay_state_factory=WanDynamicsReplayStateFactory(blueprint),
            dynamics_binding_family=component.dynamics_binding_family,
            replay_state_schema_id=component.replay_state_schema_id,
        )
