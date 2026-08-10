"""Resolution- and patch-aware SD3 scheduler materialization fixtures."""

from __future__ import annotations

import math

import pytest
import torch

from visual_rl.algorithms.dynamics.interface import DynamicsContractError
from visual_rl.algorithms.dynamics.replay import (
    DynamicsReplayRequest,
    FlowMatchDynamicShiftConfig,
    FlowMatchScheduleConditioning,
)
from visual_rl.algorithms.dynamics.sd3_flow_sde import (
    SD3DynamicsReplayStateFactory,
    SD3FlowSDEDynamics,
)
from visual_rl.algorithms.dynamics.session import DynamicsSession, PolicyStepSelection


class _SchedulerConfig(dict):
    def __getattr__(self, name):
        return self[name]


class _DynamicShiftScheduler:
    """Small Diffusers-shaped exponential time-shift fixture."""

    def __init__(self, config=None) -> None:
        values = {
            "use_dynamic_shifting": True,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
        values.update({} if config is None else config)
        self.config = _SchedulerConfig(values)
        self.last_mu = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps, device, mu=None) -> None:
        if mu is None:
            raise ValueError("mu is required")
        self.last_mu = float(mu)
        unshifted = torch.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            dtype=torch.float64,
        )
        numerator = math.exp(float(mu))
        shifted = numerator / (numerator + (1.0 / unshifted - 1.0))
        shifted = shifted.to(dtype=torch.float32, device=device)
        self.timesteps = shifted * 1000.0
        self.sigmas = torch.cat(
            [shifted, torch.zeros(1, dtype=torch.float32, device=device)]
        )


class _StaticScheduler:
    def __init__(self, config=None) -> None:
        values = {"use_dynamic_shifting": False}
        values.update({} if config is None else config)
        self.config = _SchedulerConfig(values)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps, device) -> None:
        self.timesteps = torch.linspace(
            1000.0,
            100.0,
            num_inference_steps,
            dtype=torch.float32,
            device=device,
        )
        self.sigmas = torch.linspace(
            1.0,
            0.0,
            num_inference_steps + 1,
            dtype=torch.float32,
            device=device,
        )


def _conditioning(
    factory: SD3DynamicsReplayStateFactory,
    *,
    latent_size: int,
) -> FlowMatchScheduleConditioning:
    return factory.schedule_conditioning(
        latent_height=latent_size,
        latent_width=latent_size,
        patch_size=2,
    )


def _selection(num_steps: int) -> PolicyStepSelection:
    return PolicyStepSelection.fixed(
        (0,),
        num_steps=num_steps,
        generator=torch.Generator(device="cpu").manual_seed(17),
        policy="dynamic-shift-fixture",
    )


def test_flow_factory_shape_formula_is_typed_and_canonical() -> None:
    policy = FlowMatchDynamicShiftConfig(
        base_image_seq_len=256,
        max_image_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    )
    conditioning = FlowMatchScheduleConditioning.from_latent_geometry(
        latent_height=64,
        latent_width=64,
        patch_size=2,
        dynamic_shift=policy,
    )

    # Upstream SD3: (latent_h // patch) * (latent_w // patch).
    assert conditioning.image_seq_len == 1024
    # Upstream calculate_shift: linear interpolation between scheduler anchors.
    assert conditioning.mu == pytest.approx(0.63, abs=1e-12)
    assert (
        FlowMatchScheduleConditioning(
            latent_height=64,
            latent_width=64,
            patch_size=2,
            image_seq_len=1024,
            dynamic_shift=policy,
        )
        == conditioning
    )

    with pytest.raises(DynamicsContractError, match="image_seq_len"):
        FlowMatchScheduleConditioning(
            latent_height=64,
            latent_width=64,
            patch_size=2,
            image_seq_len=1023,
            dynamic_shift=policy,
        )
    with pytest.raises(DynamicsContractError, match="divisible"):
        FlowMatchScheduleConditioning.from_latent_geometry(
            latent_height=63,
            latent_width=64,
            patch_size=2,
            dynamic_shift=policy,
        )


def test_dynamic_scheduler_requires_matching_geometry_and_mu_capability() -> None:
    factory = SD3DynamicsReplayStateFactory.from_scheduler(_DynamicShiftScheduler())
    conditioning = _conditioning(factory, latent_size=64)
    request = DynamicsReplayRequest(
        "dynamic-rollout",
        4,
        schedule_conditioning=conditioning,
    )
    binding = factory.create(request)

    assert binding.request.schedule_conditioning is conditioning
    assert binding.replay_state.num_steps == 4
    expected_unshifted = torch.linspace(1.0, 0.25, 4, dtype=torch.float64)
    numerator = math.exp(0.63)
    expected = (numerator / (numerator + (1.0 / expected_unshifted - 1.0))).to(
        torch.float32
    )
    torch.testing.assert_close(binding.replay_state.sigmas[:-1], expected)

    with pytest.raises(DynamicsContractError, match="requires latent"):
        factory.create(DynamicsReplayRequest("missing-context", 4))

    drifted = FlowMatchScheduleConditioning.from_latent_geometry(
        latent_height=64,
        latent_width=64,
        patch_size=2,
        dynamic_shift=FlowMatchDynamicShiftConfig(max_shift=1.16),
    )
    with pytest.raises(DynamicsContractError, match="does not match"):
        factory.create(
            DynamicsReplayRequest(
                "drifted-policy",
                4,
                schedule_conditioning=drifted,
            )
        )


def test_conditioning_covers_request_binding_snapshot_and_resume_identity() -> None:
    factory = SD3DynamicsReplayStateFactory.from_scheduler(_DynamicShiftScheduler())
    small_request = DynamicsReplayRequest(
        "same-rollout",
        4,
        schedule_conditioning=_conditioning(factory, latent_size=64),
    )
    large_request = DynamicsReplayRequest(
        "same-rollout",
        4,
        schedule_conditioning=_conditioning(factory, latent_size=96),
    )
    small = factory.create(small_request)
    small_resume = factory.create(small_request)
    large = factory.create(large_request)

    assert small_request.request_identity != large_request.request_identity
    assert small.binding_identity != large.binding_identity
    assert small == small_resume
    assert small.replay_state_identity == small_resume.replay_state_identity

    small_session = DynamicsSession.create(
        SD3FlowSDEDynamics(small.replay_state, replay_binding=small),
        num_steps=4,
        device="cpu",
        selection=_selection(4),
    )
    resumed_session = DynamicsSession.create(
        SD3FlowSDEDynamics(
            small_resume.replay_state,
            replay_binding=small_resume,
        ),
        num_steps=4,
        device="cpu",
        selection=_selection(4),
    )
    large_session = DynamicsSession.create(
        SD3FlowSDEDynamics(large.replay_state, replay_binding=large),
        num_steps=4,
        device="cpu",
        selection=_selection(4),
    )

    assert small_session.snapshot == resumed_session.snapshot
    assert (
        small_session.snapshot.schedule_identity
        != large_session.snapshot.schedule_identity
    )
    restored = type(small_session.snapshot).from_payload(
        small_session.snapshot.to_payload()
    )
    assert restored == small_session.snapshot
    with pytest.raises(DynamicsContractError, match="does not match"):
        DynamicsSession.from_snapshot(
            large_session.dynamics,
            restored,
        )


def test_non_dynamic_scheduler_keeps_legacy_materialization_semantics() -> None:
    factory = SD3DynamicsReplayStateFactory.from_scheduler(_StaticScheduler())
    plain = factory.create(DynamicsReplayRequest("plain", 3))
    conditioned = factory.create(
        DynamicsReplayRequest(
            "conditioned",
            3,
            schedule_conditioning=_conditioning(factory, latent_size=64),
        )
    )

    assert plain.replay_state_identity == conditioned.replay_state_identity
    torch.testing.assert_close(
        plain.replay_state.timesteps,
        conditioned.replay_state.timesteps,
    )
    torch.testing.assert_close(
        plain.replay_state.sigmas,
        conditioned.replay_state.sigmas,
    )


def test_diffusers_flow_match_scheduler_matches_native_mu_schedule() -> None:
    diffusers = pytest.importorskip("diffusers")
    scheduler_type = diffusers.FlowMatchEulerDiscreteScheduler
    source = scheduler_type(
        use_dynamic_shifting=True,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    )
    factory = SD3DynamicsReplayStateFactory.from_scheduler(source)
    request = DynamicsReplayRequest(
        "native-diffusers-fixture",
        4,
        schedule_conditioning=_conditioning(factory, latent_size=64),
    )

    observed = factory.create(request).replay_state
    expected = scheduler_type.from_config(source.config)
    expected.set_timesteps(num_inference_steps=4, device="cpu", mu=0.63)

    torch.testing.assert_close(observed.timesteps, expected.timesteps)
    torch.testing.assert_close(observed.sigmas, expected.sigmas)
