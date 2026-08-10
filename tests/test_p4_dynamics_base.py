"""Analytical tests for the P4 Dynamics template and replay records."""

from __future__ import annotations

import pytest
import torch

from visual_rl.algorithms.dynamics.interface import (
    DeterministicTransitionOutput,
    Dynamics,
    DynamicsContractError,
    TransitionInput,
    TransitionMeanStd,
)
from visual_rl.data.samples import LikelihoodSemantics


class _AnalyticalDynamics(Dynamics):
    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure
        self.mean_std_calls = 0

    def timesteps(self, *, num_steps: int, device):
        return torch.linspace(1.0, 0.25, num_steps, device=device)

    def terminal_timestep(self, *, device):
        return torch.tensor(-0.125, device=device)

    def add_noise(self, clean, noise, timestep):
        scale = timestep.reshape(-1, *([1] * (clean.ndim - 1)))
        return clean + scale * noise

    def transition_mean_std(self, transition):
        self.mean_std_calls += 1
        batch = transition.batch_size
        broadcast = (batch, *([1] * (transition.x_t.ndim - 1)))
        dt = (transition.t_next - transition.t).to(dtype=transition.x_t.dtype)
        mean = transition.x_t + transition.model_prediction * dt.reshape(broadcast)
        std = torch.full(
            broadcast,
            0.25,
            dtype=transition.x_t.dtype,
            device=transition.x_t.device,
        )
        if self.failure == "mean_shape":
            mean = mean[0]
        elif self.failure == "nan_mean":
            mean = mean.clone()
            mean[0, 0, 0] = torch.nan
        elif self.failure == "zero_std":
            std = torch.zeros_like(std)
        elif self.failure == "wrong_std":
            std = torch.ones(batch, 2, 3, device=transition.x_t.device)
        elif self.failure == "zero_dt":
            dt = torch.zeros_like(dt)
        return TransitionMeanStd(mean=mean, std=std, dt=dt)


class _AnalyticalODEDynamics(_AnalyticalDynamics):
    def _deterministic_ode_step(self, transition):
        batch = transition.batch_size
        broadcast = (batch, *([1] * (transition.x_t.ndim - 1)))
        dt = (transition.t_next - transition.t).to(dtype=transition.x_t.dtype)
        next_state = transition.x_t + transition.model_prediction * dt.reshape(
            broadcast
        )
        return DeterministicTransitionOutput(
            next_state=next_state.detach(),
            dt=dt.detach(),
        )


class _AttachedODEDynamics(_AnalyticalDynamics):
    def _deterministic_ode_step(self, transition):
        dt = (transition.t_next - transition.t).to(dtype=transition.x_t.dtype)
        broadcast = (
            transition.batch_size,
            *([1] * (transition.x_t.ndim - 1)),
        )
        return DeterministicTransitionOutput(
            next_state=(
                transition.x_t + transition.model_prediction * dt.reshape(broadcast)
            ),
            dt=dt,
        )


def _input(*, requires_grad: bool = True) -> TransitionInput:
    x_t = torch.tensor(
        [[[0.25, -0.5]], [[1.0, 0.75]]],
        dtype=torch.float32,
    )
    prediction = torch.tensor(
        [[[0.1, 0.2]], [[-0.3, 0.4]]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    return TransitionInput(
        x_t=x_t,
        model_prediction=prediction,
        t=torch.tensor([1.0, 0.8]),
        t_next=torch.tensor([0.5, 0.2]),
        mask=torch.tensor([True, True]),
        transition_index=torch.tensor([0, 2], dtype=torch.int64),
        condition_identity=("none", "camera-v1"),
        guidance_identity=("cfg:1", "cfg:4.5"),
        storage_dtype_identity=("torch.float32", "torch.float32"),
        quantization_identity=("none", "none"),
    )


def test_sample_and_replay_share_one_mean_std_path_and_are_differentiable() -> None:
    dynamics = _AnalyticalDynamics()
    transition = _input()
    generator = torch.Generator(device="cpu").manual_seed(17)

    output = dynamics.sample_transition(transition, generator=generator)
    replay = dynamics.transition_log_prob(transition, output.sampled_next)

    assert dynamics.mean_std_calls == 2
    torch.testing.assert_close(replay, output.log_prob)
    replay.sum().backward()
    assert transition.model_prediction.grad is not None
    assert bool(torch.isfinite(transition.model_prediction.grad).all())
    assert float(transition.model_prediction.grad.abs().sum()) > 0


def test_transition_schedule_uses_explicit_nonzero_terminal_value() -> None:
    schedule = _AnalyticalDynamics().transition_schedule(
        num_steps=3,
        device="cpu",
    )
    torch.testing.assert_close(
        schedule.next_timesteps[:-1],
        schedule.timesteps[1:],
    )
    assert schedule.next_timesteps[-1].item() == pytest.approx(-0.125)
    assert schedule.num_steps == 3


def test_explicit_ode_port_is_detached_and_default_fails_closed() -> None:
    transition = _input()
    output = _AnalyticalODEDynamics().deterministic_ode_step(transition)
    dt = (transition.t_next - transition.t).to(dtype=transition.x_t.dtype)
    expected = transition.x_t + transition.model_prediction.detach() * dt.reshape(
        transition.batch_size,
        *([1] * (transition.x_t.ndim - 1)),
    )

    torch.testing.assert_close(output.next_state, expected, rtol=0, atol=0)
    torch.testing.assert_close(output.dt, dt, rtol=0, atol=0)
    assert output.next_state.dtype == transition.x_t.dtype
    assert output.next_state.device == transition.x_t.device
    assert not output.next_state.requires_grad
    assert output.next_state.grad_fn is None
    assert not output.dt.requires_grad
    assert output.dt.grad_fn is None

    with pytest.raises(NotImplementedError, match="explicit deterministic ODE"):
        _AnalyticalDynamics().deterministic_ode_step(transition)
    with pytest.raises(DynamicsContractError, match="must be detached"):
        _AttachedODEDynamics().deterministic_ode_step(transition)


def test_generator_is_explicit_and_deterministically_controls_sampling() -> None:
    dynamics = _AnalyticalDynamics()
    transition = _input(requires_grad=False)

    first = dynamics.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(31),
    )
    second = dynamics.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(31),
    )
    torch.testing.assert_close(first.sampled_next, second.sampled_next)

    with pytest.raises(TypeError, match="explicit torch.Generator"):
        dynamics.sample_transition(transition, generator=None)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("mean_shape", "mean must match"),
        ("nan_mean", "mean must be finite"),
        ("zero_std", "strictly positive"),
        ("wrong_std", "not broadcastable"),
        ("zero_dt", "dt must be non-zero"),
    ],
)
def test_invalid_mean_std_and_ode_paths_fail_before_log_prob(
    failure: str,
    message: str,
) -> None:
    dynamics = _AnalyticalDynamics(failure)
    with pytest.raises(DynamicsContractError, match=message):
        dynamics.sample_transition(
            _input(requires_grad=False),
            generator=torch.Generator().manual_seed(1),
        )


def test_arbitrary_action_shape_dtype_and_finite_checks_are_closed() -> None:
    dynamics = _AnalyticalDynamics()
    transition = _input()

    with pytest.raises(DynamicsContractError, match="match x_t shape"):
        dynamics.transition_log_prob(transition, torch.zeros(2, 2))
    with pytest.raises(DynamicsContractError, match="dtype mismatch"):
        dynamics.transition_log_prob(
            transition,
            torch.zeros_like(transition.x_t, dtype=torch.float64),
        )
    action = torch.zeros_like(transition.x_t)
    action[0, 0, 0] = torch.nan
    with pytest.raises(DynamicsContractError, match="must be finite"):
        dynamics.transition_log_prob(transition, action)


def test_record_keeps_pre_hook_action_post_hook_state_and_semantics() -> None:
    dynamics = _AnalyticalDynamics()
    transition = _input()
    output = dynamics.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(5),
    )
    conditioned = output.sampled_next + 0.125

    exact = dynamics.make_record(
        transition,
        output,
        conditioned_next=conditioned,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
    )
    surrogate = dynamics.make_record(
        transition,
        output,
        conditioned_next=conditioned,
        likelihood_semantics=(LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE),
    )

    assert exact.scoring_target is exact.sampled_action
    assert surrogate.scoring_target is surrogate.conditioned_next
    assert torch.equal(exact.conditioned_next, conditioned.detach())
    assert torch.equal(exact.sampled_action, output.sampled_next.detach())
    assert exact.old_log_prob.grad_fn is None
    assert not torch.equal(exact.old_log_prob, surrogate.old_log_prob)
    expected_surrogate = dynamics.transition_log_prob(
        transition,
        conditioned,
    )
    torch.testing.assert_close(
        surrogate.old_log_prob,
        expected_surrogate.detach(),
    )

    sliced = exact.slice([1])
    assert sliced.batch_size == 1
    converted = sliced.to("cpu", dtype=torch.float64)
    assert converted.x_t.dtype == torch.float64
    assert converted.t.dtype == torch.float32
    assert converted.storage_dtype_identity == ("torch.float64",)


def test_transition_input_rejects_hidden_batch_or_identity_drift() -> None:
    transition = _input(requires_grad=False)
    with pytest.raises(DynamicsContractError, match="model_prediction"):
        TransitionInput(
            x_t=transition.x_t,
            model_prediction=transition.model_prediction[:1],
            t=transition.t,
            t_next=transition.t_next,
            mask=transition.mask,
            transition_index=transition.transition_index,
            condition_identity=transition.condition_identity,
            guidance_identity=transition.guidance_identity,
            storage_dtype_identity=transition.storage_dtype_identity,
            quantization_identity=transition.quantization_identity,
        )
    with pytest.raises(DynamicsContractError, match="storage_dtype_identity"):
        TransitionInput(
            x_t=transition.x_t,
            model_prediction=transition.model_prediction,
            t=transition.t,
            t_next=transition.t_next,
            mask=transition.mask,
            transition_index=transition.transition_index,
            condition_identity=transition.condition_identity,
            guidance_identity=transition.guidance_identity,
            storage_dtype_identity=("torch.float16", "torch.float16"),
            quantization_identity=transition.quantization_identity,
        )


def test_subclass_cannot_replace_sample_or_replay_templates() -> None:
    with pytest.raises(TypeError, match="must not override"):

        class _InvalidDynamics(Dynamics):
            def timesteps(self, *, num_steps, device):
                return None

            def add_noise(self, clean, noise, timestep):
                return clean

            def transition_mean_std(self, transition):
                return None

            def sample_transition(self, transition, *, generator):
                return None

    with pytest.raises(TypeError, match="must not override"):

        class _InvalidODEDynamics(Dynamics):
            def timesteps(self, *, num_steps, device):
                return None

            def add_noise(self, clean, noise, timestep):
                return clean

            def transition_mean_std(self, transition):
                return None

            def deterministic_ode_step(self, transition):
                return None
