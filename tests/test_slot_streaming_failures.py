"""Failure atomicity and graph lifetime for slot-streamed policy updates."""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager

import pytest
import torch

import visual_rl.algorithms.optimization.kernel as update_kernel_module
from visual_rl.algorithms.dynamics.interface import Dynamics
from visual_rl.algorithms.dynamics.session import (
    DynamicsSelectionState,
    ScheduleSnapshot,
)
from visual_rl.algorithms.optimization.advantage import AdvantageGrouping
from visual_rl.algorithms.optimization.execution import (
    PreparedLoss,
    UpdateExecutionPlan,
    UpdateTransactionPoisonedError,
)
from visual_rl.algorithms.optimization.kernel import PolicyUpdateKernel
from visual_rl.algorithms.optimization.objective import (
    ClippedSurrogateObjective,
    PolicyLossInputs,
)
from visual_rl.algorithms.optimization.recompute import (
    PolicyRecomputer,
    PolicyRecomputeRequest,
    PolicyStats,
    ReferencePolicyStats,
)
from visual_rl.algorithms.optimization.slots import UpdateSlot
from visual_rl.algorithms.rollout.interface import RolloutExecution
from visual_rl.core.contracts import LatentLayout
from visual_rl.models import ModelLatentSpec
from visual_rl.data.samples import (
    BatchRowContext,
    NoConditionBatchState,
    TrajectoryBatch,
    TrajectoryContext,
)


class _InjectedSlotFailure(RuntimeError):
    pass


class _UnusedAdapter:
    def predict(self, model_input):
        del model_input
        raise AssertionError("the injected recomputer owns this test forward")

    def predict_reference(self, model_input):
        del model_input
        raise AssertionError("the injected recomputer owns this test reference")


class _UnusedDynamics(Dynamics):
    def timesteps(self, *, num_steps, device):
        return torch.arange(num_steps, 0, -1, device=device, dtype=torch.float32)

    def terminal_timestep(self, *, device):
        return torch.tensor(0.0, device=device)

    def add_noise(self, clean, noise, timestep):
        del timestep
        return clean + noise

    def transition_mean_std(self, transition):
        del transition
        raise AssertionError("the injected recomputer owns this test transition")


class _ContextProbe:
    def __init__(self) -> None:
        self.active = False
        self.enter_calls = 0
        self.exit_calls = 0
        self.forward_checks = 0
        self.backward_checks = 0
        self.events: list[str] = []

    @contextmanager
    def open(self):
        assert not self.active
        self.active = True
        self.enter_calls += 1
        self.events.append("context.enter")
        try:
            yield
        finally:
            assert self.active
            self.events.append("context.exit")
            self.exit_calls += 1
            self.active = False


class _ContextCheckedGraph(torch.autograd.Function):
    """A checkpoint-like graph whose recompute happens during backward."""

    @staticmethod
    def forward(ctx, parameter, rows, transitions, slot_index, probe):
        assert probe.active
        probe.forward_checks += 1
        probe.events.append(f"forward[{slot_index}]")
        ctx.probe = probe
        ctx.slot_index = slot_index
        return parameter.expand(rows, transitions).clone()

    @staticmethod
    def backward(ctx, gradient):
        assert ctx.probe.active
        ctx.probe.backward_checks += 1
        ctx.probe.events.append(f"backward[{ctx.slot_index}]")
        return gradient.sum(), None, None, None, None


class _InjectedRecomputer(PolicyRecomputer):
    def __init__(
        self,
        parameter: torch.nn.Parameter,
        probe: _ContextProbe,
        *,
        fail_at: int | None = None,
        checkpoint_like: bool = False,
        prior_loss_refs: list[weakref.ReferenceType[torch.Tensor]] | None = None,
    ) -> None:
        self.parameter = parameter
        self.probe = probe
        self.fail_at = fail_at
        self.checkpoint_like = checkpoint_like
        self.prior_loss_refs = prior_loss_refs
        self.calls = 0

    def compute_current_slot(
        self,
        request,
        slot,
        *,
        reference_stats=None,
    ) -> PolicyStats:
        del reference_stats
        assert isinstance(request, PolicyRecomputeRequest)
        assert isinstance(slot, UpdateSlot)
        assert self.probe.active
        slot_index = self.calls
        self.calls += 1
        if self.prior_loss_refs is not None:
            gc.collect()
            assert all(reference() is None for reference in self.prior_loss_refs)
        if self.fail_at == slot_index:
            raise _InjectedSlotFailure(f"forward slot {slot_index}")
        rows = len(slot.row_indices)
        transitions = slot.transition_count
        if self.checkpoint_like:
            current = _ContextCheckedGraph.apply(
                self.parameter,
                rows,
                transitions,
                slot_index,
                self.probe,
            )
        else:
            self.probe.forward_checks += 1
            self.probe.events.append(f"forward[{slot_index}]")
            current = self.parameter.expand(rows, transitions)
        return PolicyStats(
            grouping=AdvantageGrouping.from_trajectory(
                request.rollout.trajectory
            ).select_rows(slot.row_indices),
            current_log_probs=current,
        )


class _RetentionProbeRecomputer(_InjectedRecomputer):
    def __init__(
        self,
        parameter: torch.nn.Parameter,
        probe: _ContextProbe,
        reference_refs: list[weakref.ReferenceType[torch.Tensor]],
    ) -> None:
        super().__init__(parameter, probe, fail_at=0)
        self.reference_refs = reference_refs

    def compute_reference_slot(self, request, slot) -> ReferencePolicyStats:
        assert isinstance(request, PolicyRecomputeRequest)
        assert isinstance(slot, UpdateSlot)
        assert self.probe.active
        transition_mean = torch.ones(
            len(slot.row_indices),
            slot.transition_count,
            4,
            8,
            dtype=torch.float32,
        )
        self.reference_refs.append(weakref.ref(transition_mean))
        return ReferencePolicyStats(
            slot_id=slot.slot_id,
            transition_mean=transition_mean,
        )


class _ReferenceOrderingRecomputer(PolicyRecomputer):
    def __init__(
        self,
        parameter: torch.nn.Parameter,
        probe: _ContextProbe,
        expected_slot_count: int,
    ) -> None:
        self.parameter = parameter
        self.probe = probe
        self.expected_slot_count = expected_slot_count
        self.events: list[str] = []

    def compute_reference_slot(self, request, slot) -> ReferencePolicyStats:
        assert isinstance(request, PolicyRecomputeRequest)
        assert isinstance(slot, UpdateSlot)
        assert self.probe.active
        self.events.append(f"reference[{slot.slot_index}]")
        return ReferencePolicyStats(
            slot_id=slot.slot_id,
            transition_mean=torch.ones(
                len(slot.row_indices),
                slot.transition_count,
                1,
            ),
        )

    def compute_current_slot(
        self,
        request,
        slot,
        *,
        reference_stats=None,
    ) -> PolicyStats:
        assert isinstance(request, PolicyRecomputeRequest)
        assert isinstance(slot, UpdateSlot)
        assert isinstance(reference_stats, ReferencePolicyStats)
        assert self.probe.active
        assert self.events == [
            f"reference[{index}]" for index in range(self.expected_slot_count)
        ] + [f"current[{index}]" for index in range(slot.slot_index)]
        self.events.append(f"current[{slot.slot_index}]")
        rows = len(slot.row_indices)
        transitions = slot.transition_count
        current = self.parameter.expand(rows, transitions)
        current_mean = self.parameter.expand(rows, transitions, 1)
        return PolicyStats(
            grouping=AdvantageGrouping.from_trajectory(
                request.rollout.trajectory
            ).select_rows(slot.row_indices),
            current_log_probs=current,
            current_transition_mean=current_mean,
            transition_std=torch.ones_like(current_mean).detach(),
            reference_transition_mean=reference_stats.transition_mean,
        )


class _TrackingAdamW(torch.optim.AdamW):
    def __init__(self, parameters) -> None:
        self.step_calls = 0
        self.zero_grad_calls = 0
        super().__init__(parameters, lr=0.05)

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.zero_grad_calls += 1
        return super().zero_grad(set_to_none=set_to_none)


class _InjectedAccelerator:
    sync_gradients = True

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.backward_calls = 0
        self.unscale_calls = 0

    @contextmanager
    def accumulate(self, root):
        assert root is not None
        yield

    def backward(self, loss):
        slot_index = self.backward_calls
        self.backward_calls += 1
        loss.backward()
        if self.fail_at == slot_index:
            raise _InjectedSlotFailure(f"backward slot {slot_index}")

    def unscale_gradients(self, optimizer):
        assert optimizer is not None
        self.unscale_calls += 1


class _StepCounter:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1


class _CallbackCounter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def _trajectory() -> TrajectoryBatch:
    batch_size = 2
    transitions = 3
    contexts = tuple(
        TrajectoryContext(
            sample_id=f"sample-{row}",
            trajectory_id=f"trajectory-{row}",
            batch_row=BatchRowContext(
                occurrence_id="occurrence-0",
                group_id="group-0",
                member_id=row,
                phase="main",
                optimizer_step=0,
                source_item_id="source-0",
            ),
        )
        for row in range(batch_size)
    )
    latent = torch.zeros(batch_size, transitions, 1, 1, 1)
    timesteps = torch.tensor((3.0, 2.0, 1.0)).expand(batch_size, -1).clone()
    next_timesteps = torch.tensor((2.0, 1.0, 0.0)).expand(batch_size, -1).clone()
    transition_index = torch.arange(transitions, dtype=torch.int64).expand(
        batch_size,
        -1,
    )
    row_strings = tuple(("none",) * transitions for _ in range(batch_size))
    storage = tuple((str(latent.dtype),) * transitions for _ in range(batch_size))
    return TrajectoryBatch(
        kind="full_trajectory",
        contexts=contexts,
        x_t=latent,
        sampled_action=torch.ones_like(latent),
        conditioned_next=torch.ones_like(latent),
        timesteps=timesteps,
        next_timesteps=next_timesteps,
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.ones(batch_size, transitions, dtype=torch.bool),
        transition_index=transition_index,
        likelihood_semantics="exact_env_action",
        condition_identity=row_strings,
        guidance_identity=row_strings,
        storage_dtype_identity=storage,
        quantization_identity=row_strings,
        media=torch.zeros(batch_size, 1, 1, 1),
        media_layout="BCHW",
        condition_state=NoConditionBatchState(batch_size),
    )


def _request(
    trajectory: TrajectoryBatch,
    probe: _ContextProbe,
    *,
    require_reference_statistics: bool = False,
) -> PolicyRecomputeRequest:
    state = DynamicsSelectionState.from_generator(torch.Generator().manual_seed(7))
    snapshot = ScheduleSnapshot(
        torch.tensor((3.0, 2.0, 1.0)),
        torch.tensor((2.0, 1.0, 0.0)),
        sigmas=None,
        dt=torch.tensor((-1.0, -1.0, -1.0)),
        dynamics_config_identity="slot-failure-test-dynamics",
        scheduler_identity="slot-failure-test-scheduler",
        selection_policy="all",
        selected_policy_step_indices=(0, 1, 2),
        randomness_identity="slot-failure-test-selection",
        next_selection_state=state,
    )
    rollout = RolloutExecution(
        trajectory=trajectory,
        schedule_snapshot=snapshot,
        encoded_conditioning=object(),
        model_condition_identity=("encoded-0", "encoded-1"),
    )
    return PolicyRecomputeRequest(
        adapter=_UnusedAdapter(),
        dynamics=_UnusedDynamics(),
        rollout=rollout,
        latent_spec=ModelLatentSpec(
            shape=(trajectory.batch_size, 1, 1, 1),
            layout=LatentLayout.BCHW,
            axis_semantics=("batch", "channel", "height", "width"),
            device="cpu",
            dtype=torch.float32,
        ),
        current_context=probe.open,
        reference_context=(probe.open if require_reference_statistics else None),
        require_reference_statistics=require_reference_statistics,
    )


def _loss_inputs(
    trajectory: TrajectoryBatch,
    *,
    reference_kl_weight: float = 0.0,
) -> PolicyLossInputs:
    return PolicyLossInputs(
        base_advantage=torch.ones_like(trajectory.old_log_probs),
        algorithm_weight=torch.ones_like(trajectory.old_log_probs),
        active_mask=trajectory.transition_mask.clone(),
        clip_range=0.2,
        reference_kl_weight=reference_kl_weight,
    )


@pytest.mark.parametrize("fail_at", (0, 1, 2), ids=("early", "middle", "last"))
@pytest.mark.parametrize("failure_phase", ("forward", "loss", "backward"))
def test_any_slot_failure_aborts_every_update_side_effect_and_clears_gradients(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    fail_at: int,
) -> None:
    trajectory = _trajectory()
    probe = _ContextProbe()
    request = _request(trajectory, probe)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    before = parameter.detach().clone()
    optimizer = _TrackingAdamW((parameter,))
    scheduler = _StepCounter()
    ema = _CallbackCounter()
    reference = _CallbackCounter()
    committed_steps: list[int] = []
    recomputer = _InjectedRecomputer(
        parameter,
        probe,
        fail_at=fail_at if failure_phase == "forward" else None,
    )
    accelerator = _InjectedAccelerator(
        fail_at=fail_at if failure_phase == "backward" else None
    )

    if failure_phase == "loss":
        original_compute = ClippedSurrogateObjective.compute
        loss_calls = 0

        def injected_compute(self, **kwargs):
            nonlocal loss_calls
            slot_index = loss_calls
            loss_calls += 1
            if slot_index == fail_at:
                raise _InjectedSlotFailure(f"loss slot {slot_index}")
            return original_compute(self, **kwargs)

        monkeypatch.setattr(ClippedSurrogateObjective, "compute", injected_compute)

    with pytest.raises(
        _InjectedSlotFailure,
        match=rf"{failure_phase} slot {fail_at}",
    ):
        PolicyUpdateKernel().step_slots(
            trajectory=trajectory,
            loss_inputs=_loss_inputs(trajectory),
            recompute_request=request,
            recomputer=recomputer,
            optimizer=optimizer,
            scaler=None,
            optimizer_step=0,
            accelerator=accelerator,
            prepared_root=object(),
            lr_scheduler=scheduler,
            ema_update=ema,
            reference_update=reference,
            logical_commit=committed_steps.append,
            execution_plan=UpdateExecutionPlan(transition_window_size=1),
        )

    assert optimizer.step_calls == 0
    assert scheduler.calls == 0
    assert ema.calls == 0
    assert reference.calls == 0
    assert committed_steps == []
    assert optimizer.zero_grad_calls == 2
    assert parameter.grad is None
    assert torch.equal(parameter.detach(), before)
    assert not optimizer.state
    assert accelerator.unscale_calls == 0
    assert probe.enter_calls == 1
    assert probe.exit_calls == 1
    assert not probe.active


def test_current_train_context_spans_each_forward_and_checkpoint_like_backward() -> (
    None
):
    trajectory = _trajectory()
    probe = _ContextProbe()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TrackingAdamW((parameter,))

    result = PolicyUpdateKernel().step_slots(
        trajectory=trajectory,
        loss_inputs=_loss_inputs(trajectory),
        recompute_request=_request(trajectory, probe),
        recomputer=_InjectedRecomputer(
            parameter,
            probe,
            checkpoint_like=True,
        ),
        optimizer=optimizer,
        scaler=None,
        optimizer_step=0,
        accelerator=_InjectedAccelerator(),
        prepared_root=object(),
        execution_plan=UpdateExecutionPlan(transition_window_size=1),
    )

    assert result.transaction.committed
    assert optimizer.step_calls == 1
    assert probe.enter_calls == probe.exit_calls == 1
    assert probe.forward_checks == probe.backward_checks == 3
    assert probe.events == [
        "context.enter",
        "forward[0]",
        "backward[0]",
        "forward[1]",
        "backward[1]",
        "forward[2]",
        "backward[2]",
        "context.exit",
    ]
    assert not probe.active


def test_all_reference_slots_finish_before_any_current_slot_graph() -> None:
    trajectory = _trajectory()
    probe = _ContextProbe()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    recomputer = _ReferenceOrderingRecomputer(parameter, probe, 3)

    result = PolicyUpdateKernel().step_slots(
        trajectory=trajectory,
        loss_inputs=_loss_inputs(trajectory, reference_kl_weight=0.25),
        recompute_request=_request(
            trajectory,
            probe,
            require_reference_statistics=True,
        ),
        recomputer=recomputer,
        optimizer=_TrackingAdamW((parameter,)),
        scaler=None,
        optimizer_step=0,
        accelerator=_InjectedAccelerator(),
        prepared_root=object(),
        execution_plan=UpdateExecutionPlan(transition_window_size=1),
    )

    assert result.transaction.committed
    assert recomputer.events == [
        "reference[0]",
        "reference[1]",
        "reference[2]",
        "current[0]",
        "current[1]",
        "current[2]",
    ]
    assert probe.events == [
        "context.enter",
        "context.exit",
        "context.enter",
        "context.exit",
    ]


def test_prepared_loss_graph_is_released_before_the_next_slot_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory()
    probe = _ContextProbe()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TrackingAdamW((parameter,))
    loss_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    real_prepared_loss = PreparedLoss

    def tracking_prepared_loss(*, loss, payload):
        loss_refs.append(weakref.ref(loss))
        return real_prepared_loss(loss=loss, payload=payload)

    monkeypatch.setattr(
        update_kernel_module,
        "PreparedLoss",
        tracking_prepared_loss,
    )

    result = PolicyUpdateKernel().step_slots(
        trajectory=trajectory,
        loss_inputs=_loss_inputs(trajectory),
        recompute_request=_request(trajectory, probe),
        recomputer=_InjectedRecomputer(
            parameter,
            probe,
            prior_loss_refs=loss_refs,
        ),
        optimizer=optimizer,
        scaler=None,
        optimizer_step=0,
        accelerator=_InjectedAccelerator(),
        prepared_root=object(),
        execution_plan=UpdateExecutionPlan(transition_window_size=1),
    )

    gc.collect()
    assert len(loss_refs) == 3
    assert all(reference() is None for reference in loss_refs)
    assert result.transaction.committed
    assert result.policy_summary is not None
    assert not result.policy_summary.new_log_probs.requires_grad
    assert result.policy_summary.new_log_probs.grad_fn is None
    assert result.policy_summary.materialized_mask.dtype == torch.bool


def test_backward_exception_does_not_retain_slot_graph_through_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory()
    probe = _ContextProbe()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TrackingAdamW((parameter,))
    loss_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    real_prepared_loss = PreparedLoss

    def tracking_prepared_loss(*, loss, payload):
        loss_refs.append(weakref.ref(loss))
        return real_prepared_loss(loss=loss, payload=payload)

    monkeypatch.setattr(
        update_kernel_module,
        "PreparedLoss",
        tracking_prepared_loss,
    )

    with pytest.raises(_InjectedSlotFailure, match="backward slot 1") as captured:
        PolicyUpdateKernel().step_slots(
            trajectory=trajectory,
            loss_inputs=_loss_inputs(trajectory),
            recompute_request=_request(trajectory, probe),
            recomputer=_InjectedRecomputer(parameter, probe),
            optimizer=optimizer,
            scaler=None,
            optimizer_step=0,
            accelerator=_InjectedAccelerator(fail_at=1),
            prepared_root=object(),
            execution_plan=UpdateExecutionPlan(transition_window_size=1),
        )

    # Keep both ExceptionInfo and the exception itself alive while collecting.
    assert captured.value is not None
    gc.collect()
    assert len(loss_refs) == 2
    assert all(reference() is None for reference in loss_refs)
    assert parameter.grad is None


def test_retained_forward_error_releases_trajectory_references_accumulator_and_closures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    reference_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    accumulator_refs: list[weakref.ReferenceType[object]] = []
    accumulator_tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    closure_refs: list[weakref.ReferenceType[object]] = []

    real_accumulator_init = update_kernel_module._StreamingObjectiveAccumulator.__init__

    def tracking_accumulator_init(self, trajectory, plan):
        real_accumulator_init(self, trajectory, plan)
        accumulator_refs.append(weakref.ref(self))
        accumulator_tensor_refs.append(weakref.ref(self.new_log_probs))

    monkeypatch.setattr(
        update_kernel_module._StreamingObjectiveAccumulator,
        "__init__",
        tracking_accumulator_init,
    )
    real_execute = UpdateExecutionPlan.execute

    def tracking_execute(self, **kwargs):
        closure_refs.extend(weakref.ref(closure) for closure in kwargs["loss_closures"])
        return real_execute(self, **kwargs)

    monkeypatch.setattr(UpdateExecutionPlan, "execute", tracking_execute)

    def invoke() -> BaseException:
        trajectory = _trajectory()
        trajectory_refs.append(weakref.ref(trajectory.x_t))
        probe = _ContextProbe()
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = _TrackingAdamW((parameter,))
        try:
            PolicyUpdateKernel().step_slots(
                trajectory=trajectory,
                loss_inputs=_loss_inputs(
                    trajectory,
                    reference_kl_weight=0.25,
                ),
                recompute_request=_request(
                    trajectory,
                    probe,
                    require_reference_statistics=True,
                ),
                recomputer=_RetentionProbeRecomputer(
                    parameter,
                    probe,
                    reference_refs,
                ),
                optimizer=optimizer,
                scaler=None,
                optimizer_step=0,
                accelerator=_InjectedAccelerator(),
                prepared_root=object(),
                execution_plan=UpdateExecutionPlan(transition_window_size=1),
            )
        except _InjectedSlotFailure as error:
            # Model a controller that drops its live iteration state but keeps
            # the exception for audit/logging.
            del trajectory, probe, parameter, optimizer
            return error
        raise AssertionError("the injected slot failure did not propagate")

    error = invoke()
    assert type(error) is _InjectedSlotFailure
    assert str(error) == "forward slot 0"
    gc.collect()

    assert len(trajectory_refs) == 1
    assert len(reference_refs) == 3
    assert len(accumulator_refs) == 1
    assert len(accumulator_tensor_refs) == 1
    assert len(closure_refs) == 3
    assert all(reference() is None for reference in trajectory_refs)
    assert all(reference() is None for reference in reference_refs)
    assert all(reference() is None for reference in accumulator_refs)
    assert all(reference() is None for reference in accumulator_tensor_refs)
    assert all(reference() is None for reference in closure_refs)
    traceback_names = []
    traceback = error.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "_step_slots_impl" not in traceback_names
    assert "execute" not in traceback_names


def test_retained_last_backward_error_releases_finalized_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    real_finalize = update_kernel_module._StreamingObjectiveAccumulator.finalize

    def tracking_finalize(self):
        metrics, summary = real_finalize(self)
        summary_refs.extend(
            (
                weakref.ref(summary.new_log_probs),
                weakref.ref(summary.materialized_mask),
            )
        )
        return metrics, summary

    monkeypatch.setattr(
        update_kernel_module._StreamingObjectiveAccumulator,
        "finalize",
        tracking_finalize,
    )

    def invoke() -> BaseException:
        trajectory = _trajectory()
        probe = _ContextProbe()
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        try:
            PolicyUpdateKernel().step_slots(
                trajectory=trajectory,
                loss_inputs=_loss_inputs(trajectory),
                recompute_request=_request(trajectory, probe),
                recomputer=_InjectedRecomputer(parameter, probe),
                optimizer=_TrackingAdamW((parameter,)),
                scaler=None,
                optimizer_step=0,
                accelerator=_InjectedAccelerator(fail_at=2),
                prepared_root=object(),
                execution_plan=UpdateExecutionPlan(transition_window_size=1),
            )
        except _InjectedSlotFailure as error:
            return error
        raise AssertionError("the injected backward failure did not propagate")

    error = invoke()
    assert type(error) is _InjectedSlotFailure
    assert str(error) == "backward slot 2"
    gc.collect()

    assert len(summary_refs) == 2
    assert all(reference() is None for reference in summary_refs)


def test_summary_finalization_failure_happens_before_any_update_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory()
    probe = _ContextProbe()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TrackingAdamW((parameter,))
    scheduler = _StepCounter()
    commits: list[int] = []

    def fail_finalize(self):
        del self
        raise _InjectedSlotFailure("summary finalize")

    monkeypatch.setattr(
        update_kernel_module._StreamingObjectiveAccumulator,
        "finalize",
        fail_finalize,
    )

    with pytest.raises(_InjectedSlotFailure, match="summary finalize"):
        PolicyUpdateKernel().step_slots(
            trajectory=trajectory,
            loss_inputs=_loss_inputs(trajectory),
            recompute_request=_request(trajectory, probe),
            recomputer=_InjectedRecomputer(parameter, probe),
            optimizer=optimizer,
            scaler=None,
            optimizer_step=0,
            accelerator=_InjectedAccelerator(),
            prepared_root=object(),
            lr_scheduler=scheduler,
            logical_commit=commits.append,
            execution_plan=UpdateExecutionPlan(transition_window_size=1),
        )

    assert optimizer.step_calls == 0
    assert scheduler.calls == 0
    assert commits == []
    assert parameter.grad is None


def test_post_commit_result_failure_is_typed_as_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory()
    probe = _ContextProbe()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TrackingAdamW((parameter,))
    commits: list[int] = []
    summary_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def fail_result(**kwargs):
        summary = kwargs["policy_summary"]
        summary_refs.extend(
            (
                weakref.ref(summary.new_log_probs),
                weakref.ref(summary.materialized_mask),
            )
        )
        raise _InjectedSlotFailure("result materialization")

    monkeypatch.setattr(update_kernel_module, "PolicyUpdateResult", fail_result)

    with pytest.raises(UpdateTransactionPoisonedError) as captured:
        PolicyUpdateKernel().step_slots(
            trajectory=trajectory,
            loss_inputs=_loss_inputs(trajectory),
            recompute_request=_request(trajectory, probe),
            recomputer=_InjectedRecomputer(parameter, probe),
            optimizer=optimizer,
            scaler=None,
            optimizer_step=0,
            accelerator=_InjectedAccelerator(),
            prepared_root=object(),
            logical_commit=commits.append,
            execution_plan=UpdateExecutionPlan(transition_window_size=1),
        )

    assert captured.value.optimizer_step_applied
    assert captured.value.failed_phase == "result_materialization"
    assert type(captured.value.cause) is _InjectedSlotFailure
    assert captured.value.cause.__traceback__ is None
    assert optimizer.step_calls == 1
    assert commits == [1]
    gc.collect()
    assert len(summary_refs) == 2
    assert all(reference() is None for reference in summary_refs)
