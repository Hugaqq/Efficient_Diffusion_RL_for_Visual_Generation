"""Lifecycle, residency, and rollback tests for model component ownership."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from visual_rl.core.contracts import (
    ArtifactBoundContract,
    ComputePrecision,
    DeclaredContract,
    LatentLayout,
    MediaKind,
    ModelContract,
    PredictionType,
    RuntimeBoundContract,
    TaskKind,
    TimeCoordinate,
    TrainingMode,
)
from visual_rl.models import (
    ComponentBinding,
    ComponentLifecycleError,
    ComponentManager,
    ComponentManagerError,
    ComponentRole,
    ExecutionMode,
    ModelAdapter,
    ModelComponents,
    OwnershipState,
    Residency,
    ResourcePlan,
)


def _declared() -> DeclaredContract:
    return DeclaredContract(
        component_kind="model",
        component_id="fake-owned-model-v1",
        model=ModelContract(
            tasks=(TaskKind.T2V,),
            output_media=(MediaKind.VIDEO,),
            latent_layouts=(LatentLayout.BCTHW,),
            latent_ranks=(5,),
            axis_semantics=(("batch", "channel", "time", "height", "width"),),
            prediction_types=(PredictionType.FLOW,),
            time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
            training_modes=(TrainingMode.LORA,),
            supported_precisions=(ComputePrecision.FP32,),
            provides_reference_policy=True,
        ),
    )


def _runtime_bound() -> RuntimeBoundContract:
    return RuntimeBoundContract(
        artifact=ArtifactBoundContract(
            declared=_declared(),
            artifact_identity="fake-artifact-sha256",
            resolved_fields=(),
        ),
        runtime_identity="fake-runtime-g3",
        verified_fields=(
            ("model.component_topology", "verified"),
            ("model.reference_forward", "verified"),
        ),
    )


class _TrackedModule(torch.nn.Module):
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str, str | None]],
        *,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0]),
            requires_grad=trainable,
        )

    def to(self, device):
        self.events.append(("move", self.name, str(torch.device(device))))
        return self

    def close(self):
        self.events.append(("close", self.name, None))


class _StaticComponent:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def close(self):
        self.events.append(("close", self.name, None))


class _IdentityAccelerator:
    def __init__(self) -> None:
        self.prepare_calls = []
        self.accumulation_roots = []

    def prepare(self, *items):
        self.prepare_calls.append(items)
        return items

    @contextmanager
    def accumulate(self, root):
        self.accumulation_roots.append(root)
        yield


class _OwnedAdapter(ModelAdapter):
    def __init__(self, events=None) -> None:
        self.events = [] if events is None else events

    @classmethod
    def describe(cls, config):
        del config
        return _declared()

    @classmethod
    def from_config(cls, config, *, runtime_context):
        del config, runtime_context
        return cls()

    def load_components(self, session):
        self.tokenizer = session.acquire(
            "tokenizer",
            lambda: _StaticComponent("tokenizer", self.events),
            roles=(ComponentRole.PREPROCESS,),
            managed_residency=False,
        )
        self.transformer = session.acquire(
            "transformer",
            lambda: _TrackedModule(
                "transformer",
                self.events,
                trainable=True,
            ),
            roles=(ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
        )
        self.reference = session.acquire(
            "reference",
            lambda: _TrackedModule("reference", self.events, trainable=False),
            roles=(ComponentRole.REFERENCE,),
        )
        self.vae = session.acquire(
            "vae",
            lambda: _TrackedModule("vae", self.events, trainable=False),
            roles=(ComponentRole.PREPROCESS, ComponentRole.DECODER),
        )
        return session.freeze()

    def encode(self, batch):
        return batch

    def prepare_latents(self, latent_spec, *, generator):
        del latent_spec, generator

    def predict(self, model_input):
        del model_input

    def predict_reference(self, model_input):
        del model_input

    def decode(self, latents, latent_spec):
        del latent_spec
        return latents


class _FailingAdapter(_OwnedAdapter):
    def load_components(self, session):
        session.acquire(
            "first",
            lambda: _StaticComponent("first", self.events),
            roles=(ComponentRole.PREPROCESS,),
            managed_residency=False,
        )
        session.acquire(
            "second",
            lambda: _StaticComponent("second", self.events),
            roles=(ComponentRole.DECODER,),
            managed_residency=False,
        )

        def fail():
            raise RuntimeError("third component failed to load")

        session.acquire(
            "third",
            fail,
            roles=(ComponentRole.INFERENCE,),
        )
        raise AssertionError("unreachable")


class _InvalidRegistrationAdapter(_OwnedAdapter):
    def load_components(self, session):
        session.acquire(
            "duplicate",
            lambda: _StaticComponent("owned", self.events),
            roles=(ComponentRole.PREPROCESS,),
            managed_residency=False,
        )
        session.acquire(
            "duplicate",
            lambda: _StaticComponent("unbound", self.events),
            roles=(ComponentRole.DECODER,),
            managed_residency=False,
        )
        raise AssertionError("unreachable")


def _prepared_manager(adapter=None) -> ComponentManager:
    manager = ComponentManager(
        _OwnedAdapter() if adapter is None else adapter,
        execution_device="cpu",
        offload_device="cpu",
    )
    manager.load()
    manager.configure()
    optimizer = torch.optim.SGD(manager.parameter_state.parameters(), lr=0.1)
    manager.prepare(
        accelerator=_IdentityAccelerator(),
        optimizer=optimizer,
    )
    manager.bind_runtime(_runtime_bound())
    return manager


def test_full_lifecycle_and_g3_gate_are_explicit() -> None:
    manager = ComponentManager(
        _OwnedAdapter(),
        execution_device="cpu",
        offload_device="cpu",
    )
    assert len(manager.resource_plan.plan_id) == 64
    assert {item["mode"] for item in manager.resource_plan.to_payload()["modes"]} == {
        "preprocess",
        "rollout",
        "train",
        "eval",
    }
    assert manager.state is OwnershipState.UNLOADED
    with pytest.raises(
        ComponentLifecycleError,
        match="PREPARED",
    ), manager.preprocess():
        pass

    components = manager.load()
    assert manager.state is OwnershipState.LOADED
    assert components.inference == (manager.component("transformer"),)
    assert components.trainable == (manager.component("transformer"),)
    assert components.reference == (manager.component("reference"),)
    assert components.decoder == (manager.component("vae"),)
    assert manager.residency("tokenizer") is Residency.STATIC
    assert manager.residency("transformer") is Residency.OFFLOADED

    manager.configure()
    assert manager.state is OwnershipState.CONFIGURED
    optimizer = torch.optim.SGD(manager.parameter_state.parameters(), lr=0.1)
    accelerator = _IdentityAccelerator()
    manager.prepare(accelerator=accelerator, optimizer=optimizer)
    assert manager.state is OwnershipState.PREPARED
    assert len(accelerator.prepare_calls) == 1
    assert manager.residency("transformer") is Residency.PREPARED
    assert manager.residency("reference") is Residency.PREPARED
    with pytest.raises(ComponentLifecycleError, match="G3"), manager.rollout():
        pass
    manager.bind_runtime(_runtime_bound())
    assert manager.runtime_bound.runtime_identity == "fake-runtime-g3"

    manager.close()
    assert manager.state is OwnershipState.CLOSED
    manager.close()
    with pytest.raises(ComponentLifecycleError, match="UNLOADED"):
        manager.load()


def test_resource_plan_covers_every_execution_mode() -> None:
    plan = ResourcePlan.default()
    assert plan.roles_for(ExecutionMode.PREPROCESS) == (ComponentRole.PREPROCESS,)
    assert ComponentRole.TRAINABLE in plan.roles_for(ExecutionMode.TRAIN)
    with pytest.raises(ComponentManagerError, match="must cover"):
        ResourcePlan(
            mode_roles=((ExecutionMode.PREPROCESS, (ComponentRole.PREPROCESS,)),)
        )


def test_modes_are_reentrant_nonconflicting_and_restore_residency() -> None:
    manager = _prepared_manager()
    assert manager.mode is ExecutionMode.IDLE

    with manager.preprocess():
        assert manager.mode is ExecutionMode.PREPROCESS
        assert manager.residency("transformer") is Residency.PREPARED
        assert manager.residency("reference") is Residency.PREPARED
        assert manager.residency("vae") is Residency.RESIDENT

    with manager.rollout():
        assert manager.mode is ExecutionMode.ROLLOUT
        assert manager.residency("transformer") is Residency.PREPARED
        assert manager.residency("reference") is Residency.PREPARED
        assert manager.residency("vae") is Residency.RESIDENT
        with manager.rollout():
            assert manager.mode is ExecutionMode.ROLLOUT
        with pytest.raises(
            ComponentLifecycleError,
            match="while rollout is active",
        ), manager.train():
            pass
        with pytest.raises(ComponentLifecycleError, match="execution mode is active"):
            manager.close()

    assert manager.mode is ExecutionMode.IDLE
    assert manager.residency("transformer") is Residency.PREPARED
    assert manager.residency("vae") is Residency.OFFLOADED

    with manager.train():
        assert manager.mode is ExecutionMode.TRAIN
        assert manager.residency("transformer") is Residency.PREPARED
        assert manager.residency("reference") is Residency.PREPARED
        assert manager.residency("vae") is Residency.OFFLOADED

    with manager.evaluate():
        assert manager.mode is ExecutionMode.EVAL
        assert manager.residency("transformer") is Residency.PREPARED
        assert manager.residency("reference") is Residency.PREPARED
        assert manager.residency("vae") is Residency.RESIDENT

    manager.close()


def test_partial_load_failure_closes_acquired_components_in_reverse_order() -> None:
    events: list[tuple[str, str, str | None]] = []
    manager = ComponentManager(
        _FailingAdapter(events),
        execution_device="cpu",
        offload_device="cpu",
    )
    with pytest.raises(RuntimeError, match="third component"):
        manager.load()

    assert manager.state is OwnershipState.UNLOADED
    assert events == [
        ("close", "second", None),
        ("close", "first", None),
    ]


def test_failed_registration_closes_unbound_then_rolls_back_owned_object() -> None:
    events: list[tuple[str, str, str | None]] = []
    manager = ComponentManager(
        _InvalidRegistrationAdapter(events),
        execution_device="cpu",
    )
    with pytest.raises(ComponentManagerError, match="duplicate component name"):
        manager.load()
    assert events == [
        ("close", "unbound", None),
        ("close", "owned", None),
    ]


def test_close_uses_reverse_ownership_order() -> None:
    events: list[tuple[str, str, str | None]] = []
    manager = _prepared_manager(_OwnedAdapter(events))
    manager.close()
    closed = [name for operation, name, _device in events if operation == "close"]
    assert closed == ["vae", "reference", "transformer", "tokenizer"]


def test_model_components_reject_duplicate_object_or_conflicting_role() -> None:
    shared = object()
    with pytest.raises(ComponentManagerError, match="multiple ownership"):
        ModelComponents(
            (
                ComponentBinding(
                    "first",
                    shared,
                    (ComponentRole.PREPROCESS,),
                    managed_residency=False,
                ),
                ComponentBinding(
                    "second",
                    shared,
                    (ComponentRole.DECODER,),
                    managed_residency=False,
                ),
            )
        )
    with pytest.raises(ComponentManagerError, match="frozen reference"):
        ComponentBinding(
            "invalid",
            object(),
            (ComponentRole.TRAINABLE, ComponentRole.REFERENCE),
            managed_residency=False,
        )
