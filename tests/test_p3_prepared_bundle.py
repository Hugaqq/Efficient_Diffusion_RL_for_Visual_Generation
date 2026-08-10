"""Single-root distributed preparation and prepared-forward routing tests."""

from __future__ import annotations

import gc
import weakref
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
    ComponentRole,
    ModelAdapter,
    ModelComponents,
    ModelInput,
    ModelLatentSpec,
    ModelPortError,
    ModelPrediction,
    OwnershipState,
    Residency,
    ResourceOwner,
)
from visual_rl.models.lifecycle.prepared import (
    PreparedBundleError,
    PreparedComponentHandle,
    PreparedModelBundle,
)


def _declared() -> DeclaredContract:
    return DeclaredContract(
        component_kind="model",
        component_id="prepared-fake-v1",
        model=ModelContract(
            tasks=(TaskKind.T2I,),
            output_media=(MediaKind.IMAGE,),
            latent_layouts=(LatentLayout.PACKED_SEQUENCE,),
            latent_ranks=(3,),
            axis_semantics=(("batch", "sequence", "channel"),),
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
            artifact_identity="prepared-fake-artifact",
            resolved_fields=(),
        ),
        runtime_identity="prepared-fake-runtime",
        verified_fields=(
            ("model.prepared_root", "verified"),
            ("model.reference_forward", "verified"),
        ),
    )


class _GuardedModule(torch.nn.Module):
    def __init__(self, name, events, *, trainable):
        super().__init__()
        self.name = name
        self.events = events
        self.linear = torch.nn.Linear(2, 2)
        self.linear.requires_grad_(trainable)
        self._through_prepared_root = False
        self.direct_calls = 0
        self.prepared_calls = 0
        self.move_calls = 0
        self._no_split_modules = [f"{name.title()}Block", "SharedBlock"]

    def forward(self, value):
        if self._through_prepared_root:
            self.prepared_calls += 1
            self.events.append(("raw-prepared", self.name))
        else:
            self.direct_calls += 1
            self.events.append(("raw-direct", self.name))
        return self.linear(value)

    def to(self, device):
        self.move_calls += 1
        self.events.append(("move", self.name))
        return self


class _StaticDecoder:
    def __init__(self, events):
        self.events = events
        self.move_calls = 0

    def to(self, device):
        del device
        self.move_calls += 1
        self.events.append(("move", "decoder"))
        return self


class _RoutingAdapter(ModelAdapter):
    def __init__(self, events=None):
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
        self.raw_policy = session.acquire(
            "policy",
            lambda: _GuardedModule(
                "policy",
                self.events,
                trainable=True,
            ),
            roles=(ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
        )
        self.raw_reference = session.acquire(
            "reference",
            lambda: _GuardedModule(
                "reference",
                self.events,
                trainable=False,
            ),
            roles=(ComponentRole.REFERENCE,),
        )
        self.decoder = session.acquire(
            "decoder",
            lambda: _StaticDecoder(self.events),
            roles=(ComponentRole.DECODER,),
        )
        return session.freeze()

    def encode(self, batch):
        return batch

    def prepare_latents(self, latent_spec, *, generator):
        return torch.randn(
            latent_spec.shape,
            generator=generator,
            device=latent_spec.device,
            dtype=latent_spec.dtype,
        )

    def predict(self, model_input):
        value = self._forward_prepared("policy", model_input.latents)
        return ModelPrediction(
            value=value,
            prediction_type=PredictionType.FLOW,
            condition_identity=model_input.condition_identity,
            guidance_identity=model_input.guidance_identity,
        )

    def predict_reference(self, model_input):
        value = self._forward_prepared("reference", model_input.latents)
        return ModelPrediction(
            value=value,
            prediction_type=PredictionType.FLOW,
            condition_identity=model_input.condition_identity,
            guidance_identity=model_input.guidance_identity,
        )

    def decode(self, latents, latent_spec):
        del latent_spec
        return latents


class _PreparedRoot(torch.nn.Module):
    def __init__(self, bundle, events):
        super().__init__()
        self.bundle = bundle
        self.events = events

    def forward(self, component_name, *args, **kwargs):
        self.events.append(("prepared-root", component_name))
        component = self.bundle.component(component_name)
        component._through_prepared_root = True
        try:
            return self.bundle(component_name, *args, **kwargs)
        finally:
            component._through_prepared_root = False


class _FakeAccelerator:
    def __init__(self, events, *, fail=False, invalid_root=False):
        self.events = events
        self.fail = fail
        self.invalid_root = invalid_root
        self.prepare_calls = []
        self.accumulation_roots = []

    def prepare(self, *items):
        self.prepare_calls.append(items)
        if self.fail:
            raise RuntimeError("fake prepare failure")
        if self.invalid_root:
            return (object(), *items[1:])
        return (_PreparedRoot(items[0], self.events), *items[1:])

    @contextmanager
    def accumulate(self, root):
        self.accumulation_roots.append(root)
        yield


class _WeakResource:
    pass


def _input() -> ModelInput:
    spec = ModelLatentSpec(
        shape=(2, 1, 2),
        layout=LatentLayout.PACKED_SEQUENCE,
        axis_semantics=("batch", "sequence", "channel"),
        device="cpu",
        dtype=torch.float32,
    )
    return ModelInput(
        latents=torch.tensor([[[1.0, 2.0]], [[-1.0, 0.5]]]),
        timestep=torch.tensor([900.5, 400.25]),
        conditioning=None,
        guidance=None,
        latent_spec=spec,
        condition_identity=("prompt:a", "prompt:b"),
        guidance_identity=("none", "none"),
    )


def _configured_manager(events=None):
    adapter = _RoutingAdapter(events)
    manager = ComponentManager(
        adapter,
        execution_device="cpu",
        offload_device="cpu",
    )
    manager.load()
    manager.configure()
    return adapter, manager


def test_one_bundle_one_prepare_call_and_prepared_forward_only() -> None:
    events = []
    adapter, manager = _configured_manager(events)
    optimizer = torch.optim.AdamW(manager.parameter_state.parameters(), lr=1e-4)
    scheduler = object()
    accelerator = _FakeAccelerator(events)

    handle = manager.prepare(
        accelerator=accelerator,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert len(accelerator.prepare_calls) == 1
    prepared_arguments = accelerator.prepare_calls[0]
    assert len(prepared_arguments) == 3
    assert isinstance(prepared_arguments[0], PreparedModelBundle)
    assert handle.optimizer is optimizer
    assert handle.scheduler is scheduler
    assert handle.component_names == ("policy", "reference")
    assert handle.accumulation_root is handle.prepared_root
    assert handle.prepared_root is not prepared_arguments[0]
    assert prepared_arguments[0]._no_split_modules == [
        "PolicyBlock",
        "SharedBlock",
        "ReferenceBlock",
    ]

    manager.bind_runtime(_runtime_bound())
    policy_moves_before = adapter.raw_policy.move_calls
    with manager.train(), handle.accumulate() as accumulation_root:
        assert accumulation_root is handle.prepared_root
        prediction = adapter.predict(_input())
    prediction.validate_against(_input())

    assert adapter.raw_policy.direct_calls == 0
    assert adapter.raw_policy.prepared_calls == 1
    assert events[-2:] == [
        ("prepared-root", "policy"),
        ("raw-prepared", "policy"),
    ]
    assert accelerator.accumulation_roots == [handle.prepared_root]
    assert manager.residency("policy") is Residency.PREPARED
    assert manager.residency("reference") is Residency.PREPARED
    assert adapter.raw_policy.move_calls == policy_moves_before
    assert (
        manager.resource_plan.owner_for(
            manager.components.binding("policy"),
            prepared_component_names=frozenset(handle.component_names),
        )
        is ResourceOwner.PREPARED_BACKEND
    )
    with pytest.raises(PreparedBundleError, match="bypassed"):
        adapter.raw_policy(_input().latents)
    assert adapter.raw_policy.direct_calls == 0

    with pytest.raises(ComponentLifecycleError, match="exactly one"):
        manager.prepare(accelerator=accelerator, optimizer=optimizer)
    assert len(accelerator.prepare_calls) == 1

    manager.close()
    assert adapter.raw_policy.move_calls == policy_moves_before
    with pytest.raises(PreparedBundleError, match="closed"):
        handle.forward("policy", _input().latents)
    adapter.raw_policy(_input().latents)
    assert adapter.raw_policy.direct_calls == 1


def test_optimizer_must_exactly_match_requires_grad_subset_before_prepare() -> None:
    events = []
    adapter, manager = _configured_manager(events)
    parameters = manager.parameter_state.parameters()
    incomplete = torch.optim.SGD(parameters[:1], lr=0.1)
    accelerator = _FakeAccelerator(events)

    with pytest.raises(PreparedBundleError, match="exactly equal"):
        manager.prepare(accelerator=accelerator, optimizer=incomplete)
    assert accelerator.prepare_calls == []
    assert manager.state is OwnershipState.CONFIGURED
    assert isinstance(manager.prepare_error, PreparedBundleError)
    assert manager.residency("policy") is Residency.OFFLOADED
    adapter.raw_policy(_input().latents)
    assert adapter.raw_policy.direct_calls == 1
    with pytest.raises(ModelPortError, match="not been prepared"):
        _ = adapter.prepared_components
    with pytest.raises(ComponentLifecycleError, match="exactly one"):
        manager.prepare(
            accelerator=_FakeAccelerator(events),
            optimizer=torch.optim.SGD(parameters, lr=0.1),
        )
    manager.close()


def test_accelerator_prepare_failure_leaves_no_handle_and_clear_state() -> None:
    events = []
    adapter, manager = _configured_manager(events)
    optimizer = torch.optim.SGD(manager.parameter_state.parameters(), lr=0.1)
    accelerator = _FakeAccelerator(events, fail=True)

    with pytest.raises(RuntimeError, match="fake prepare failure"):
        manager.prepare(accelerator=accelerator, optimizer=optimizer)
    assert len(accelerator.prepare_calls) == 1
    assert manager.state is OwnershipState.CONFIGURED
    assert isinstance(manager.prepare_error, RuntimeError)
    with pytest.raises(ComponentLifecycleError, match="not prepared"):
        _ = manager.prepared_handle
    with pytest.raises(ModelPortError, match="not been prepared"):
        _ = adapter.prepared_components
    assert manager.residency("policy") is Residency.OFFLOADED
    manager.close()


def test_invalid_prepared_root_releases_forward_guard_during_rollback() -> None:
    events = []
    adapter, manager = _configured_manager(events)
    optimizer = torch.optim.SGD(manager.parameter_state.parameters(), lr=0.1)
    accelerator = _FakeAccelerator(events, invalid_root=True)

    with pytest.raises(TypeError, match="prepared_root must be torch.nn.Module"):
        manager.prepare(accelerator=accelerator, optimizer=optimizer)
    assert len(accelerator.prepare_calls) == 1
    assert manager.state is OwnershipState.CONFIGURED
    assert isinstance(manager.prepare_error, TypeError)
    with pytest.raises(ComponentLifecycleError, match="not prepared"):
        _ = manager.prepared_handle
    adapter.raw_policy(_input().latents)
    assert adapter.raw_policy.direct_calls == 1
    manager.close()


def test_closed_prepared_handle_rejects_access_and_releases_owned_refs() -> None:
    events = []
    component = _GuardedModule("policy", events, trainable=True)
    components = ModelComponents(
        (
            ComponentBinding(
                "policy",
                component,
                (ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
            ),
        )
    )
    source_bundle = PreparedModelBundle(components)
    prepared_root = _PreparedRoot(source_bundle, events)
    optimizer = torch.optim.SGD(source_bundle.trainable_parameters(), lr=0.1)
    scheduler = _WeakResource()
    accelerator = _FakeAccelerator(events)
    handle = PreparedComponentHandle(
        source_bundle=source_bundle,
        prepared_root=prepared_root,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
    )
    resource_refs = tuple(
        weakref.ref(resource)
        for resource in (
            source_bundle,
            prepared_root,
            optimizer,
            scheduler,
            accelerator,
        )
    )
    del source_bundle, prepared_root, optimizer, scheduler, accelerator

    handle.close()
    handle.close()

    assert handle._source_bundle is None
    assert handle._prepared_root is None
    assert handle._optimizer is None
    assert handle._scheduler is None
    assert handle._accelerator is None
    gc.collect()
    assert all(resource_ref() is None for resource_ref in resource_refs)

    for access in (
        lambda: handle.component_names,
        lambda: handle.prepared_root,
        lambda: handle.optimizer,
        lambda: handle.scheduler,
        lambda: handle.accumulation_root,
        lambda: handle.owns("policy"),
        lambda: handle.forward("policy", torch.ones(1, 2)),
    ):
        with pytest.raises(PreparedBundleError, match="closed"):
            access()
    with pytest.raises(PreparedBundleError, match="closed"), handle.accumulate():
        pass


def test_bundle_rejects_tied_parameter_with_multiple_registered_names() -> None:
    shared = torch.nn.Parameter(torch.ones(2))
    tied = torch.nn.Module()
    tied.register_parameter("first", shared)
    tied.register_parameter("second", shared)
    components = ModelComponents(
        (
            ComponentBinding(
                "tied",
                tied,
                (ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
            ),
        )
    )
    with pytest.raises(PreparedBundleError, match="multiple prepared paths"):
        PreparedModelBundle(components)
