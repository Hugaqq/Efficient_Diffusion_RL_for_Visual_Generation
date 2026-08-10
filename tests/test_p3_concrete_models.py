"""Lightweight SD3/Wan model-port and catalog tests with no model download."""

from __future__ import annotations

import gc
import hashlib
import subprocess
import sys
import weakref
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from visual_rl.algorithms.dynamics.interface import Dynamics
from visual_rl.algorithms.rollout.interface import RolloutRequest
from visual_rl.composition.config.compiler import compile_recipe_v2, default_catalog
from visual_rl.composition.config.source import load_source_recipe
from visual_rl.composition.recipes.builtins import builtin_recipe_definitions
from visual_rl.composition.registry import (
    DeclarationResolver,
    RegistryError,
    build_catalog,
)
from visual_rl.core.contracts import (
    ComponentArtifactBindingSet,
    ComponentLoadPlan,
    ComputePrecision,
    LatentLayout,
    LikelihoodSemantics,
    MediaKind,
    PredictionType,
    TaskKind,
    TimeCoordinate,
)
from visual_rl.data.media import DecodedMediaBatch
from visual_rl.models import (
    BatchProjectableModelPayload,
    BatchRowProjection,
    ComponentLifecycleError,
    ComponentLoadSession,
    ComponentManager,
    ComponentRole,
    ModelAdapter,
    ModelInput,
    ModelPortError,
    ModelScheduleContext,
    SchedulerArtifactBlueprint,
)
from visual_rl.models.catalog import model_catalog_fragment
from visual_rl.models.implementations.sd3 import (
    SD3Adapter,
    SD3Conditioning,
    SD3Config,
    SD3RuntimeParts,
)
from visual_rl.models.implementations.wan import (
    WanConditioning,
    WanConfig,
    WanRuntimeParts,
    WanT2VAdapter,
)
from visual_rl.models.lifecycle.prepared import PreparedBundleError
from visual_rl.composition.preflight import (
    RuntimeBindResult,
    RuntimeFacts,
    runtime_launch_payload_id,
)
from visual_rl.runtime.component_loader import (
    RuntimeComponentLoader,
    RuntimeComponentLoadGate,
    build_component_artifact_binding,
)
from visual_rl.data.samples import (
    BatchRowContext,
    ExplicitCollator,
    SourceItemContext,
    T2IItem,
    T2VItem,
)

MODEL_CATALOG = build_catalog((model_catalog_fragment(),))
MODEL_DECLARATIONS = MODEL_CATALOG.for_kind("model")
_FIXTURE_RECIPE_ID = (
    "materialized-recipe.v2:" + hashlib.sha256(b"p3-concrete-model-fixture").hexdigest()
)
_FIXTURE_CODE_ID = hashlib.sha256(b"p3-concrete-model-code").hexdigest()
_FIXTURE_RUNTIME_ID = hashlib.sha256(b"p3-concrete-model-runtime").hexdigest()


class _PromptEncoder:
    def __init__(self, family: str) -> None:
        self.family = family
        self.move_calls = 0
        self.closed = False

    def to(self, device):
        del device
        self.move_calls += 1
        return self

    def encode(self, prompts, max_sequence_length, guidance_scale):
        del max_sequence_length
        batch_size = len(prompts)
        positive = torch.full((batch_size, 3, 2), 0.75)
        negative = torch.full((batch_size, 3, 2), -0.25)
        if self.family == "sd3":
            pooled = torch.full((batch_size, 2), 0.5)
            negative_pooled = torch.full((batch_size, 2), -0.5)
            return positive, negative, pooled, negative_pooled
        return positive, negative if guidance_scale > 1.0 else None

    def close(self):
        self.closed = True


class _Decoder:
    def __init__(self, family: str) -> None:
        self.family = family
        self.move_calls = 0
        self.closed = False

    def to(self, device):
        del device
        self.move_calls += 1
        return self

    def decode(self, latents, latent_spec):
        assert tuple(latents.shape) == latent_spec.shape
        if self.family == "sd3":
            return latents[:, :3].detach().clone()
        return latents[:, :3].permute(0, 2, 1, 3, 4).detach().clone()

    def close(self):
        self.closed = True


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.base_scale = torch.nn.Parameter(
            torch.tensor(0.5),
            requires_grad=False,
        )
        self.frozen_bias = torch.nn.Parameter(
            torch.tensor(0.125),
            requires_grad=False,
        )
        self.forward_calls = 0
        self.cache_context_calls = []
        self.forward_cache_contexts = []
        self._active_cache_context = None
        self.move_calls = 0
        self._adapter_disabled = 0
        self._no_split_modules = ["FakeTransformerBlock"]

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict,
        pooled_projections=None,
        attention_kwargs=None,
    ):
        del timestep, return_dict, pooled_projections, attention_kwargs
        self.forward_calls += 1
        self.forward_cache_contexts.append(self._active_cache_context)
        batch_values = encoder_hidden_states.mean(
            dim=tuple(range(1, encoder_hidden_states.ndim))
        )
        batch_values = batch_values.reshape(
            hidden_states.shape[0],
            *([1] * (hidden_states.ndim - 1)),
        )
        scale = self.base_scale
        if self._adapter_disabled == 0:
            scale = scale + self.scale
        return (hidden_states * scale + self.frozen_bias + batch_values,)

    def to(self, *args, **kwargs):
        self.move_calls += 1
        return super().to(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        self._adapter_disabled += 1
        try:
            yield
        finally:
            self._adapter_disabled -= 1

    @contextmanager
    def cache_context(self, name):
        assert self._active_cache_context is None
        self.cache_context_calls.append(name)
        self._active_cache_context = name
        try:
            yield
        finally:
            self._active_cache_context = None


class _Scheduler:
    def __init__(self, config=None) -> None:
        self.config = {} if config is None else dict(config)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps, device) -> None:
        self.num_inference_steps = num_inference_steps
        self.device = device


class _RequestOnlyDynamics(Dynamics):
    def timesteps(self, *, num_steps, device):
        return torch.linspace(1.0, 0.25, num_steps, device=device)

    def terminal_timestep(self, *, device):
        return torch.tensor(0.0, device=device)

    def add_noise(self, clean, noise, timestep):
        raise AssertionError("request construction must not execute dynamics")

    def transition_mean_std(self, transition):
        raise AssertionError("request construction must not execute dynamics")


class _FakeModelLoader:
    def __init__(self) -> None:
        self.calls = []
        self.transformers = []

    def __call__(self, family, artifact_path, config, precision):
        self.calls.append((family, artifact_path, config, precision))
        transformer = _Transformer()
        self.transformers.append(transformer)
        if family == "sd3":
            return SD3RuntimeParts(
                prompt_encoder=_PromptEncoder("sd3"),
                transformer=transformer,
                decoder=_Decoder("sd3"),
                reference_context=transformer.disable_adapter,
                latent_channels=4,
                scheduler_artifact_blueprint=(
                    SchedulerArtifactBlueprint.from_scheduler(_Scheduler())
                ),
                transformer_patch_size=1,
            )
        if family == "wan-t2v":
            return WanRuntimeParts(
                prompt_encoder=_PromptEncoder("wan"),
                transformer=transformer,
                decoder=_Decoder("wan"),
                latent_channels=4,
                scheduler_artifact_blueprint=(
                    SchedulerArtifactBlueprint.from_scheduler(_Scheduler())
                ),
                expand_timesteps=True,
            )
        raise AssertionError(family)


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


def _batch(task: str, *, batch_size: int = 2):
    item_type = T2IItem if task == "t2i" else T2VItem
    items = tuple(
        item_type(
            prompt=f"prompt {index}",
            source=SourceItemContext(
                source_item_id=f"source-{task}-{index}",
                dataset_source_id="main",
                dataset_index=index,
                dataset_revision="revision-1",
            ),
        )
        for index in range(batch_size)
    )
    rows = tuple(
        BatchRowContext(
            occurrence_id=f"occurrence-{task}-{index}",
            group_id=f"group-{task}",
            member_id=index,
            phase="main",
            optimizer_step=0,
            source_item_id=item.source.source_item_id,
        )
        for index, item in enumerate(items)
    )
    return ExplicitCollator().collate_samples(items, rows)


def _runtime_binding() -> RuntimeBindResult:
    runtime_facts = RuntimeFacts(
        distribution_mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        device="cpu",
        precision="fp32",
        backend=None,
    )
    return RuntimeBindResult(
        recipe_id=_FIXTURE_RECIPE_ID,
        launch_id=runtime_launch_payload_id(_FIXTURE_RECIPE_ID, runtime_facts),
        runtime_facts=runtime_facts,
    )


def _load_declaration(tmp_path, declaration):
    loader = _FakeModelLoader()
    model_artifact_id = hashlib.sha256(
        f"{declaration.alias}-fake-model-artifact".encode()
    ).hexdigest()
    binding = build_component_artifact_binding(
        declaration,
        recipe_id=_FIXTURE_RECIPE_ID,
        slot="model",
        artifact_content_identities={
            "code": _FIXTURE_CODE_ID,
            "model": model_artifact_id,
        },
        code_identity=_FIXTURE_CODE_ID,
    )
    binding_set = ComponentArtifactBindingSet(binding.recipe_id, (binding,))
    load_plan = ComponentLoadPlan.create(
        binding_set,
        required_artifact_names_by_slot={"model": ("code", "model")},
    )
    loaded = RuntimeComponentLoader().load(
        declaration,
        gate=RuntimeComponentLoadGate(
            runtime_binding=_runtime_binding(),
            artifact_binding=binding,
        ),
        binding_set=binding_set,
        load_plan=load_plan,
        runtime_context={
            "precision": "fp32",
            "model_artifacts": {"main": tmp_path},
            "model_loader": loader,
        },
    )
    adapter = loaded.instance
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("canonical model loader did not return ModelAdapter")
    return adapter, loader, loaded


def _load_adapter(tmp_path, alias, params):
    declaration = DeclarationResolver().resolve(
        MODEL_DECLARATIONS,
        alias,
        params,
    )
    adapter, loader, loaded = _load_declaration(tmp_path, declaration)
    return declaration, adapter, loader, loaded


def _prepare_manager(adapter, loaded):
    manager = ComponentManager(
        adapter,
        execution_device="cpu",
        offload_device="cpu",
    )
    components = manager.load()
    manager.configure()
    optimizer = torch.optim.SGD(manager.parameter_state.parameters(), lr=0.1)
    accelerator = _IdentityAccelerator()
    handle = manager.prepare(accelerator=accelerator, optimizer=optimizer)
    declared = adapter.describe(adapter.config)
    verified_fields = [("model.component_topology", "verified")]
    if declared.model.provides_reference_policy is True:
        verified_fields.append(("model.reference_forward", "verified"))
    manager.bind_runtime(
        loaded.attest_prepared(
            runtime_identity=_FIXTURE_RUNTIME_ID,
            verified_fields=tuple(verified_fields),
        )
    )
    return manager, components, optimizer, accelerator, handle


def test_sd3_components_exact_parameters_and_typed_forward(tmp_path) -> None:
    resolved, adapter, loader, loaded = _load_adapter(
        tmp_path,
        "sd3",
        {"artifact_ref": "main", "resolution": 16},
    )
    assert isinstance(resolved.config, SD3Config)
    assert isinstance(adapter, SD3Adapter)
    manager, components, optimizer, accelerator, handle = _prepare_manager(
        adapter,
        loaded,
    )
    assert tuple(item.name for item in components.bindings) == (
        "prompt_encoder",
        "transformer",
        "decoder",
    )
    assert components.binding("prompt_encoder").roles == (ComponentRole.PREPROCESS,)
    assert components.binding("prompt_encoder").managed_residency is False
    prompt_encoder = components.binding("prompt_encoder").component
    assert components.binding("transformer").roles == (
        ComponentRole.INFERENCE,
        ComponentRole.TRAINABLE,
    )
    assert components.binding("decoder").roles == (ComponentRole.DECODER,)
    transformer = loader.transformers[0]
    assert manager.parameter_state.parameters() == (transformer.scale,)
    assert optimizer.param_groups[0]["params"] == [transformer.scale]
    assert len(accelerator.prepare_calls) == 1
    assert handle.component_names == ("transformer",)

    batch = _batch("t2i")
    with manager.preprocess():
        conditioning = adapter.encode(batch)
    assert prompt_encoder.move_calls == 0
    assert isinstance(conditioning, SD3Conditioning)
    assert isinstance(conditioning, BatchProjectableModelPayload)
    selected_conditioning = conditioning.project_rows(
        BatchRowProjection(conditioning.batch_size, (1,))
    )
    assert selected_conditioning.condition_identity == (
        conditioning.condition_identity[1],
    )
    torch.testing.assert_close(
        selected_conditioning.prompt_embeds,
        conditioning.prompt_embeds[1:2],
    )
    spec = adapter.latent_spec_for_batch(
        batch,
        device="cpu",
        dtype=torch.float32,
    )
    assert isinstance(spec, ModelScheduleContext)
    assert spec.spatial_stride == (8, 8)
    assert spec.temporal_stride is None
    assert spec.scheduler_patch_size == 1
    assert adapter.model_schedule_context(spec) is spec
    assert isinstance(
        adapter.scheduler_artifact_blueprint,
        SchedulerArtifactBlueprint,
    )
    latents = adapter.prepare_latents(
        spec,
        generator=torch.Generator().manual_seed(7),
    )
    model_input = ModelInput(
        latents=latents,
        timestep=torch.tensor([0.8, 0.4]),
        conditioning=conditioning,
        guidance=None,
        latent_spec=spec,
        condition_identity=conditioning.condition_identity,
        guidance_identity=("cfg:4.5", "cfg:4.5"),
    )
    with manager.rollout():
        prediction = adapter.predict(model_input)
        reference_before = adapter.predict_reference(model_input)
        with torch.no_grad():
            transformer.scale.add_(1.0)
        reference_after = adapter.predict_reference(model_input)
        changed_policy = adapter.predict(model_input)
    prediction.validate_against(model_input)
    reference_before.validate_against(model_input)
    torch.testing.assert_close(reference_before.value, reference_after.value)
    assert not torch.equal(prediction.value, changed_policy.value)
    assert transformer.forward_calls == 4
    with manager.evaluate():
        media = adapter.decode(prediction.value, spec)
    assert isinstance(media, DecodedMediaBatch)
    assert media.layout == "BCHW"
    assert tuple(media.tensor.shape) == (2, 3, 2, 2)
    with pytest.raises(PreparedBundleError, match="bypassed"):
        transformer(
            hidden_states=latents,
            timestep=model_input.timestep,
            encoder_hidden_states=conditioning.prompt_embeds,
            pooled_projections=conditioning.pooled_prompt_embeds,
            return_dict=False,
        )
    manager.close()


def test_wan_components_exact_parameters_and_typed_forward(tmp_path) -> None:
    resolved, adapter, loader, loaded = _load_adapter(
        tmp_path,
        "wan-t2v",
        {
            "artifact_ref": "main",
            "height": 16,
            "width": 16,
            "frames": 5,
        },
    )
    assert isinstance(resolved.config, WanConfig)
    assert isinstance(adapter, WanT2VAdapter)
    manager, components, optimizer, accelerator, handle = _prepare_manager(
        adapter,
        loaded,
    )
    transformer = loader.transformers[0]
    assert manager.parameter_state.parameters() == (transformer.scale,)
    assert optimizer.param_groups[0]["params"] == [transformer.scale]
    assert len(accelerator.prepare_calls) == 1
    assert handle.component_names == ("transformer",)
    assert components.binding("prompt_encoder").roles == (ComponentRole.PREPROCESS,)
    assert components.binding("prompt_encoder").managed_residency is False
    prompt_encoder = components.binding("prompt_encoder").component
    assert components.binding("transformer").roles == (
        ComponentRole.INFERENCE,
        ComponentRole.TRAINABLE,
    )

    batch = _batch("t2v")
    with manager.preprocess():
        conditioning = adapter.encode(batch)
    assert prompt_encoder.move_calls == 0
    assert isinstance(conditioning, WanConditioning)
    assert isinstance(conditioning, BatchProjectableModelPayload)
    selected_conditioning = conditioning.project_rows(
        BatchRowProjection(conditioning.batch_size, (1,))
    )
    assert selected_conditioning.condition_identity == (
        conditioning.condition_identity[1],
    )
    torch.testing.assert_close(
        selected_conditioning.prompt_embeds,
        conditioning.prompt_embeds[1:2],
    )
    spec = adapter.latent_spec_for_batch(
        batch,
        device="cpu",
        dtype=torch.float32,
    )
    assert isinstance(spec, ModelScheduleContext)
    assert spec.spatial_stride == (8, 8)
    assert spec.temporal_stride == 4
    assert spec.scheduler_patch_size is None
    assert adapter.model_schedule_context(spec) is spec
    assert isinstance(
        adapter.scheduler_artifact_blueprint,
        SchedulerArtifactBlueprint,
    )
    latents = adapter.prepare_latents(
        spec,
        generator=torch.Generator().manual_seed(11),
    )
    model_input = ModelInput(
        latents=latents,
        timestep=torch.tensor([0.75, 0.25]),
        conditioning=conditioning,
        guidance=None,
        latent_spec=spec,
        condition_identity=conditioning.condition_identity,
        guidance_identity=("cfg:5", "cfg:5"),
    )
    with manager.rollout():
        prediction = adapter.predict(model_input)
    prediction.validate_against(model_input)
    assert transformer.forward_calls == 2
    assert transformer.cache_context_calls == ["cond", "uncond"]
    assert transformer.forward_cache_contexts == ["cond", "uncond"]
    transformer.cache_context = None
    with manager.rollout():
        adapter.predict(model_input)
    assert transformer.forward_calls == 4
    assert transformer.cache_context_calls == ["cond", "uncond"]
    assert transformer.forward_cache_contexts == ["cond", "uncond", None, None]
    with manager.evaluate():
        media = adapter.decode(prediction.value, spec)
    assert isinstance(media, DecodedMediaBatch)
    assert media.layout == "BFCHW"
    assert tuple(media.tensor.shape) == (2, 2, 3, 2, 2)
    with pytest.raises(ModelPortError, match="explicitly provides no reference"):
        adapter.predict_reference(model_input)
    manager.close()


def test_wan_non_cfg_forward_uses_only_cond_cache_context(tmp_path) -> None:
    _resolved, adapter, loader, loaded = _load_adapter(
        tmp_path,
        "wan-t2v",
        {
            "artifact_ref": "main",
            "guidance_scale": 1.0,
            "height": 16,
            "width": 16,
            "frames": 5,
        },
    )
    manager, _components, _optimizer, _accelerator, _handle = _prepare_manager(
        adapter,
        loaded,
    )
    transformer = loader.transformers[0]
    batch = _batch("t2v")
    with manager.preprocess():
        conditioning = adapter.encode(batch)
    assert conditioning.negative_prompt_embeds is None
    spec = adapter.latent_spec_for_batch(
        batch,
        device="cpu",
        dtype=torch.float32,
    )
    latents = adapter.prepare_latents(
        spec,
        generator=torch.Generator().manual_seed(13),
    )
    model_input = ModelInput(
        latents=latents,
        timestep=torch.tensor([0.75, 0.25]),
        conditioning=conditioning,
        guidance=None,
        latent_spec=spec,
        condition_identity=conditioning.condition_identity,
        guidance_identity=("cfg:1", "cfg:1"),
    )

    with manager.rollout():
        adapter.predict(model_input)

    assert transformer.forward_calls == 1
    assert transformer.cache_context_calls == ["cond"]
    assert transformer.forward_cache_contexts == ["cond"]
    manager.close()


@pytest.mark.parametrize(
    ("alias", "params"),
    (
        ("sd3", {"artifact_ref": "main", "resolution": 16}),
        (
            "wan-t2v",
            {
                "artifact_ref": "main",
                "height": 16,
                "width": 16,
                "frames": 5,
            },
        ),
    ),
)
def test_concrete_adapter_close_releases_loaded_runtime_parts(
    tmp_path,
    alias,
    params,
) -> None:
    _resolved, adapter, loader, _loaded = _load_adapter(tmp_path, alias, params)
    manager = ComponentManager(
        adapter,
        execution_device="cpu",
        offload_device="cpu",
    )
    components = manager.load()
    prompt_encoder = components.binding("prompt_encoder").component
    transformer = components.binding("transformer").component
    decoder = components.binding("decoder").component
    resource_refs = tuple(
        weakref.ref(resource) for resource in (prompt_encoder, transformer, decoder)
    )

    manager.close()
    assert prompt_encoder.closed is True
    assert decoder.closed is True
    if isinstance(adapter, SD3Adapter):
        assert adapter._reference_context is None

    adapter.close()
    adapter.close()
    assert adapter.closed is True
    assert adapter._prompt_encoder is None
    assert adapter._decoder is None
    if isinstance(adapter, WanT2VAdapter):
        assert adapter._transformer is None
    assert adapter._model_loader is None
    assert adapter._scheduler_artifact_blueprint is None
    with pytest.raises(ModelPortError, match="closed"):
        adapter.load_components(ComponentLoadSession())
    with pytest.raises(ModelPortError, match="closed"):
        _ = adapter.scheduler_artifact_blueprint

    loader.transformers.clear()
    del components, prompt_encoder, transformer, decoder
    gc.collect()
    assert all(resource_ref() is None for resource_ref in resource_refs)


@pytest.mark.parametrize(
    (
        "alias",
        "params",
        "task",
        "wrong_task",
        "expected_layout",
        "expected_shape",
    ),
    (
        (
            "sd3",
            {"artifact_ref": "main", "resolution": 24},
            "t2i",
            "t2v",
            LatentLayout.BCHW,
            (3, 4, 3, 3),
        ),
        (
            "wan-t2v",
            {
                "artifact_ref": "main",
                "height": 24,
                "width": 32,
                "frames": 9,
            },
            "t2v",
            "t2i",
            LatentLayout.BCTHW,
            (3, 4, 3, 3, 4),
        ),
    ),
)
def test_model_latent_port_uses_loaded_artifact_channels_without_alias_branch(
    tmp_path,
    alias,
    params,
    task,
    wrong_task,
    expected_layout,
    expected_shape,
) -> None:
    _resolved, adapter, _loader, loaded = _load_adapter(tmp_path, alias, params)
    batch = _batch(task, batch_size=3)

    with pytest.raises(ModelPortError, match="must be loaded"):
        adapter.latent_spec_for_batch(
            batch,
            device="cpu",
            dtype=torch.float16,
        )

    manager, _components, _optimizer, _accelerator, _handle = _prepare_manager(
        adapter,
        loaded,
    )
    spec = adapter.latent_spec_for_batch(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert spec.shape == expected_shape
    assert spec.batch_size == batch.batch_size
    assert spec.layout is expected_layout
    assert spec.device == torch.device("cpu")
    assert spec.dtype is torch.float16
    assert spec.shape[1] == 4  # supplied by fake runtime parts, not model config
    request = RolloutRequest(
        adapter=adapter,
        dynamics=_RequestOnlyDynamics(),
        samples=batch,
        latent_spec=spec,
        generator=torch.Generator().manual_seed(19),
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
    )
    assert request.latent_spec is spec

    with pytest.raises(ModelPortError, match="sample batch"):
        adapter.latent_spec_for_batch(
            _batch(wrong_task),
            device="cpu",
            dtype=torch.float32,
        )
    with pytest.raises(TypeError, match="latent dtype"):
        adapter.latent_spec_for_batch(
            batch,
            device="cpu",
            dtype="fp16",
        )
    manager.close()


class _DeclaredButMissingReference(WanT2VAdapter):
    @classmethod
    def describe(cls, config):
        declared = super().describe(config)
        return replace(
            declared,
            model=replace(declared.model, provides_reference_policy=True),
        )


def test_g3_rejects_declared_reference_without_real_typed_port(tmp_path) -> None:
    declaration = DeclarationResolver().resolve(
        MODEL_DECLARATIONS,
        "wan-t2v",
        {
            "artifact_ref": "main",
            "height": 16,
            "width": 16,
            "frames": 5,
        },
    )
    declaration = replace(
        declaration,
        descriptor=replace(
            declaration.descriptor,
            implementation_class_path=(f"{__name__}:_DeclaredButMissingReference"),
        ),
        declared_contract=_DeclaredButMissingReference.describe(declaration.config),
    )
    adapter, _loader, loaded = _load_declaration(tmp_path, declaration)
    assert type(adapter).predict_reference is ModelAdapter.predict_reference
    manager = ComponentManager(adapter, execution_device="cpu")
    manager.load()
    manager.configure()
    optimizer = torch.optim.SGD(manager.parameter_state.parameters(), lr=0.1)
    manager.prepare(
        accelerator=_IdentityAccelerator(),
        optimizer=optimizer,
    )
    runtime_contract = loaded.attest_prepared(
        runtime_identity=hashlib.sha256(
            b"p3-declared-reference-without-port"
        ).hexdigest(),
        verified_fields=(
            ("model.component_topology", "verified"),
            ("model.reference_forward", "verified"),
        ),
    )
    with pytest.raises(ComponentLifecycleError, match="predict_reference"):
        manager.bind_runtime(runtime_contract)
    manager.close()


def test_g3_rejects_sd3_without_verified_reference_forward_probe(tmp_path) -> None:
    _resolved, adapter, _loader, loaded = _load_adapter(
        tmp_path,
        "sd3",
        {"artifact_ref": "main", "resolution": 16},
    )
    manager = ComponentManager(adapter, execution_device="cpu")
    manager.load()
    manager.configure()
    optimizer = torch.optim.SGD(manager.parameter_state.parameters(), lr=0.1)
    manager.prepare(
        accelerator=_IdentityAccelerator(),
        optimizer=optimizer,
    )
    missing_probe = loaded.attest_prepared(
        runtime_identity=hashlib.sha256(
            b"p3-concrete-model-runtime-missing-reference"
        ).hexdigest(),
        verified_fields=(("model.component_topology", "verified"),),
    )
    with pytest.raises(ComponentLifecycleError, match="reference_forward probe"):
        manager.bind_runtime(missing_probe)
    manager.close()


def test_model_catalog_contracts_are_strict_and_match_recipe_ports() -> None:
    assert MODEL_DECLARATIONS.aliases == ("sd3", "wan-t2v")
    resolver = DeclarationResolver()
    sd3 = resolver.resolve(
        MODEL_DECLARATIONS,
        "sd3",
        {"artifact_ref": "main"},
    )
    wan = resolver.resolve(
        MODEL_DECLARATIONS,
        "wan-t2v",
        {"artifact_ref": "main"},
    )
    assert sd3.declared_contract.model.tasks == (TaskKind.T2I,)
    assert sd3.declared_contract.model.output_media == (MediaKind.IMAGE,)
    assert sd3.declared_contract.model.latent_layouts == (LatentLayout.BCHW,)
    assert sd3.declared_contract.model.prediction_types == (PredictionType.FLOW,)
    assert sd3.declared_contract.model.time_coordinates == (
        TimeCoordinate.FRACTIONAL_TIMESTEP,
    )
    assert ComputePrecision.BF16 in sd3.declared_contract.model.supported_precisions
    assert wan.declared_contract.model.tasks == (TaskKind.T2V,)
    assert wan.declared_contract.model.output_media == (MediaKind.VIDEO,)
    assert wan.declared_contract.model.latent_layouts == (LatentLayout.BCTHW,)
    assert wan.declared_contract.model.temporal_stride == 4
    assert ComputePrecision.BF16 in wan.declared_contract.model.supported_precisions
    with pytest.raises(RegistryError) as excinfo:
        resolver.resolve(
            MODEL_DECLARATIONS,
            "sd3",
            {"artifact_ref": "main", "rollout_kind": "branching"},
        )
    assert excinfo.value.code == "provider_failed"
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "unknown sd3 params" in str(excinfo.value.__cause__)


def test_all_six_recipes_resolve_model_descriptors_in_static_preflight(
    tmp_path,
) -> None:
    observed_models = []
    catalog = default_catalog()
    for definition in builtin_recipe_definitions():
        path = tmp_path / f"{definition.definition_id}.yaml"
        path.write_text(
            f"schema_version: 2\nrecipe: {definition.definition_id}\n",
            encoding="utf-8",
        )
        resolved = compile_recipe_v2(
            load_source_recipe(path),
            catalog=catalog,
        )
        assert resolved.compatibility.status == "compatible"
        observed_models.append((definition.name, resolved.model.declaration.alias))
    assert observed_models == [
        ("flow_grpo", "sd3"),
        ("tempflow_grpo", "sd3"),
        ("flash_grpo", "wan-t2v"),
        ("world_r1_core", "wan-t2v"),
        ("world_r1_release_surrogate", "wan-t2v"),
        ("world_r1_exact_env_hook", "wan-t2v"),
    ]


def test_concrete_model_modules_import_without_heavy_optional_dependencies() -> None:
    script = r"""
import builtins
original = builtins.__import__
blocked = {'torch', 'diffusers', 'transformers', 'peft'}
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in blocked:
        raise RuntimeError('heavy dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import visual_rl.models.implementations.sd3
import visual_rl.models.implementations.wan
print('import-safe')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "import-safe"
