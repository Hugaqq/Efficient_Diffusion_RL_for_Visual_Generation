"""World-R1 camera Conditioner lifecycle and explicit RNG contracts."""

from __future__ import annotations

import hashlib

import pytest
import torch

from visual_rl.algorithms.conditioning.config import (
    CONDITIONING_CATALOG_FRAGMENT,
    WorldR1CameraConfig,
)
from visual_rl.algorithms.conditioning.interface import (
    LatentConditioner,
    LatentSpec,
)
from visual_rl.algorithms.conditioning.world_r1_camera import (
    CameraConditionState,
    WorldR1CameraConditioner,
)
from visual_rl.composition.registry import DeclarationResolver, build_catalog
from visual_rl.core.contracts import (
    ComponentArtifactBindingSet,
    ComponentLoadPlan,
    DeclaredContract,
    LatentLayout,
)
from visual_rl.models import ModelLatentSpec
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


def _spec(batch_size: int = 2) -> LatentSpec:
    return LatentSpec(
        batch_size=batch_size,
        channels=2,
        latent_frames=3,
        latent_height=4,
        latent_width=4,
        output_frames=5,
        output_height=8,
        output_width=8,
        temporal_compression=2,
        spatial_compression=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _generator(seed: int = 7) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _model_geometry(batch_size: int = 2) -> ModelLatentSpec:
    return ModelLatentSpec(
        shape=(batch_size, 2, 3, 4, 4),
        layout=LatentLayout.BCTHW,
        axis_semantics=("batch", "channel", "time", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(2, 2),
        temporal_stride=2,
    )


def _runtime_binding(recipe_id: str) -> RuntimeBindResult:
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
        recipe_id=recipe_id,
        launch_id=runtime_launch_payload_id(recipe_id, runtime_facts),
        runtime_facts=runtime_facts,
    )


def test_camera_conditioner_owns_generic_model_geometry_binding() -> None:
    conditioner = WorldR1CameraConditioner(WorldR1CameraConfig(frames_per_trajectory=5))

    assert conditioner.bind_model_geometry(_model_geometry()) == _spec()


def test_camera_conditioner_rejects_incompatible_generic_geometry() -> None:
    conditioner = WorldR1CameraConditioner(WorldR1CameraConfig(frames_per_trajectory=5))
    image = ModelLatentSpec(
        shape=(2, 2, 4, 4),
        layout=LatentLayout.BCHW,
        axis_semantics=("batch", "channel", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(2, 2),
    )
    wrong_length = ModelLatentSpec(
        shape=(2, 2, 4, 4, 4),
        layout=LatentLayout.BCTHW,
        axis_semantics=("batch", "channel", "time", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(2, 2),
        temporal_stride=2,
    )

    with pytest.raises(ValueError, match="BCTHW"):
        conditioner.bind_model_geometry(image)
    with pytest.raises(ValueError, match="trajectory length"):
        conditioner.bind_model_geometry(wrong_length)


def test_typed_config_is_strict_and_declares_t2v_bcthw_contract():
    config = WorldR1CameraConfig.from_mapping(
        {"guidance_steps": 3, "wrap_strength": 0.5},
        context=None,
    )
    declared = WorldR1CameraConditioner.describe(config)
    assert isinstance(declared, DeclaredContract)
    assert declared.component_kind == "conditioner"
    assert declared.conditioner.payload_type == "camera_trajectory_v1"
    assert declared.conditioner.has_initialize_hook
    assert declared.conditioner.has_after_step_hook
    with pytest.raises(ValueError, match="unknown"):
        WorldR1CameraConfig.from_mapping({"surprise": 1}, context=None)


def test_camera_conditioner_is_a_real_lazy_catalog_component():
    catalog = build_catalog((CONDITIONING_CATALOG_FRAGMENT,))
    registry = catalog.for_kind("conditioner")
    assert registry.aliases == ("world-r1-camera",)
    descriptor = registry.lookup("world-r1-camera")
    assert descriptor is not None
    assert descriptor.implementation_class_path == (
        "visual_rl.algorithms.conditioning.world_r1_camera:WorldR1CameraConditioner"
    )
    assert descriptor.declaration_provider_path == (
        "visual_rl.algorithms.conditioning.config:WorldR1CameraDeclarationProvider"
    )
    declaration = DeclarationResolver().resolve(
        registry,
        "world-r1-camera",
        {},
    )
    assert declaration.config == WorldR1CameraConfig()
    assert declaration.implementation_class_path == (
        descriptor.implementation_class_path
    )


def test_provider_contract_constructs_canonical_runtime_without_legacy_owner():
    catalog = build_catalog((CONDITIONING_CATALOG_FRAGMENT,))
    declaration = DeclarationResolver().resolve(
        catalog.for_kind("conditioner"),
        "world-r1-camera",
        {"frames_per_trajectory": 5, "guidance_steps": 3},
    )
    code_identity = hashlib.sha256(b"camera-conditioner-code").hexdigest()
    recipe_id = (
        "materialized-recipe.v2:"
        + hashlib.sha256(b"camera-conditioner-recipe").hexdigest()
    )
    binding = build_component_artifact_binding(
        declaration,
        recipe_id=recipe_id,
        slot="conditioner",
        artifact_content_identities={"code": code_identity},
        code_identity=code_identity,
    )
    gate = RuntimeComponentLoadGate(
        runtime_binding=_runtime_binding(recipe_id),
        artifact_binding=binding,
    )
    binding_set = ComponentArtifactBindingSet(binding.recipe_id, (binding,))
    load_plan = ComponentLoadPlan.create(
        binding_set,
        required_artifact_names_by_slot={"conditioner": ("code",)},
    )

    loaded = RuntimeComponentLoader().load(
        declaration,
        gate=gate,
        binding_set=binding_set,
        load_plan=load_plan,
        runtime_context={},
    )
    instance = loaded.instance

    assert isinstance(instance, LatentConditioner)
    assert loaded.artifact_binding is binding
    assert type(instance).__module__ == (
        "visual_rl.algorithms.conditioning.world_r1_camera"
    )
    assert type(instance.config).__module__ == (
        "visual_rl.algorithms.conditioning.config"
    )
    assert instance.describe(instance.config) == declaration.declared_contract


def test_prepare_produces_one_fp32_camera_payload_per_prompt():
    conditioner = WorldR1CameraConditioner(WorldR1CameraConfig(flow_scale=1))
    state = conditioner.prepare(
        ("a quiet forest", "camera push in through a forest"),
        _spec(),
        generator=_generator(),
    )
    assert isinstance(state, CameraConditionState)
    assert state.camera_trajectory.shape == (2, 5, 4, 4)
    assert state.camera_trajectory.dtype == torch.float32
    assert state.movement_names[0] == ()
    assert state.movement_names[1] == ("push_in",)
    assert len(state.row_condition_identities) == 2
    assert state.row_condition_identities[0] != state.row_condition_identities[1]
    assert len(state.condition_identity) == 64
    assert state.camera_delta is None


def test_no_motion_is_gaussian_base_fallback_without_resampling():
    conditioner = WorldR1CameraConditioner(WorldR1CameraConfig(flow_scale=1))
    state = conditioner.prepare(
        ("a still landscape",),
        _spec(batch_size=1),
        generator=_generator(),
    )
    base = torch.randn((1, 2, 3, 4, 4), generator=_generator(3))
    result = conditioner.initialize_latents(base, state, generator=_generator(9))
    torch.testing.assert_close(result.latents, base)
    torch.testing.assert_close(result.state.camera_delta, torch.zeros_like(base))
    assert len(result.condition_payloads) == 1
    assert (
        result.condition_payloads[0].condition_identity
        == state.row_condition_identities[0]
    )


def test_motion_initialization_uses_explicit_generator_and_preserves_global_rng(
    monkeypatch,
):
    from visual_rl.algorithms.conditioning import camera_math

    observed = []

    def fake_warped(_trajectory, **kwargs):
        shape = (
            kwargs["batch_size"],
            kwargs["num_channels_latents"],
            (kwargs["num_frames"] - 1) // kwargs["temporal_compression"] + 1,
            kwargs["height"] // kwargs["spatial_compression"],
            kwargs["width"] // kwargs["spatial_compression"],
        )
        value = torch.randn(shape, device=kwargs["device"], dtype=kwargs["dtype"])
        observed.append(value.clone())
        return value

    monkeypatch.setattr(camera_math, "generate_camera_warped_latents", fake_warped)
    conditioner = WorldR1CameraConditioner(
        WorldR1CameraConfig(
            flow_scale=1,
            wrap_strength=0.6,
            delta_lowpass_kernel=3,
        )
    )
    state = conditioner.prepare(
        ("camera push in through a forest",),
        _spec(batch_size=1),
        generator=_generator(),
    )
    base = torch.randn((1, 2, 3, 4, 4), generator=_generator(2))
    generator = _generator(123)
    before_generator = generator.get_state().clone()
    torch.manual_seed(999)
    global_before = torch.random.get_rng_state().clone()
    result = conditioner.initialize_latents(base, state, generator=generator)

    assert observed
    assert not torch.equal(generator.get_state(), before_generator)
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert result.latents.shape == base.shape
    assert result.state.camera_delta.dtype == torch.float32
    base_mean = base.mean(dim=(1, 2, 3, 4))
    base_std = base.std(dim=(1, 2, 3, 4))
    result_mean = result.latents.mean(dim=(1, 2, 3, 4))
    result_std = result.latents.std(dim=(1, 2, 3, 4))
    torch.testing.assert_close(result_mean, base_mean, atol=2e-6, rtol=0)
    torch.testing.assert_close(result_std, base_std, atol=2e-6, rtol=0)


def test_after_step_uses_saved_delta_and_preserves_each_sample_statistics():
    conditioner = WorldR1CameraConditioner(
        WorldR1CameraConfig(
            guidance_steps=3,
            wrap_strength=0.7,
            flow_scale=1,
        )
    )
    state = conditioner.prepare(
        ("camera push in", "camera pan left"),
        _spec(),
        generator=_generator(),
    )
    delta = torch.randn((2, 2, 3, 4, 4), generator=_generator(4))
    state = type(state)(
        **{
            name: getattr(state, name)
            for name in state.__dataclass_fields__
            if name != "camera_delta"
        },
        camera_delta=delta,
    )
    next_latents = torch.randn((2, 2, 3, 4, 4), generator=_generator(5))
    result = conditioner.after_step(0, torch.tensor(1.0), next_latents, state)
    assert not torch.equal(result, next_latents)
    reduce = (1, 2, 3, 4)
    torch.testing.assert_close(
        result.mean(dim=reduce),
        next_latents.mean(dim=reduce),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        result.std(dim=reduce),
        next_latents.std(dim=reduce),
        atol=2e-6,
        rtol=0,
    )
    assert (
        conditioner.after_step(3, torch.tensor(0.5), next_latents, state)
        is next_latents
    )


def test_prepare_and_initialize_require_explicit_matching_generator():
    conditioner = WorldR1CameraConditioner(WorldR1CameraConfig(flow_scale=1))
    with pytest.raises(TypeError, match="explicit torch.Generator"):
        conditioner.prepare(("prompt",), _spec(batch_size=1), generator=None)
