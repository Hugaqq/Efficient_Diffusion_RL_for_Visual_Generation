"""Focused tests for W02's final typed component contracts."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
from collections.abc import Mapping

import pytest
import torch

from visual_rl.builtins import builtin_components
from visual_rl.core.types import (
    FrozenMapping,
    MetricContribution,
    PolicyRecomputeStats,
    ResolutionContext,
    RewardBatch,
    RewardVector,
    RolloutBatch,
    RolloutRequest,
    RuntimeBuildContext,
    StepContext,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
    to_plain_dict,
)
from visual_rl.feedback.base import RewardClient
from visual_rl.errors import ConfigError, RunError
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
from visual_rl.model_adapters.wan import (
    WanFlashAdapter,
    WanWorldR1Adapter,
)
from visual_rl.optimizers.base import PolicyAlgorithm
from visual_rl.rollout.base import RolloutEngine

COMPONENT_BASES = {
    "model": ModelAdapter,
    "rollout": RolloutEngine,
    "reward": RewardClient,
    "algorithm": PolicyAlgorithm,
}


def _specs(kind: str):
    return tuple(spec for spec in builtin_components() if spec.kind == kind)


def _request(*, kind: str = "full_trajectory") -> RolloutRequest:
    selected = (0, 1) if kind == "single_step" else None
    branch = (0, 0) if kind == "branching" else None
    return RolloutRequest(
        prompts=("red", "blue"),
        metadata=({"dataset_epoch": 0}, {"dataset_epoch": 0}),
        sample_id=("s0", "s1"),
        prompt_id=("p0", "p1"),
        group_id=("g0", "g1"),
        branch_id=None,
        context=StepContext(step=0, seed=7),
        kind=kind,
        num_steps=2,
        group_size=1,
        selected_timestep_index=selected,
        branch_step_index=branch,
    )


def _batch(request: RolloutRequest | None = None) -> RolloutBatch:
    request = request or _request()
    transitions = request.num_steps if request.kind == "full_trajectory" else 1
    selected = (
        torch.tensor(request.selected_timestep_index, dtype=torch.int64)
        if request.selected_timestep_index is not None
        else None
    )
    branch = (
        torch.tensor(request.branch_step_index, dtype=torch.int64)
        if request.branch_step_index is not None
        else None
    )
    return RolloutBatch(
        prompts=request.prompts,
        metadata=request.metadata,
        media=torch.zeros(2, 3, 4, 4),
        latents=torch.zeros(2, transitions, 3, 2, 2),
        next_latents=torch.ones(2, transitions, 3, 2, 2),
        timesteps=torch.arange(transitions).repeat(2, 1),
        old_log_probs=-torch.ones(2, transitions),
        transition_mask=torch.ones(2, transitions, dtype=torch.bool),
        sample_id=request.sample_id,
        prompt_id=request.prompt_id,
        group_id=request.group_id,
        branch_id=request.branch_id,
        media_layout="BCHW",
        camera_trajectory=None,
        context=request.context,
        selected_timestep_index=selected,
        flash_coefficient=None,
        branch_step_index=branch,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={"noise": torch.zeros(2, 1)},
        artifact_metadata={"source": "test"},
    )


def test_every_manifest_factory_implements_its_final_base_contract():
    for spec in builtin_components():
        base = COMPONENT_BASES[spec.kind]
        assert issubclass(spec.factory, base), (spec.kind, spec.name)
        assert not inspect.isabstract(spec.factory), (spec.kind, spec.name)
        for method in (
            "resolve_params",
            "check_environment",
            "from_config",
            "required_capabilities",
        ):
            assert callable(getattr(spec.factory, method)), (spec.name, method)
        assert "resolve_params" in spec.factory.__dict__, spec.name
        assert "from_config" in spec.factory.__dict__, spec.name
        assert callable(getattr(spec.factory, "close")), spec.name


def test_base_factory_protocol_is_one_typed_classmethod_quartet():
    for base in COMPONENT_BASES.values():
        for method in (
            "resolve_params",
            "check_environment",
            "from_config",
            "required_capabilities",
        ):
            assert isinstance(inspect.getattr_static(base, method), classmethod)
        context = ResolutionContext(
            config_path=Path("/tmp/config.yaml"),
            config_dir=Path("/tmp"),
        )
        raw = {"nested": [1, 2]}
        resolved = base.resolve_params(raw, context)
        raw["nested"].append(3)
        assert resolved == FrozenMapping({"nested": (1, 2)})
        with pytest.raises(NotImplementedError):
            base.from_config(
                resolved,
                RuntimeBuildContext(
                    rank=0,
                    local_rank=0,
                    world_size=1,
                    backend=None,
                    device=torch.device("cpu"),
                    precision="fp32",
                ),
            )


def test_model_adapter_surface_has_no_legacy_aliases():
    assert tuple(inspect.signature(ModelAdapter.sample).parameters) == (
        "self",
        "request",
    )
    assert tuple(
        inspect.signature(ModelAdapter.recompute_policy_stats).parameters
    ) == ("self", "batch", "require_reference")
    for name in (
        "recompute_log_probs",
        "prepare_for_sampling",
        "prepare_for_training",
        "state_dict",
        "load_state_dict",
        "save_pretrained",
        "branch_transition_count",
    ):
        assert not hasattr(ModelAdapter, name)
    for spec in _specs("model"):
        assert spec.factory.MEDIA_TYPE in {"image", "video"}
        assert "MEDIA_TYPE" in spec.factory.__dict__
        assert "recompute_log_probs" not in spec.factory.__dict__


def test_rollout_reward_and_algorithm_base_surfaces_are_final():
    assert tuple(inspect.signature(RolloutEngine.sample).parameters) == (
        "self",
        "adapter",
        "prompts",
        "metadata",
        "context",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in tuple(inspect.signature(RolloutEngine.sample).parameters.values())[1:]
    )
    assert tuple(inspect.signature(RewardClient.score).parameters) == (
        "self",
        "batch",
        "context",
    )
    for name in ("resolve_context", "runtime_config", "finalize_batch"):
        assert not hasattr(RolloutEngine, name)
    for spec in _specs("algorithm"):
        assert type(spec.factory.TRAINING_CONTRACT_VERSION) is int
        assert spec.factory.ADVANTAGE_DTYPE in {"float32", "float64"}
        assert type(spec.factory.MIN_GROUP_SIZE) is int
        assert spec.factory.MIN_GROUP_SIZE >= 2


EXPECTED_FIELDS = {
    RolloutRequest: (
        "prompts",
        "metadata",
        "sample_id",
        "prompt_id",
        "group_id",
        "branch_id",
        "context",
        "kind",
        "num_steps",
        "group_size",
        "selected_timestep_index",
        "branch_step_index",
    ),
    RolloutBatch: (
        "prompts",
        "metadata",
        "media",
        "latents",
        "next_latents",
        "timesteps",
        "old_log_probs",
        "transition_mask",
        "sample_id",
        "prompt_id",
        "group_id",
        "branch_id",
        "media_layout",
        "camera_trajectory",
        "context",
        "selected_timestep_index",
        "flash_coefficient",
        "branch_step_index",
        "trajectory_step_index",
        "transition_std_dev",
        "recompute_payload",
        "artifact_metadata",
    ),
    RewardVector: ("sample_id", "values", "shared_metadata", "sample_metadata"),
    RewardBatch: (
        "sample_id",
        "raw",
        "weighted",
        "weighted_total",
        "valid_mask",
        "shared_metadata",
        "sample_metadata",
    ),
    PolicyRecomputeStats: (
        "new_log_probs",
        "current_transition_mean",
        "transition_std",
        "reference_transition_mean",
    ),
    MetricContribution: ("numerator", "denominator"),
    ValidationCheck: ("level", "code", "path", "message", "volatile"),
    ResolutionContext: ("config_path", "config_dir"),
    ValidationContext: (
        "phase",
        "config_dir",
        "distributed_mode",
        "world_size",
        "backend",
        "device",
        "timeout_s",
    ),
    ValidatedRuntimeEnv: (
        "mode",
        "rank",
        "local_rank",
        "world_size",
        "local_world_size",
        "group_rank",
        "group_world_size",
        "master_addr",
        "master_port",
        "visible_gpu_count",
        "raw_launch_env",
    ),
    RuntimeBuildContext: (
        "rank",
        "local_rank",
        "world_size",
        "backend",
        "device",
        "precision",
    ),
}


@pytest.mark.parametrize(
    ("contract", "expected"),
    tuple(EXPECTED_FIELDS.items()),
    ids=tuple(contract.__name__ for contract in EXPECTED_FIELDS),
)
def test_cross_component_dataclasses_are_frozen_and_exact(contract, expected):
    assert dataclasses.is_dataclass(contract)
    assert contract.__dataclass_params__.frozen
    assert tuple(field.name for field in dataclasses.fields(contract)) == expected


def test_rollout_request_and_batch_validate_the_same_identity_object():
    request = _request()
    batch = _batch(request)
    batch.validate_against(request)
    assert batch.context is request.context
    assert isinstance(batch.metadata[0], FrozenMapping)
    assert isinstance(batch.artifact_metadata, FrozenMapping)
    with pytest.raises(ValueError, match="same StepContext object"):
        dataclasses.replace(
            batch,
            context=StepContext(step=0, seed=7),
        ).validate_against(request)


def test_rollout_batch_slice_moves_all_sample_fields_but_not_t_axis():
    request = _request()
    batch = dataclasses.replace(
        _batch(request),
        trajectory_step_index=torch.tensor([0, 1], dtype=torch.int64),
    )
    selected = batch.slice((1,))
    assert selected.sample_id == ("s1",)
    assert selected.latents.shape[:2] == (1, 2)
    assert selected.recompute_payload["noise"].shape[0] == 1
    assert selected.trajectory_step_index is batch.trajectory_step_index
    assert selected.artifact_metadata is batch.artifact_metadata


def test_reward_contract_is_cpu_float32_detached_and_ordered():
    batch = _batch()
    rewards = RewardBatch(
        sample_id=batch.sample_id,
        raw={"mock": torch.tensor([1.0, 2.0])},
        weighted={"mock": torch.tensor([0.5, 1.0])},
        weighted_total=torch.tensor([0.5, 1.0]),
        valid_mask=torch.ones(2, dtype=torch.bool),
        shared_metadata={"mock": {"revision": "v1"}},
        sample_metadata={"mock": ({"row": 0}, {"row": 1})},
    )
    rewards.validate_against(batch)
    assert isinstance(rewards.raw, Mapping)
    assert isinstance(rewards.shared_metadata, FrozenMapping)
    assert rewards.slice((1,)).sample_id == ("s1",)
    with pytest.raises(ValueError, match="every row"):
        dataclasses.replace(
            rewards,
            valid_mask=torch.tensor([True, False]),
        )


def test_policy_recompute_stats_enforces_active_mask_and_reference_contract():
    batch = _batch()
    new_log_probs = (-torch.ones_like(batch.old_log_probs)).requires_grad_(True)
    PolicyRecomputeStats(new_log_probs=new_log_probs).validate_against(
        batch,
        require_reference=False,
    )
    with pytest.raises(ValueError, match="reference statistics"):
        PolicyRecomputeStats(
            new_log_probs=new_log_probs,
            transition_std=torch.ones_like(batch.old_log_probs),
        ).validate_against(batch, require_reference=False)

    current = torch.zeros(2, 2, 3, requires_grad=True)
    reference = torch.zeros(2, 2, 3)
    std = torch.ones(2, 2)
    PolicyRecomputeStats(
        new_log_probs=new_log_probs,
        current_transition_mean=current,
        transition_std=std,
        reference_transition_mean=reference,
    ).validate_against(batch, require_reference=True)


def test_frozen_mapping_and_projector_are_pickle_safe_and_strict():
    source = {"nested": [1, {"x": 2}]}
    frozen = FrozenMapping(source)
    source["nested"].append(3)
    assert frozen["nested"] == (1, FrozenMapping({"x": 2}))
    assert pickle.loads(pickle.dumps(frozen)) == frozen
    with pytest.raises(TypeError, match="JSON-safe"):
        FrozenMapping({"tensor": torch.zeros(1)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        to_plain_dict(torch.zeros(1))


def test_projector_uses_the_one_dataclass_plain_name_rule():
    @dataclasses.dataclass(frozen=True)
    class _Resume:
        from_: Path | None = dataclasses.field(
            metadata={"plain_name": "from"},
        )

    assert to_plain_dict(_Resume(from_=Path("/tmp/checkpoint"))) == {
        "from": "/tmp/checkpoint"
    }


def test_importing_projector_does_not_import_training_libraries():
    script = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('_types_import_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert module.to_plain_dict({'ok': (1, 2)}) == {'ok': [1, 2]}
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parents[1] / "visual_rl" / "core" / "types.py"),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
    )


class _ToyAdapter(ModelAdapter):
    MEDIA_TYPE = "image"

    def __init__(self):
        self.module = torch.nn.Linear(2, 1)

    @property
    def train_module(self):
        return self.module

    def sample(self, request):
        raise NotImplementedError

    def recompute_policy_stats(self, batch, *, require_reference=False):
        raise NotImplementedError


def test_base_adapter_checkpoint_is_two_file_deterministic_and_atomic(tmp_path):
    torch.manual_seed(3)
    adapter = _ToyAdapter()
    first = tmp_path / "first"
    second = tmp_path / "second"
    adapter.save_checkpoint(first)
    adapter.save_checkpoint(second)
    assert sorted(path.name for path in first.iterdir()) == [
        "adapter.json",
        "adapter_state.pt",
    ]
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in ("adapter.json", "adapter_state.pt")
    )
    adapter.validate_checkpoint(first)
    expected = tuple(parameter.detach().clone() for parameter in adapter.parameters())
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.add_(10)
    adapter.load_checkpoint(first)
    assert all(
        torch.equal(parameter, value)
        for parameter, value in zip(adapter.parameters(), expected, strict=True)
    )
    snapshot = tuple(parameter.detach().clone() for parameter in adapter.parameters())
    state_path = first / "adapter_state.pt"
    state_path.write_bytes(state_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.load_checkpoint(first)
    assert all(
        torch.equal(parameter, value)
        for parameter, value in zip(adapter.parameters(), snapshot, strict=True)
    )


def _resolution_context(tmp_path: Path) -> ResolutionContext:
    config_path = (tmp_path / "config.yaml").resolve()
    return ResolutionContext(
        config_path=config_path,
        config_dir=config_path.parent,
    )


def _runtime_context() -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )


def _heavy_model_params(*, frames: int | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "checkpoint": "checkpoint",
        "reference_repo": "reference",
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_target_modules": ["to_q", "to_v"],
        "gradient_checkpointing": True,
        "guidance_scale": 4.5,
        "height": 64,
        "width": 96,
        "max_sequence_length": 32,
    }
    if frames is not None:
        params["frames"] = frames
    return params


@pytest.mark.parametrize(
    ("factory", "raw", "expected_keys"),
    [
        (
            TinyDiffusionAdapter,
            {"image_size": 8},
            {"image_size"},
        ),
        (
            SD3TempFlowAdapter,
            {
                key: value
                for key, value in {
                    **_heavy_model_params(),
                    "resolution": 64,
                }.items()
                if key not in {"height", "width"}
            },
            {
                "checkpoint",
                "reference_repo",
                "lora_rank",
                "lora_alpha",
                "lora_target_modules",
                "gradient_checkpointing",
                "guidance_scale",
                "resolution",
                "max_sequence_length",
                "local_files_only",
                "low_cpu_mem_usage",
            },
        ),
        (
            WanFlashAdapter,
            _heavy_model_params(frames=9),
            {
                "checkpoint",
                "reference_repo",
                "lora_rank",
                "lora_alpha",
                "lora_target_modules",
                "gradient_checkpointing",
                "guidance_scale",
                "height",
                "width",
                "frames",
                "max_sequence_length",
                "local_files_only",
                "low_cpu_mem_usage",
            },
        ),
        (
            WanWorldR1Adapter,
            _heavy_model_params(),
            {
                "checkpoint",
                "reference_repo",
                "lora_rank",
                "lora_alpha",
                "lora_target_modules",
                "gradient_checkpointing",
                "guidance_scale",
                "height",
                "width",
                "max_sequence_length",
                "local_files_only",
                "low_cpu_mem_usage",
            },
        ),
    ],
)
def test_model_resolvers_have_one_exact_parameter_table(
    tmp_path,
    factory,
    raw,
    expected_keys,
):
    resolved = factory.resolve_params(raw, _resolution_context(tmp_path))
    assert set(resolved) == expected_keys
    if "checkpoint" in resolved:
        assert resolved["checkpoint"] == (tmp_path / "checkpoint").resolve()
        assert resolved["reference_repo"] == (tmp_path / "reference").resolve()
        assert resolved["lora_target_modules"] == ("to_q", "to_v")
        assert resolved["local_files_only"] is True
        assert resolved["low_cpu_mem_usage"] is True

    for key_to_remove in raw:
        invalid = dict(raw)
        invalid.pop(key_to_remove)
        with pytest.raises(ConfigError):
            factory.resolve_params(invalid, _resolution_context(tmp_path))
    with pytest.raises(ConfigError, match="Unknown"):
        factory.resolve_params(
            {**raw, "legacy_alias": True},
            _resolution_context(tmp_path),
        )


@pytest.mark.parametrize(
    "factory",
    [SD3TempFlowAdapter, WanFlashAdapter, WanWorldR1Adapter],
)
@pytest.mark.parametrize(
    "legacy_key",
    [
        "device",
        "dtype",
        "lora_path",
        "wan_backend",
        "adapter_name",
        "attention_kwargs",
        "defer_load",
        "extra",
        "use_lora",
        "noise_level",
        "output_type",
        "train_cfg",
        "use_camera_trajectory",
    ],
)
def test_heavy_model_resolvers_reject_every_retired_alias(
    tmp_path,
    factory,
    legacy_key,
):
    raw = (
        {
            key: value
            for key, value in {
                **_heavy_model_params(),
                "resolution": 64,
            }.items()
            if key not in {"height", "width"}
        }
        if factory is SD3TempFlowAdapter
        else _heavy_model_params(
            frames=9 if factory is WanFlashAdapter else None
        )
    )
    with pytest.raises(ConfigError, match="Unknown"):
        factory.resolve_params(
            {**raw, legacy_key: object()},
            _resolution_context(tmp_path),
        )


def test_model_resolvers_reject_bool_numeric_and_cross_field_edges(tmp_path):
    context = _resolution_context(tmp_path)
    with pytest.raises(ConfigError, match="positive integer"):
        TinyDiffusionAdapter.resolve_params({"image_size": True}, context)

    sd3 = {
        key: value
        for key, value in {
            **_heavy_model_params(),
            "resolution": 64,
        }.items()
        if key not in {"height", "width"}
    }
    with pytest.raises(ConfigError, match="positive integer"):
        SD3TempFlowAdapter.resolve_params({**sd3, "lora_rank": True}, context)
    with pytest.raises(ConfigError, match="duplicates"):
        SD3TempFlowAdapter.resolve_params(
            {**sd3, "lora_target_modules": ["to_q", "to_q"]},
            context,
        )
    with pytest.raises(ConfigError, match="finite"):
        SD3TempFlowAdapter.resolve_params(
            {**sd3, "guidance_scale": float("nan")},
            context,
        )

    flash = _heavy_model_params(frames=9)
    with pytest.raises(ConfigError, match="multiples of 8"):
        WanFlashAdapter.resolve_params({**flash, "height": 65}, context)
    with pytest.raises(ConfigError, match=r"\(frames - 1\) % 4"):
        WanFlashAdapter.resolve_params({**flash, "frames": 8}, context)
    with pytest.raises(ConfigError, match="finite and > 0"):
        WanFlashAdapter.resolve_params(
            {**flash, "guidance_scale": 0.0},
            context,
        )


@pytest.mark.parametrize(
    ("factory", "raw"),
    [
        (
            SD3TempFlowAdapter,
            {
                key: value
                for key, value in {
                    **_heavy_model_params(),
                    "resolution": 64,
                }.items()
                if key not in {"height", "width"}
            },
        ),
        (WanFlashAdapter, _heavy_model_params(frames=9)),
        (WanWorldR1Adapter, _heavy_model_params()),
    ],
)
def test_heavy_from_config_consumes_only_resolved_fields(
    tmp_path,
    monkeypatch,
    factory,
    raw,
):
    resolved = factory.resolve_params(raw, _resolution_context(tmp_path))
    monkeypatch.setattr(factory, "_load_base_pipeline", lambda self: None)
    adapter = factory.from_config(resolved, _runtime_context())
    assert adapter.checkpoint is resolved["checkpoint"]
    assert adapter.reference_repo is resolved["reference_repo"]
    assert adapter.lora_target_modules == resolved["lora_target_modules"]
    assert adapter.device == torch.device("cpu")
    assert adapter.dtype == torch.float32
    assert adapter.pipeline is None


def test_wan_components_are_two_real_classes_without_backend_alias():
    assert WanFlashAdapter is not WanWorldR1Adapter
    assert WanFlashAdapter.__base__ is WanWorldR1Adapter.__base__
    assert WanFlashAdapter.MEDIA_TYPE == WanWorldR1Adapter.MEDIA_TYPE == "video"
    assert WanWorldR1Adapter.WORLD_R1_FRAMES == 81
    assert dict(WanWorldR1Adapter.WORLD_R1_CAMERA_NOISE_WRAP) == {
        "remove_camera_keywords_from_prompt": False,
        "force_camera_movement": None,
        "noise_wrap_compute_dtype": "fp32",
        "noise_downtemp_interp": "nearest",
        "noise_downspatial_mode": "resize_noise",
        "noise_degradation": 0.35,
        "noise_wrap_flow_scale": 16,
        "wrap_strength": 0.35,
        "wrap_injection_mode": "stepwise_delta",
        "delta_lowpass_kernel": 9,
        "stepwise_guidance_steps": 8,
    }


def test_world_r1_camera_path_uses_one_scoped_exact_helper_contract(tmp_path):
    fixture_repo = (
        Path(__file__).parent
        / "fixtures"
        / "reference_repos"
        / "world_r1_camera_v1"
    ).resolve()
    adapter = WanWorldR1Adapter(
        checkpoint=(tmp_path / "checkpoint").resolve(),
        reference_repo=fixture_repo,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q", "to_v"),
        gradient_checkpointing=True,
        guidance_scale=4.5,
        height=64,
        width=96,
        frames=WanWorldR1Adapter.WORLD_R1_FRAMES,
        max_sequence_length=32,
        local_files_only=True,
        low_cpu_mem_usage=True,
        context=_runtime_context(),
    )
    transformer = torch.nn.Module()
    transformer.config = SimpleNamespace(in_channels=4)
    adapter.transformer = transformer
    adapter.pipeline = SimpleNamespace(vae_scale_factor_temporal=4)
    generator = torch.Generator(device="cpu").manual_seed(7)

    base, callback, camera = adapter._prepare_world_camera(
        (
            "push in toward a red cube",
            "orbit right around a blue sculpture",
        ),
        generator,
    )
    assert tuple(base.shape) == (2, 4, 21, 8, 12)
    assert callable(callback)
    assert camera.dtype == torch.float64
    assert tuple(camera.shape) == (2, 81, 4, 4)
    assert not torch.equal(camera[0], camera[1])
    updated = callback(
        None,
        0,
        None,
        {"latents": base.clone()},
    )
    assert not torch.equal(updated["latents"], base)

    with pytest.raises(RunError, match="camera movement"):
        adapter._prepare_world_camera(("a still red cube",), generator)


def _tiny_request(kind: str) -> RolloutRequest:
    if kind == "branching":
        return RolloutRequest(
            prompts=("red", "red"),
            metadata=({"source": "unit"}, {"source": "unit"}),
            sample_id=("sample-0", "sample-1"),
            prompt_id=("prompt-0", "prompt-0"),
            group_id=("group-0", "group-0"),
            branch_id=(0, 1),
            context=StepContext(step=3, seed=17),
            kind="branching",
            num_steps=3,
            group_size=2,
            branch_step_index=(1, 1),
        )
    return RolloutRequest(
        prompts=("red", "blue"),
        metadata=({"source": "unit"}, {"source": "unit"}),
        sample_id=("sample-0", "sample-1"),
        prompt_id=("prompt-0", "prompt-1"),
        group_id=("group-0", "group-1"),
        branch_id=None,
        context=StepContext(step=3, seed=17),
        kind=kind,
        num_steps=3,
        group_size=1,
        selected_timestep_index=(0, 2) if kind == "single_step" else None,
    )


@pytest.mark.parametrize("kind", ["full_trajectory", "single_step", "branching"])
def test_tiny_adapter_uses_one_typed_rollout_and_recompute_path(kind):
    adapter = TinyDiffusionAdapter.from_config(
        FrozenMapping({"image_size": 8}),
        _runtime_context(),
    )
    request = _tiny_request(kind)
    adapter.train_module.train(True)
    batch = adapter.sample(request)
    assert adapter.train_module.training is True
    batch.validate_against(request)
    assert batch.context is request.context
    assert batch.sample_id == request.sample_id
    assert batch.transition_count == (
        request.num_steps if kind == "full_trajectory" else 1
    )
    if kind == "full_trajectory":
        assert batch.trajectory_step_index is None
        assert batch.flash_coefficient is None
    elif kind == "single_step":
        assert batch.trajectory_step_index is None
        torch.testing.assert_close(
            batch.flash_coefficient,
            torch.ones(2, 1),
        )
    else:
        assert batch.trajectory_step_index.tolist() == [1]
        assert batch.transition_std_dev.shape == (2, 1)
        assert torch.equal(batch.latents[0], batch.latents[1])

    stats = adapter.recompute_policy_stats(batch)
    stats.validate_against(batch, require_reference=False)
    assert stats.new_log_probs.requires_grad
    with pytest.raises(RunError, match="stage 4"):
        adapter.recompute_policy_stats(batch, require_reference=True)
