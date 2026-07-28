"""Contract-shape tests for the four component factory kinds (v0.7 W02).

Model/rollout factories already inherit their base classes; reward and
algorithm concrete classes predate the new ``RewardClient``/``PolicyAlgorithm``
ABCs, so their conformance is verified through the lightweight adaptation the
atomic cutover will perform: a test-local subclass mixing the existing
concrete class with the new base must satisfy the unified protocol without
breaking instantiation or the legacy surface.
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle

import pytest
import torch

from visual_rl.core.components import builtin_components
from visual_rl.core.types import (
    AdvantageResult,
    FrozenMapping,
    MetricContribution,
    ObjectiveOutput,
    PolicyLossInputs,
    PolicyRecomputeStats,
    ResolutionContext,
    RewardVector,
    RolloutBatch,
    RolloutRequest,
    RuntimeBuildContext,
    SampleRecord,
    StepArtifacts,
    StepMetrics,
    StepResult,
    UpdateResult,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
    to_plain_dict,
)
from visual_rl.feedback.base import RewardClient
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.optimizers.base import PolicyAlgorithm
from visual_rl.rollout.base import RolloutEngine

COMPONENT_BASES = (ModelAdapter, RolloutEngine, RewardClient, PolicyAlgorithm)


class _NullTransport:
    """Transport stand-in so reward clients construct without `requests`."""

    def post(self, *args, **kwargs):  # pragma: no cover - never called here
        raise RuntimeError("contract tests never perform network calls")


def _specs(kind):
    return [spec for spec in builtin_components() if spec.kind == kind]


# ---------------------------------------------------------------------------
# Unified resolve_params/check_environment/from_config protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", COMPONENT_BASES)
def test_factory_protocol_is_a_classmethod_trio_on_every_base(base):
    for method in (
        "resolve_params",
        "check_environment",
        "from_config",
        "required_capabilities",
    ):
        assert isinstance(inspect.getattr_static(base, method), classmethod), (
            f"{base.__name__}.{method} must be a classmethod"
        )


@pytest.mark.parametrize("base", COMPONENT_BASES)
def test_factory_protocol_default_shapes(base):
    resolved = base.resolve_params({"alpha": 1, "nested": {"beta": [1, 2]}}, object())
    assert isinstance(resolved, FrozenMapping)
    assert resolved["alpha"] == 1
    assert resolved["nested"] == FrozenMapping({"beta": (1, 2)})
    with pytest.raises(TypeError, match="requires a mapping"):
        base.resolve_params(["not", "a", "mapping"], object())
    assert base.check_environment(resolved, object()) == ()
    with pytest.raises(NotImplementedError, match=base.__name__):
        base.from_config(resolved, object())
    assert base.required_capabilities(resolved) == frozenset()


def test_resolve_params_result_is_isolated_from_caller_mutation():
    raw = {"items": [1, 2]}
    resolved = ModelAdapter.resolve_params(raw, object())
    raw["items"].append(3)
    raw["added"] = True
    assert resolved == FrozenMapping({"items": (1, 2)})


# ---------------------------------------------------------------------------
# Kind-specific conformance
# ---------------------------------------------------------------------------


def test_model_factories_satisfy_model_adapter_contract():
    for spec in _specs("model"):
        assert issubclass(spec.factory, ModelAdapter), spec.name
        assert inspect.isabstract(spec.factory) is False, spec.name
        assert callable(spec.factory.sample)
        assert callable(spec.factory.recompute_log_probs)
        assert callable(spec.factory.recompute_policy_stats)
        assert spec.factory.check_environment({}, object()) == ()
    # The base declares the final class-body MEDIA_TYPE direction; concrete
    # adapters set the value during the cutover.
    assert "MEDIA_TYPE" in ModelAdapter.__annotations__
    adapter = _specs("model")[0].factory  # tiny_diffusion is first in the manifest
    assert adapter.__name__ == "TinyDiffusionAdapter"


def test_tiny_adapter_keeps_working_and_declares_recompute_direction():
    from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter

    adapter = TinyDiffusionAdapter({"extra": {"image_size": 8, "device": "cpu"}})
    assert adapter.media_type == "image"
    assert adapter.train_module is not None
    batch = _tiny_batch()
    with pytest.raises(NotImplementedError, match="recompute_policy_stats"):
        adapter.recompute_policy_stats(batch)
    # Legacy entry points still work unchanged.
    stats = adapter.recompute_log_probs(batch)
    assert tuple(stats.shape) == tuple(batch.old_log_probs.shape)


def test_rollout_factories_satisfy_rollout_engine_contract():
    for spec in _specs("rollout"):
        assert issubclass(spec.factory, RolloutEngine), spec.name
        assert inspect.isabstract(spec.factory) is False, spec.name
        engine = spec.factory({})
        assert callable(engine.sample)
        assert spec.factory.check_environment({}, object()) == ()
        with pytest.raises(NotImplementedError):
            spec.factory.from_config({}, object())


def test_reward_factories_adapt_to_reward_client_contract():
    for spec in _specs("reward"):
        factory = spec.factory
        assert inspect.isclass(factory) and inspect.isabstract(factory) is False
        assert callable(getattr(factory, "score", None)), spec.name
        assert hasattr(factory, "name"), spec.name
        adapted = type(f"Adapted{factory.__name__}", (factory, RewardClient), {})
        assert issubclass(adapted, RewardClient)
        assert inspect.isabstract(adapted) is False
        assert adapted.check_environment({}, object()) == ()
        assert adapted.required_capabilities({}) == frozenset()
        resolved = adapted.resolve_params({"weight": 1.0}, object())
        assert isinstance(resolved, FrozenMapping)
        with pytest.raises(NotImplementedError):
            adapted.from_config({}, object())
        if spec.name in {"reward_general", "reward_3d"}:
            instance = adapted(
                url="https://reward.example.com",
                transport=_NullTransport(),
            )
        else:
            instance = adapted()
        assert instance.name == spec.name


def test_algorithm_factories_adapt_to_policy_algorithm_contract():
    for spec in _specs("algorithm"):
        factory = spec.factory
        assert inspect.isclass(factory) and inspect.isabstract(factory) is False
        adapted = type(
            f"Adapted{factory.__name__}",
            (factory, PolicyAlgorithm),
            {"TRAINING_CONTRACT_VERSION": 1, "ADVANTAGE_DTYPE": "float32"},
        )
        assert issubclass(adapted, PolicyAlgorithm)
        assert inspect.isabstract(adapted) is False
        instance = adapted()
        assert instance.TRAINING_CONTRACT_VERSION == 1
        assert instance.ADVANTAGE_DTYPE == "float32"
        assert instance.MIN_GROUP_SIZE == 2
        assert adapted.check_environment({}, object()) == ()
        assert adapted.required_capabilities({}) == frozenset()
        resolved = adapted.resolve_params({"clip_range": 0.1}, object())
        assert resolved["clip_range"] == 0.1
        # The legacy from_config surface keeps working on the concrete class.
        legacy = factory.from_config({})
        assert type(legacy).__name__ == factory.__name__


def test_policy_algorithm_class_constants():
    assert PolicyAlgorithm.MIN_GROUP_SIZE == 2
    assert "TRAINING_CONTRACT_VERSION" in PolicyAlgorithm.__annotations__
    assert "ADVANTAGE_DTYPE" in PolicyAlgorithm.__annotations__


# ---------------------------------------------------------------------------
# Cross-component typed contract field sets (plan "shared data contract")
# ---------------------------------------------------------------------------

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
    PolicyRecomputeStats: (
        "new_log_probs",
        "current_transition_mean",
        "transition_std",
        "reference_transition_mean",
    ),
    RewardVector: ("sample_id", "values", "shared_metadata", "sample_metadata"),
    MetricContribution: ("numerator", "denominator"),
    AdvantageResult: ("base_advantage", "diagnostics"),
    PolicyLossInputs: (
        "base_advantage",
        "algorithm_weight",
        "active_mask",
        "clip_range",
        "reference_kl_weight",
    ),
    ObjectiveOutput: (
        "loss",
        "policy_loss",
        "reference_kl",
        "approx_kl",
        "clipfrac",
        "active_transition_count",
    ),
    UpdateResult: (
        "loss",
        "policy_loss",
        "reference_kl",
        "approx_kl",
        "clipfrac",
        "active_transition_count",
        "diagnostics",
    ),
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
    StepMetrics: ("values", "sample_count", "active_transition_count"),
    StepArtifacts: ("local_records",),
    StepResult: ("context", "metrics", "artifacts"),
    SampleRecord: (
        "run_id",
        "sample_id",
        "sample_index",
        "step",
        "rank",
        "prompt",
        "media_type",
        "prompt_metadata",
        "seed",
        "rollout_type",
        "timestep_summary",
        "reward_values",
        "media_path",
        "rollout_cache_path",
        "checkpoint_path",
        "model_metadata",
        "prompt_id",
        "group_id",
        "branch_id",
    ),
}


@pytest.mark.parametrize(
    ("contract", "expected"),
    list(EXPECTED_FIELDS.items()),
    ids=[contract.__name__ for contract in EXPECTED_FIELDS],
)
def test_cross_component_field_sets_match_plan(contract, expected):
    assert dataclasses.is_dataclass(contract)
    assert contract.__dataclass_params__.frozen
    assert tuple(item.name for item in dataclasses.fields(contract)) == expected


def test_mapping_fields_are_deep_frozen_on_construction():
    env = ValidatedRuntimeEnv(
        mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        group_rank=None,
        group_world_size=None,
        master_addr=None,
        master_port=None,
        visible_gpu_count=0,
        raw_launch_env={"RANK": "0"},
    )
    assert isinstance(env.raw_launch_env, FrozenMapping)
    metrics = StepMetrics(values={"loss": 1.0}, sample_count=2, active_transition_count=8)
    assert isinstance(metrics.values, FrozenMapping)
    result = UpdateResult(
        loss=1.0,
        policy_loss=1.0,
        reference_kl=0.0,
        approx_kl=0.0,
        clipfrac=0.0,
        active_transition_count=8,
        diagnostics={"algorithm/x": 2.0},
    )
    assert isinstance(result.diagnostics, FrozenMapping)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.loss = 2.0


def test_metric_contribution_denominator_contract():
    MetricContribution(numerator=torch.tensor(1.0), denominator=None)
    MetricContribution(numerator=torch.tensor(1.0), denominator=4)
    with pytest.raises(ValueError, match="positive int or None"):
        MetricContribution(numerator=torch.tensor(1.0), denominator=0)
    with pytest.raises(TypeError, match="not bool"):
        MetricContribution(numerator=torch.tensor(1.0), denominator=True)


# ---------------------------------------------------------------------------
# FrozenMapping and the single strict to_plain_dict projector
# ---------------------------------------------------------------------------


def test_frozen_mapping_deep_freezes_and_round_trips_pickle():
    source = {"a": [1, {"b": 2}], "c": {"d": (3, 4)}}
    frozen = FrozenMapping(source)
    source["a"].append(99)
    assert frozen["a"] == (1, FrozenMapping({"b": 2}))
    assert frozen["c"]["d"] == (3, 4)
    restored = pickle.loads(pickle.dumps(frozen))
    assert restored == frozen
    assert hash(restored) == hash(frozen)
    with pytest.raises(TypeError, match="keys must be strings"):
        FrozenMapping({1: "x"})
    with pytest.raises(TypeError, match="JSON-safe"):
        FrozenMapping({"bad": {1, 2}})
    with pytest.raises(TypeError, match="JSON-safe"):
        FrozenMapping({"bad": torch.zeros(1)})


def test_to_plain_dict_projects_plain_containers_and_scalars():
    from pathlib import Path

    @dataclasses.dataclass(frozen=True)
    class _Nested:
        value: int
        tags: tuple[str, ...]

    projected = to_plain_dict(
        {
            "nested": _Nested(value=1, tags=("a", "b")),
            "path": Path("/tmp/x"),
            "scalars": [1, 2.5, True, "s", None],
            "mapping": FrozenMapping({"k": ("v",)}),
        }
    )
    assert projected == {
        "nested": {"value": 1, "tags": ["a", "b"]},
        "path": "/tmp/x",
        "scalars": [1, 2.5, True, "s", None],
        "mapping": {"k": ["v"]},
    }
    assert type(projected["nested"]["tags"]) is list


def test_to_plain_dict_rejects_non_plain_values_before_writing():
    import numpy as np

    with pytest.raises(TypeError, match="torch.Tensor"):
        to_plain_dict(torch.zeros(1))
    with pytest.raises(TypeError, match="numpy.ndarray"):
        to_plain_dict(np.zeros(1))
    with pytest.raises(TypeError, match="set"):
        to_plain_dict({1, 2})
    with pytest.raises(TypeError, match="callables"):
        to_plain_dict(lambda: None)
    with pytest.raises(ValueError, match="non-finite"):
        to_plain_dict(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        to_plain_dict({"x": float("inf")})
    with pytest.raises(TypeError, match="arbitrary|does not accept"):
        to_plain_dict(object())

    @dataclasses.dataclass
    class _Mutable:
        value: int

    with pytest.raises(TypeError, match="frozen dataclass"):
        to_plain_dict(_Mutable(value=1))


# ---------------------------------------------------------------------------
# PolicyRecomputeStats.validate_against
# ---------------------------------------------------------------------------


def _tiny_batch():
    prompts = ["a", "b"]
    metadata = [{}, {}]
    latents = torch.zeros(2, 2, 3, 2, 2)
    next_latents = torch.ones(2, 2, 3, 2, 2)
    timesteps = torch.arange(2).repeat(2, 1)
    old_log_probs = -torch.ones(2, 2)
    mask = torch.tensor([[True, False], [True, True]])
    return RolloutBatch(
        prompts=prompts,
        metadata=metadata,
        latents=latents,
        next_latents=next_latents,
        timesteps=timesteps,
        old_log_probs=old_log_probs,
        transition_mask=mask,
    )


def test_policy_recompute_stats_validate_against_happy_path():
    batch = _tiny_batch()
    new_log_probs = (-torch.ones(2, 2)).requires_grad_(True)
    stats = PolicyRecomputeStats(new_log_probs=new_log_probs)
    stats.validate_against(batch, require_reference=False)
    # Non-finite padding at masked-out positions is tolerated.
    padded = new_log_probs.detach().clone()
    padded[0, 1] = float("nan")
    stats = PolicyRecomputeStats(new_log_probs=padded.requires_grad_(True))
    stats.validate_against(batch, require_reference=False)


def test_policy_recompute_stats_validate_against_rejects_contract_violations():
    batch = _tiny_batch()
    with pytest.raises(ValueError, match="same shape as old_log_probs"):
        PolicyRecomputeStats(new_log_probs=-torch.ones(2, 3)).validate_against(
            batch, require_reference=False
        )
    with pytest.raises(ValueError, match="require_reference=True"):
        PolicyRecomputeStats(
            new_log_probs=-torch.ones(2, 2),
            reference_transition_mean=torch.zeros(2, 2),
        ).validate_against(batch, require_reference=False)
    with pytest.raises(ValueError, match="requires current_transition_mean"):
        PolicyRecomputeStats(new_log_probs=-torch.ones(2, 2)).validate_against(
            batch, require_reference=True
        )
    bad = -torch.ones(2, 2)
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite at active"):
        PolicyRecomputeStats(new_log_probs=bad).validate_against(
            batch, require_reference=False
        )


def test_policy_recompute_stats_reference_contract():
    batch = _tiny_batch()
    new_log_probs = (-torch.ones(2, 2)).requires_grad_(True)
    current = torch.zeros(2, 2, requires_grad=True)
    reference = torch.zeros(2, 2)
    std = torch.ones(2, 2)
    stats = PolicyRecomputeStats(
        new_log_probs=new_log_probs,
        current_transition_mean=current,
        transition_std=std,
        reference_transition_mean=reference,
    )
    stats.validate_against(batch, require_reference=True)

    with pytest.raises(ValueError, match="current_transition_mean must require"):
        PolicyRecomputeStats(
            new_log_probs=new_log_probs,
            current_transition_mean=torch.zeros(2, 2),
            transition_std=std,
            reference_transition_mean=reference,
        ).validate_against(batch, require_reference=True)
    with pytest.raises(ValueError, match="reference_transition_mean must be detached"):
        PolicyRecomputeStats(
            new_log_probs=new_log_probs,
            current_transition_mean=current,
            transition_std=std,
            reference_transition_mean=torch.zeros(2, 2, requires_grad=True),
        ).validate_against(batch, require_reference=True)
    with pytest.raises(ValueError, match="strictly positive"):
        PolicyRecomputeStats(
            new_log_probs=new_log_probs,
            current_transition_mean=current,
            transition_std=torch.zeros(2, 2),
            reference_transition_mean=reference,
        ).validate_against(batch, require_reference=True)


# ---------------------------------------------------------------------------
# RolloutBatch incremental typed fields
# ---------------------------------------------------------------------------


def test_rollout_batch_typed_fields_slice_and_detach_rules():
    batch = RolloutBatch(
        prompts=["a", "b"],
        metadata=[{}, {}],
        latents=torch.zeros(2, 2, 3),
        next_latents=torch.ones(2, 2, 3),
        timesteps=torch.arange(2).repeat(2, 1),
        old_log_probs=-torch.ones(2, 2),
        selected_timestep_index=torch.tensor([0, 1]),
        flash_coefficient=torch.ones(2, 1),
        branch_step_index=torch.tensor([0, 1]),
        # Deliberately T == B: a batch-shared [T] value must still not be
        # sliced by sample indices.
        trajectory_step_index=torch.tensor([5, 6]),
        transition_std_dev=torch.ones(2, 2),
        camera_trajectory=torch.zeros(2, 1, 4, 4, dtype=torch.float64),
        recompute_payload={"extra": torch.arange(2.0, requires_grad=True)},
        artifact_metadata={"note": "batch-shared"},
    )
    sliced = batch.slice([1])
    assert sliced.selected_timestep_index.tolist() == [1]
    assert tuple(sliced.flash_coefficient.shape) == (1, 1)
    assert sliced.branch_step_index.tolist() == [1]
    assert tuple(sliced.transition_std_dev.shape) == (1, 2)
    assert tuple(sliced.camera_trajectory.shape) == (1, 1, 4, 4)
    assert sliced.camera_trajectory.dtype == torch.float64
    assert sliced.recompute_payload["extra"].tolist() == [1.0]
    assert sliced.trajectory_step_index.tolist() == [5, 6]
    assert sliced.artifact_metadata == {"note": "batch-shared"}

    detached = batch.detach()
    assert detached.recompute_payload["extra"].requires_grad is False
    assert detached.artifact_metadata == {"note": "batch-shared"}


def test_rollout_batch_typed_fields_default_to_absent():
    batch = _tiny_batch()
    assert batch.selected_timestep_index is None
    assert batch.flash_coefficient is None
    assert batch.branch_step_index is None
    assert batch.trajectory_step_index is None
    assert batch.transition_std_dev is None
    assert batch.camera_trajectory is None
    assert batch.recompute_payload == {}
    assert batch.artifact_metadata == {}
