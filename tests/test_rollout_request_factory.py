"""Iteration-scoped RolloutRequest composition without model-name branching."""

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager

import pytest
import torch

from visual_rl.algorithms.conditioning.config import WorldR1CameraConfig
from visual_rl.algorithms.conditioning.world_r1_camera import (
    WorldR1CameraConditioner,
)
from visual_rl.algorithms.dynamics.config import (
    FlowSDEConfig,
    WanFlowSDEConfig,
    WanFlowSDEProfile,
)
from visual_rl.algorithms.dynamics.replay import DynamicsReplayRequest
from visual_rl.algorithms.dynamics.sd3_flow_sde import (
    RegisteredSD3FlowSDE,
    SD3DynamicsReplayStateFactory,
    SD3ScheduleReplayState,
)
from visual_rl.algorithms.dynamics.selection import (
    DYNAMICS_SELECTION_SEED_DERIVATION_SCHEMA,
    DYNAMICS_SELECTION_SEED_DERIVATION_VERSION,
    DynamicsSelectionPolicyState,
)
from visual_rl.algorithms.dynamics.wan_flow_sde import (
    RegisteredWanFlowSDE,
    WanDynamicsReplayStateFactory,
    WanScheduleReplayState,
)
from visual_rl.algorithms.rollout.request import (
    IterationRolloutRequestFactory,
    RolloutRequestFactoryError,
)
from visual_rl.algorithms.trainer.interface import IterationIdentity
from visual_rl.core.contracts import ComputePrecision, LikelihoodSemantics
from visual_rl.core.contracts import ReplayTarget
from visual_rl.models import ComponentManager, SchedulerArtifactBlueprint
from visual_rl.models.implementations.sd3 import SD3Adapter, SD3Config, SD3RuntimeParts
from visual_rl.models.implementations.wan import (
    WanConfig,
    WanRuntimeParts,
    WanT2VAdapter,
)
from visual_rl.data.samples import (
    BatchRowContext,
    ExplicitCollator,
    SourceItemContext,
    T2IItem,
    T2VItem,
)


class _SchedulerConfig(dict):
    def __getattr__(self, name):
        return self[name]


class _Scheduler:
    def __init__(self, config=None) -> None:
        values = {"stochastic_sampling": True}
        values.update({} if config is None else config)
        self.config = _SchedulerConfig(values)
        self.set_timesteps(num_inference_steps=2, device="cpu")

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps, device) -> None:
        self.timesteps = torch.linspace(
            900.5,
            100.25,
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


class _DynamicScheduler:
    def __init__(self, config=None) -> None:
        values = {
            "stochastic_sampling": True,
            "use_dynamic_shifting": True,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
        values.update({} if config is None else config)
        self.config = _SchedulerConfig(values)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps, device, mu=None) -> None:
        if mu is None:
            raise AssertionError("dynamic SD3 fixture requires mu")
        sigmas = torch.linspace(
            1.0,
            0.1,
            num_inference_steps,
            dtype=torch.float32,
            device=device,
        )
        sigmas = sigmas * (1.0 + float(mu) / 10.0)
        self.timesteps = sigmas * 1000.0
        self.sigmas = torch.cat(
            [sigmas, torch.zeros(1, dtype=torch.float32, device=device)]
        )


class _PromptEncoder:
    def to(self, device):
        del device
        return self

    def encode(self, prompts, max_sequence_length, guidance_scale):
        raise AssertionError(
            f"request construction encoded prompts: {prompts}, "
            f"{max_sequence_length}, {guidance_scale}"
        )


class _Decoder:
    def to(self, device):
        del device
        return self

    def decode(self, latents, latent_spec):
        raise AssertionError(f"request construction decoded {latents}, {latent_spec}")


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    @contextmanager
    def disable_adapter(self):
        yield


class _RuntimeLoader:
    def __call__(self, family, artifact_path, config, precision):
        del artifact_path, config, precision
        transformer = _Transformer()
        if family == "sd3":
            return SD3RuntimeParts(
                prompt_encoder=_PromptEncoder(),
                transformer=transformer,
                decoder=_Decoder(),
                reference_context=transformer.disable_adapter,
                latent_channels=4,
                scheduler_artifact_blueprint=(
                    SchedulerArtifactBlueprint.from_scheduler(_Scheduler())
                ),
            )
        if family == "wan-t2v":
            return WanRuntimeParts(
                prompt_encoder=_PromptEncoder(),
                transformer=transformer,
                decoder=_Decoder(),
                latent_channels=4,
                scheduler_artifact_blueprint=(
                    SchedulerArtifactBlueprint.from_scheduler(_Scheduler())
                ),
            )
        raise AssertionError(family)


class _DynamicSD3RuntimeLoader:
    def __call__(self, family, artifact_path, config, precision):
        del artifact_path, config, precision
        if family != "sd3":
            raise AssertionError(family)
        transformer = _Transformer()
        return SD3RuntimeParts(
            prompt_encoder=_PromptEncoder(),
            transformer=transformer,
            decoder=_Decoder(),
            reference_context=transformer.disable_adapter,
            latent_channels=4,
            scheduler_artifact_blueprint=(
                SchedulerArtifactBlueprint.from_scheduler(_DynamicScheduler())
            ),
            transformer_patch_size=1,
        )


class _PerIterationDynamics:
    """Test implementation of the narrow training-side factory protocol."""

    def __init__(self, component, replay_state_factory) -> None:
        self.component = component
        self.replay_state_factory = replay_state_factory

    def schedule_conditioning(self, context):
        return self.component.schedule_conditioning(
            self.replay_state_factory.scheduler_blueprint,
            context,
        )

    def create(self, request: DynamicsReplayRequest):
        binding = self.replay_state_factory.create(request)
        if type(binding.replay_state) is not self.component.replay_state_type:
            raise TypeError(
                f"{type(self.component).__name__} requires "
                f"{self.component.replay_state_type.__name__}"
            )
        return self.component.create(binding)


def _loaded_sd3(tmp_path):
    adapter = SD3Adapter(
        SD3Config(artifact_ref="main", resolution=24),
        artifact_path=tmp_path,
        precision=ComputePrecision.FP32,
        model_loader=_RuntimeLoader(),
    )
    manager = ComponentManager(adapter, execution_device="cpu")
    manager.load()
    return adapter, manager


def _loaded_wan(tmp_path):
    adapter = WanT2VAdapter(
        WanConfig(
            artifact_ref="main",
            height=24,
            width=32,
            frames=9,
        ),
        artifact_path=tmp_path,
        precision=ComputePrecision.FP32,
        model_loader=_RuntimeLoader(),
    )
    manager = ComponentManager(adapter, execution_device="cpu")
    manager.load()
    return adapter, manager


def _loaded_dynamic_sd3(tmp_path):
    adapter = SD3Adapter(
        SD3Config(artifact_ref="main", resolution=24),
        artifact_path=tmp_path,
        precision=ComputePrecision.FP32,
        model_loader=_DynamicSD3RuntimeLoader(),
    )
    manager = ComponentManager(adapter, execution_device="cpu")
    manager.load()
    return adapter, manager


def _batch(task: str, *, optimizer_step: int):
    item_type = T2IItem if task == "t2i" else T2VItem
    items = []
    rows = []
    for index in range(2):
        source = SourceItemContext(
            source_item_id=f"source-{optimizer_step}-{index}",
            dataset_source_id="dataset-main",
            dataset_index=index,
            dataset_revision="revision-1",
        )
        items.append(item_type(prompt=f"prompt-{index}", source=source))
        rows.append(
            BatchRowContext(
                occurrence_id=f"occurrence-{optimizer_step}-{index}",
                group_id=f"group-{optimizer_step}",
                member_id=index,
                phase="main",
                optimizer_step=optimizer_step,
                source_item_id=source.source_item_id,
            )
        )
    return ExplicitCollator().collate_samples(tuple(items), tuple(rows))


def _identity(samples) -> IterationIdentity:
    return IterationIdentity(
        optimizer_step=samples.rows[0].optimizer_step,
        source_id="dataset-main",
        phase_id=samples.rows[0].phase,
        row_identities=tuple(row.identity for row in samples.rows),
        group_ids=tuple(row.group_id for row in samples.rows),
        member_ids=tuple(row.member_id for row in samples.rows),
    )


def _sd3_component_factory() -> RegisteredSD3FlowSDE:
    return RegisteredSD3FlowSDE.from_config(
        FlowSDEConfig(noise_level=0.7),
        runtime_context={},
    )


def _wan_component_factory() -> RegisteredWanFlowSDE:
    return RegisteredWanFlowSDE.from_config(
        WanFlowSDEConfig(
            profile=WanFlowSDEProfile.CONDITIONED,
            likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
            replay_target=ReplayTarget.SAMPLED_ACTION,
        ),
        runtime_context={},
    )


def _sd3_rollout_factory(adapter):
    return _PerIterationDynamics(
        _sd3_component_factory(),
        SD3DynamicsReplayStateFactory(adapter.scheduler_artifact_blueprint),
    )


def _wan_rollout_factory(adapter):
    return _PerIterationDynamics(
        _wan_component_factory(),
        WanDynamicsReplayStateFactory(adapter.scheduler_artifact_blueprint),
    )


def test_consecutive_iterations_receive_distinct_bindings_instances_and_rng(
    tmp_path,
) -> None:
    adapter, manager = _loaded_wan(tmp_path)
    factory = IterationRolloutRequestFactory(
        adapter=adapter,
        dynamics_factory=_wan_rollout_factory(adapter),
        num_steps=3,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=2718,
        device="cpu",
        dtype=torch.float32,
    )
    first_samples = _batch("t2v", optimizer_step=0)
    second_samples = _batch("t2v", optimizer_step=1)

    first = factory(first_samples, _identity(first_samples))
    second = factory(second_samples, _identity(second_samples))

    assert first.dynamics is not second.dynamics
    assert first.dynamics_replay_binding is not second.dynamics_replay_binding
    assert (
        first.dynamics_replay_binding.binding_identity
        != second.dynamics_replay_binding.binding_identity
    )
    assert not torch.equal(first.generator.get_state(), second.generator.get_state())
    assert not torch.equal(
        first.selection_generator.get_state(),
        second.selection_generator.get_state(),
    )
    assert not torch.equal(
        first.generator.get_state(),
        first.selection_generator.get_state(),
    )
    manager.close()


def test_abort_retry_and_fresh_factory_resume_recreate_seed_and_identity(
    tmp_path,
) -> None:
    adapter, manager = _loaded_sd3(tmp_path)
    kwargs = {
        "adapter": adapter,
        "dynamics_factory": _sd3_rollout_factory(adapter),
        "num_steps": 3,
        "likelihood_semantics": LikelihoodSemantics.EXACT_ENV_ACTION,
        "base_seed": 31415,
        "device": "cpu",
        "dtype": torch.float32,
    }
    factory = IterationRolloutRequestFactory(**kwargs)
    samples = _batch("t2i", optimizer_step=7)
    identity = _identity(samples)
    first = factory(samples, identity)
    first_rollout_state = first.generator.get_state()
    first_selection_state = first.selection_generator.get_state()
    torch.randn(first.latent_spec.shape, generator=first.generator)

    retry = factory(samples, identity)
    resumed = IterationRolloutRequestFactory(**kwargs)(samples, identity)

    for rebuilt in (retry, resumed):
        assert rebuilt.dynamics is not first.dynamics
        assert rebuilt.dynamics_replay_binding is not first.dynamics_replay_binding
        assert rebuilt.dynamics_replay_binding == first.dynamics_replay_binding
        assert isinstance(
            rebuilt.dynamics_replay_binding.replay_state, SD3ScheduleReplayState
        )
        assert torch.equal(rebuilt.generator.get_state(), first_rollout_state)
        assert torch.equal(
            rebuilt.selection_generator.get_state(),
            first_selection_state,
        )
    manager.close()


def test_sd3_request_factory_binds_loaded_patch_and_latent_geometry(tmp_path) -> None:
    adapter, manager = _loaded_dynamic_sd3(tmp_path)
    factory = IterationRolloutRequestFactory(
        adapter=adapter,
        dynamics_factory=_sd3_rollout_factory(adapter),
        num_steps=3,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=17,
        device="cpu",
        dtype=torch.float32,
    )
    samples = _batch("t2i", optimizer_step=2)

    request = factory(samples, _identity(samples))

    conditioning = request.dynamics_replay_binding.request.schedule_conditioning
    assert conditioning is not None
    assert conditioning.latent_height == 3
    assert conditioning.latent_width == 3
    assert conditioning.patch_size == 1
    assert conditioning.image_seq_len == 9
    assert conditioning.dynamic_shift is not None
    assert request.dynamics.dynamics_config_identity.endswith(
        conditioning.conditioning_identity
    )
    manager.close()


def test_checkpointed_policy_recreates_next_iteration_request_and_schedule(
    tmp_path,
) -> None:
    adapter, manager = _loaded_sd3(tmp_path)
    original = IterationRolloutRequestFactory(
        adapter=adapter,
        dynamics_factory=_sd3_rollout_factory(adapter),
        num_steps=3,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=31415,
        device="cpu",
        dtype=torch.float32,
        selection_contract_identity="test.single-step-selection.v1",
    )
    checkpoint_payload = original.dynamics_selection_policy.to_checkpoint_payload()
    assert set(checkpoint_payload) == {
        "schema_version",
        "base_seed",
        "selection_contract_identity",
        "seed_derivation_schema",
        "seed_derivation_version",
        "policy_identity",
    }
    assert "next_optimizer_step" not in checkpoint_payload
    assert "generator_state" not in checkpoint_payload
    restored_policy = DynamicsSelectionPolicyState.from_checkpoint_payload(
        checkpoint_payload
    )
    resumed = IterationRolloutRequestFactory(
        adapter=adapter,
        dynamics_factory=_sd3_rollout_factory(adapter),
        num_steps=3,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=restored_policy.base_seed,
        device="cpu",
        dtype=torch.float32,
        selection_contract_identity=restored_policy.selection_contract_identity,
        seed_derivation_schema=restored_policy.seed_derivation_schema,
        seed_derivation_version=restored_policy.seed_derivation_version,
    )
    samples = _batch("t2i", optimizer_step=8)
    identity = _identity(samples)

    expected = original(samples, identity)
    observed = resumed(samples, identity)

    assert resumed.dynamics_selection_policy == restored_policy
    assert observed.selection_contract_identity == (
        restored_policy.selection_contract_identity
    )
    assert observed.dynamics_replay_binding == expected.dynamics_replay_binding
    assert (
        observed.dynamics_replay_binding.replay_state_identity
        == expected.dynamics_replay_binding.replay_state_identity
    )
    assert torch.equal(observed.generator.get_state(), expected.generator.get_state())
    assert torch.equal(
        observed.selection_generator.get_state(),
        expected.selection_generator.get_state(),
    )
    manager.close()


def test_base_seed_or_derivation_schema_changes_policy_and_request(tmp_path) -> None:
    adapter, manager = _loaded_sd3(tmp_path)
    common = {
        "adapter": adapter,
        "num_steps": 3,
        "likelihood_semantics": LikelihoodSemantics.EXACT_ENV_ACTION,
        "device": "cpu",
        "dtype": torch.float32,
    }
    baseline = IterationRolloutRequestFactory(
        **common,
        dynamics_factory=_sd3_rollout_factory(adapter),
        base_seed=7,
    )
    changed_seed = IterationRolloutRequestFactory(
        **common,
        dynamics_factory=_sd3_rollout_factory(adapter),
        base_seed=8,
    )
    changed_schema = IterationRolloutRequestFactory(
        **common,
        dynamics_factory=_sd3_rollout_factory(adapter),
        base_seed=7,
        seed_derivation_schema=(
            DYNAMICS_SELECTION_SEED_DERIVATION_SCHEMA + ".test-next"
        ),
        seed_derivation_version=DYNAMICS_SELECTION_SEED_DERIVATION_VERSION,
    )
    samples = _batch("t2i", optimizer_step=9)
    identity = _identity(samples)
    requests = tuple(
        factory(samples, identity)
        for factory in (baseline, changed_seed, changed_schema)
    )

    assert (
        len(
            {
                baseline.dynamics_selection_policy_identity,
                changed_seed.dynamics_selection_policy_identity,
                changed_schema.dynamics_selection_policy_identity,
            }
        )
        == 3
    )
    assert (
        len({request.dynamics_replay_binding.binding_identity for request in requests})
        == 3
    )
    assert not torch.equal(
        requests[0].selection_generator.get_state(),
        requests[1].selection_generator.get_state(),
    )
    assert not torch.equal(
        requests[0].selection_generator.get_state(),
        requests[2].selection_generator.get_state(),
    )
    manager.close()


def test_selection_contract_changes_checkpoint_policy_and_iteration_stream(
    tmp_path,
) -> None:
    adapter, manager = _loaded_sd3(tmp_path)
    common = {
        "adapter": adapter,
        "dynamics_factory": _sd3_rollout_factory(adapter),
        "num_steps": 3,
        "likelihood_semantics": LikelihoodSemantics.EXACT_ENV_ACTION,
        "base_seed": 17,
        "device": "cpu",
        "dtype": torch.float32,
    }
    first = IterationRolloutRequestFactory(
        **common,
        selection_contract_identity="test.selection-contract.first.v1",
    )
    second = IterationRolloutRequestFactory(
        **common,
        selection_contract_identity="test.selection-contract.second.v1",
    )
    samples = _batch("t2i", optimizer_step=10)
    identity = _identity(samples)

    first_request = first(samples, identity)
    second_request = second(samples, identity)

    assert first.dynamics_selection_policy_identity != (
        second.dynamics_selection_policy_identity
    )
    assert first_request.selection_contract_identity != (
        second_request.selection_contract_identity
    )
    assert not torch.equal(
        first_request.selection_generator.get_state(),
        second_request.selection_generator.get_state(),
    )
    manager.close()


def test_model_and_dynamics_replay_state_type_mismatch_is_rejected(tmp_path) -> None:
    adapter, manager = _loaded_sd3(tmp_path)
    dynamics_factory = _PerIterationDynamics(
        _wan_component_factory(),
        SD3DynamicsReplayStateFactory(adapter.scheduler_artifact_blueprint),
    )

    with pytest.raises(TypeError, match="WanScheduleReplayState"):
        dynamics_factory.create(
            DynamicsReplayRequest(
                rollout_identity="mismatched-replay-state",
                num_steps=2,
            )
        )
    manager.close()


def test_camera_conditioner_binds_generic_wan_geometry_itself(tmp_path) -> None:
    adapter, manager = _loaded_wan(tmp_path)
    conditioner = WorldR1CameraConditioner(WorldR1CameraConfig(frames_per_trajectory=9))
    factory = IterationRolloutRequestFactory(
        adapter=adapter,
        dynamics_factory=_wan_rollout_factory(adapter),
        num_steps=2,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=99,
        device="cpu",
        dtype=torch.float16,
        conditioner=conditioner,
    )
    samples = _batch("t2v", optimizer_step=0)

    request = factory(samples, _identity(samples))
    spec = request.conditioner_latent_spec

    assert spec is not None
    assert (
        spec.batch_size,
        spec.channels,
        spec.latent_frames,
        spec.latent_height,
        spec.latent_width,
    ) == (2, 4, 3, 3, 4)
    assert (
        spec.output_frames,
        spec.output_height,
        spec.output_width,
        spec.temporal_compression,
        spec.spatial_compression,
    ) == (9, 24, 32, 4, 8)
    assert spec.device == torch.device("cpu")
    assert spec.dtype is torch.float16
    assert request.latent_spec.shape == (2, 4, 3, 3, 4)
    assert isinstance(
        request.dynamics_replay_binding.replay_state, WanScheduleReplayState
    )
    manager.close()


def test_non_conditionable_model_refuses_camera_geometry(tmp_path) -> None:
    adapter, manager = _loaded_sd3(tmp_path)
    factory = IterationRolloutRequestFactory(
        adapter=adapter,
        dynamics_factory=_sd3_rollout_factory(adapter),
        num_steps=2,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=99,
        device="cpu",
        dtype=torch.float32,
        conditioner=WorldR1CameraConditioner(WorldR1CameraConfig()),
    )
    samples = _batch("t2i", optimizer_step=0)

    with pytest.raises(ValueError, match="requires BCTHW"):
        factory(samples, _identity(samples))
    manager.close()


def test_factory_source_has_no_model_alias_branch() -> None:
    tree = ast.parse(inspect.getsource(IterationRolloutRequestFactory))
    string_constants = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "sd3" not in string_constants
    assert "wan" not in string_constants
    assert "wan-t2v" not in string_constants


def test_iteration_identity_must_match_exact_sample_rows(tmp_path) -> None:
    adapter, manager = _loaded_sd3(tmp_path)
    factory = IterationRolloutRequestFactory(
        adapter=adapter,
        dynamics_factory=_sd3_rollout_factory(adapter),
        num_steps=2,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=4,
        device="cpu",
        dtype=torch.float32,
    )
    samples = _batch("t2i", optimizer_step=0)
    other = _batch("t2i", optimizer_step=1)

    with pytest.raises(RolloutRequestFactoryError, match="canonical iteration"):
        factory(samples, _identity(other))
    manager.close()
