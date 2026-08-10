"""Fake-leaf vertical slices for every checked-in schema-v2 demo recipe.

These tests prove that each official config can traverse the sole default
composition root, perform one optimizer update, write its final checkpoint,
and restore that completed checkpoint without executing training work.  They
deliberately do *not* claim native model, reward, or numerical parity: model
geometry and training length are reduced in a derived temporary config, and
only the heavyweight model/reward leaves are replaced.
"""

from __future__ import annotations

import hashlib
import importlib
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from tests import test_default_composition_e2e as e2e_fixture
import torch
import yaml

from visual_rl.algorithms.rollout.config import SingleStepRolloutConfig
from visual_rl.algorithms.rewards.clients.input import (
    pointwise_reward_output,
    resolve_pointwise_reward_input,
)
from visual_rl.composition.config.compiler import default_catalog
from visual_rl.composition.registry import Catalog
from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    ComponentDeclaration,
    ComponentDescriptor,
)
from visual_rl.core.types import to_plain_dict
from visual_rl.models import SchedulerArtifactBlueprint
from visual_rl.models.catalog import WanConfig
from visual_rl.models.implementations.sd3 import SD3RuntimeParts
from visual_rl.models.implementations.wan import WanRuntimeParts, WanT2VAdapter
from visual_rl.composition.preflight import FilesystemArtifactIdentityResolver
from visual_rl.runtime import (
    AcquiredRewardResource,
    ControllerStage,
    ControllerState,
    CoordinatorCheckpointSink,
    CoordinatorRestoreService,
    CoordinatorRunFinalizer,
    DefaultComponentRuntimeBinder,
    DefaultModelRuntimeProbe,
    DefaultPreprocessIdentityProvider,
    DefaultRuntimeContextProvider,
    DefaultRuntimeSessionFactory,
    DefaultStageAssembler,
    RewardResourceAcquireRequest,
    RewardResourceBindingFacts,
)
from visual_rl.runtime.composition import create_default_run_controller
from visual_rl.runtime.composition import create_run_controller
from visual_rl.runtime.types import RunResult

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_OFFICIAL_CONFIG_ROOT = _REPOSITORY_ROOT / "configs" / "v2"


@dataclass(frozen=True, slots=True)
class _DemoCase:
    filename: str
    recipe_id: str
    model_family: str
    source_ids: tuple[str, ...]
    reward_ids: tuple[str, ...]
    expects_reference_state: bool


class _RecordingComponentBinder:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.evidence: list[object] = []
        self.public_axis_identities: list[
            tuple[dict[str, object], dict[str, object]]
        ] = []

    def bind(self, request: object) -> object:
        graph = request.graph
        self.public_axis_identities.append(
            (
                graph.components.binding("model").declaration.to_identity_payload(),
                graph.components.binding("algorithm").declaration.to_identity_payload(),
            )
        )
        result = self.delegate.bind(request)
        self.evidence.append(result)
        return result


_DEMO_CASES = (
    _DemoCase(
        "flow_grpo_sd3.yaml",
        "flow_grpo_v1",
        "sd3",
        ("main",),
        ("reward_quality",),
        True,
    ),
    _DemoCase(
        "flow_grpo_wan.yaml",
        "flow_grpo_v1",
        "wan-t2v",
        ("main",),
        ("reward_general",),
        False,
    ),
    _DemoCase(
        "tempflow_sd3.yaml",
        "tempflow_grpo_v1",
        "sd3",
        ("main",),
        ("reward_quality",),
        False,
    ),
    _DemoCase(
        "flash_wan.yaml",
        "flash_grpo_v1",
        "wan-t2v",
        ("main",),
        ("reward_general",),
        False,
    ),
    _DemoCase(
        "world_r1_core_wan.yaml",
        "world_r1_core_v1",
        "wan-t2v",
        ("main",),
        ("reward_3d", "reward_general"),
        False,
    ),
    _DemoCase(
        "world_r1_release_surrogate_wan.yaml",
        "world_r1_release_surrogate_v1",
        "wan-t2v",
        ("dynamic", "main"),
        ("reward_3d", "reward_general"),
        False,
    ),
)


@dataclass(frozen=True, slots=True)
class _PublicAxisCase:
    base_filename: str
    recipe_id: str
    model_family: str
    algorithm_id: str
    reward_id: str


class FakeWanPolicyA(WanT2VAdapter):
    """First fake public model identity over the same heavyweight Wan leaf."""

    @classmethod
    def describe(cls, config: object):
        return replace(
            super().describe(config),
            component_id="fake-wan-a",
        )

    def describe_preprocess(self):
        return replace(
            super().describe_preprocess(),
            implementation_id=("tests.test_v2_demo_default_composition:FakeWanPolicyA"),
        )


class FakeWanPolicyB(WanT2VAdapter):
    """Second fake public model identity over the same heavyweight Wan leaf."""

    @classmethod
    def describe(cls, config: object):
        return replace(
            super().describe(config),
            component_id="fake-wan-b",
        )

    def describe_preprocess(self):
        return replace(
            super().describe_preprocess(),
            implementation_id=("tests.test_v2_demo_default_composition:FakeWanPolicyB"),
        )


class _FakeWanDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.models.catalog:WanConfig"
    COMPONENT_ID = ""

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, object],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        config = WanConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config,
            declared_contract=replace(
                config.describe_contract(),
                component_id=cls.COMPONENT_ID,
            ),
        )


class FakeWanADeclarationProvider(_FakeWanDeclarationProvider):
    COMPONENT_ID = "fake-wan-a"


class FakeWanBDeclarationProvider(_FakeWanDeclarationProvider):
    COMPONENT_ID = "fake-wan-b"


def _public_axis_catalog() -> Catalog:
    descriptors = (
        ComponentDescriptor(
            alias="fake-wan-a",
            implementation_class_path=(
                "tests.test_v2_demo_default_composition:FakeWanPolicyA"
            ),
            declaration_provider_path=(
                "tests.test_v2_demo_default_composition:FakeWanADeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
        ComponentDescriptor(
            alias="fake-wan-b",
            implementation_class_path=(
                "tests.test_v2_demo_default_composition:FakeWanPolicyB"
            ),
            declaration_provider_path=(
                "tests.test_v2_demo_default_composition:FakeWanBDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
    )
    base = default_catalog()
    return Catalog(
        tuple(
            registry.register(*descriptors) if registry.kind == "model" else registry
            for registry in base.registries
        )
    )


def _create_catalog_run_controller(
    *,
    catalog: Catalog,
    code_root: Path,
    model_loader: object,
    reward_resource_factory: object,
):
    """Use the explicit production root for a test-owned canonical catalog."""

    runtime_factory = DefaultRuntimeSessionFactory(
        model_loader=model_loader,
        reward_resource_factory=reward_resource_factory,
    )
    binder = DefaultComponentRuntimeBinder(
        model_probe=DefaultModelRuntimeProbe(),
        preprocess_identity_provider=DefaultPreprocessIdentityProvider(),
    )
    finalizer = CoordinatorRunFinalizer()
    return create_run_controller(
        catalog=catalog,
        artifact_resolver=FilesystemArtifactIdentityResolver(code_root),
        runtime_factory=runtime_factory,
        runtime_context_provider=DefaultRuntimeContextProvider(),
        stage_assembler=DefaultStageAssembler(),
        component_binder=binder,
        checkpoint_sink=CoordinatorCheckpointSink(finalizer=finalizer),
        restore_service=CoordinatorRestoreService(finalizer=finalizer),
    )


_PUBLIC_AXIS_CASES = (
    _PublicAxisCase(
        "flow_grpo_wan.yaml",
        "flow_grpo_v1",
        "fake-wan-a",
        "flow-grpo",
        "reward_general",
    ),
    _PublicAxisCase(
        "flow_grpo_wan.yaml",
        "flow_grpo_v1",
        "fake-wan-b",
        "flow-grpo",
        "reward_general",
    ),
    _PublicAxisCase(
        "flash_wan.yaml",
        "flash_grpo_v1",
        "fake-wan-a",
        "flash-grpo",
        "reward_general",
    ),
    _PublicAxisCase(
        "flash_wan.yaml",
        "flash_grpo_v1",
        "fake-wan-b",
        "flash-grpo",
        "reward_general",
    ),
)


class _WanSchedulerConfig(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeWanScheduler:
    scheduler_identity = "fake-wan-scheduler.v1"

    def __init__(self, config=None) -> None:
        values = {"stochastic_sampling": True}
        values.update({} if config is None else config)
        self.config = _WanSchedulerConfig(values)
        self.set_timesteps(num_inference_steps=2, device="cpu")

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps: int, device: object) -> None:
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


class _FakeWanPromptEncoder:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[tuple[tuple[str, ...], int, float, bool]] = []

    def to(self, device: object):
        self.device = torch.device(device)
        return self

    def encode(
        self,
        prompts: tuple[str, ...],
        max_sequence_length: int,
        guidance_scale: float,
    ):
        batch_size = len(prompts)
        positive = (
            torch.linspace(
                0.2,
                0.8,
                batch_size,
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(batch_size, 1, 1)
            .expand(-1, 3, 2)
        )
        negative = torch.full_like(positive, -0.2) if guidance_scale > 1.0 else None
        self.calls.append(
            (
                prompts,
                max_sequence_length,
                guidance_scale,
                negative is not None,
            )
        )
        return positive, negative


class _FakeWanTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_scale = torch.nn.Parameter(torch.tensor(0.2))
        self.register_buffer("base_scale", torch.tensor(0.45))

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: object,
        return_dict: bool,
    ):
        del timestep, attention_kwargs, return_dict
        conditioning = encoder_hidden_states.mean(
            dim=tuple(range(1, encoder_hidden_states.ndim))
        ).reshape(hidden_states.shape[0], *([1] * (hidden_states.ndim - 1)))
        return (hidden_states * (self.base_scale + self.policy_scale) + conditioning,)


class _FakeWanDecoder:
    def __init__(self, output_frames: int) -> None:
        self.output_frames = output_frames

    def to(self, device: object):
        del device
        return self

    def decode(self, latents: torch.Tensor, latent_spec: object) -> torch.Tensor:
        assert tuple(latents.shape) == latent_spec.shape
        frame_indices = (
            torch.linspace(
                0,
                latents.shape[2] - 1,
                self.output_frames,
                device=latents.device,
            )
            .round()
            .to(dtype=torch.int64)
        )
        # Wan's public decoded-media contract is BFCHW, not latent BCTHW.
        return (
            latents[:, :3]
            .index_select(2, frame_indices)
            .permute(0, 2, 1, 3, 4)
            .detach()
            .clone()
        )


class _FakeModelLoader:
    """One test leaf that dispatches only on the model family requested by G2."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.transformers: list[torch.nn.Module] = []
        self.wan_prompt_encoders: list[_FakeWanPromptEncoder] = []

    def __call__(self, family, artifact_path, config, precision):
        self.calls.append((family, artifact_path, config, precision))
        if family == "sd3":
            transformer = e2e_fixture._FakeTransformer()
            self.transformers.append(transformer)
            return SD3RuntimeParts(
                prompt_encoder=e2e_fixture._FakePromptEncoder(),
                transformer=transformer,
                decoder=e2e_fixture._FakeDecoder(),
                reference_context=transformer.disable_adapter,
                latent_channels=4,
                scheduler_artifact_blueprint=(
                    SchedulerArtifactBlueprint.from_scheduler(
                        e2e_fixture._FakeScheduler()
                    )
                ),
            )
        if family == "wan-t2v":
            transformer = _FakeWanTransformer()
            prompt_encoder = _FakeWanPromptEncoder()
            self.transformers.append(transformer)
            self.wan_prompt_encoders.append(prompt_encoder)
            return WanRuntimeParts(
                prompt_encoder=prompt_encoder,
                transformer=transformer,
                decoder=_FakeWanDecoder(config.frames),
                latent_channels=4,
                scheduler_artifact_blueprint=(
                    SchedulerArtifactBlueprint.from_scheduler(_FakeWanScheduler())
                ),
            )
        raise AssertionError(f"unexpected model family: {family}")


class _FakeRewardResource:
    """Deterministic scoring leaf with observable pool ownership semantics."""

    def __init__(self) -> None:
        self.activate_calls = 0
        self.close_calls = 0
        self.score_calls = 0

    def activate(self) -> None:
        self.activate_calls += 1

    def score(self, *, batch):
        resolved = resolve_pointwise_reward_input(batch)
        self.score_calls += 1
        values = np.linspace(
            -1.0,
            1.0,
            resolved.flat_size,
            dtype=np.float64,
        )
        return pointwise_reward_output(
            resolved,
            values,
            shared_metadata={"leaf": "fake-composition-smoke"},
            sample_metadata=tuple(
                {"row": index} for index in range(resolved.flat_size)
            ),
        )

    def close(self) -> None:
        self.close_calls += 1


class _FakeRewardResourceFactory:
    def __init__(self) -> None:
        self.requests: list[RewardResourceAcquireRequest] = []
        self.resources: list[_FakeRewardResource] = []

    def __call__(
        self,
        request: RewardResourceAcquireRequest,
    ) -> AcquiredRewardResource:
        self.requests.append(request)
        resource = _FakeRewardResource()
        self.resources.append(resource)
        policy = request.descriptor.allowed_runtime_policy
        runtime_device = request.runtime_facts.device
        device_domain = runtime_device.split(":", 1)[0]
        device = (
            runtime_device
            if device_domain in policy.allowed_devices
            else policy.allowed_devices[0]
        )
        dtype = (
            request.runtime_facts.precision
            if request.runtime_facts.precision in policy.allowed_dtypes
            else policy.allowed_dtypes[0]
        )
        endpoint_identity = hashlib.sha256(
            request.reward_resource_spec_id.encode("ascii")
        ).hexdigest()
        return AcquiredRewardResource(
            resource=resource,
            binding_facts=RewardResourceBindingFacts(
                endpoint_identity=endpoint_identity,
                protocol=request.descriptor.protocol,
                protocol_version=request.descriptor.protocol_version,
                device=device,
                dtype=dtype,
                worker_domain=policy.allowed_worker_domains[0],
            ),
        )


class _UnavailableRewardResourceFactory:
    """Fail the readiness/revision gate before any policy allocation."""

    def __init__(self) -> None:
        self.requests: list[RewardResourceAcquireRequest] = []

    def __call__(
        self,
        request: RewardResourceAcquireRequest,
    ) -> AcquiredRewardResource:
        self.requests.append(request)
        raise RuntimeError("injected reward readiness/revision failure")


def _write_fake_artifacts(root: Path, case: _DemoCase) -> dict[str, object]:
    model = root / "model"
    model.mkdir(parents=True)
    (model / "weights.fake").write_bytes(case.model_family.encode("ascii"))

    datasets: dict[str, Path] = {}
    for source_id in case.source_ids:
        path = root / f"{source_id}-prompts.txt"
        if source_id == "dynamic":
            prompts = "Camera orbits around a statue\n"
        elif case.model_family in {"wan-t2v", "fake-wan-a", "fake-wan-b"}:
            prompts = (
                "Camera pushes forward through a forest\n"
                "Camera pans left across a valley\n"
            )
        else:
            prompts = "a red cube\na blue sphere\n"
        path.write_text(prompts, encoding="utf-8")
        datasets[source_id] = path

    rewards: dict[str, Path] = {}
    for reward_id in case.reward_ids:
        path = root / reward_id
        path.mkdir()
        (path / "revision.txt").write_text(
            f"{reward_id}-fake-v1\n",
            encoding="utf-8",
        )
        rewards[reward_id] = path
    return {"model": model, "datasets": datasets, "rewards": rewards}


def _write_smoke_config(
    root: Path,
    case: _DemoCase,
    *,
    output_dir: Path,
    artifacts: dict[str, object],
    resume_from: Path | None = None,
    max_optimizer_steps: int = 1,
) -> Path:
    source_path = _OFFICIAL_CONFIG_ROOT / case.filename
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["recipe"] == case.recipe_id

    overrides = payload.setdefault("overrides", {})
    model_override = overrides.setdefault("model", {})
    if case.model_family == "sd3":
        model_override["params"] = {
            "artifact_ref": "main",
            "resolution": 16,
            "guidance_scale": 1.0,
            "gradient_checkpointing": False,
            "max_sequence_length": 8,
        }
    else:
        model_override["params"] = {
            "artifact_ref": "main",
            "height": 16,
            "width": 16,
            "frames": (
                81
                if case.recipe_id
                in {"world_r1_core_v1", "world_r1_release_surrogate_v1"}
                else 5
            ),
            "gradient_checkpointing": False,
            "max_sequence_length": 512,
            "vae_tiling": False,
        }

    if case.recipe_id == "tempflow_grpo_v1":
        rollout_params = {
            "num_steps": 2,
            "branch_count": 6,
        }
    elif case.recipe_id == "flash_grpo_v1":
        rollout_params = {
            "selected_timestep_policy": "uniform",
            "num_steps": 2,
            "candidate_timestep_window": [0, 10],
            "selection_key": "prompt",
            "selection_domain": "single_process",
        }
    else:
        rollout_params = {"num_steps": 2}
    algorithm_params = overrides.setdefault("algorithm", {}).setdefault("params", {})
    algorithm_params.update(rollout_params)

    training = overrides.setdefault("training", {})
    training.update(
        {
            "global_prompt_batch_size": 1,
            "max_optimizer_steps": max_optimizer_steps,
            "gradient_accumulation_steps": 1,
            "adamw": {"learning_rate": 0.05, "weight_decay": 0.0},
            "lr_schedule": {"warmup_steps": 0},
            "update_safety": {
                "require_finite_gradients": True,
                "require_nonzero_gradients": True,
                "max_grad_norm": 1.0,
            },
        }
    )
    overrides.setdefault("execution", {})["precision"] = "fp32"

    launch = payload["launch"]
    launch["output_dir"] = output_dir.as_posix()
    launch["resume_from"] = None if resume_from is None else resume_from.as_posix()
    launch["checkpoint_every_optimizer_steps"] = 1
    launch["artifacts"] = {
        "model": artifacts["model"].as_posix(),
        "datasets": {
            key: value.as_posix() for key, value in artifacts["datasets"].items()
        },
        "rewards": {
            key: value.as_posix() for key, value in artifacts["rewards"].items()
        },
    }

    suffix = "smoke" if resume_from is None else "resume"
    path = root / f"{case.filename.removesuffix('.yaml')}-{suffix}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("case", _DEMO_CASES, ids=lambda case: case.filename)
def test_official_v2_demo_continuation_exactly_matches_continuous_two_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _DemoCase,
) -> None:
    """G7a: every official route restores into the exact next update.

    A completed-checkpoint no-op only proves that restore can inspect terminal
    state.  This gate instead commits step 1, crashes immediately after that
    durable safe point, resumes in a fresh controller, and compares the step-2
    state with an uninterrupted run.  Heavy model/reward leaves remain fake;
    real GPU resume is still a separate M7 gate.
    """

    e2e_fixture._patch_default_runtime(monkeypatch)
    # Prime the optional dependency before either seeded controller starts so
    # one-time module initialization cannot perturb only the first branch.
    importlib.import_module("diffusers")

    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "runtime.py").write_text(
        "FAKE_SIX_ROUTE_CONTINUATION = True\n",
        encoding="utf-8",
    )
    artifacts = _write_fake_artifacts(tmp_path / "artifacts", case)
    continuous_root = tmp_path / "continuous-config"
    interrupted_root = tmp_path / "interrupted-config"
    resume_root = tmp_path / "resume-config"
    continuous_root.mkdir()
    interrupted_root.mkdir()
    resume_root.mkdir()
    continuous_output = tmp_path / "continuous-output"
    interrupted_output = tmp_path / "interrupted-output"
    continuous_config = _write_smoke_config(
        continuous_root,
        case,
        output_dir=continuous_output,
        artifacts=artifacts,
        max_optimizer_steps=2,
    )
    interrupted_config = _write_smoke_config(
        interrupted_root,
        case,
        output_dir=interrupted_output,
        artifacts=artifacts,
        max_optimizer_steps=2,
    )

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    try:
        continuous_controller = create_default_run_controller(
            code_root=code_root,
            model_loader=_FakeModelLoader(),
            reward_resource_factory=_FakeRewardResourceFactory(),
        )
        continuous_result = continuous_controller.run(continuous_config)
        continuous_step1 = e2e_fixture._checkpoint_snapshot(
            continuous_output / "checkpoints" / "step-1"
        )
        continuous_step2 = e2e_fixture._checkpoint_snapshot(
            continuous_result.authoritative_checkpoint
        )

        interrupted_controller = create_default_run_controller(
            code_root=code_root,
            model_loader=_FakeModelLoader(),
            reward_resource_factory=_FakeRewardResourceFactory(),
        )
        checkpoint_sink = interrupted_controller._backend.checkpoint_sink
        checkpoint_safe_point = checkpoint_sink.checkpoint_safe_point
        committed_receipts: list[object] = []

        def commit_then_crash(request: object):
            receipt = checkpoint_safe_point(request)
            committed_receipts.append(receipt)
            raise e2e_fixture._InjectedCrash(
                "injected after durable route step-1 checkpoint"
            )

        checkpoint_sink.checkpoint_safe_point = commit_then_crash
        with pytest.raises(e2e_fixture._InjectedCrash, match="route step-1"):
            interrupted_controller.run(interrupted_config)

        step1_path = interrupted_output / "checkpoints" / "step-1"
        interrupted_step1 = e2e_fixture._checkpoint_snapshot(step1_path)
        step1_files_before_resume = e2e_fixture._file_tree_identity(step1_path)
        assert len(committed_receipts) == 1
        assert committed_receipts[0].checkpoint_path == step1_path
        assert interrupted_controller.state is ControllerState.FAILED
        assert not (interrupted_output / "SUCCESS").exists()
        assert not (interrupted_output / "checkpoints" / "step-2").exists()
        e2e_fixture._assert_checkpoint_state_exact(
            continuous_step1,
            interrupted_step1,
            include_run_summary=True,
        )

        resume_config = _write_smoke_config(
            resume_root,
            case,
            output_dir=interrupted_output,
            artifacts=artifacts,
            resume_from=step1_path,
            max_optimizer_steps=2,
        )
        resume_controller = create_default_run_controller(
            code_root=code_root,
            model_loader=_FakeModelLoader(),
            reward_resource_factory=_FakeRewardResourceFactory(),
        )
        resumed_result = resume_controller.run(resume_config)

        assert e2e_fixture._file_tree_identity(step1_path) == (
            step1_files_before_resume
        )
        assert {
            path.name
            for path in (interrupted_output / "checkpoints").iterdir()
            if path.is_dir()
        } == {"step-1", "step-2"}
        assert resumed_result.authoritative_checkpoint == (
            interrupted_output / "checkpoints" / "step-2"
        )
        resumed_step2 = e2e_fixture._checkpoint_snapshot(
            resumed_result.authoritative_checkpoint
        )
        e2e_fixture._assert_checkpoint_state_exact(
            continuous_step2,
            resumed_step2,
            include_run_summary=False,
        )
        assert continuous_result.last_metrics == resumed_result.last_metrics
        assert continuous_controller.completed_stages == tuple(ControllerStage)
        assert resume_controller.completed_stages == tuple(ControllerStage)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)


def _write_public_axis_matrix_config(
    root: Path,
    case: _PublicAxisCase,
    *,
    output_dir: Path,
    artifacts: dict[str, object],
) -> Path:
    """Derive one orthogonal public-axis case from an official recipe."""

    demo_case = _DemoCase(
        filename=case.base_filename,
        recipe_id=case.recipe_id,
        model_family=case.model_family,
        source_ids=("main",),
        reward_ids=(case.reward_id,),
        expects_reference_state=False,
    )
    path = _write_smoke_config(
        root,
        demo_case,
        output_dir=output_dir,
        artifacts=artifacts,
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    overrides = payload["overrides"]
    # Each official base recipe already owns its internal rollout, Dynamics,
    # credit, trainer, reward, and source choices.  Crossing the public model
    # axis therefore changes only the canonical model declaration.
    overrides["model"]["id"] = case.model_family

    # Flow uses one identical beta-free public identity on both fake models.
    if case.algorithm_id == "flow-grpo":
        overrides["algorithm"] = {"params": {"beta": 0.0}}

    matrix_path = root / f"{case.algorithm_id}-{case.model_family}.yaml"
    matrix_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return matrix_path


def test_public_model_and_algorithm_axes_cross_2x2_via_production_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1: one production path binds the symmetric public 2x2 matrix."""

    e2e_fixture._patch_default_runtime(monkeypatch)
    catalog = _public_axis_catalog()
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "runtime.py").write_text(
        "FAKE_PUBLIC_AXIS_MATRIX = True\n",
        encoding="utf-8",
    )
    loader = _FakeModelLoader()
    reward_factory = _FakeRewardResourceFactory()
    observed: dict[tuple[str, str], tuple[dict[str, object], dict[str, object]]] = {}
    lifecycle_paths: set[tuple[ControllerStage, ...]] = set()
    controller_types: set[type[object]] = set()
    backend_types: set[type[object]] = set()

    for case in _PUBLIC_AXIS_CASES:
        pair_root = tmp_path / f"{case.algorithm_id}-{case.model_family}"
        pair_root.mkdir()
        demo_case = _DemoCase(
            filename=case.base_filename,
            recipe_id=case.recipe_id,
            model_family=case.model_family,
            source_ids=("main",),
            reward_ids=(case.reward_id,),
            expects_reference_state=False,
        )
        artifacts = _write_fake_artifacts(pair_root / "artifacts", demo_case)
        config = _write_public_axis_matrix_config(
            pair_root,
            case,
            output_dir=pair_root / "output",
            artifacts=artifacts,
        )

        # RunController is deliberately one-shot.  Every cell therefore uses
        # a fresh instance from the same sole composition root, with only the
        # documented heavyweight model/reward leaf seams replaced.
        controller = _create_catalog_run_controller(
            catalog=catalog,
            code_root=code_root,
            model_loader=loader,
            reward_resource_factory=reward_factory,
        )
        recording_binder = _RecordingComponentBinder(
            controller._backend.component_binder
        )
        controller._backend.component_binder = recording_binder
        result = controller.run(config)

        assert isinstance(result, RunResult)
        assert result.committed_steps == 1
        assert len(recording_binder.evidence) == 1
        assert len(recording_binder.public_axis_identities) == 1
        model_identity, algorithm_identity = recording_binder.public_axis_identities[0]
        assert model_identity["alias"] == case.model_family
        assert model_identity["config_type_path"] == (
            "visual_rl.models.catalog:WanConfig"
        )
        assert algorithm_identity["alias"] == case.algorithm_id
        observed[(case.model_family, case.algorithm_id)] = (
            model_identity,
            algorithm_identity,
        )
        lifecycle_paths.add(controller.completed_stages)
        controller_types.add(type(controller))
        backend_types.add(type(controller._backend))

    expected_pairs = {
        (model_id, algorithm_id)
        for model_id in ("fake-wan-a", "fake-wan-b")
        for algorithm_id in ("flow-grpo", "flash-grpo")
    }
    assert set(observed) == expected_pairs
    assert lifecycle_paths == {tuple(ControllerStage)}
    assert len(controller_types) == 1
    assert len(backend_types) == 1

    # Full resolved identities, not just aliases, remain constant when the
    # opposite public axis changes.  A model/algorithm-name special case in
    # the composition root would make at least one of these equalities fail.
    for model_id in ("fake-wan-a", "fake-wan-b"):
        assert (
            observed[(model_id, "flow-grpo")][0]
            == observed[(model_id, "flash-grpo")][0]
        )
    for algorithm_id in ("flow-grpo", "flash-grpo"):
        assert (
            observed[("fake-wan-a", algorithm_id)][1]
            == observed[("fake-wan-b", algorithm_id)][1]
        )
    assert (
        observed[("fake-wan-a", "flow-grpo")][0]
        != observed[("fake-wan-b", "flow-grpo")][0]
    )
    assert (
        observed[("fake-wan-a", "flow-grpo")][1]
        != observed[("fake-wan-a", "flash-grpo")][1]
    )

    assert [call[0] for call in loader.calls] == [
        "wan-t2v",
        "wan-t2v",
        "wan-t2v",
        "wan-t2v",
    ]
    assert len(reward_factory.requests) == 4


def test_reward_attestation_failure_precedes_model_load_and_accelerator_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable remote reward must leave policy weights unmaterialized."""

    e2e_fixture._patch_default_runtime(monkeypatch)
    prepare_calls = 0

    def record_prepare(self: object, *values: object):
        nonlocal prepare_calls
        del self
        prepare_calls += 1
        return values

    monkeypatch.setattr(
        e2e_fixture._FakeAccelerator,
        "prepare",
        record_prepare,
    )
    case = next(item for item in _DEMO_CASES if item.filename == "flash_wan.yaml")
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "runtime.py").write_text(
        "FAKE_REWARD_READINESS_FAILURE = True\n",
        encoding="utf-8",
    )
    artifacts = _write_fake_artifacts(tmp_path / "artifacts", case)
    output = tmp_path / "output"
    config = _write_smoke_config(
        tmp_path,
        case,
        output_dir=output,
        artifacts=artifacts,
    )
    loader = _FakeModelLoader()
    reward_factory = _UnavailableRewardResourceFactory()
    controller = create_default_run_controller(
        code_root=code_root,
        model_loader=loader,
        reward_resource_factory=reward_factory,
    )

    with pytest.raises(RuntimeError) as caught:
        controller.run(config)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == ("injected reward readiness/revision failure")
    assert len(reward_factory.requests) == 1
    assert loader.calls == []
    assert prepare_calls == 0
    assert controller.state is ControllerState.FAILED
    assert controller.attempted_stages[-1] is ControllerStage.PREPARE
    assert controller.completed_stages[-1] is ControllerStage.CONSTRUCT_GRAPH
    assert len(e2e_fixture._FakeAccelerator.instances) == 1
    assert e2e_fixture._FakeAccelerator.instances[0].end_training_calls == 1
    assert not (output / "checkpoints").exists()


@pytest.mark.parametrize("case", _DEMO_CASES, ids=lambda case: case.filename)
def test_official_v2_demo_reaches_update_checkpoint_and_completed_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _DemoCase,
) -> None:
    """G0-G6 plus terminal/completed-no-op lifecycle smoke."""

    e2e_fixture._patch_default_runtime(monkeypatch)
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "runtime.py").write_text(
        "FAKE_FIVE_RECIPE_SMOKE = True\n",
        encoding="utf-8",
    )
    artifacts = _write_fake_artifacts(tmp_path / "artifacts", case)
    output = tmp_path / "output"
    config = _write_smoke_config(
        tmp_path,
        case,
        output_dir=output,
        artifacts=artifacts,
    )
    loader = _FakeModelLoader()
    reward_factory = _FakeRewardResourceFactory()
    controller = create_default_run_controller(
        code_root=code_root,
        model_loader=loader,
        reward_resource_factory=reward_factory,
    )
    recording_binder = _RecordingComponentBinder(controller._backend.component_binder)
    controller._backend.component_binder = recording_binder

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    try:
        result = controller.run(config)

        assert isinstance(result, RunResult)
        assert result.committed_steps == 1
        assert result.output_dir == output
        assert result.authoritative_checkpoint == output / "checkpoints" / "step-1"
        assert result.marker_path == output / "SUCCESS"
        assert len(recording_binder.evidence) == 1
        assert len(recording_binder.public_axis_identities) == 1
        resolved_model_identity, _resolved_algorithm_identity = (
            recording_binder.public_axis_identities[0]
        )
        assert resolved_model_identity["config_type_path"] == (
            "visual_rl.models.catalog:SD3Config"
            if case.model_family == "sd3"
            else "visual_rl.models.catalog:WanConfig"
        )
        bound_evidence = recording_binder.evidence[0]
        reference_state = bound_evidence.reference_policy_state_evidence
        assert reference_state.has_reference_capability is (case.model_family == "sd3")
        assert reference_state.has_active_reference_owner is (
            case.expects_reference_state
        )
        assert reference_state.state_schema == (
            "derived-from-model-artifact.v1"
            if case.model_family == "sd3"
            else "none.v1"
        )
        assert reference_state.checkpoint_state_schema == (
            "derived-from-model-artifact.v1"
            if case.expects_reference_state
            else "none.v1"
        )
        assert (
            to_plain_dict(bound_evidence.verified_fields["reference_policy_state"])
            == reference_state.to_payload()
        )
        component_bound_contract_ids = {
            slot: contract.contract_id
            for slot, contract in bound_evidence.runtime_bound_contracts
        }
        assert "algorithm" in component_bound_contract_ids
        algorithm_evidence = bound_evidence.verified_fields["algorithm_module"]
        assert algorithm_evidence["resolved_declaration"]["kind"] == "algorithm"
        assert algorithm_evidence["requirement_id"].startswith(
            "algorithm-requirements.v1:"
        )
        assert len(algorithm_evidence["execution_plan_id"]) == 64
        inspection, snapshot = e2e_fixture._checkpoint_snapshot(
            result.authoritative_checkpoint
        )
        # Checkpoints intentionally carry the materialized recipe digest; the
        # human-readable definition id was already asserted from the official
        # source config while deriving this smoke variant.
        recipe_schema, recipe_digest = inspection.contract.recipe_id.split(":", 1)
        assert recipe_schema == "materialized-recipe.v2"
        assert len(recipe_digest) == 64
        assert int(recipe_digest, 16) >= 0
        algorithm_refs = tuple(
            item for item in inspection.contract.components if item.kind == "algorithm"
        )
        assert len(algorithm_refs) == 1
        assert algorithm_refs[0].slot == "algorithm"
        assert (
            algorithm_refs[0].runtime_bound_contract_id
            == component_bound_contract_ids["algorithm"]
        )
        execution_numerics = inspection.contract.model_execution_numerics
        assert inspection.contract.model_execution_numerics_id == (
            execution_numerics.execution_numerics_id
        )
        assert (
            execution_numerics.parameter_dtype_policy.trainable_parameter_dtype
            == "float32"
        )
        assert (
            execution_numerics.parameter_dtype_policy.frozen_parameter_policy.value
            == "preserve_loaded"
        )
        assert (
            execution_numerics.parameter_dtype_policy.floating_buffer_policy.value
            == "preserve_loaded"
        )
        views = {
            item.parameter_view.value: item.mode.value
            for item in execution_numerics.parameter_view_evidence
        }
        assert views == (
            {"current": "current", "reference": "lora_disable"}
            if case.model_family == "sd3"
            else {"current": "current"}
        )
        assert inspection.progress.next_optimizer_step == 1
        assert inspection.contract.reference_state_schema == (
            "derived-from-model-artifact.v1"
            if case.expects_reference_state
            else "none.v1"
        )
        assert inspection.progress.reference_state_saved is case.expects_reference_state
        assert snapshot.safe_point.committed_optimizer_step == 1
        if case.recipe_id == "flash_grpo_v1":
            selection_config = SingleStepRolloutConfig(
                selected_timestep_policy="uniform",
                num_steps=2,
                candidate_timestep_window=(0, 10),
                selection_key="prompt",
                selection_domain="single_process",
            )
            assert (
                snapshot.dynamics_selection_policy.selection_contract_identity
                == selection_config.selection_contract_identity
            )
        assert snapshot.component_names == (
            "data_plane",
            "lr_scheduler",
            "model",
            "optimizer",
            "run_checkpoint_summary",
        )
        assert [call[0] for call in loader.calls] == [case.model_family]
        if case.model_family == "wan-t2v":
            assert loader.calls[0][2].max_sequence_length == 512
            assert len(loader.wan_prompt_encoders) == 1
            assert loader.wan_prompt_encoders[0].calls
            assert {
                (max_length, guidance > 1.0, has_negative)
                for _prompts, max_length, guidance, has_negative in (
                    loader.wan_prompt_encoders[0].calls
                )
            } == {(512, True, True)}
        assert len(loader.transformers) == 1
        assert loader.transformers[0].policy_scale.item() != pytest.approx(0.2)
        assert len(reward_factory.requests) == len(case.reward_ids)
        assert {
            request.descriptor.artifact_ref for request in reward_factory.requests
        } == set(case.reward_ids)
        assert all(
            resource.activate_calls == 1
            and resource.score_calls >= 1
            and resource.close_calls == 1
            for resource in reward_factory.resources
        )
        assert controller.state is ControllerState.CLOSED
        assert controller.completed_stages == tuple(ControllerStage)

        recipe_manifest = output / "recipe.resolved.json"
        launch_manifest = output / "launch.resolved.json"
        assert recipe_manifest.is_file() and not recipe_manifest.is_symlink()
        assert launch_manifest.is_file() and not launch_manifest.is_symlink()
        recipe_manifest_before_resume = recipe_manifest.read_bytes()
        launch_manifest_before_resume = launch_manifest.read_bytes()
        checkpoint_tree_before = e2e_fixture._file_tree_identity(
            result.authoritative_checkpoint
        )
        resume_config = _write_smoke_config(
            tmp_path,
            case,
            output_dir=output,
            artifacts=artifacts,
            resume_from=result.authoritative_checkpoint,
        )
        resume_loader = _FakeModelLoader()
        resume_reward_factory = _FakeRewardResourceFactory()
        resume_controller = create_default_run_controller(
            code_root=code_root,
            model_loader=resume_loader,
            reward_resource_factory=resume_reward_factory,
        )
        resume_recording_binder = _RecordingComponentBinder(
            resume_controller._backend.component_binder
        )
        resume_controller._backend.component_binder = resume_recording_binder
        resumed = resume_controller.run(resume_config)

        assert resumed == result
        assert len(resume_recording_binder.evidence) == 1
        assert (
            resume_recording_binder.evidence[
                0
            ].reference_policy_state_evidence.to_payload()
            == reference_state.to_payload()
        )
        assert resumed.authoritative_checkpoint == result.authoritative_checkpoint
        assert recipe_manifest.read_bytes() == recipe_manifest_before_resume
        assert launch_manifest.read_bytes() == launch_manifest_before_resume
        assert (
            e2e_fixture._file_tree_identity(result.authoritative_checkpoint)
            == checkpoint_tree_before
        )
        assert {
            path.name for path in (output / "checkpoints").iterdir() if path.is_dir()
        } == {"step-1"}
        assert len(resume_reward_factory.requests) == len(case.reward_ids)
        assert {
            request.descriptor.artifact_ref
            for request in resume_reward_factory.requests
        } == set(case.reward_ids)
        assert all(
            resource.activate_calls == 0
            and resource.score_calls == 0
            and resource.close_calls == 1
            for resource in resume_reward_factory.resources
        )
        assert resume_controller.state is ControllerState.CLOSED
        assert resume_controller.completed_stages[-1] is ControllerStage.RESTORE_BOUND
        assert ControllerStage.PREPARE_RUN not in resume_controller.completed_stages
        assert ControllerStage.RUN not in resume_controller.completed_stages
        assert ControllerStage.CHECKPOINT not in resume_controller.completed_stages

    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
