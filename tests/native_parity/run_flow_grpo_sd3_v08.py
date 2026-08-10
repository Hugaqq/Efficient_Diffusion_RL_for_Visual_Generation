"""V0.8 public-boundary harness for Flow-GRPO/SD3 native parity.

This test-only entrypoint deliberately does not reuse the legacy v0.7 internal
parity implementation.  It binds the v0.8 public model and algorithm
descriptors, derives the exact ``PolicyRuntimePort`` contract identity, and
validates a pinned upstream checkout before any model weights are loaded.

The real native executor remains an identity-checked injected boundary because
the pinned upstream checkout is itself part of the oracle.  This module also
contains a deterministic CPU contract executor.  That executor traverses the
canonical SD3Adapter, PolicyRuntimePort, scheduler-bound Dynamics, rollout,
policy replay, objective, optimizer, and the canonical Flow-GRPO execution
plan/materializer boundary, but is explicitly typed as non-native evidence.
It can therefore exercise all fourteen comparison categories without ever
manufacturing a native-parity claim when real SD3 artifacts or CUDA are absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from visual_rl.composition.recipes.schema import ResolvedRecipe

_HARNESS_ID = "flow-grpo-sd3-native-parity.v08"
_PARITY_PROTOCOL_ID = "flow-grpo-sd3.v08-native-kernel.full-trajectory-single-commit.v1"
_DEFAULT_CONFIG = "configs/v2/flow_grpo_sd3.yaml"
_DEFAULT_CASE = "tests/fixtures/native_parity/flow_grpo_sd3_v08_case_v1.json"
_REFERENCE_FILES = (
    "scripts/train_sd3.py",
    "flow_grpo/stat_tracking.py",
    "flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py",
    "flow_grpo/diffusers_patch/sd3_sde_with_logprob.py",
)
_EXECUTION_DEPENDENCIES = (
    "absl",
    "accelerate",
    "diffusers",
    "ml_collections",
    "numpy",
    "peft",
    "PIL",
    "torch",
    "transformers",
    "wandb",
)
_CASE_KEYS = frozenset(
    {
        "schema_version",
        "prompt",
        "seed",
        "logical_step",
        "reward_values",
        "expected_advantages",
    }
)
_NATIVE_COMPARISON_KEYS = (
    "prompt_encoding",
    "initial_latent",
    "timestep",
    "rollout_latent",
    "old_log_prob",
    "current_log_prob",
    "transition_statistics",
    "group_advantage",
    "policy_loss",
    "reference_kl",
    "total_loss",
    "gradient",
    "parameter_delta",
    "checkpoint_resume",
)
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POLICY_PORT_PROTOCOL = "visual_rl.core.contracts.runtime:PolicyRuntimePort"
_POLICY_PORT_IMPLEMENTATION = "visual_rl.runtime.model_binding:DefaultPolicyRuntimePort"
_EXPECTED_MODEL_CLASS = "visual_rl.models.implementations.sd3:SD3Adapter"
_EXPECTED_ALGORITHM_CLASS = (
    "visual_rl.algorithms.modules.flow_grpo:FlowGRPOAlgorithmModule"
)
_CPU_CONTRACT_EXECUTOR_ID = "flow-grpo-sd3.v08.cpu-fake-contract.v1"
_CPU_CONTRACT_STEPS = 3
_CPU_CONTRACT_RESOLUTION = 8
_CPU_CONTRACT_RTOL = 2.0e-5
_CPU_CONTRACT_ATOL = 2.0e-6


class HarnessArgumentError(ValueError):
    """Raised when the test-only command contract is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class HarnessArguments:
    """All filesystem and upstream identities are explicit harness inputs."""

    repo_root: Path
    config_path: Path
    case_path: Path
    reference_repo: Path
    reference_revision: str
    reference_digest: str
    preflight_only: bool = False

    def __post_init__(self) -> None:
        for name in ("repo_root", "config_path", "case_path", "reference_repo"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{name} must be an absolute Path")
        if _GIT_REVISION_RE.fullmatch(self.reference_revision) is None:
            raise ValueError("reference_revision must be a full lowercase Git digest")
        if _SHA256_RE.fullmatch(self.reference_digest) is None:
            raise ValueError("reference_digest must be a lowercase SHA-256 digest")
        if type(self.preflight_only) is not bool:
            raise TypeError("preflight_only must be bool")


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    code: str
    status: Literal["pass", "error", "blocked"]
    detail: str
    observed: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("preflight check code must be non-empty")
        if self.status not in {"pass", "error", "blocked"}:
            raise ValueError("invalid preflight check status")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("preflight check detail must be non-empty")

    def to_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "status": self.status,
            "detail": self.detail,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class ReferenceIdentity:
    path: Path
    revision: str
    digest: str

    def to_payload(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "revision": self.revision,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class CompositionIdentity:
    config_path: Path
    resolved_fingerprint: str
    model_requested_id: str
    model_class_path: str
    algorithm_requested_id: str
    algorithm_class_path: str
    model_capability_id: str
    algorithm_requirement_id: str
    algorithm_module_identity: str
    algorithm_binding_id: str
    reference_kl_weight: float
    policy_port_protocol: str
    policy_port_implementation: str

    def to_payload(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "resolved_fingerprint": self.resolved_fingerprint,
            "model_requested_id": self.model_requested_id,
            "model_class_path": self.model_class_path,
            "algorithm_requested_id": self.algorithm_requested_id,
            "algorithm_class_path": self.algorithm_class_path,
            "model_capability_id": self.model_capability_id,
            "algorithm_requirement_id": self.algorithm_requirement_id,
            "algorithm_module_identity": self.algorithm_module_identity,
            "algorithm_binding_id": self.algorithm_binding_id,
            "reference_kl_weight": self.reference_kl_weight,
            "policy_port_protocol": self.policy_port_protocol,
            "policy_port_implementation": self.policy_port_implementation,
        }


@dataclass(frozen=True, slots=True)
class NativeParityExecution:
    """Result returned only by a real, externally supplied parity executor."""

    passed: bool
    recipe_id: str
    bound_contract_id: str
    algorithm_module_identity: str
    algorithm_binding_id: str
    reference_revision: str
    reference_digest: str
    comparisons: Mapping[str, object]
    evidence_scope: Literal["pinned_upstream_native"]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be bool")
        for name in (
            "recipe_id",
            "bound_contract_id",
            "algorithm_module_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "algorithm_binding_id",
            "reference_revision",
            "reference_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.comparisons, Mapping):
            raise TypeError("comparisons must be a mapping")
        if self.evidence_scope != "pinned_upstream_native":
            raise ValueError("native execution must use pinned_upstream_native scope")
        _validate_comparison_set(self.passed, self.comparisons)


@dataclass(frozen=True, slots=True)
class ContractParityExecution:
    """Fourteen-item CPU contract result which is never native evidence."""

    passed: bool
    recipe_id: str
    bound_contract_id: str
    algorithm_module_identity: str
    algorithm_binding_id: str
    comparisons: Mapping[str, object]
    evidence_scope: Literal["cpu_fake_contract"] = "cpu_fake_contract"
    executor_identity: str = _CPU_CONTRACT_EXECUTOR_ID

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be bool")
        for name in (
            "recipe_id",
            "bound_contract_id",
            "algorithm_module_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.algorithm_binding_id, str)
            or not self.algorithm_binding_id
        ):
            raise ValueError("algorithm_binding_id must be non-empty")
        if self.evidence_scope != "cpu_fake_contract":
            raise ValueError("contract execution must use cpu_fake_contract scope")
        if self.executor_identity != _CPU_CONTRACT_EXECUTOR_ID:
            raise ValueError("contract executor identity drifted")
        _validate_comparison_set(self.passed, self.comparisons)


def _validate_comparison_set(
    passed: bool,
    comparisons: Mapping[str, object],
) -> None:
    """Validate one exact fail-closed fourteen-item comparison envelope."""

    if not isinstance(comparisons, Mapping):
        raise TypeError("comparisons must be a mapping")
    if set(comparisons) != set(_NATIVE_COMPARISON_KEYS):
        raise ValueError("comparisons must contain the exact v0.8 parity item set")
    outcomes: list[bool] = []
    for name in _NATIVE_COMPARISON_KEYS:
        item = comparisons[name]
        if not isinstance(item, Mapping) or type(item.get("passed")) is not bool:
            raise TypeError(f"comparison {name!r} must map a boolean passed field")
        outcomes.append(item["passed"])
    if passed is not all(outcomes):
        raise ValueError("passed must equal the conjunction of all comparisons")
    json.dumps(dict(comparisons), allow_nan=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class NativeExecutionRequest:
    arguments: HarnessArguments
    case: Mapping[str, object]
    reference: ReferenceIdentity
    composition: CompositionIdentity


NativeExecutor = Callable[[NativeExecutionRequest], NativeParityExecution]


class NativeFlowReferenceOracle:
    """Independent CPU statement of the Flow-GRPO scalar objective."""

    @staticmethod
    def group_advantages(reward_values: Any, epsilon: float):
        import torch

        rewards = torch.as_tensor(reward_values, dtype=torch.float64)
        if rewards.ndim != 1 or rewards.numel() < 2:
            raise ValueError("reward_values must be a one-dimensional group")
        if not bool(torch.isfinite(rewards).all()):
            raise ValueError("reward_values must be finite")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)

    @staticmethod
    def evaluate(
        *,
        old_log_probs: Any,
        new_log_probs: Any,
        advantages: Any,
        current_mean: Any,
        reference_mean: Any,
        std_dev: Any,
        clip_range: float,
        beta: float,
    ) -> dict[str, Any]:
        import torch

        old = torch.as_tensor(old_log_probs, dtype=torch.float64)
        new = torch.as_tensor(new_log_probs, dtype=torch.float64)
        advantage = torch.as_tensor(advantages, dtype=torch.float64)
        current = torch.as_tensor(current_mean, dtype=torch.float64)
        reference = torch.as_tensor(reference_mean, dtype=torch.float64)
        std = torch.as_tensor(std_dev, dtype=torch.float64)
        if old.ndim != 2 or new.shape != old.shape or advantage.shape != old.shape:
            raise ValueError("log-probabilities and advantages must share [B,T]")
        if current.shape != reference.shape or current.shape[:2] != old.shape:
            raise ValueError("transition means must share [B,T,...X]")
        if std.shape == old.shape:
            std = std.reshape(*old.shape, *((1,) * (current.ndim - 2)))
        elif (
            std.ndim != current.ndim
            or std.shape[:2] != old.shape
            or any(size != 1 for size in std.shape[2:])
        ):
            raise ValueError("std_dev must be [B,T] or [B,T,1,...,1]")
        if not 0.0 < clip_range < 1.0:
            raise ValueError("clip_range must satisfy 0 < value < 1")
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (old, new, advantage, current, reference, std)
        ):
            raise ValueError("oracle tensors must be finite")
        if not bool((std > 0.0).all()):
            raise ValueError("std_dev must be strictly positive")

        ratio = torch.exp(new - old)
        clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.minimum(
            ratio * advantage,
            clipped_ratio * advantage,
        ).mean()
        reference_kl = (
            ((current - reference).square() / (2.0 * std.square()))
            .flatten(start_dim=2)
            .mean(dim=2)
            .mean()
        )
        return {
            "policy_loss": policy_loss,
            "reference_kl": reference_kl,
            "total_loss": policy_loss + beta * reference_kl,
        }


def _tensor_comparison(
    tensor_name: str,
    observed: Any,
    expected: Any,
    *,
    rtol: float = _CPU_CONTRACT_RTOL,
    atol: float = _CPU_CONTRACT_ATOL,
) -> dict[str, object]:
    """Return one finite, JSON-safe tensor comparison."""

    import torch

    left = torch.as_tensor(observed).detach().to(device="cpu").contiguous()
    right = torch.as_tensor(expected).detach().to(device="cpu").contiguous()
    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    finite = True
    for value in (left, right):
        if value.is_floating_point() or value.is_complex():
            finite = finite and bool(torch.isfinite(value).all())
    max_value = float(sys.float_info.max)
    max_abs = max_value
    max_rel = max_value
    passed = False
    if shape_equal and dtype_equal and finite:
        if left.is_floating_point() or left.is_complex():
            delta = (left - right).abs()
            max_abs = float(delta.max()) if delta.numel() else 0.0
            denominator = right.abs().clamp_min(torch.finfo(right.dtype).tiny)
            max_rel = float((delta / denominator).max()) if delta.numel() else 0.0
            passed = bool(torch.allclose(left, right, rtol=rtol, atol=atol))
        else:
            passed = bool(torch.equal(left, right))
            max_abs = 0.0 if passed else 1.0
            max_rel = max_abs
            rtol = 0.0
            atol = 0.0
    return {
        "tensor_name": tensor_name,
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "passed": passed,
    }


def _comparison_item(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    rtol: float = _CPU_CONTRACT_RTOL,
    atol: float = _CPU_CONTRACT_ATOL,
) -> dict[str, object]:
    """Compare an exact named tensor set and fail closed on missing names."""

    comparisons: list[dict[str, object]] = []
    for name in sorted(set(observed) | set(expected)):
        if name not in observed or name not in expected:
            comparisons.append(
                {
                    "tensor_name": name,
                    "shape": [],
                    "dtype": "missing",
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "max_abs_error": float(sys.float_info.max),
                    "max_rel_error": float(sys.float_info.max),
                    "passed": False,
                }
            )
            continue
        comparisons.append(
            _tensor_comparison(
                name,
                observed[name],
                expected[name],
                rtol=rtol,
                atol=atol,
            )
        )
    return {
        "passed": bool(comparisons)
        and all(bool(item["passed"]) for item in comparisons),
        "comparisons": comparisons,
    }


def _deep_equal(left: Any, right: Any) -> bool:
    """Exact recursive comparison for detached checkpoint state."""

    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_deep_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_deep_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


class _CPUContractSchedulerConfig(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        return self[name]


class _CPUContractScheduler:
    """Small Diffusers-shaped scheduler used only by the CPU contract."""

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        values: dict[str, object] = {
            "stochastic_sampling": True,
            "use_dynamic_shifting": False,
        }
        values.update({} if config is None else config)
        self.config = _CPUContractSchedulerConfig(values)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> _CPUContractScheduler:
        return cls(config)

    def set_timesteps(self, *, num_inference_steps: int, device: object) -> None:
        import torch

        self.timesteps = torch.linspace(
            900.0,
            100.0,
            num_inference_steps,
            dtype=torch.float32,
            device=device,
        )
        # Keep every policy sigma strictly positive while retaining one next
        # sigma per transition.  This exercises the real SD3 SDE equation.
        self.sigmas = torch.linspace(
            0.9,
            0.1,
            num_inference_steps + 1,
            dtype=torch.float32,
            device=device,
        )


class _CPUContractPromptEncoder:
    def to(self, device: object) -> _CPUContractPromptEncoder:
        del device
        return self

    def encode(
        self,
        prompts: tuple[str, ...],
        max_sequence_length: int,
        guidance_scale: float,
    ) -> tuple[Any, Any, Any, Any]:
        import torch

        del max_sequence_length, guidance_scale
        batch_size = len(prompts)
        positive = torch.full((batch_size, 3, 2), 0.75)
        negative = torch.full((batch_size, 3, 2), -0.25)
        pooled = torch.full((batch_size, 2), 0.5)
        negative_pooled = torch.full((batch_size, 2), -0.5)
        return positive, negative, pooled, negative_pooled

    def close(self) -> None:
        return None


class _CPUContractDecoder:
    def to(self, device: object) -> _CPUContractDecoder:
        del device
        return self

    def decode(self, latents: Any, latent_spec: object) -> Any:
        if tuple(latents.shape) != tuple(latent_spec.shape):
            raise ValueError("CPU contract decoder received foreign latent geometry")
        return latents[:, :3].detach().clone()

    def close(self) -> None:
        return None


class _CPUContractModelLoader:
    """Inject deterministic runtime parts without bypassing SD3Adapter."""

    def __init__(self) -> None:
        self.transformer: object | None = None

    def __call__(
        self,
        family: str,
        artifact_path: Path,
        config: object,
        precision: object,
    ) -> object:
        import torch

        from visual_rl.models import SchedulerArtifactBlueprint
        from visual_rl.models.implementations.sd3 import SD3RuntimeParts

        del artifact_path, config, precision
        if family != "sd3":
            raise ValueError("CPU Flow-GRPO contract accepts only SD3")

        class ContractTransformer(torch.nn.Module):
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
                self._adapter_disabled = 0
                self._no_split_modules: list[str] = []

            def forward(
                self,
                *,
                hidden_states: Any,
                timestep: Any,
                encoder_hidden_states: Any,
                return_dict: bool,
                pooled_projections: Any = None,
                attention_kwargs: Any = None,
            ) -> tuple[Any]:
                del (
                    timestep,
                    return_dict,
                    pooled_projections,
                    attention_kwargs,
                )
                batch_values = encoder_hidden_states.mean(
                    dim=tuple(range(1, encoder_hidden_states.ndim))
                ).reshape(
                    hidden_states.shape[0],
                    *([1] * (hidden_states.ndim - 1)),
                )
                scale = self.base_scale
                if self._adapter_disabled == 0:
                    scale = scale + self.scale
                return (hidden_states * scale + self.frozen_bias + batch_values,)

            @contextmanager
            def disable_adapter(self):
                self._adapter_disabled += 1
                try:
                    yield
                finally:
                    self._adapter_disabled -= 1

        transformer = ContractTransformer()
        self.transformer = transformer
        return SD3RuntimeParts(
            prompt_encoder=_CPUContractPromptEncoder(),
            transformer=transformer,
            decoder=_CPUContractDecoder(),
            reference_context=transformer.disable_adapter,
            latent_channels=4,
            scheduler_artifact_blueprint=(
                SchedulerArtifactBlueprint.from_scheduler(_CPUContractScheduler())
            ),
            transformer_patch_size=1,
        )


class _CPUContractAccelerator:
    def prepare(self, *items: object) -> tuple[object, ...]:
        return items

    @contextmanager
    def accumulate(self, root: object):
        del root
        yield


@dataclass(slots=True)
class _CPUContractRuntime:
    adapter: object
    manager: object
    optimizer: object
    policy: object
    algorithm: object
    dynamics_component: object
    rollout_component: object
    credit_component: object
    resolved_recipe: ResolvedRecipe
    loader: _CPUContractModelLoader

    def close(self) -> None:
        self.manager.close()


@dataclass(frozen=True, slots=True)
class _CPUContractIteration:
    optimizer_step: int
    comparisons: Mapping[str, object]
    bound_contract_id: str


def _cpu_contract_materializer(
    runtime: _CPUContractRuntime,
    *,
    execute: Callable[[int], _CPUContractIteration],
) -> tuple[object, object]:
    """Build the real canonical materializer around one test-only Trainer."""

    from visual_rl.algorithms.trainer.interface import (
        IterationIdentity,
        IterationResult,
        PrepareRunContext,
        StageValue,
        TrainerComponent,
        TrainerState,
    )
    from visual_rl.core.contracts import AlgorithmComponentRole
    from visual_rl.runtime.algorithm_binding import (
        AlgorithmRuntimeComponent,
        AlgorithmRuntimeComponents,
        CanonicalAlgorithmMaterializer,
    )

    class CPUContractTrainer(TrainerComponent):
        """Lifecycle oracle only; it is not production or upstream evidence."""

        def __init__(self) -> None:
            self.state = TrainerState.NEW
            self.next_optimizer_step: int | None = None

        @classmethod
        def describe(cls, config: object) -> object:
            del cls, config
            raise RuntimeError("CPU contract Trainer has no public declaration")

        @classmethod
        def from_config(
            cls,
            config: object,
            *,
            runtime_context: Mapping[str, Any],
        ) -> TrainerComponent:
            del cls, config, runtime_context
            raise RuntimeError("CPU contract Trainer is fixture-injected only")

        def prepare_run(self, context: object) -> None:
            if not isinstance(context, PrepareRunContext):
                raise TypeError("CPU contract requires PrepareRunContext")
            if self.state is not TrainerState.NEW:
                raise RuntimeError("CPU contract Trainer may be prepared once")
            self.next_optimizer_step = context.start_optimizer_step
            self.state = TrainerState.PREPARED

        def run_iteration(self, optimizer_step: int) -> IterationResult[object]:
            if self.state is not TrainerState.PREPARED:
                raise RuntimeError("CPU contract Trainer is not prepared")
            if optimizer_step != self.next_optimizer_step:
                raise ValueError("CPU contract Trainer optimizer step drifted")
            payload = execute(optimizer_step)
            identity = IterationIdentity(
                optimizer_step=optimizer_step,
                source_id="cpu-contract-source",
                phase_id="cpu-contract-phase",
                row_identities=(f"cpu-contract-row-{optimizer_step}",),
                group_ids=("cpu-contract-group",),
                member_ids=(0,),
            )
            self.next_optimizer_step = optimizer_step + 1
            return IterationResult(
                optimizer_step=optimizer_step,
                value=StageValue(identity=identity, payload=payload),
                stage_order=(
                    "prelude",
                    "rollout",
                    "reward",
                    "advantage",
                    "credit",
                    "optimize",
                ),
            )

        def close(self) -> None:
            self.state = TrainerState.CLOSED

    trainer = CPUContractTrainer()
    instances = {
        AlgorithmComponentRole.TRAINER: trainer,
        AlgorithmComponentRole.DYNAMICS: runtime.dynamics_component,
        AlgorithmComponentRole.ROLLOUT: runtime.rollout_component,
        AlgorithmComponentRole.CREDIT: runtime.credit_component,
    }
    selections = runtime.resolved_recipe.algorithm_spec.components
    unexpected = tuple(
        selection.role.value
        for selection in selections
        if selection.role not in instances
    )
    if unexpected:
        raise ValueError(
            f"CPU Flow-GRPO contract has unexpected component roles {list(unexpected)}"
        )
    components = AlgorithmRuntimeComponents(
        tuple(
            AlgorithmRuntimeComponent(
                selection=selection,
                instance=instances[selection.role],
            )
            for selection in selections
        )
    )
    return CanonicalAlgorithmMaterializer(components), trainer


def _build_cpu_contract_runtime(
    request: NativeExecutionRequest,
) -> _CPUContractRuntime:
    """Build canonical v0.8 ports over tiny deterministic SD3 runtime parts."""

    import torch

    from visual_rl.algorithms.dynamics.config import FlowSDEConfig
    from visual_rl.algorithms.dynamics.sd3_flow_sde import RegisteredSD3FlowSDE
    from visual_rl.algorithms.modules.config import FlowGRPOAlgorithmConfig
    from visual_rl.algorithms.modules.flow_grpo import FlowGRPOAlgorithmModule
    from visual_rl.algorithms.optimization.config import GRPOCreditConfig
    from visual_rl.algorithms.optimization.credit import RegisteredGRPOCredit
    from visual_rl.algorithms.rollout.config import FullTrajectoryRolloutConfig
    from visual_rl.algorithms.rollout.full_trajectory import FullTrajectoryRollout
    from visual_rl.core.contracts import (
        ArtifactBoundContract,
        ComputePrecision,
        RuntimeBoundContract,
    )
    from visual_rl.core.contracts.composition import BoundPolicyCapabilities
    from visual_rl.models import ComponentManager
    from visual_rl.models.catalog import SD3Config
    from visual_rl.models.implementations.sd3 import SD3Adapter
    from visual_rl.runtime.model_binding import DefaultPolicyRuntimePort

    resolved = _compile_v08_recipe(request.arguments.config_path)
    artifact = _resolved_model_artifact(request.arguments.config_path)
    if not artifact.is_dir():
        raise ValueError("CPU contract requires an existing artifact directory")
    declared_model_config = resolved.model.declaration.config
    if not isinstance(declared_model_config, SD3Config):
        raise TypeError("CPU contract recipe did not resolve canonical SD3Config")
    # The fixture keeps the compiler-owned model semantics and capability graph,
    # but shrinks only the live latent geometry and disables checkpointing.  This
    # is why its evidence scope remains cpu_fake_contract rather than native.
    model_config = replace(
        declared_model_config,
        gradient_checkpointing=False,
        resolution=_CPU_CONTRACT_RESOLUTION,
    )
    loader = _CPUContractModelLoader()
    adapter = SD3Adapter.from_config(
        model_config,
        runtime_context={
            "precision": ComputePrecision.FP32.value,
            "model_artifacts": {"main": artifact},
            "model_loader": loader,
        },
    )
    manager = ComponentManager(
        adapter,
        execution_device="cpu",
        offload_device="cpu",
    )
    try:
        manager.load()
        manager.configure()
        optimizer = torch.optim.AdamW(
            manager.parameter_state.parameters(),
            lr=3.0e-4,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=1.0e-4,
        )
        handle = manager.prepare(
            accelerator=_CPUContractAccelerator(),
            optimizer=optimizer,
        )
        declared_model = adapter.describe(model_config)
        manager.bind_runtime(
            RuntimeBoundContract(
                artifact=ArtifactBoundContract(
                    declared=declared_model,
                    artifact_identity="cpu-fake-sd3-runtime-parts.v1",
                    resolved_fields=(("model.artifact_ref", "main"),),
                ),
                runtime_identity=_CPU_CONTRACT_EXECUTOR_ID,
                verified_fields=(
                    ("model.component_topology", "verified"),
                    ("model.reference_forward", "verified"),
                ),
            )
        )
        dynamics_declaration = resolved.component("dynamics").declaration
        rollout_declaration = resolved.component("rollout").declaration
        credit_declaration = resolved.component("credit").declaration
        trainer_declaration = resolved.component("trainer").declaration
        dynamics_config = dynamics_declaration.config
        rollout_config = rollout_declaration.config
        credit_config = credit_declaration.config
        algorithm_config = resolved.algorithm.config
        if not isinstance(dynamics_config, FlowSDEConfig):
            raise TypeError("CPU contract recipe did not resolve FlowSDEConfig")
        if not isinstance(rollout_config, FullTrajectoryRolloutConfig):
            raise TypeError(
                "CPU contract recipe did not resolve FullTrajectoryRolloutConfig"
            )
        if not isinstance(credit_config, GRPOCreditConfig):
            raise TypeError("CPU contract recipe did not resolve GRPOCreditConfig")
        if not isinstance(algorithm_config, FlowGRPOAlgorithmConfig):
            raise TypeError(
                "CPU contract recipe did not resolve FlowGRPOAlgorithmConfig"
            )
        execution_policy = resolved.execution_policy.to_receipt()
        dynamics_component = RegisteredSD3FlowSDE.from_config(
            dynamics_config,
            runtime_context={},
        )
        rollout_component = FullTrajectoryRollout(
            rollout_config,
            execution_policy=execution_policy,
            expected_policy_id=resolved.algorithm_spec.execution_policy_id,
        )
        credit_component = RegisteredGRPOCredit.from_config(
            credit_config,
            runtime_context={"beta": resolved.algorithm_spec.beta},
        )
        declared_model_detail = resolved.model.declaration.declared_contract.model
        dynamics_detail = dynamics_declaration.declared_contract.dynamics
        trainer_detail = trainer_declaration.declared_contract.trainer
        if (
            declared_model_detail is None
            or dynamics_detail is None
            or trainer_detail is None
        ):
            raise TypeError("CPU contract descriptor graph is incomplete")
        capabilities = BoundPolicyCapabilities.from_contracts(
            declared_model_detail,
            dynamics=dynamics_detail,
            trainer=trainer_detail,
        )
        algorithm = FlowGRPOAlgorithmModule.from_config(
            algorithm_config,
            runtime_context={},
        )
        policy = DefaultPolicyRuntimePort(
            _adapter=adapter,
            _manager=manager,
            _prepared_handle=handle,
            capabilities=capabilities,
            algorithm_requirements=algorithm.requirements,
            runtime_capabilities={"evidence_scope": "cpu_fake_contract"},
        )
        if policy.binding.binding_id != request.composition.algorithm_binding_id:
            raise ValueError(
                "CPU contract binding differs from the resolved v0.8 composition"
            )
        return _CPUContractRuntime(
            adapter=adapter,
            manager=manager,
            optimizer=optimizer,
            policy=policy,
            algorithm=algorithm,
            dynamics_component=dynamics_component,
            rollout_component=rollout_component,
            credit_component=credit_component,
            resolved_recipe=resolved,
            loader=loader,
        )
    except BaseException:
        manager.close()
        raise


def _cpu_contract_samples(prompt: str, batch_size: int) -> tuple[object, object]:
    from visual_rl.algorithms.trainer.interface import IterationIdentity
    from visual_rl.data.samples import (
        BatchRowContext,
        ExplicitCollator,
        SourceItemContext,
        T2IItem,
    )

    source = SourceItemContext(
        source_item_id="cpu-contract-source",
        dataset_source_id="main",
        dataset_index=0,
        dataset_revision="cpu-contract-v1",
    )
    items = tuple(T2IItem(prompt=prompt, source=source) for _ in range(batch_size))
    rows = tuple(
        BatchRowContext(
            occurrence_id="cpu-contract-occurrence",
            group_id="cpu-contract-group",
            member_id=index,
            phase="main",
            optimizer_step=0,
            source_item_id=source.source_item_id,
        )
        for index in range(batch_size)
    )
    samples = ExplicitCollator().collate_samples(items, rows)
    identity = IterationIdentity(
        optimizer_step=0,
        source_id="main",
        phase_id="main",
        row_identities=tuple(row.identity for row in samples.rows),
        group_ids=tuple(row.group_id for row in samples.rows),
        member_ids=tuple(row.member_id for row in samples.rows),
    )
    return samples, identity


def _clone_generator(generator: object) -> object:
    import torch

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")
    result = torch.Generator(device=generator.device)
    result.set_state(generator.get_state().clone())
    return result


def _cpu_contract_reward(trajectory: object, values: Sequence[float]) -> object:
    import numpy as np

    from visual_rl.algorithms.rewards import RewardBatchIdentity, RewardResult

    contexts = trajectory.contexts
    score = np.asarray(values, dtype=np.float64)
    valid = np.ones(len(contexts), dtype=np.bool_)
    identity = RewardBatchIdentity(
        source_id="main",
        phase_id="main",
        batch_row_ids=tuple(item.batch_row_identity for item in contexts),
        sample_ids=tuple(item.sample_id for item in contexts),
        trajectory_ids=tuple(item.trajectory_id for item in contexts),
        condition_payload_ids=("none",) * len(contexts),
        group_ids=tuple(item.batch_row.group_id for item in contexts),
    )
    return RewardResult(
        identity=identity,
        component_scores={"fixture_reward": score},
        weighted_scores={"fixture_reward": score},
        component_valid_masks={"fixture_reward": valid},
        weighted_total=score,
        valid_mask=valid,
        resource_identities={"fixture_reward": "cpu-contract-fixture.v1"},
    )


def _cpu_contract_checkpoint_item(
    runtime: _CPUContractRuntime,
    request: NativeExecutionRequest,
    *,
    non_timing_metrics: Mapping[str, float],
    next_step_inputs: Mapping[str, Any],
) -> dict[str, object]:
    """Round-trip detached model/AdamW/RNG state into one fresh runtime."""

    import torch

    source_root = runtime.manager.prepared_handle.prepared_root
    model_state = deepcopy(source_root.state_dict())
    optimizer_state = deepcopy(runtime.optimizer.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    torch.randn((3,), generator=generator)
    generator_state = generator.get_state().clone()
    expected_rng_next = torch.randn((5,), generator=generator)

    resumed = _build_cpu_contract_runtime(request)
    try:
        target_root = resumed.manager.prepared_handle.prepared_root
        target_root.load_state_dict(model_state, strict=True)
        resumed.optimizer.load_state_dict(optimizer_state)
        observed_model = deepcopy(target_root.state_dict())
        observed_optimizer = deepcopy(resumed.optimizer.state_dict())
        resumed_generator = torch.Generator(device="cpu")
        resumed_generator.set_state(generator_state.clone())
        observed_rng_next = torch.randn((5,), generator=resumed_generator)
        observed_inputs = {
            name: value.detach().clone() for name, value in next_step_inputs.items()
        }
        observed_metrics = dict(non_timing_metrics)
        flags = {
            "adapter_tensors": _deep_equal(model_state, observed_model),
            "optimizer_state": _deep_equal(
                optimizer_state,
                observed_optimizer,
            ),
            "grad_scaler_state": _deep_equal(None, None),
            "rng_state": bool(torch.equal(expected_rng_next, observed_rng_next)),
            "next_step_inputs": _deep_equal(next_step_inputs, observed_inputs),
            "global_step": True,
            "non_timing_metrics": _deep_equal(
                non_timing_metrics,
                observed_metrics,
            ),
        }
    finally:
        resumed.close()
    return {
        "passed": all(flags.values()),
        "comparisons": flags,
    }


def _execute_cpu_contract_iteration(
    runtime: _CPUContractRuntime,
    native_request: NativeExecutionRequest,
    optimizer_step: int,
) -> _CPUContractIteration:
    """Execute all fourteen categories through canonical v0.8 runtime ports."""

    import torch

    from tests.support.policy_recompute_oracle import (
        compute_full_policy_stats_oracle,
    )
    from visual_rl.algorithms.optimization.advantage import (
        AdvantageGrouping,
        GroupZScoreAdvantageProcessor,
    )
    from visual_rl.algorithms.optimization.credit import GRPOCreditStrategy
    from visual_rl.algorithms.optimization.objective import ClippedSurrogateObjective
    from visual_rl.algorithms.optimization.recompute import PolicyRecomputeRequest
    from visual_rl.algorithms.rollout.config import FullTrajectoryRolloutConfig
    from visual_rl.algorithms.rollout.full_trajectory import FullTrajectoryRollout
    from visual_rl.algorithms.rollout.request import IterationRolloutRequestFactory
    from visual_rl.core.contracts import LikelihoodSemantics
    from visual_rl.runtime import PerRolloutDynamicsFactory

    if optimizer_step != int(native_request.case["logical_step"]):
        raise ValueError("CPU contract logical step differs from the fixture")
    rewards = tuple(float(value) for value in native_request.case["reward_values"])
    expected_advantages = tuple(
        float(value) for value in native_request.case["expected_advantages"]
    )
    batch_size = len(rewards)
    samples, iteration_identity = _cpu_contract_samples(
        str(native_request.case["prompt"]),
        batch_size,
    )
    policy = runtime.policy
    dynamics_component = runtime.dynamics_component
    blueprint = runtime.adapter.scheduler_artifact_blueprint
    replay_factory = dynamics_component.bind_replay_state_factory(blueprint)
    dynamics_factory = PerRolloutDynamicsFactory(
        component=dynamics_component,
        scheduler_blueprint=blueprint,
        replay_state_factory=replay_factory,
        dynamics_binding_family=dynamics_component.dynamics_binding_family,
        replay_state_schema_id=dynamics_component.replay_state_schema_id,
    )
    request_factory = IterationRolloutRequestFactory(
        adapter=policy,
        dynamics_factory=dynamics_factory,
        num_steps=_CPU_CONTRACT_STEPS,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        base_seed=int(native_request.case["seed"]),
        device="cpu",
        dtype=torch.float32,
    )
    first_request = request_factory(samples, iteration_identity)
    second_request = request_factory(samples, iteration_identity)
    encoded = policy.encode(samples)
    first_request = replace(
        first_request,
        encoded_conditioning=encoded,
        model_condition_identity=encoded.condition_identity,
    )
    second_request = replace(
        second_request,
        encoded_conditioning=encoded,
        model_condition_identity=encoded.condition_identity,
    )

    expected_prompt = {
        "prompt_embeds": torch.full((batch_size, 3, 2), 0.75),
        "negative_prompt_embeds": torch.full((batch_size, 3, 2), -0.25),
        "pooled_prompt_embeds": torch.full((batch_size, 2), 0.5),
        "negative_pooled_prompt_embeds": torch.full((batch_size, 2), -0.5),
    }
    comparisons: dict[str, dict[str, object]] = {
        "prompt_encoding": _comparison_item(
            {
                "prompt_embeds": encoded.prompt_embeds,
                "negative_prompt_embeds": encoded.negative_prompt_embeds,
                "pooled_prompt_embeds": encoded.pooled_prompt_embeds,
                "negative_pooled_prompt_embeds": (
                    encoded.negative_pooled_prompt_embeds
                ),
            },
            expected_prompt,
            rtol=0.0,
            atol=0.0,
        )
    }

    direct_generator = _clone_generator(first_request.generator)
    expected_generator = _clone_generator(first_request.generator)
    direct_initial = policy.prepare_latents(
        first_request.latent_spec,
        generator=direct_generator,
    )
    expected_initial = torch.randn(
        first_request.latent_spec.shape,
        device=first_request.latent_spec.device,
        dtype=first_request.latent_spec.dtype,
        generator=expected_generator,
    )
    strategy = FullTrajectoryRollout(
        FullTrajectoryRolloutConfig(num_steps=_CPU_CONTRACT_STEPS),
        execution_policy=runtime.resolved_recipe.execution_policy.to_receipt(),
        expected_policy_id=runtime.resolved_recipe.algorithm_spec.execution_policy_id,
    )
    first_rollout = strategy.run_with_snapshot(first_request)
    second_rollout = strategy.run_with_snapshot(second_request)
    trajectory = first_rollout.trajectory
    repeated = second_rollout.trajectory
    comparisons["initial_latent"] = _comparison_item(
        {
            "adapter_initial_latent": direct_initial,
            "rollout_initial_latent": trajectory.x_t[:, 0],
        },
        {
            "adapter_initial_latent": expected_initial,
            "rollout_initial_latent": expected_initial,
        },
        rtol=0.0,
        atol=0.0,
    )

    schedule = first_request.dynamics.timesteps(
        num_steps=_CPU_CONTRACT_STEPS,
        device="cpu",
    )
    expected_timesteps = schedule.reshape(1, -1).expand(batch_size, -1)
    terminal = first_request.dynamics.terminal_timestep(device="cpu")
    expected_next = (
        torch.cat((schedule[1:], terminal.reshape(1)))
        .reshape(
            1,
            -1,
        )
        .expand(batch_size, -1)
    )
    comparisons["timestep"] = _comparison_item(
        {
            "timesteps": trajectory.timesteps,
            "next_timesteps": trajectory.next_timesteps,
            "transition_index": trajectory.transition_index,
        },
        {
            "timesteps": expected_timesteps,
            "next_timesteps": expected_next,
            "transition_index": torch.arange(
                _CPU_CONTRACT_STEPS,
                dtype=torch.int64,
            )
            .reshape(1, -1)
            .expand(batch_size, -1),
        },
        rtol=0.0,
        atol=0.0,
    )
    comparisons["rollout_latent"] = _comparison_item(
        {
            "x_t": trajectory.x_t,
            "sampled_action": trajectory.sampled_action,
            "conditioned_next": trajectory.conditioned_next,
            "terminal_media": trajectory.media,
        },
        {
            "x_t": repeated.x_t,
            "sampled_action": repeated.sampled_action,
            "conditioned_next": repeated.conditioned_next,
            "terminal_media": repeated.media,
        },
        rtol=0.0,
        atol=0.0,
    )
    comparisons["old_log_prob"] = _comparison_item(
        {"old_log_prob": trajectory.old_log_probs},
        {"old_log_prob": repeated.old_log_probs},
        rtol=0.0,
        atol=0.0,
    )

    transformer = runtime.loader.transformer
    parameter = getattr(transformer, "scale", None)
    if not isinstance(parameter, torch.nn.Parameter):
        raise TypeError("CPU contract transformer lost its trainable scale")
    with torch.no_grad():
        parameter.add_(0.02)
    recompute_request = PolicyRecomputeRequest(
        adapter=policy,
        dynamics=first_request.dynamics,
        rollout=first_rollout,
        latent_spec=first_request.latent_spec,
        require_reference_statistics=True,
    )
    policy_stats = compute_full_policy_stats_oracle(recompute_request)
    repeat_stats = compute_full_policy_stats_oracle(recompute_request)
    comparisons["current_log_prob"] = _comparison_item(
        {"current_log_prob": policy_stats.current_log_probs},
        {"current_log_prob": repeat_stats.current_log_probs},
        rtol=0.0,
        atol=0.0,
    )
    if any(
        value is None
        for value in (
            policy_stats.current_transition_mean,
            policy_stats.reference_transition_mean,
            policy_stats.transition_std,
            repeat_stats.current_transition_mean,
            repeat_stats.reference_transition_mean,
            repeat_stats.transition_std,
        )
    ):
        raise RuntimeError("CPU Flow-GRPO contract lost reference statistics")
    comparisons["transition_statistics"] = _comparison_item(
        {
            "current_transition_mean": policy_stats.current_transition_mean,
            "reference_transition_mean": (policy_stats.reference_transition_mean),
            "transition_std": policy_stats.transition_std,
        },
        {
            "current_transition_mean": repeat_stats.current_transition_mean,
            "reference_transition_mean": repeat_stats.reference_transition_mean,
            "transition_std": repeat_stats.transition_std,
        },
        rtol=0.0,
        atol=0.0,
    )
    del repeat_stats

    reward = _cpu_contract_reward(trajectory, rewards)
    advantage = GroupZScoreAdvantageProcessor(
        epsilon=1.0e-4,
        std_domain="batch",
        output_dtype="float32",
    ).normalize(
        reward,
        AdvantageGrouping.from_trajectory(trajectory),
        device="cpu",
    )
    oracle_advantage = NativeFlowReferenceOracle.group_advantages(rewards, 1.0e-4)
    comparisons["group_advantage"] = _comparison_item(
        {
            "runtime_group_advantage": advantage.values,
            "oracle_group_advantage": oracle_advantage,
        },
        {
            "runtime_group_advantage": torch.tensor(
                expected_advantages,
                dtype=torch.float32,
            ),
            "oracle_group_advantage": torch.tensor(
                expected_advantages,
                dtype=torch.float64,
            ),
        },
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    credit = GRPOCreditStrategy(
        advantage_epsilon=1.0e-4,
        advantage_std_domain="batch",
        clip_range=1.0e-4,
        advantage_clip=5.0,
        reference_kl_weight=native_request.composition.reference_kl_weight,
    ).plan(trajectory=trajectory, advantage=advantage)
    objective = ClippedSurrogateObjective().compute(
        old_log_probs=trajectory.old_log_probs,
        policy_stats=policy_stats,
        loss_inputs=credit,
    )
    assert policy_stats.current_transition_mean is not None
    assert policy_stats.reference_transition_mean is not None
    assert policy_stats.transition_std is not None
    oracle = NativeFlowReferenceOracle.evaluate(
        old_log_probs=trajectory.old_log_probs,
        new_log_probs=policy_stats.current_log_probs,
        advantages=credit.base_advantage,
        current_mean=policy_stats.current_transition_mean,
        reference_mean=policy_stats.reference_transition_mean,
        std_dev=policy_stats.transition_std,
        clip_range=credit.clip_range,
        beta=credit.reference_kl_weight,
    )
    for item_name, runtime_value, oracle_name in (
        ("policy_loss", objective.policy_loss, "policy_loss"),
        ("reference_kl", objective.reference_kl, "reference_kl"),
        ("total_loss", objective.loss, "total_loss"),
    ):
        comparisons[item_name] = _comparison_item(
            {item_name: runtime_value},
            {item_name: oracle[oracle_name].to(dtype=runtime_value.dtype)},
            rtol=2.0e-5,
            atol=2.0e-6,
        )

    runtime_gradient = torch.autograd.grad(
        objective.loss,
        parameter,
        retain_graph=True,
    )[0]
    oracle_gradient = torch.autograd.grad(
        oracle["total_loss"],
        parameter,
    )[0]
    comparisons["gradient"] = _comparison_item(
        {"transformer.scale": runtime_gradient},
        {"transformer.scale": oracle_gradient},
        rtol=5.0e-5,
        atol=5.0e-6,
    )

    parameter_before = parameter.detach().clone()
    runtime.optimizer.zero_grad(set_to_none=True)
    parameter.grad = runtime_gradient.detach().clone()
    runtime.optimizer.step()
    runtime.optimizer.zero_grad(set_to_none=True)
    runtime_delta = parameter.detach() - parameter_before
    oracle_parameter = torch.nn.Parameter(parameter_before.clone())
    oracle_optimizer = torch.optim.AdamW(
        (oracle_parameter,),
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-4,
    )
    oracle_parameter.grad = oracle_gradient.detach().clone()
    oracle_optimizer.step()
    oracle_delta = oracle_parameter.detach() - parameter_before
    comparisons["parameter_delta"] = _comparison_item(
        {"transformer.scale": runtime_delta},
        {"transformer.scale": oracle_delta},
        rtol=5.0e-5,
        atol=5.0e-7,
    )

    comparisons["checkpoint_resume"] = _cpu_contract_checkpoint_item(
        runtime,
        native_request,
        non_timing_metrics={
            "policy_loss": float(objective.policy_loss.detach()),
            "reference_kl": float(objective.reference_kl.detach()),
            "total_loss": float(objective.loss.detach()),
        },
        next_step_inputs={
            "latents": trajectory.x_t.detach().cpu(),
            "timesteps": trajectory.timesteps.detach().cpu(),
            "old_log_probs": trajectory.old_log_probs.detach().cpu(),
            "advantages": credit.base_advantage.detach().cpu(),
        },
    )
    if set(comparisons) != set(_NATIVE_COMPARISON_KEYS):
        raise RuntimeError("CPU contract did not produce all fourteen comparisons")

    blueprint_id = runtime.algorithm.blueprint.blueprint_id
    bound_payload = {
        "executor_identity": _CPU_CONTRACT_EXECUTOR_ID,
        "recipe_id": native_request.composition.resolved_fingerprint,
        "algorithm_binding_id": runtime.policy.binding.binding_id,
        "algorithm_blueprint_id": blueprint_id,
        "scheduler_blueprint_identity": blueprint.blueprint_identity,
        "parameter_topology_identity": (
            runtime.manager.parameter_state.topology.identity
        ),
    }
    bound_contract_id = hashlib.sha256(
        json.dumps(
            bound_payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return _CPUContractIteration(
        optimizer_step=optimizer_step,
        comparisons=comparisons,
        bound_contract_id=bound_contract_id,
    )


def _cpu_contract_prepare_context(
    recipe_id: str,
    *,
    start_optimizer_step: int,
) -> object:
    from visual_rl.algorithms.trainer.interface import PrepareRunContext

    return PrepareRunContext(
        run_id=_CPU_CONTRACT_EXECUTOR_ID,
        recipe_id=recipe_id,
        start_optimizer_step=start_optimizer_step,
        runtime_facts=(
            ("evidence_scope", "cpu_fake_contract"),
            ("executor_identity", _CPU_CONTRACT_EXECUTOR_ID),
        ),
    )


def run_cpu_contract_executor(
    request: NativeExecutionRequest,
) -> ContractParityExecution:
    """Run the canonical CPU chain while explicitly refusing native scope."""

    if not isinstance(request, NativeExecutionRequest):
        raise TypeError("request must be NativeExecutionRequest")
    composition = request.composition
    if (
        composition.model_class_path != _EXPECTED_MODEL_CLASS
        or composition.algorithm_class_path != _EXPECTED_ALGORITHM_CLASS
        or composition.policy_port_protocol != _POLICY_PORT_PROTOCOL
        or composition.policy_port_implementation != _POLICY_PORT_IMPLEMENTATION
    ):
        raise ValueError("CPU contract request is not the canonical SD3/Flow-GRPO port")

    runtime = _build_cpu_contract_runtime(request)
    bound = None
    trainer = None
    try:
        materializer, trainer = _cpu_contract_materializer(
            runtime,
            execute=lambda step: _execute_cpu_contract_iteration(
                runtime,
                request,
                step,
            ),
        )
        resolved = runtime.resolved_recipe
        execution_policy = resolved.execution_policy.to_receipt()
        bound = runtime.algorithm.materialize(
            runtime.policy,
            runtime.policy.binding,
            resolved.algorithm_spec,
            materializer,
            execution_policy=execution_policy,
        )
        if bound.binding.binding_id != composition.algorithm_binding_id:
            raise ValueError("bound Flow-GRPO module identity drifted")
        bound.prepare_run(
            _cpu_contract_prepare_context(
                composition.resolved_fingerprint,
                start_optimizer_step=int(request.case["logical_step"]),
            )
        )
        result = bound.run_iteration(int(request.case["logical_step"]))
        trainer_iteration = result.iteration
        from visual_rl.algorithms.trainer.interface import IterationResult

        if not isinstance(trainer_iteration, IterationResult):
            raise TypeError("canonical materializer returned an invalid iteration")
        iteration = trainer_iteration.value.payload
        if not isinstance(iteration, _CPUContractIteration):
            raise TypeError("CPU contract module returned an invalid iteration")
        if result.algorithm_binding_id != composition.algorithm_binding_id:
            raise ValueError("CPU contract result binding identity drifted")
        comparisons = dict(iteration.comparisons)
        passed = all(bool(comparisons[name]["passed"]) for name in comparisons)
        return ContractParityExecution(
            passed=passed,
            recipe_id=composition.resolved_fingerprint,
            bound_contract_id=iteration.bound_contract_id,
            algorithm_module_identity=composition.algorithm_module_identity,
            algorithm_binding_id=composition.algorithm_binding_id,
            comparisons=comparisons,
        )
    finally:
        if bound is not None:
            bound.close()
        if trainer is not None:
            trainer.close()
        runtime.close()


def parse_arguments(
    argv: Sequence[str],
    *,
    repo_root: Path | None = None,
) -> HarnessArguments:
    """Parse a strict test-only flag surface without hidden path defaults."""

    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise TypeError("argv must be a non-string sequence")
    root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    values: dict[str, str] = {}
    preflight_only = False
    index = 0
    allowed = {
        "--config",
        "--case",
        "--reference-repo",
        "--reference-revision",
        "--reference-digest",
    }
    while index < len(argv):
        flag = argv[index]
        if flag == "--preflight-only":
            if preflight_only:
                raise HarnessArgumentError("--preflight-only may appear only once")
            preflight_only = True
            index += 1
            continue
        if flag not in allowed:
            raise HarnessArgumentError(f"unknown native harness argument: {flag!r}")
        if flag in values:
            raise HarnessArgumentError(f"duplicate native harness argument: {flag}")
        if index + 1 >= len(argv):
            raise HarnessArgumentError(f"{flag} requires one value")
        value = argv[index + 1]
        if not isinstance(value, str) or not value or value.startswith("--"):
            raise HarnessArgumentError(f"{flag} requires one non-empty value")
        values[flag] = value
        index += 2

    missing = tuple(
        flag
        for flag in (
            "--reference-repo",
            "--reference-revision",
            "--reference-digest",
        )
        if flag not in values
    )
    if missing:
        raise HarnessArgumentError(
            "native reference identity must be explicit; missing " + ", ".join(missing)
        )

    def rooted(flag: str, default: str) -> Path:
        raw = Path(values.get(flag, default)).expanduser()
        return (raw if raw.is_absolute() else root / raw).resolve(strict=False)

    return HarnessArguments(
        repo_root=root,
        config_path=rooted("--config", _DEFAULT_CONFIG),
        case_path=rooted("--case", _DEFAULT_CASE),
        reference_repo=rooted("--reference-repo", ""),
        reference_revision=values["--reference-revision"],
        reference_digest=values["--reference-digest"],
        preflight_only=preflight_only,
    )


def compute_reference_digest(reference_repo: Path) -> str:
    """Hash the paths, modes, and bytes of every Git-tracked stage-0 entry."""

    root = Path(reference_repo).resolve(strict=True)
    records = _git_bytes(root, "ls-files", "--stage", "-z").split(b"\0")
    records = [record for record in records if record]
    if not records:
        raise ValueError("reference repo has no Git-tracked files")
    hasher = hashlib.sha256()
    for record in records:
        try:
            metadata, relative_bytes = record.split(b"\t", 1)
            mode, object_id, stage = metadata.split(b" ", 2)
        except ValueError as exc:
            raise ValueError("git ls-files returned a malformed record") from exc
        if stage != b"0":
            raise ValueError("reference repo contains an unresolved index stage")
        relative = Path(os.fsdecode(relative_bytes))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("reference repo returned an unsafe tracked path")
        source = root / relative
        if mode == b"160000":
            content = b"gitlink:" + object_id
        elif mode == b"120000":
            content = os.fsencode(os.readlink(source))
        elif source.is_file():
            content = source.read_bytes()
        else:
            raise ValueError(
                f"tracked reference file is missing: {relative.as_posix()}"
            )
        for part in (mode, relative_bytes, content):
            hasher.update(len(part).to_bytes(8, "big"))
            hasher.update(part)
    return hasher.hexdigest()


def inspect_reference_identity(reference_repo: Path) -> ReferenceIdentity:
    """Observe one clean checkout without accepting an implicit revision."""

    root = Path(reference_repo).resolve(strict=True)
    revision = _git_text(root, "rev-parse", "HEAD")
    if _GIT_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("reference HEAD is not a full lowercase Git digest")
    dirty = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if dirty:
        raise ValueError("reference repo must be clean, including untracked files")
    return ReferenceIdentity(
        path=root,
        revision=revision,
        digest=compute_reference_digest(root),
    )


def run_preflight(
    arguments: HarnessArguments,
    *,
    dependency_probe: Callable[[str], bool] | None = None,
    cuda_probe: Callable[[], tuple[bool, str]] | None = None,
) -> tuple[
    tuple[PreflightCheck, ...],
    Mapping[str, object] | None,
    ReferenceIdentity | None,
    CompositionIdentity | None,
]:
    """Run every read-only gate and return all failures as structured checks."""

    if not isinstance(arguments, HarnessArguments):
        raise TypeError("arguments must be HarnessArguments")
    dependency_probe = dependency_probe or _dependency_available
    cuda_probe = cuda_probe or _cuda_available
    checks: list[PreflightCheck] = []

    case: Mapping[str, object] | None = None
    try:
        case = _load_case(arguments.case_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        checks.append(_error("fixture", exc))
    else:
        checks.append(
            PreflightCheck(
                "fixture",
                "pass",
                "native fixture schema and finite values validated",
                hashlib.sha256(arguments.case_path.read_bytes()).hexdigest(),
            )
        )

    missing_dependencies = tuple(
        name for name in _EXECUTION_DEPENDENCIES if not dependency_probe(name)
    )
    if missing_dependencies:
        checks.append(
            PreflightCheck(
                "python_dependencies",
                "error",
                "native parity Python dependencies are missing",
                list(missing_dependencies),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "python_dependencies",
                "pass",
                "all native parity Python dependencies are importable",
                list(_EXECUTION_DEPENDENCIES),
            )
        )

    cuda_available, cuda_detail = cuda_probe()
    checks.append(
        PreflightCheck(
            "cuda",
            "pass" if cuda_available else "error",
            (
                "CUDA runtime is available"
                if cuda_available
                else "native parity requires an available CUDA runtime"
            ),
            cuda_detail,
        )
    )

    reference: ReferenceIdentity | None = None
    try:
        reference = inspect_reference_identity(arguments.reference_repo)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        checks.append(_error("reference_checkout", exc))
    else:
        checks.append(
            PreflightCheck(
                "reference_checkout",
                "pass",
                "reference checkout is a clean Git worktree",
                reference.to_payload(),
            )
        )
        if reference.revision != arguments.reference_revision:
            checks.append(
                PreflightCheck(
                    "reference_revision",
                    "error",
                    "observed reference revision differs from the explicit pin",
                    reference.revision,
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "reference_revision",
                    "pass",
                    "reference revision matches the explicit pin",
                    reference.revision,
                )
            )
        if reference.digest != arguments.reference_digest:
            checks.append(
                PreflightCheck(
                    "reference_digest",
                    "error",
                    "observed reference tree differs from the explicit SHA-256 pin",
                    reference.digest,
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "reference_digest",
                    "pass",
                    "reference tree matches the explicit SHA-256 pin",
                    reference.digest,
                )
            )
        missing_files = tuple(
            relative
            for relative in _REFERENCE_FILES
            if not (reference.path / relative).is_file()
        )
        checks.append(
            PreflightCheck(
                "reference_modules",
                "error" if missing_files else "pass",
                (
                    "reference checkout is missing required Flow-GRPO modules"
                    if missing_files
                    else "required Flow-GRPO reference modules are present"
                ),
                list(missing_files or _REFERENCE_FILES),
            )
        )

    # Descriptor binding is deliberately independent from the native execution
    # environment.  Missing upstream-only packages such as wandb/absl and even
    # missing model-runtime extras must not hide whether the public SD3 x Flow
    # contracts bind.  The resolver used below records optional dependencies
    # without probing them; the execution-dependency check above remains the
    # authoritative environment gate before any weights or upstream code run.
    composition: CompositionIdentity | None = None
    try:
        composition, model_artifact = _resolve_v08_composition(arguments.config_path)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        checks.append(_error("v08_composition", exc))
    else:
        checks.append(
            PreflightCheck(
                "v08_composition",
                "pass",
                "runtime validated the SD3 and Flow-GRPO binding before materialization",
                composition.to_payload(),
            )
        )
        artifact_ok = model_artifact.is_dir()
        checks.append(
            PreflightCheck(
                "model_artifact",
                "pass" if artifact_ok else "error",
                (
                    "SD3 model artifact directory exists"
                    if artifact_ok
                    else "SD3 model artifact directory is missing"
                ),
                str(model_artifact),
            )
        )
        if case is not None:
            expected_group = _resolved_group_size(arguments.config_path)
            actual_group = len(case["reward_values"])
            checks.append(
                PreflightCheck(
                    "fixture_group_size",
                    "pass" if actual_group == expected_group else "error",
                    (
                        "fixture group size matches the resolved recipe"
                        if actual_group == expected_group
                        else "fixture group size differs from the resolved recipe"
                    ),
                    {"expected": expected_group, "observed": actual_group},
                )
            )

    return tuple(checks), case, reference, composition


def run_harness(
    arguments: HarnessArguments,
    *,
    executor: NativeExecutor | None = None,
    dependency_probe: Callable[[str], bool] | None = None,
    cuda_probe: Callable[[], tuple[bool, str]] | None = None,
) -> dict[str, object]:
    """Run preflight and optionally dispatch one identity-checked executor."""

    checks, case, reference, composition = run_preflight(
        arguments,
        dependency_probe=dependency_probe,
        cuda_probe=cuda_probe,
    )
    preflight_passed = all(item.status == "pass" for item in checks)
    execution: dict[str, object]
    if arguments.preflight_only:
        execution = {
            "status": "not_run",
            "reason": "explicit preflight-only mode",
            "comparisons": {},
        }
    elif not preflight_passed:
        execution = {
            "status": "not_run",
            "reason": "preflight failed",
            "comparisons": {},
        }
    elif case is None or reference is None or composition is None:
        raise RuntimeError("passed preflight did not retain its validated identities")
    elif executor is None:
        artifact = _resolved_model_artifact(arguments.config_path)
        missing_marker = not (artifact / "model_index.json").is_file()
        execution = {
            "status": "blocked",
            "reason": (
                "native execution is blocked because the SD3 artifact lacks "
                "model_index.json"
                if missing_marker
                else "native execution requires an explicit pinned-upstream "
                "executor; the CPU fake contract is not native evidence"
            ),
            "comparisons": {},
        }
    else:
        try:
            result = executor(
                NativeExecutionRequest(
                    arguments=arguments,
                    case=case,
                    reference=reference,
                    composition=composition,
                )
            )
            _validate_execution_identity(result, reference, composition)
        except BaseException as exc:  # noqa: BLE001 - executable failure boundary
            execution = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "comparisons": {},
            }
        else:
            execution = {
                "status": "passed" if result.passed else "failed",
                "reason": "executor completed",
                "identity": {
                    "recipe_id": result.recipe_id,
                    "bound_contract_id": result.bound_contract_id,
                    "algorithm_module_identity": (result.algorithm_module_identity),
                    "algorithm_binding_id": result.algorithm_binding_id,
                    "evidence_scope": result.evidence_scope,
                },
                "comparisons": dict(result.comparisons),
            }

    native_parity_passed = execution["status"] == "passed"
    return {
        "schema_version": 2,
        "harness_id": _HARNESS_ID,
        "parity_protocol": _parity_protocol_payload(),
        "mode": "preflight" if arguments.preflight_only else "execute",
        "preflight": {
            "passed": preflight_passed,
            "checks": [item.to_payload() for item in checks],
        },
        "reference": None if reference is None else reference.to_payload(),
        "composition": None if composition is None else composition.to_payload(),
        "execution": execution,
        "native_parity_passed": native_parity_passed,
    }


def _validate_execution_identity(
    result: NativeParityExecution,
    reference: ReferenceIdentity,
    composition: CompositionIdentity,
) -> None:
    if not isinstance(result, NativeParityExecution):
        raise TypeError("native executor must return NativeParityExecution")
    if result.algorithm_module_identity != composition.algorithm_module_identity:
        raise ValueError("executor returned a different algorithm module identity")
    if result.algorithm_binding_id != composition.algorithm_binding_id:
        raise ValueError("executor returned a different algorithm binding identity")
    if result.reference_revision != reference.revision:
        raise ValueError("executor returned a different reference revision")
    if result.reference_digest != reference.digest:
        raise ValueError("executor returned a different reference digest")


def _resolve_v08_composition(config_path: Path) -> tuple[CompositionIdentity, Path]:
    from visual_rl.algorithms.trainer.execution_plan import AlgorithmExecutionPlan
    from visual_rl.composition.compatibility import bind_model_algorithm
    from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
    from visual_rl.composition.config.source import load_source_recipe
    from visual_rl.composition.registry import ResolvedAlgorithmDeclaration
    from visual_rl.core.contracts.composition import BoundPolicyCapabilities

    source = load_source_recipe(config_path)
    resolved = _compile_v08_recipe(config_path)
    algorithm_declaration = resolved.algorithm
    if not isinstance(algorithm_declaration, ResolvedAlgorithmDeclaration):
        raise TypeError("v0.8 recipe omitted its typed algorithm declaration")
    if resolved.compatibility.status == "invalid":
        issue_codes = tuple(issue["code"] for issue in resolved.compatibility.issues)
        raise ValueError(f"v0.8 recipe is statically incompatible: {issue_codes}")
    model_declaration = resolved.model.declaration
    if (
        model_declaration.alias != "sd3"
        or model_declaration.implementation_class_path != _EXPECTED_MODEL_CLASS
    ):
        raise ValueError("native scope requires the v0.8 SD3Adapter descriptor")
    if (
        algorithm_declaration.alias != "flow-grpo"
        or algorithm_declaration.component.implementation_class_path
        != _EXPECTED_ALGORITHM_CLASS
    ):
        raise ValueError(
            "native scope requires the v0.8 FlowGRPOAlgorithmModule descriptor"
        )

    dynamics_declaration = resolved.component("dynamics").declaration
    trainer_declaration = resolved.component("trainer").declaration
    plan = AlgorithmExecutionPlan.from_spec(
        resolved.algorithm_spec,
        execution_policy=resolved.execution_policy.to_receipt(),
    )
    if not math.isfinite(plan.beta) or plan.beta <= 0.0:
        raise ValueError("native Flow-GRPO scope requires a finite beta > 0")
    declared_model = model_declaration.declared_contract.model
    declared_dynamics = dynamics_declaration.declared_contract.dynamics
    declared_trainer = trainer_declaration.declared_contract.trainer
    if any(
        value is None
        for value in (
            declared_model,
            declared_dynamics,
            declared_trainer,
        )
    ):
        raise TypeError("v0.8 descriptor graph omitted a typed public-port contract")
    capabilities = BoundPolicyCapabilities.from_contracts(
        declared_model,
        dynamics=declared_dynamics,
        trainer=declared_trainer,
    )
    requirements = algorithm_declaration.requirements
    binding = bind_model_algorithm(capabilities, requirements)
    launch = bootstrap_recipe_v2(source).require_launch()
    return (
        CompositionIdentity(
            config_path=config_path,
            resolved_fingerprint=_canonical_identity_digest(
                resolved.resolved_fingerprint,
                namespace="resolved-recipe.v2",
            ),
            model_requested_id=model_declaration.alias,
            model_class_path=model_declaration.implementation_class_path,
            algorithm_requested_id=algorithm_declaration.alias,
            algorithm_class_path=(
                algorithm_declaration.component.implementation_class_path
            ),
            model_capability_id=capabilities.capability_id,
            algorithm_requirement_id=requirements.requirement_id,
            algorithm_module_identity=_canonical_identity_digest(
                algorithm_declaration.declaration_id,
                namespace="algorithm-declaration.v1",
            ),
            algorithm_binding_id=binding.binding_id,
            reference_kl_weight=plan.beta,
            policy_port_protocol=_POLICY_PORT_PROTOCOL,
            policy_port_implementation=_POLICY_PORT_IMPLEMENTATION,
        ),
        launch.artifacts.model,
    )


def _resolved_model_artifact(config_path: Path) -> Path:
    """Resolve the sole model artifact through the v0.8 config frontend."""

    from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
    from visual_rl.composition.config.source import load_source_recipe

    return (
        bootstrap_recipe_v2(load_source_recipe(config_path))
        .require_launch()
        .artifacts.model
    )


def _resolved_group_size(config_path: Path) -> int:
    resolved = _compile_v08_recipe(config_path)
    value = resolved.execution_policy.group_size
    if type(value) is not int or value < 2:
        raise ValueError("resolved Flow-GRPO group_size must be an integer >= 2")
    return value


def _compile_v08_recipe(config_path: Path) -> ResolvedRecipe:
    """Compile one exact typed recipe without importing runtime implementations."""

    from visual_rl.composition.config.compiler import compile_recipe_v2, default_catalog
    from visual_rl.composition.config.source import load_source_recipe
    from visual_rl.composition.recipes.schema import ResolvedRecipe

    resolved = compile_recipe_v2(
        load_source_recipe(config_path),
        catalog=default_catalog(),
    )
    if not isinstance(resolved, ResolvedRecipe):
        raise TypeError("canonical compiler did not return ResolvedRecipe")
    return resolved


def _canonical_identity_digest(value: str, *, namespace: str) -> str:
    """Project one exact namespaced canonical identity to its SHA-256 digest."""

    prefix = f"{namespace}:"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"identity must use namespace {namespace!r}")
    digest = value.removeprefix(prefix)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"identity in namespace {namespace!r} has an invalid digest")
    return digest


def _load_case(case_path: Path) -> Mapping[str, object]:
    value = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _CASE_KEYS:
        raise ValueError("native v0.8 case has an invalid exact key set")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError("native v0.8 case schema_version must be integer 1")
    if not isinstance(value["prompt"], str) or not value["prompt"].strip():
        raise ValueError("native v0.8 case prompt must be non-empty")
    seed = value["seed"]
    if type(seed) is not int or not 0 <= seed <= 0xFFFF_FFFF:
        raise ValueError("native v0.8 case seed must be uint32")
    if value["logical_step"] != 0:
        raise ValueError("native v0.8 case logical_step must be zero")
    for key in ("reward_values", "expected_advantages"):
        values = value[key]
        if (
            type(values) is not list
            or len(values) < 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in values
            )
        ):
            raise ValueError(f"native v0.8 case {key} must be a finite scalar list")
    if len(value["reward_values"]) != len(value["expected_advantages"]):
        raise ValueError("native v0.8 rewards and advantages must share group size")
    rewards = tuple(float(item) for item in value["reward_values"])
    mean = sum(rewards) / len(rewards)
    if sum((item - mean) ** 2 for item in rewards) == 0.0:
        raise ValueError("native v0.8 case rewards must have non-zero variance")
    return value


def _dependency_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _cuda_available() -> tuple[bool, str]:
    try:
        import torch
    except ImportError as exc:
        return False, f"ImportError: {exc}"
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return count > 0, f"torch={torch.__version__}, cuda_devices={count}"


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    return _git_bytes(root, *arguments).decode("ascii").strip()


def _error(code: str, exc: BaseException) -> PreflightCheck:
    return PreflightCheck(
        code=code,
        status="error",
        detail=f"{type(exc).__name__}: {exc}",
    )


def _parity_protocol_payload() -> dict[str, object]:
    """Name the exact claim so kernel parity cannot imply script-loop parity."""

    return {
        "protocol_id": _PARITY_PROTOCOL_ID,
        "reference_surface": "pinned_upstream_numerical_kernels",
        "configuration_source": "resolved_v08_recipe",
        "profile_interpretation": (
            "chosen_kernel_parity_profile_not_upstream_experiment_defaults"
        ),
        "schedule_source": "resolved_v08_recipe",
        "update_cadence": "v08_full_trajectory_single_adamw_commit",
        "excluded_claims": [
            "upstream_train_sd3_end_to_end_update_cadence",
            "upstream_geneval_experiment_default_hyperparameters",
        ],
    }


def _argument_failure_report(exc: BaseException) -> dict[str, object]:
    return {
        "schema_version": 2,
        "harness_id": _HARNESS_ID,
        "parity_protocol": _parity_protocol_payload(),
        "mode": "argument_error",
        "preflight": {
            "passed": False,
            "checks": [_error("arguments", exc).to_payload()],
        },
        "reference": None,
        "composition": None,
        "execution": {
            "status": "not_run",
            "reason": "argument validation failed",
            "comparisons": {},
        },
        "native_parity_passed": False,
    }


def _canonical_report_json(report: Mapping[str, object]) -> str:
    expected = {
        "schema_version",
        "harness_id",
        "parity_protocol",
        "mode",
        "preflight",
        "reference",
        "composition",
        "execution",
        "native_parity_passed",
    }
    if not isinstance(report, Mapping) or set(report) != expected:
        raise ValueError("v0.8 native report has an invalid top-level key set")
    return json.dumps(
        dict(report),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments: HarnessArguments | None = None
    try:
        arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
        report = run_harness(arguments)
    except BaseException as exc:  # noqa: BLE001 - executable failure boundary
        report = _argument_failure_report(exc)
    try:
        payload = _canonical_report_json(report)
    except BaseException as exc:  # noqa: BLE001 - serializer must fail closed
        report = _argument_failure_report(exc)
        payload = _canonical_report_json(report)
    sys.stdout.write(payload + "\n")
    if arguments is not None and arguments.preflight_only:
        return 0 if report["preflight"]["passed"] else 1
    return 0 if report["native_parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
