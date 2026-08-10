"""Typed semantic training and location-only launch specifications."""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visual_rl.core.contracts import (
    ComputePrecision,
    DistributionMode,
    ExecutionPolicyReceipt,
    ExecutionTransformPlan,
    TrainingMode,
)
from visual_rl.core.identity import canonical_identity
from visual_rl.core.types import ResolutionContext, validate_step_seed_budget

__all__ = (
    "AdamWSpec",
    "ArtifactLocations",
    "ExecutionPolicySpec",
    "LaunchSpec",
    "LearningRateScheduleSpec",
    "PolicyRecomputeSpec",
    "RewardRuntimeBindingSpec",
    "RolloutExecutionPolicySpec",
    "SpecValidationError",
    "TrainingSpec",
    "UpdateSafetySpec",
)


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_TORCH_DEVICE = re.compile(r"^(?P<type>[a-z][a-z0-9_]*)(?::(?P<index>0|[1-9][0-9]*))?$")
_TORCH_DEVICE_TYPES = frozenset(
    {
        "cpu",
        "cuda",
        "fpga",
        "hip",
        "hpu",
        "ideep",
        "ipu",
        "lazy",
        "maia",
        "meta",
        "mkldnn",
        "mps",
        "mtia",
        "opencl",
        "opengl",
        "privateuseone",
        "ve",
        "vulkan",
        "xla",
        "xpu",
    }
)


class SpecValidationError(ValueError):
    """Strict typed-spec error carrying the exact dotted input path."""

    def __init__(self, message: str, *, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("spec validation path must be non-empty")
        self.path = path
        super().__init__(f"{path}: {message}")


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecValidationError("expected a mapping", path=path)
    if any(not isinstance(key, str) for key in value):
        raise SpecValidationError("mapping keys must be strings", path=path)
    return value


def _exact(
    value: object,
    expected: frozenset[str],
    *,
    path: str,
) -> Mapping[str, Any]:
    result = _mapping(value, path=path)
    missing = sorted(expected - set(result))
    unknown = sorted(set(result) - expected)
    if missing or unknown:
        error_path = path
        if len(unknown) == 1:
            error_path = f"{path}.{unknown[0]}"
        elif not unknown and len(missing) == 1:
            error_path = f"{path}.{missing[0]}"
        raise SpecValidationError(
            f"invalid exact key set: missing={missing}, unknown={unknown}",
            path=error_path,
        )
    return result


def _integer(value: object, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if minimum == 0 else f">= {minimum}"
        raise SpecValidationError(f"expected an integer {qualifier}", path=path)
    return value


def _finite_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecValidationError("expected a finite number", path=path)
    result = float(value)
    if not math.isfinite(result):
        raise SpecValidationError("expected a finite number", path=path)
    return result


def _positive_float(value: object, *, path: str) -> float:
    result = _finite_float(value, path=path)
    if result <= 0.0:
        raise SpecValidationError("expected a positive finite number", path=path)
    return result


def _bool(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        raise SpecValidationError("expected a boolean", path=path)
    return value


def _literal(value: object, allowed: frozenset[str], *, path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SpecValidationError(
            f"expected one of {sorted(allowed)}",
            path=path,
        )
    return value


def _canonical_torch_device(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecValidationError(
            "expected a canonical non-empty torch device",
            path=path,
        )
    match = _TORCH_DEVICE.fullmatch(value)
    if match is None or match.group("type") not in _TORCH_DEVICE_TYPES:
        raise SpecValidationError(
            "expected a canonical torch device",
            path=path,
        )
    return value


def _canonical_host(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SpecValidationError(
            "trusted host values must be canonical non-empty strings",
            path=path,
        )
    if "%" in value:
        raise SpecValidationError(
            "trusted host must not contain an IPv6 zone identifier",
            path=path,
        )
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        try:
            host = value.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise SpecValidationError(
                "trusted host is not a valid hostname",
                path=path,
            ) from exc
        host = host.removesuffix(".")
        labels = host.split(".")
        if (
            not host
            or len(host) > 253
            or any(_HOST_LABEL.fullmatch(label) is None for label in labels)
        ):
            raise SpecValidationError(
                "trusted host is not a valid hostname",
                path=path,
            )
        return host


def _trusted_hosts(value: object, *, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SpecValidationError(
            "trusted_hosts must be a non-string sequence of hostnames",
            path=path,
        )
    result = tuple(
        _canonical_host(item, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise SpecValidationError(
            "remote trusted_hosts must be non-empty",
            path=path,
        )
    if len(result) != len(set(result)):
        raise SpecValidationError(
            "trusted_hosts must be unique after canonicalization",
            path=path,
        )
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class AdamWSpec:
    """Exact AdamW hyperparameters which affect optimizer semantics."""

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    amsgrad: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "learning_rate",
            _positive_float(self.learning_rate, path="training.adamw.learning_rate"),
        )
        for name in ("beta1", "beta2"):
            value = _finite_float(
                getattr(self, name),
                path=f"training.adamw.{name}",
            )
            if not 0.0 <= value < 1.0:
                raise SpecValidationError(
                    "expected a value in [0, 1)",
                    path=f"training.adamw.{name}",
                )
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "epsilon",
            _positive_float(self.epsilon, path="training.adamw.epsilon"),
        )
        weight_decay = _finite_float(
            self.weight_decay,
            path="training.adamw.weight_decay",
        )
        if weight_decay < 0.0:
            raise SpecValidationError(
                "expected a non-negative finite number",
                path="training.adamw.weight_decay",
            )
        object.__setattr__(self, "weight_decay", weight_decay)
        _bool(self.amsgrad, path="training.adamw.amsgrad")

    @classmethod
    def from_mapping(cls, value: object) -> AdamWSpec:
        raw = _exact(
            value,
            frozenset(
                {
                    "learning_rate",
                    "beta1",
                    "beta2",
                    "epsilon",
                    "weight_decay",
                    "amsgrad",
                }
            ),
            path="training.adamw",
        )
        return cls(**raw)

    def to_payload(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }


@dataclass(frozen=True, slots=True)
class LearningRateScheduleSpec:
    """Optimizer-step LR schedule; total steps come from ``TrainingSpec``."""

    kind: str
    warmup_steps: int
    min_lr_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _literal(
                self.kind,
                frozenset({"constant", "linear", "cosine"}),
                path="training.lr_schedule.kind",
            ),
        )
        _integer(
            self.warmup_steps,
            path="training.lr_schedule.warmup_steps",
        )
        ratio = _finite_float(
            self.min_lr_ratio,
            path="training.lr_schedule.min_lr_ratio",
        )
        if not 0.0 <= ratio <= 1.0:
            raise SpecValidationError(
                "expected a value in [0, 1]",
                path="training.lr_schedule.min_lr_ratio",
            )
        object.__setattr__(self, "min_lr_ratio", ratio)

    @classmethod
    def from_mapping(cls, value: object) -> LearningRateScheduleSpec:
        raw = _exact(
            value,
            frozenset({"kind", "warmup_steps", "min_lr_ratio"}),
            path="training.lr_schedule",
        )
        return cls(**raw)

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }


@dataclass(frozen=True, slots=True)
class UpdateSafetySpec:
    """Failure and gradient-safety semantics for the update transaction."""

    require_finite_gradients: bool
    require_nonzero_gradients: bool
    max_grad_norm: float | None
    max_initial_logprob_delta: float | None
    require_initial_clipfrac_zero: bool
    zero_grad_set_to_none: bool
    scaler_skip_policy: str
    post_optimizer_failure_policy: str

    def __post_init__(self) -> None:
        for name in (
            "require_finite_gradients",
            "require_nonzero_gradients",
            "require_initial_clipfrac_zero",
            "zero_grad_set_to_none",
        ):
            _bool(getattr(self, name), path=f"training.update_safety.{name}")
        if self.max_grad_norm is not None:
            object.__setattr__(
                self,
                "max_grad_norm",
                _positive_float(
                    self.max_grad_norm,
                    path="training.update_safety.max_grad_norm",
                ),
            )
        if self.max_initial_logprob_delta is not None:
            delta = _finite_float(
                self.max_initial_logprob_delta,
                path="training.update_safety.max_initial_logprob_delta",
            )
            if delta < 0.0:
                raise SpecValidationError(
                    "expected a non-negative finite number or null",
                    path="training.update_safety.max_initial_logprob_delta",
                )
            object.__setattr__(self, "max_initial_logprob_delta", delta)
        object.__setattr__(
            self,
            "scaler_skip_policy",
            _literal(
                self.scaler_skip_policy,
                frozenset({"do_not_commit"}),
                path="training.update_safety.scaler_skip_policy",
            ),
        )
        object.__setattr__(
            self,
            "post_optimizer_failure_policy",
            _literal(
                self.post_optimizer_failure_policy,
                frozenset({"poison_and_restore"}),
                path="training.update_safety.post_optimizer_failure_policy",
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> UpdateSafetySpec:
        raw = _exact(
            value,
            frozenset(
                {
                    "require_finite_gradients",
                    "require_nonzero_gradients",
                    "max_grad_norm",
                    "max_initial_logprob_delta",
                    "require_initial_clipfrac_zero",
                    "zero_grad_set_to_none",
                    "scaler_skip_policy",
                    "post_optimizer_failure_policy",
                }
            ),
            path="training.update_safety",
        )
        return cls(**raw)

    def to_payload(self) -> dict[str, object]:
        return {
            "require_finite_gradients": self.require_finite_gradients,
            "require_nonzero_gradients": self.require_nonzero_gradients,
            "max_grad_norm": self.max_grad_norm,
            "max_initial_logprob_delta": self.max_initial_logprob_delta,
            "require_initial_clipfrac_zero": self.require_initial_clipfrac_zero,
            "zero_grad_set_to_none": self.zero_grad_set_to_none,
            "scaler_skip_policy": self.scaler_skip_policy,
            "post_optimizer_failure_policy": self.post_optimizer_failure_policy,
        }


@dataclass(frozen=True, slots=True)
class PolicyRecomputeSpec:
    """Memory-bounded current/reference policy replay geometry.

    This is an execution choice, not an algorithm choice: it may change the
    number of forward/backward calls but never the set of active policy cells
    or their objective weights. ``None`` replays all rollout rows together;
    the transition window remains independently bounded.
    """

    row_microbatch_size: int | None
    transition_window_size: int

    def __post_init__(self) -> None:
        if self.row_microbatch_size is not None:
            object.__setattr__(
                self,
                "row_microbatch_size",
                _integer(
                    self.row_microbatch_size,
                    path="training.policy_recompute.row_microbatch_size",
                    minimum=1,
                ),
            )
        object.__setattr__(
            self,
            "transition_window_size",
            _integer(
                self.transition_window_size,
                path="training.policy_recompute.transition_window_size",
                minimum=1,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> PolicyRecomputeSpec:
        raw = _exact(
            value,
            frozenset({"row_microbatch_size", "transition_window_size"}),
            path="training.policy_recompute",
        )
        return cls(**raw)

    def to_payload(self) -> dict[str, object]:
        return {
            "row_microbatch_size": self.row_microbatch_size,
            "transition_window_size": self.transition_window_size,
        }


@dataclass(frozen=True, slots=True)
class RolloutExecutionPolicySpec:
    """Physical rollout partitioning and storage policy, never algorithm math."""

    forward_microbatch_size: int | None
    decode_microbatch_size: int | None
    trajectory_storage_device: str

    def __post_init__(self) -> None:
        for name in ("forward_microbatch_size", "decode_microbatch_size"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, path=f"execution.rollout.{name}", minimum=1)
        object.__setattr__(
            self,
            "trajectory_storage_device",
            _literal(
                self.trajectory_storage_device,
                frozenset({"cpu", "model"}),
                path="execution.rollout.trajectory_storage_device",
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> RolloutExecutionPolicySpec:
        raw = _exact(
            value,
            frozenset(
                {
                    "forward_microbatch_size",
                    "decode_microbatch_size",
                    "trajectory_storage_device",
                }
            ),
            path="execution.rollout",
        )
        return cls(**raw)

    def to_payload(self) -> dict[str, object]:
        return {
            "forward_microbatch_size": self.forward_microbatch_size,
            "decode_microbatch_size": self.decode_microbatch_size,
            "trajectory_storage_device": self.trajectory_storage_device,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicySpec:
    """Typed run geometry and execution semantics compiled beside an algorithm."""

    training_mode: TrainingMode
    distribution_mode: DistributionMode
    precision: ComputePrecision
    group_size: int
    rollout: RolloutExecutionPolicySpec
    transform_plan: ExecutionTransformPlan
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise SpecValidationError(
                "expected integer 1",
                path="execution.schema_version",
            )
        for name, expected_type in (
            ("training_mode", TrainingMode),
            ("distribution_mode", DistributionMode),
            ("precision", ComputePrecision),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        _integer(self.group_size, path="execution.group_size", minimum=1)
        if not isinstance(self.rollout, RolloutExecutionPolicySpec):
            raise TypeError("rollout must be a RolloutExecutionPolicySpec")
        if not isinstance(self.transform_plan, ExecutionTransformPlan):
            raise TypeError("transform_plan must be an ExecutionTransformPlan")

    @classmethod
    def from_mapping(cls, value: object) -> ExecutionPolicySpec:
        raw = _exact(
            value,
            frozenset(
                {
                    "schema_version",
                    "training_mode",
                    "distribution_mode",
                    "precision",
                    "group_size",
                    "rollout",
                    "transform_plan",
                }
            ),
            path="execution",
        )
        try:
            training_mode = TrainingMode(raw["training_mode"])
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(
                "unsupported training mode",
                path="execution.training_mode",
            ) from exc
        try:
            distribution_mode = DistributionMode(raw["distribution_mode"])
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(
                "unsupported distribution mode",
                path="execution.distribution_mode",
            ) from exc
        try:
            precision = ComputePrecision(raw["precision"])
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(
                "unsupported compute precision",
                path="execution.precision",
            ) from exc
        try:
            transform_plan = ExecutionTransformPlan.from_mapping(
                _mapping(raw["transform_plan"], path="execution.transform_plan")
            )
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(
                str(exc),
                path="execution.transform_plan",
            ) from exc
        return cls(
            schema_version=raw["schema_version"],
            training_mode=training_mode,
            distribution_mode=distribution_mode,
            precision=precision,
            group_size=raw["group_size"],
            rollout=RolloutExecutionPolicySpec.from_mapping(raw["rollout"]),
            transform_plan=transform_plan,
        )

    @property
    def policy_id(self) -> str:
        return canonical_identity("execution-policy.v1", self.to_payload())

    def to_receipt(self) -> ExecutionPolicyReceipt:
        """Freeze the complete owner payload for algorithm-side verification."""

        receipt = ExecutionPolicyReceipt.from_payload(self.to_payload())
        if receipt.policy_id != self.policy_id:  # pragma: no cover - defensive
            raise RuntimeError("execution policy receipt identity drift")
        return receipt

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "training_mode": self.training_mode.value,
            "distribution_mode": self.distribution_mode.value,
            "precision": self.precision.value,
            "group_size": self.group_size,
            "rollout": self.rollout.to_payload(),
            "transform_plan": self.transform_plan.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """Complete semantic optimizer-run definition hashed into recipe identity."""

    seed: int
    global_prompt_batch_size: int
    max_optimizer_steps: int
    gradient_accumulation_steps: int
    adamw: AdamWSpec
    lr_schedule: LearningRateScheduleSpec
    update_safety: UpdateSafetySpec
    policy_recompute: PolicyRecomputeSpec
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise SpecValidationError(
                "expected integer 1",
                path="training.schema_version",
            )
        seed = _integer(self.seed, path="training.seed")
        if seed > 0xFFFF_FFFF:
            raise SpecValidationError(
                "expected a uint32 integer",
                path="training.seed",
            )
        _integer(
            self.global_prompt_batch_size,
            path="training.global_prompt_batch_size",
            minimum=1,
        )
        _integer(
            self.max_optimizer_steps,
            path="training.max_optimizer_steps",
            minimum=1,
        )
        _integer(
            self.gradient_accumulation_steps,
            path="training.gradient_accumulation_steps",
            minimum=1,
        )
        if self.global_prompt_batch_size % self.gradient_accumulation_steps:
            raise SpecValidationError(
                "must be divisible by gradient_accumulation_steps",
                path="training.global_prompt_batch_size",
            )
        if not isinstance(self.adamw, AdamWSpec):
            raise TypeError("adamw must be an AdamWSpec")
        if not isinstance(self.lr_schedule, LearningRateScheduleSpec):
            raise TypeError("lr_schedule must be a LearningRateScheduleSpec")
        if not isinstance(self.update_safety, UpdateSafetySpec):
            raise TypeError("update_safety must be an UpdateSafetySpec")
        if not isinstance(self.policy_recompute, PolicyRecomputeSpec):
            raise TypeError("policy_recompute must be a PolicyRecomputeSpec")
        if self.lr_schedule.warmup_steps > self.max_optimizer_steps:
            raise SpecValidationError(
                "must not exceed max_optimizer_steps",
                path="training.lr_schedule.warmup_steps",
            )
        try:
            validate_step_seed_budget(seed, self.max_optimizer_steps, 1)
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(str(exc), path="training.seed") from exc

    @classmethod
    def from_mapping(cls, value: object) -> TrainingSpec:
        raw = _exact(
            value,
            frozenset(
                {
                    "schema_version",
                    "seed",
                    "global_prompt_batch_size",
                    "max_optimizer_steps",
                    "gradient_accumulation_steps",
                    "adamw",
                    "lr_schedule",
                    "update_safety",
                    "policy_recompute",
                }
            ),
            path="training",
        )
        return cls(
            schema_version=raw["schema_version"],
            seed=raw["seed"],
            global_prompt_batch_size=raw["global_prompt_batch_size"],
            max_optimizer_steps=raw["max_optimizer_steps"],
            gradient_accumulation_steps=raw["gradient_accumulation_steps"],
            adamw=AdamWSpec.from_mapping(raw["adamw"]),
            lr_schedule=LearningRateScheduleSpec.from_mapping(raw["lr_schedule"]),
            update_safety=UpdateSafetySpec.from_mapping(raw["update_safety"]),
            policy_recompute=PolicyRecomputeSpec.from_mapping(raw["policy_recompute"]),
        )

    @classmethod
    def default(cls) -> TrainingSpec:
        return cls(
            seed=42,
            global_prompt_batch_size=1,
            max_optimizer_steps=1_000,
            gradient_accumulation_steps=1,
            adamw=AdamWSpec(
                learning_rate=1.0e-5,
                beta1=0.9,
                beta2=0.999,
                epsilon=1.0e-8,
                weight_decay=0.01,
                amsgrad=False,
            ),
            lr_schedule=LearningRateScheduleSpec(
                kind="cosine",
                warmup_steps=10,
                min_lr_ratio=0.0,
            ),
            update_safety=UpdateSafetySpec(
                require_finite_gradients=True,
                require_nonzero_gradients=True,
                max_grad_norm=1.0,
                max_initial_logprob_delta=1.0e-4,
                require_initial_clipfrac_zero=True,
                zero_grad_set_to_none=True,
                scaler_skip_policy="do_not_commit",
                post_optimizer_failure_policy="poison_and_restore",
            ),
            policy_recompute=PolicyRecomputeSpec(
                row_microbatch_size=None,
                transition_window_size=1,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "global_prompt_batch_size": self.global_prompt_batch_size,
            "max_optimizer_steps": self.max_optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "adamw": self.adamw.to_payload(),
            "lr_schedule": self.lr_schedule.to_payload(),
            "update_safety": self.update_safety.to_payload(),
            "policy_recompute": self.policy_recompute.to_payload(),
        }


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SpecValidationError(
            "expected a canonical lowercase identifier",
            path=path,
        )
    return value


def _resolved_path(value: object, *, config_dir: Path, path: str) -> Path:
    if not isinstance(config_dir, Path) or not config_dir.is_absolute():
        raise TypeError("config_dir must be an absolute Path")
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise SpecValidationError("expected a filesystem path", path=path)
    text = str(value)
    if not text.strip() or "\r" in text or "\n" in text or "\0" in text:
        raise SpecValidationError(
            "expected a non-empty single-line filesystem path",
            path=path,
        )
    try:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config_dir / candidate
        return Path(os.path.abspath(candidate))
    except (OSError, TypeError, ValueError) as exc:
        raise SpecValidationError("invalid filesystem path", path=path) from exc


def _location_mapping(
    value: object,
    *,
    config_dir: Path,
    path: str,
) -> tuple[tuple[str, Path], ...]:
    raw = _mapping(value, path=path)
    if not raw:
        raise SpecValidationError("expected a non-empty mapping", path=path)
    result = tuple(
        sorted(
            (
                _identifier(logical_id, path=f"{path}.{logical_id}"),
                _resolved_path(
                    location,
                    config_dir=config_dir,
                    path=f"{path}.{logical_id}",
                ),
            )
            for logical_id, location in raw.items()
        )
    )
    return result


@dataclass(frozen=True, slots=True)
class ArtifactLocations:
    """Location-only model/dataset/reward paths, excluded from recipe identity."""

    model: Path
    datasets: tuple[tuple[str, Path], ...]
    rewards: tuple[tuple[str, Path], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, Path) or not self.model.is_absolute():
            raise ValueError("model artifact location must be an absolute Path")
        for name in ("datasets", "rewards"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} artifact locations must be non-empty")
            keys: list[str] = []
            for logical_id, location in values:
                _identifier(logical_id, path=f"launch.artifacts.{name}")
                if not isinstance(location, Path) or not location.is_absolute():
                    raise ValueError(
                        f"{name} artifact location must be an absolute Path"
                    )
                keys.append(logical_id)
            if keys != sorted(set(keys)):
                raise ValueError(f"{name} artifact locations must be sorted and unique")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        config_dir: Path,
    ) -> ArtifactLocations:
        raw = _exact(
            value,
            frozenset({"model", "datasets", "rewards"}),
            path="launch.artifacts",
        )
        return cls(
            model=_resolved_path(
                raw["model"],
                config_dir=config_dir,
                path="launch.artifacts.model",
            ),
            datasets=_location_mapping(
                raw["datasets"],
                config_dir=config_dir,
                path="launch.artifacts.datasets",
            ),
            rewards=_location_mapping(
                raw["rewards"],
                config_dir=config_dir,
                path="launch.artifacts.rewards",
            ),
        )

    def dataset(self, logical_id: str) -> Path:
        return _lookup_location(self.datasets, logical_id, kind="dataset")

    def reward(self, logical_id: str) -> Path:
        return _lookup_location(self.rewards, logical_id, kind="reward")

    def to_payload(self) -> dict[str, object]:
        return {
            "model": str(self.model),
            "datasets": {key: str(value) for key, value in self.datasets},
            "rewards": {key: str(value) for key, value in self.rewards},
        }


@dataclass(frozen=True, slots=True)
class RewardRuntimeBindingSpec:
    """Launch-only connection facts for one physical reward artifact.

    Immutable reward weights/model/service revision remain in the content-
    addressed artifact.  Endpoint, trust, timeout, device, and worker-domain
    facts live here so changing a deployment location cannot change recipe_id.
    """

    artifact_ref: str
    execution_domain: str
    device: str
    dtype: str
    endpoint: str | None = None
    timeout_s: float | None = None
    trusted_hosts: tuple[str, ...] = ()
    ca_bundle: Path | None = None
    max_response_bytes: int | None = None

    def __post_init__(self) -> None:
        _identifier(
            self.artifact_ref,
            path="launch.reward_runtime_bindings.<key>",
        )
        domain = _literal(
            self.execution_domain,
            frozenset({"in_process", "remote"}),
            path=(
                f"launch.reward_runtime_bindings.{self.artifact_ref}.execution_domain"
            ),
        )
        device_path = f"launch.reward_runtime_bindings.{self.artifact_ref}.device"
        object.__setattr__(
            self,
            "device",
            _canonical_torch_device(self.device, path=device_path),
        )
        _literal(
            self.dtype,
            frozenset({"bf16", "fp16", "fp32"}),
            path=f"launch.reward_runtime_bindings.{self.artifact_ref}.dtype",
        )
        if domain == "in_process":
            if (
                self.endpoint is not None
                or self.timeout_s is not None
                or self.trusted_hosts
                or self.ca_bundle is not None
                or self.max_response_bytes is not None
            ):
                raise SpecValidationError(
                    "in_process bindings cannot contain remote connection fields",
                    path=f"launch.reward_runtime_bindings.{self.artifact_ref}",
                )
            return

        endpoint_path = f"launch.reward_runtime_bindings.{self.artifact_ref}.endpoint"
        if (
            not isinstance(self.endpoint, str)
            or not self.endpoint
            or self.endpoint.strip() != self.endpoint
            or "\r" in self.endpoint
            or "\n" in self.endpoint
        ):
            raise SpecValidationError(
                "remote endpoint must be a canonical non-empty string",
                path=endpoint_path,
            )
        if self.timeout_s is None:
            raise SpecValidationError(
                "remote timeout_s is required",
                path=(f"launch.reward_runtime_bindings.{self.artifact_ref}.timeout_s"),
            )
        object.__setattr__(
            self,
            "timeout_s",
            _positive_float(
                self.timeout_s,
                path=(f"launch.reward_runtime_bindings.{self.artifact_ref}.timeout_s"),
            ),
        )
        trusted_hosts_path = (
            f"launch.reward_runtime_bindings.{self.artifact_ref}.trusted_hosts"
        )
        object.__setattr__(
            self,
            "trusted_hosts",
            _trusted_hosts(self.trusted_hosts, path=trusted_hosts_path),
        )
        if self.ca_bundle is not None and (
            not isinstance(self.ca_bundle, Path) or not self.ca_bundle.is_absolute()
        ):
            raise SpecValidationError(
                "ca_bundle must be an absolute Path or null",
                path=(f"launch.reward_runtime_bindings.{self.artifact_ref}.ca_bundle"),
            )
        if type(self.max_response_bytes) is not int or self.max_response_bytes <= 0:
            raise SpecValidationError(
                "remote max_response_bytes must be a positive integer",
                path=(
                    "launch.reward_runtime_bindings."
                    f"{self.artifact_ref}.max_response_bytes"
                ),
            )

    @classmethod
    def from_mapping(
        cls,
        artifact_ref: str,
        value: object,
        *,
        config_dir: Path,
    ) -> RewardRuntimeBindingSpec:
        path = f"launch.reward_runtime_bindings.{artifact_ref}"
        raw = _mapping(value, path=path)
        domain = raw.get("execution_domain")
        if domain == "in_process":
            values = _exact(
                raw,
                frozenset({"execution_domain", "device", "dtype"}),
                path=path,
            )
            return cls(artifact_ref=artifact_ref, **values)
        if domain == "remote":
            values = _exact(
                raw,
                frozenset(
                    {
                        "execution_domain",
                        "device",
                        "dtype",
                        "endpoint",
                        "timeout_s",
                        "trusted_hosts",
                        "ca_bundle",
                        "max_response_bytes",
                    }
                ),
                path=path,
            )
            ca_bundle = values["ca_bundle"]
            return cls(
                artifact_ref=artifact_ref,
                execution_domain=values["execution_domain"],
                device=values["device"],
                dtype=values["dtype"],
                endpoint=values["endpoint"],
                timeout_s=values["timeout_s"],
                trusted_hosts=_trusted_hosts(
                    values["trusted_hosts"],
                    path=f"{path}.trusted_hosts",
                ),
                ca_bundle=(
                    None
                    if ca_bundle is None
                    else _resolved_path(
                        ca_bundle,
                        config_dir=config_dir,
                        path=f"{path}.ca_bundle",
                    )
                ),
                max_response_bytes=values["max_response_bytes"],
            )
        raise SpecValidationError(
            "expected execution_domain to be in_process or remote",
            path=f"{path}.execution_domain",
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "artifact_ref": self.artifact_ref,
            "execution_domain": self.execution_domain,
            "device": self.device,
            "dtype": self.dtype,
        }
        if self.execution_domain == "remote":
            payload.update(
                {
                    "endpoint": self.endpoint,
                    "timeout_s": self.timeout_s,
                    "trusted_hosts": list(self.trusted_hosts),
                    "ca_bundle": (
                        None if self.ca_bundle is None else str(self.ca_bundle)
                    ),
                    "max_response_bytes": self.max_response_bytes,
                }
            )
        return payload


def _lookup_location(
    values: tuple[tuple[str, Path], ...],
    logical_id: str,
    *,
    kind: str,
) -> Path:
    _identifier(logical_id, path=f"{kind}_id")
    for candidate, location in values:
        if candidate == logical_id:
            return location
    raise KeyError(f"unknown {kind} artifact location {logical_id!r}")


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Location-only execution input parsed relative to the YAML directory."""

    output_dir: Path
    resume_from: Path | None
    checkpoint_every_optimizer_steps: int
    artifacts: ArtifactLocations
    reward_runtime_bindings: tuple[tuple[str, RewardRuntimeBindingSpec], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path) or not self.output_dir.is_absolute():
            raise ValueError("output_dir must be an absolute Path")
        if self.resume_from is not None and (
            not isinstance(self.resume_from, Path) or not self.resume_from.is_absolute()
        ):
            raise ValueError("resume_from must be an absolute Path or None")
        _integer(
            self.checkpoint_every_optimizer_steps,
            path="launch.checkpoint_every_optimizer_steps",
            minimum=1,
        )
        if not isinstance(self.artifacts, ArtifactLocations):
            raise TypeError("artifacts must be ArtifactLocations")
        if type(self.reward_runtime_bindings) is not tuple:
            raise TypeError("reward_runtime_bindings must be a tuple")
        binding_refs: list[str] = []
        for item in self.reward_runtime_bindings:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(
                    "reward_runtime_bindings entries must be (artifact_ref, spec)"
                )
            artifact_ref, spec = item
            _identifier(
                artifact_ref,
                path="launch.reward_runtime_bindings.<key>",
            )
            if not isinstance(spec, RewardRuntimeBindingSpec):
                raise TypeError(
                    "reward_runtime_bindings values must be RewardRuntimeBindingSpec"
                )
            if spec.artifact_ref != artifact_ref:
                raise ValueError(
                    "reward runtime binding key differs from its artifact_ref"
                )
            binding_refs.append(artifact_ref)
        if binding_refs != sorted(set(binding_refs)):
            raise ValueError(
                "reward_runtime_bindings must use sorted unique artifact refs"
            )
        available_refs = {key for key, _location in self.artifacts.rewards}
        unknown_refs = sorted(set(binding_refs) - available_refs)
        if unknown_refs:
            raise SpecValidationError(
                f"bindings reference unknown reward artifacts: {unknown_refs}",
                path="launch.reward_runtime_bindings",
            )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: ResolutionContext,
    ) -> LaunchSpec:
        if not isinstance(context, ResolutionContext):
            raise TypeError("context must be a ResolutionContext")
        raw = _mapping(value, path="launch")
        required = {
            "output_dir",
            "resume_from",
            "checkpoint_every_optimizer_steps",
            "artifacts",
        }
        allowed = required | {"reward_runtime_bindings"}
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - allowed)
        if missing or unknown:
            error_path = "launch"
            if len(unknown) == 1:
                error_path = f"launch.{unknown[0]}"
            elif not unknown and len(missing) == 1:
                error_path = f"launch.{missing[0]}"
            raise SpecValidationError(
                f"invalid exact key set: missing={missing}, unknown={unknown}",
                path=error_path,
            )
        resume = raw["resume_from"]
        artifacts = ArtifactLocations.from_mapping(
            raw["artifacts"],
            config_dir=context.config_dir,
        )
        binding_values = raw.get("reward_runtime_bindings", {})
        binding_mapping = _mapping(
            binding_values,
            path="launch.reward_runtime_bindings",
        )
        bindings = tuple(
            sorted(
                (
                    _identifier(
                        artifact_ref,
                        path=(f"launch.reward_runtime_bindings.{artifact_ref}"),
                    ),
                    RewardRuntimeBindingSpec.from_mapping(
                        artifact_ref,
                        binding,
                        config_dir=context.config_dir,
                    ),
                )
                for artifact_ref, binding in binding_mapping.items()
            )
        )
        return cls(
            output_dir=_resolved_path(
                raw["output_dir"],
                config_dir=context.config_dir,
                path="launch.output_dir",
            ),
            resume_from=(
                None
                if resume is None
                else _resolved_path(
                    resume,
                    config_dir=context.config_dir,
                    path="launch.resume_from",
                )
            ),
            checkpoint_every_optimizer_steps=_integer(
                raw["checkpoint_every_optimizer_steps"],
                path="launch.checkpoint_every_optimizer_steps",
                minimum=1,
            ),
            artifacts=artifacts,
            reward_runtime_bindings=bindings,
        )

    def reward_runtime_binding(
        self,
        artifact_ref: str,
    ) -> RewardRuntimeBindingSpec | None:
        _identifier(artifact_ref, path="reward_artifact_ref")
        for candidate, binding in self.reward_runtime_bindings:
            if candidate == artifact_ref:
                return binding
        return None

    def to_payload(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "resume_from": (
                None if self.resume_from is None else str(self.resume_from)
            ),
            "checkpoint_every_optimizer_steps": (self.checkpoint_every_optimizer_steps),
            "artifacts": self.artifacts.to_payload(),
            "reward_runtime_bindings": {
                key: value.to_payload() for key, value in self.reward_runtime_bindings
            },
        }
