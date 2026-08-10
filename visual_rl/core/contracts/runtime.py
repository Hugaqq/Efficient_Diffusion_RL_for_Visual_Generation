"""Import-safe contracts at the algorithm-to-policy runtime boundary.

The values in this module deliberately keep tensor and implementation details
opaque.  Algorithms can therefore depend on this leaf contract without
importing a concrete model, scheduler, trainer, or numerical framework.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from visual_rl.core.contracts.algorithm import TrainingParadigm
from visual_rl.core.contracts.composition import BoundPolicyCapabilities
from visual_rl.core.identity import canonical_identity
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict
from visual_rl.errors import ExecutionTransformCompatibilityError

__all__ = (
    "AlgorithmStepResult",
    "CheckpointOwnerSnapshot",
    "CheckpointParticipant",
    "CheckpointRestorePhase",
    "CheckpointScope",
    "CheckpointSnapshotBundle",
    "ExecutionPolicyReceipt",
    "ExecutionPolicyView",
    "ExecutionTransformContract",
    "ExecutionTransformPlan",
    "ExecutionTransformSafety",
    "ExecutionTransformStage",
    "PolicyRuntimePort",
    "PolicyTransitionRequest",
    "PolicyTransitionResult",
    "RolloutExecutionPolicy",
    "TrainingParadigm",
)

_CLASS_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_CHECKPOINT_OWNER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class ExecutionTransformStage(str, Enum):
    """The runtime stages at which a transform is actually active."""

    BOTH = "both"
    ROLLOUT = "rollout"
    TRAIN = "train"


class ExecutionTransformSafety(str, Enum):
    """Whether a transform preserves exact numerical execution semantics."""

    LOSSLESS = "lossless"
    LOSSY = "lossy"


class CheckpointScope(str, Enum):
    GLOBAL = "global"
    RANK_LOCAL = "rank_local"


class CheckpointRestorePhase(str, Enum):
    PREPARED = "prepared"
    BOUND = "bound"
    RNG_FINAL = "rng_final"


@dataclass(frozen=True, slots=True)
class CheckpointOwnerSnapshot:
    """Immutable envelope around one owner-produced detached typed payload.

    The envelope does not claim that an opaque payload is deeply immutable.
    Each participant must own a frozen/detached payload type and prove its
    alias-safety in domain tests before M4 persistence is considered complete.
    """

    owner_id: str
    schema_version: int
    scope: CheckpointScope
    restore_phase: CheckpointRestorePhase
    restore_after: tuple[str, ...]
    payload: object

    def __post_init__(self) -> None:
        _checkpoint_owner_id(self.owner_id)
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("checkpoint schema_version must be positive")
        if not isinstance(self.scope, CheckpointScope):
            raise TypeError("checkpoint scope must be a CheckpointScope")
        if not isinstance(self.restore_phase, CheckpointRestorePhase):
            raise TypeError("restore_phase must be a CheckpointRestorePhase")
        if type(self.restore_after) is not tuple:
            raise TypeError("restore_after must be a tuple")
        for dependency in self.restore_after:
            _checkpoint_owner_id(dependency)
        if len(self.restore_after) != len(set(self.restore_after)):
            raise ValueError("restore_after owners must be unique")
        if self.restore_after != tuple(sorted(self.restore_after)):
            raise ValueError("restore_after owners must be sorted")
        if self.owner_id in self.restore_after:
            raise ValueError("checkpoint owner cannot depend on itself")
        if self.payload is None or callable(self.payload):
            raise TypeError("checkpoint payload must be detached owner state")

    @property
    def metadata_identity(self) -> str:
        return canonical_identity(
            "checkpoint-owner-metadata.v1",
            {
                "owner_id": self.owner_id,
                "schema_version": self.schema_version,
                "scope": self.scope.value,
                "restore_phase": self.restore_phase.value,
                "restore_after": self.restore_after,
            },
        )


@runtime_checkable
class CheckpointParticipant(Protocol):
    """Domain-owned capture/validate/restore port called by runtime."""

    @property
    def checkpoint_owner_id(self) -> str: ...

    @property
    def checkpoint_schema_version(self) -> int: ...

    @property
    def checkpoint_scope(self) -> CheckpointScope: ...

    @property
    def checkpoint_restore_phase(self) -> CheckpointRestorePhase: ...

    @property
    def checkpoint_restore_after(self) -> tuple[str, ...]: ...

    def capture_checkpoint_snapshot(self) -> CheckpointOwnerSnapshot: ...

    def validate_checkpoint_snapshot(
        self,
        snapshot: CheckpointOwnerSnapshot,
    ) -> None: ...

    def restore_checkpoint_snapshot(
        self,
        snapshot: CheckpointOwnerSnapshot,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckpointSnapshotBundle:
    """One deterministic owner set with a validated restore dependency DAG."""

    snapshots: tuple[CheckpointOwnerSnapshot, ...]

    def __post_init__(self) -> None:
        if type(self.snapshots) is not tuple or not self.snapshots:
            raise ValueError("checkpoint snapshots must be a non-empty tuple")
        if any(
            not isinstance(item, CheckpointOwnerSnapshot) for item in self.snapshots
        ):
            raise TypeError("snapshots must contain CheckpointOwnerSnapshot values")
        by_owner = {item.owner_id: item for item in self.snapshots}
        if len(by_owner) != len(self.snapshots):
            raise ValueError("checkpoint snapshot owner ids must be unique")
        phase_order = {
            phase: index for index, phase in enumerate(CheckpointRestorePhase)
        }
        for snapshot in self.snapshots:
            unknown = tuple(
                dependency
                for dependency in snapshot.restore_after
                if dependency not in by_owner
            )
            if unknown:
                raise ValueError(
                    f"checkpoint owner {snapshot.owner_id!r} has unknown "
                    f"dependencies {list(unknown)}"
                )
            later = tuple(
                dependency
                for dependency in snapshot.restore_after
                if phase_order[by_owner[dependency].restore_phase]
                > phase_order[snapshot.restore_phase]
            )
            if later:
                raise ValueError(
                    f"checkpoint owner {snapshot.owner_id!r} depends on later "
                    f"restore phase owners {list(later)}"
                )
        order = _checkpoint_restore_order(by_owner)
        object.__setattr__(
            self,
            "snapshots",
            tuple(by_owner[owner_id] for owner_id in order),
        )

    @property
    def restore_order(self) -> tuple[str, ...]:
        return tuple(item.owner_id for item in self.snapshots)


@dataclass(frozen=True, slots=True)
class ExecutionTransformContract:
    """One immutable transform declaration, without importing its class."""

    transform_id: str
    class_path: str
    config: FrozenMapping
    stage: ExecutionTransformStage
    safety: ExecutionTransformSafety
    supported_training_paradigms: tuple[TrainingParadigm, ...]
    preserves_parameter_identity: bool
    preserves_state_dict_keys: bool
    deterministic: bool
    _contract_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _canonical_identifier(self.transform_id, field_name="transform_id")
        _class_path(self.class_path)
        if not isinstance(self.config, FrozenMapping):
            raise TypeError("config must be a FrozenMapping")
        if not isinstance(self.stage, ExecutionTransformStage):
            raise TypeError("stage must be an ExecutionTransformStage")
        if not isinstance(self.safety, ExecutionTransformSafety):
            raise TypeError("safety must be an ExecutionTransformSafety")
        paradigms = self.supported_training_paradigms
        if type(paradigms) is not tuple or not paradigms:
            raise ValueError("supported_training_paradigms must be a non-empty tuple")
        if any(not isinstance(item, TrainingParadigm) for item in paradigms):
            raise TypeError(
                "supported_training_paradigms must contain TrainingParadigm values"
            )
        if len(set(paradigms)) != len(paradigms):
            raise ValueError("supported_training_paradigms must be unique")
        object.__setattr__(
            self,
            "supported_training_paradigms",
            tuple(sorted(paradigms, key=lambda item: item.value)),
        )
        for field_name in (
            "preserves_parameter_identity",
            "preserves_state_dict_keys",
            "deterministic",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool")
        digest = hashlib.sha256(_canonical_json(self.to_payload())).hexdigest()
        object.__setattr__(self, "_contract_id", digest)

    @property
    def contract_id(self) -> str:
        return self._contract_id

    def supports(self, paradigm: TrainingParadigm) -> bool:
        if not isinstance(paradigm, TrainingParadigm):
            raise TypeError("paradigm must be a TrainingParadigm")
        return paradigm in self.supported_training_paradigms

    def to_payload(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "class_path": self.class_path,
            "config": to_plain_dict(self.config),
            "stage": self.stage.value,
            "safety": self.safety.value,
            "supported_training_paradigms": [
                item.value for item in self.supported_training_paradigms
            ],
            "preserves_parameter_identity": self.preserves_parameter_identity,
            "preserves_state_dict_keys": self.preserves_state_dict_keys,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> ExecutionTransformContract:
        required = frozenset(
            {
                "transform_id",
                "class_path",
                "config",
                "stage",
                "safety",
                "supported_training_paradigms",
                "preserves_parameter_identity",
                "preserves_state_dict_keys",
                "deterministic",
            }
        )
        _exact_keys(values, required=required, context="execution transform")
        raw_config = values["config"]
        if not isinstance(raw_config, Mapping):
            raise TypeError("execution transform config must be a mapping")
        raw_paradigms = values["supported_training_paradigms"]
        if isinstance(raw_paradigms, (str, bytes)) or not isinstance(
            raw_paradigms,
            Sequence,
        ):
            raise TypeError("supported_training_paradigms must be a sequence")
        paradigms = tuple(
            _enum_value(
                item,
                TrainingParadigm,
                field_name="supported_training_paradigms item",
            )
            for item in raw_paradigms
        )
        return cls(
            transform_id=values["transform_id"],
            class_path=values["class_path"],
            config=FrozenMapping(raw_config),
            stage=_enum_value(
                values["stage"],
                ExecutionTransformStage,
                field_name="stage",
            ),  # type: ignore[arg-type]
            safety=_enum_value(
                values["safety"],
                ExecutionTransformSafety,
                field_name="safety",
            ),  # type: ignore[arg-type]
            supported_training_paradigms=paradigms,  # type: ignore[arg-type]
            preserves_parameter_identity=values["preserves_parameter_identity"],
            preserves_state_dict_keys=values["preserves_state_dict_keys"],
            deterministic=values["deterministic"],
        )


@dataclass(frozen=True, slots=True)
class ExecutionTransformPlan:
    """An ordered transform chain validated for one training paradigm."""

    paradigm: TrainingParadigm
    transforms: tuple[ExecutionTransformContract, ...]
    _plan_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.paradigm, TrainingParadigm):
            raise TypeError("paradigm must be a TrainingParadigm")
        if type(self.transforms) is not tuple:
            raise TypeError("transforms must be a tuple")
        if any(
            not isinstance(item, ExecutionTransformContract) for item in self.transforms
        ):
            raise TypeError("transforms must contain ExecutionTransformContract values")
        transform_ids = tuple(item.transform_id for item in self.transforms)
        if len(set(transform_ids)) != len(transform_ids):
            raise ValueError("execution transform ids must be unique within a plan")
        self.validate_static()
        digest = hashlib.sha256(_canonical_json(self.to_payload())).hexdigest()
        object.__setattr__(self, "_plan_id", digest)

    @property
    def plan_id(self) -> str:
        return self._plan_id

    @property
    def transform_ids(self) -> tuple[str, ...]:
        return tuple(item.transform_id for item in self.transforms)

    def validate_static(self) -> None:
        for transform in self.transforms:
            if not transform.supports(self.paradigm):
                raise ExecutionTransformCompatibilityError(
                    f"execution transform {transform.transform_id!r} does not "
                    f"support training paradigm {self.paradigm.value!r}"
                )
            if (
                self.paradigm is TrainingParadigm.COUPLED
                and transform.stage is ExecutionTransformStage.ROLLOUT
                and transform.safety is ExecutionTransformSafety.LOSSY
            ):
                raise ExecutionTransformCompatibilityError(
                    "coupled training forbids lossy rollout-only execution "
                    f"transform {transform.transform_id!r}"
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "paradigm": self.paradigm.value,
            "transforms": [item.to_payload() for item in self.transforms],
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ExecutionTransformPlan:
        required = frozenset({"schema_version", "paradigm", "transforms"})
        _exact_keys(values, required=required, context="execution transform plan")
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("execution transform plan schema_version must be 1")
        raw_transforms = values["transforms"]
        if isinstance(raw_transforms, (str, bytes)) or not isinstance(
            raw_transforms,
            Sequence,
        ):
            raise TypeError("execution transform plan transforms must be a sequence")
        return cls(
            paradigm=_enum_value(
                values["paradigm"],
                TrainingParadigm,
                field_name="paradigm",
            ),  # type: ignore[arg-type]
            transforms=tuple(
                ExecutionTransformContract.from_mapping(item) for item in raw_transforms
            ),
        )


@dataclass(frozen=True, slots=True)
class RolloutExecutionPolicy:
    """Validated physical rollout projection owned by an execution receipt."""

    forward_microbatch_size: int | None
    decode_microbatch_size: int | None
    trajectory_storage_device: Literal["cpu", "model"]

    def __post_init__(self) -> None:
        for name in ("forward_microbatch_size", "decode_microbatch_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.trajectory_storage_device not in {"cpu", "model"}:
            raise ValueError("trajectory_storage_device must be cpu or model")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RolloutExecutionPolicy:
        if not isinstance(payload, Mapping):
            raise TypeError("execution policy rollout payload must be a mapping")
        expected = {
            "forward_microbatch_size",
            "decode_microbatch_size",
            "trajectory_storage_device",
        }
        if set(payload) != expected:
            raise ValueError(
                "execution policy rollout payload has an invalid exact key set"
            )
        return cls(
            forward_microbatch_size=payload["forward_microbatch_size"],
            decode_microbatch_size=payload["decode_microbatch_size"],
            trajectory_storage_device=payload["trajectory_storage_device"],
        )  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, object]:
        return {
            "forward_microbatch_size": self.forward_microbatch_size,
            "decode_microbatch_size": self.decode_microbatch_size,
            "trajectory_storage_device": self.trajectory_storage_device,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicyReceipt:
    """Canonical, core-owned receipt over a composition-owned policy payload.

    The receipt deliberately does not duplicate ``ExecutionPolicySpec`` fields.
    It hashes the complete canonical payload and derives only the narrow
    projections required by algorithms and rollout execution.  Consumers call
    :meth:`validated_projection` before using any projection, so an arbitrary
    structural object cannot self-report a trusted policy identity.
    """

    canonical_payload: FrozenMapping
    _policy_id: str = field(init=False, repr=False, compare=False)
    _group_size: int = field(init=False, repr=False, compare=False)
    _transform_plan: ExecutionTransformPlan = field(
        init=False,
        repr=False,
        compare=False,
    )
    _rollout: RolloutExecutionPolicy = field(
        init=False,
        repr=False,
        compare=False,
    )
    _projection_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_payload, FrozenMapping):
            raise TypeError("canonical_payload must be a FrozenMapping")
        if self.canonical_payload.get("schema_version") != 1:
            raise ValueError("execution policy payload schema_version must be 1")
        group_size = self.canonical_payload.get("group_size")
        if type(group_size) is not int or group_size < 1:
            raise ValueError("execution policy payload group_size must be positive")
        raw_transform_plan = self.canonical_payload.get("transform_plan")
        if not isinstance(raw_transform_plan, Mapping):
            raise TypeError("execution policy payload transform_plan must be a mapping")
        transform_plan = ExecutionTransformPlan.from_mapping(raw_transform_plan)
        if to_plain_dict(raw_transform_plan) != transform_plan.to_payload():
            raise ValueError("execution policy transform_plan payload is not canonical")
        raw_rollout = self.canonical_payload.get("rollout")
        if not isinstance(raw_rollout, Mapping):
            raise TypeError("execution policy payload rollout must be a mapping")
        rollout = RolloutExecutionPolicy.from_payload(raw_rollout)
        if to_plain_dict(raw_rollout) != rollout.to_payload():
            raise ValueError("execution policy rollout payload is not canonical")
        policy_id = canonical_identity(
            "execution-policy.v1",
            to_plain_dict(self.canonical_payload),
        )
        projection_id = canonical_identity(
            "execution-policy-projection.v1",
            {
                "policy_id": policy_id,
                "group_size": group_size,
                "transform_plan_id": transform_plan.plan_id,
            },
        )
        object.__setattr__(self, "_policy_id", policy_id)
        object.__setattr__(self, "_group_size", group_size)
        object.__setattr__(self, "_transform_plan", transform_plan)
        object.__setattr__(self, "_rollout", rollout)
        object.__setattr__(self, "_projection_id", projection_id)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExecutionPolicyReceipt:
        if not isinstance(payload, Mapping):
            raise TypeError("execution policy payload must be a mapping")
        return cls(canonical_payload=FrozenMapping(payload))

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def group_size(self) -> int:
        return self._group_size

    @property
    def transform_plan(self) -> ExecutionTransformPlan:
        return self._transform_plan

    @property
    def rollout(self) -> RolloutExecutionPolicy:
        return self._rollout

    @property
    def projection_id(self) -> str:
        return self._projection_id

    def validated_projection(
        self,
        expected_policy_id: str,
    ) -> ExecutionPolicyReceipt:
        """Rebuild from the full payload and bind it to the expected identity."""

        if not isinstance(expected_policy_id, str) or not expected_policy_id:
            raise ValueError("expected_policy_id must be non-empty")
        rebuilt = ExecutionPolicyReceipt.from_payload(self.canonical_payload)
        if (
            self.policy_id != rebuilt.policy_id
            or self.group_size != rebuilt.group_size
            or self.transform_plan != rebuilt.transform_plan
            or self.rollout != rebuilt.rollout
            or self.projection_id != rebuilt.projection_id
        ):
            raise ValueError("execution policy receipt projection is not canonical")
        if rebuilt.policy_id != expected_policy_id:
            raise ValueError("execution policy identity differs from expected policy")
        return rebuilt

    def to_payload(self) -> dict[str, Any]:
        return to_plain_dict(self.canonical_payload)


# Compatibility import name; this is intentionally a nominal frozen class now,
# not the former runtime-checkable structural protocol.
ExecutionPolicyView = ExecutionPolicyReceipt


@dataclass(frozen=True, slots=True)
class PolicyTransitionRequest:
    """Opaque, typed request for sampling or replaying one policy action."""

    mode: Literal["sample", "evaluate"]
    transition_session: object
    transition_input: object
    generator: object | None = None
    action_latent: object | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"sample", "evaluate"}:
            raise ValueError("transition mode must be sample or evaluate")
        if self.transition_session is None:
            raise ValueError("transition_session must not be None")
        if self.transition_input is None:
            raise ValueError("transition_input must not be None")
        if self.mode == "sample" and self.action_latent is not None:
            raise ValueError("a sample request cannot provide action_latent")
        if self.mode == "evaluate" and self.action_latent is None:
            raise ValueError("an evaluate request requires action_latent")
        if self.mode == "evaluate" and self.generator is not None:
            raise ValueError("an evaluate request cannot consume a generator")
        _validate_metadata(self.metadata)


@dataclass(frozen=True, slots=True)
class PolicyTransitionResult:
    """Model-independent result returned by ``PolicyRuntimePort.transition``."""

    next_latents: object
    log_prob: object | None
    transition_output: object
    policy_metadata: object | None = None
    replay_state: object | None = None

    def __post_init__(self) -> None:
        if self.next_latents is None:
            raise ValueError("next_latents must not be None")
        if self.transition_output is None:
            raise ValueError("transition_output must not be None")


@dataclass(frozen=True, slots=True)
class AlgorithmStepResult:
    """One coarse algorithm iteration wrapping the internal trainer result."""

    optimizer_step: int
    iteration: object
    algorithm_binding_id: str

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if self.iteration is None:
            raise ValueError("iteration must not be None")
        if (
            not isinstance(self.algorithm_binding_id, str)
            or not self.algorithm_binding_id
        ):
            raise ValueError("algorithm_binding_id must be non-empty")
        observed = getattr(self.iteration, "optimizer_step", self.optimizer_step)
        if observed != self.optimizer_step:
            raise ValueError("iteration optimizer_step differs from the facade result")


@runtime_checkable
class PolicyRuntimePort(Protocol):
    """The sole model-facing surface available to an algorithm module."""

    @property
    def capabilities(self) -> BoundPolicyCapabilities: ...

    @property
    def trainable_parameters(self) -> tuple[object, ...]: ...

    @property
    def prepared_forward_handle(self) -> object: ...

    @property
    def state_contract(self) -> object: ...

    @property
    def runtime_capabilities(self) -> object: ...

    def preprocess(self, raw_batch: object) -> object: ...

    def encode(self, batch: object) -> object: ...

    def initialize_latents(self, batch_geometry: object, rng: object) -> object: ...

    def prepare_latents(
        self,
        latent_spec: object,
        *,
        generator: object,
    ) -> object: ...

    def latent_spec_for_batch(
        self,
        batch: object,
        *,
        device: object,
        dtype: object,
    ) -> object: ...

    def model_schedule_context(self, latent_spec: object) -> object: ...

    def predict(
        self,
        model_input: object,
        parameter_view: object | None = None,
    ) -> object: ...

    def predict_reference(self, model_input: object) -> object: ...

    def transition(
        self,
        request: PolicyTransitionRequest,
    ) -> PolicyTransitionResult: ...

    def decode(self, latents: object, latent_spec: object) -> object: ...


def _validate_metadata(value: tuple[tuple[str, str], ...]) -> None:
    if type(value) is not tuple:
        raise TypeError("metadata must be a tuple")
    if any(
        type(item) is not tuple
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        or not isinstance(item[1], str)
        for item in value
    ):
        raise ValueError("metadata entries must be non-empty string pairs")
    keys = tuple(key for key, _item in value)
    if len(keys) != len(set(keys)):
        raise ValueError("metadata keys must be unique")


def _canonical_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value.strip() != value or "\r" in value or "\n" in value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _class_path(value: Any) -> str:
    if not isinstance(value, str) or not _CLASS_PATH.fullmatch(value):
        raise ValueError("class_path must use explicit module:Class syntax")
    return value


def _exact_keys(
    values: Mapping[str, Any],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in values):
        raise TypeError(f"{context} keys must be strings")
    keys = set(values)
    missing = sorted(required.difference(keys))
    unknown = sorted(keys.difference(required))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        raise ValueError(f"{context} has " + " and ".join(details))


def _enum_value(value: Any, enum_type: type[Enum], *, field_name: str) -> Enum:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = tuple(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of {allowed}") from exc


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _checkpoint_owner_id(value: object) -> str:
    if not isinstance(value, str) or not _CHECKPOINT_OWNER.fullmatch(value):
        raise ValueError("checkpoint owner_id must be a canonical dotted identifier")
    return value


def _checkpoint_restore_order(
    by_owner: Mapping[str, CheckpointOwnerSnapshot],
) -> tuple[str, ...]:
    phase_order = {phase: index for index, phase in enumerate(CheckpointRestorePhase)}
    pending: set[str] = set()
    complete: set[str] = set()
    result: list[str] = []

    def visit(owner_id: str) -> None:
        if owner_id in complete:
            return
        if owner_id in pending:
            raise ValueError("checkpoint restore dependencies contain a cycle")
        pending.add(owner_id)
        for dependency in by_owner[owner_id].restore_after:
            visit(dependency)
        pending.remove(owner_id)
        complete.add(owner_id)
        result.append(owner_id)

    for owner_id in sorted(
        by_owner,
        key=lambda item: (
            phase_order[by_owner[item].restore_phase],
            item,
        ),
    ):
        visit(owner_id)
    return tuple(result)
