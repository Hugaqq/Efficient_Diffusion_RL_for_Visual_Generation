"""Content-addressed preprocessing contracts owned by the data plane.

The preprocessing identity is deliberately independent of recipes and runtime
row identities.  A cache entry is reusable only when the immutable model
components, geometry, ordered input transforms, and preprocessing config are
byte-for-byte equivalent after canonical serialization.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.data.samples import SourceItemContext

__all__ = (
    "PreprocessBarrier",
    "PreprocessCacheReader",
    "PreprocessCacheWriter",
    "PreprocessCompatibilityReceipt",
    "PreprocessComponentIdentity",
    "PreprocessComponentRole",
    "PreprocessConsumerRequirement",
    "PreprocessContractError",
    "PreprocessDependency",
    "PreprocessGeometry",
    "PreprocessManifest",
    "PreprocessPlan",
    "PreprocessPortContract",
    "PreprocessProducerSpec",
    "PreprocessRequirementProvider",
    "PreprocessRequirementSet",
    "PreprocessTransform",
    "PreprocessWriteLease",
    "PreprocessedItem",
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "accumulation_index",
        "algorithm",
        "algorithm_id",
        "beta",
        "conditioner",
        "conditioner_id",
        "gradient_checkpointing",
        "group_id",
        "group_size",
        "local_rank",
        "lora_alpha",
        "lora_rank",
        "lora_target_modules",
        "member_id",
        "occurrence_id",
        "optimizer",
        "optimizer_id",
        "optimizer_step",
        "phase",
        "rank",
        "reward",
        "reward_id",
        "rollout",
        "rollout_id",
        "sample_id",
        "trainer",
        "trainer_id",
        "training",
        "training_params",
        "trajectory_id",
        "world_size",
    }
)


class PreprocessContractError(ValueError):
    """Raised when a preprocessing identity or cache record is unsafe."""


class PreprocessComponentRole(str, Enum):
    """Stable roles which may affect encoded model inputs."""

    MODEL = "model"
    TOKENIZER = "tokenizer"
    TEXT_ENCODER = "text_encoder"
    VAE = "vae"
    IMAGE_ENCODER = "image_encoder"
    IMAGE_PROCESSOR = "image_processor"
    VIDEO_PROCESSOR = "video_processor"


class PreprocessRequirementProvider(str, Enum):
    """Owner which must produce fields consumed by one preprocess client."""

    MODEL = "model_preprocess"
    CONDITIONER = "conditioner"


def _canonical_text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\r" in value
        or "\n" in value
    ):
        raise PreprocessContractError(
            f"{field_name} must be a canonical non-empty string"
        )
    return value


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PreprocessContractError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise PreprocessContractError(f"{field_name} must be a positive integer")
    return value


def _canonical_string_tuple(
    value: object,
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    if not allow_empty and not value:
        raise PreprocessContractError(f"{field_name} must not be empty")
    for item in value:
        _canonical_text(item, field_name=f"{field_name} entry")
    canonical = tuple(sorted(value))
    if len(canonical) != len(set(canonical)):
        raise PreprocessContractError(f"{field_name} must contain unique values")
    return canonical


def _frozen_config(
    value: object,
    *,
    field_name: str,
    allowed_identity_fields: frozenset[str] = frozenset(),
) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        result = value if isinstance(value, FrozenMapping) else FrozenMapping(value)
    except (TypeError, ValueError) as exc:
        raise PreprocessContractError(f"invalid {field_name}: {exc}") from exc
    _reject_runtime_identity_fields(
        result,
        location=field_name,
        allowed_identity_fields=allowed_identity_fields,
    )
    # Fail here, rather than during hashing, if a nominally frozen value cannot
    # cross the durable JSON boundary.
    try:
        canonical_json_text(result)
    except (TypeError, ValueError) as exc:
        raise PreprocessContractError(f"invalid {field_name}: {exc}") from exc
    return result


def _reject_runtime_identity_fields(
    value: object,
    *,
    location: str,
    allowed_identity_fields: frozenset[str] = frozenset(),
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _RUNTIME_IDENTITY_FIELDS and key not in allowed_identity_fields:
                raise PreprocessContractError(
                    f"{location}.{key} is a runtime identity or compatibility-only "
                    "field and cannot enter the preprocessing cache identity"
                )
            _reject_runtime_identity_fields(
                item,
                location=f"{location}.{key}",
                allowed_identity_fields=allowed_identity_fields,
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_runtime_identity_fields(
                item,
                location=f"{location}[{index}]",
                allowed_identity_fields=allowed_identity_fields,
            )


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreprocessComponentIdentity:
    """Immutable identity of one model-side preprocessing dependency."""

    role: PreprocessComponentRole
    logical_name: str
    implementation_id: str
    revision: str
    content_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        try:
            role = PreprocessComponentRole(self.role)
        except (TypeError, ValueError):
            raise PreprocessContractError(
                f"unknown preprocessing component role: {self.role!r}"
            ) from None
        object.__setattr__(self, "role", role)
        for name in ("logical_name", "implementation_id", "revision"):
            _canonical_text(getattr(self, name), field_name=name)
        _sha256(self.content_sha256, field_name="content_sha256")
        _sha256(self.config_sha256, field_name="config_sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "logical_name": self.logical_name,
            "implementation_id": self.implementation_id,
            "revision": self.revision,
            "content_sha256": self.content_sha256,
            "config_sha256": self.config_sha256,
        }

    @property
    def dependency(self) -> PreprocessDependency:
        return PreprocessDependency(role=self.role, logical_name=self.logical_name)


@dataclass(frozen=True, slots=True)
class PreprocessDependency:
    """One exact dependency slot declared by a model preprocessing port."""

    role: PreprocessComponentRole
    logical_name: str

    def __post_init__(self) -> None:
        try:
            role = PreprocessComponentRole(self.role)
        except (TypeError, ValueError):
            raise PreprocessContractError(
                f"unknown preprocessing dependency role: {self.role!r}"
            ) from None
        object.__setattr__(self, "role", role)
        _canonical_text(self.logical_name, field_name="logical_name")

    @property
    def key(self) -> tuple[str, str]:
        return (self.role.value, self.logical_name)

    def to_payload(self) -> dict[str, object]:
        return {"role": self.role.value, "logical_name": self.logical_name}


@dataclass(frozen=True, slots=True)
class PreprocessPortContract:
    """Model-declared exact dependency surface for one preprocessing port."""

    port_id: str
    output_payload_type: str
    dependencies: tuple[PreprocessDependency, ...]
    producer_output_fields: tuple[str, ...] = ()
    negative_condition_fields: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _canonical_text(self.port_id, field_name="port_id")
        _canonical_text(self.output_payload_type, field_name="output_payload_type")
        if self.schema_version not in {1, 2} or type(self.schema_version) is not int:
            raise PreprocessContractError("unsupported preprocess port schema_version")
        if type(self.dependencies) is not tuple or not self.dependencies:
            raise PreprocessContractError("port dependencies must be a non-empty tuple")
        if any(
            not isinstance(item, PreprocessDependency) for item in self.dependencies
        ):
            raise TypeError(
                "port dependencies must contain PreprocessDependency values"
            )
        dependencies = tuple(sorted(self.dependencies, key=lambda item: item.key))
        keys = tuple(item.key for item in dependencies)
        if len(keys) != len(set(keys)):
            raise PreprocessContractError("port dependency keys must be unique")
        object.__setattr__(self, "dependencies", dependencies)
        output_fields = _canonical_string_tuple(
            self.producer_output_fields,
            field_name="producer_output_fields",
            allow_empty=self.schema_version == 1,
        )
        negative_fields = _canonical_string_tuple(
            self.negative_condition_fields,
            field_name="negative_condition_fields",
            allow_empty=True,
        )
        if self.schema_version == 1 and (output_fields or negative_fields):
            raise PreprocessContractError(
                "schema-v1 preprocess ports cannot declare producer output fields"
            )
        if not set(negative_fields).issubset(output_fields):
            raise PreprocessContractError(
                "negative_condition_fields must be a subset of producer_output_fields"
            )
        object.__setattr__(self, "producer_output_fields", output_fields)
        object.__setattr__(self, "negative_condition_fields", negative_fields)

    @property
    def dependency_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(item.key for item in self.dependencies)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "port_id": self.port_id,
            "output_payload_type": self.output_payload_type,
            "dependencies": [item.to_payload() for item in self.dependencies],
            "producer_output_fields": list(self.producer_output_fields),
            "negative_condition_fields": list(self.negative_condition_fields),
        }

    @property
    def contract_id(self) -> str:
        return _digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class PreprocessConsumerRequirement:
    """One independently identified consumer and its exact field demand."""

    consumer_identity: str
    provider: PreprocessRequirementProvider
    payload_type: str
    required_modalities: tuple[str, ...]
    required_output_fields: tuple[str, ...]
    required_negative_condition_fields: tuple[str, ...] = ()
    requires_negative_condition: bool = False

    def __post_init__(self) -> None:
        _canonical_text(self.consumer_identity, field_name="consumer_identity")
        try:
            provider = PreprocessRequirementProvider(self.provider)
        except (TypeError, ValueError):
            raise PreprocessContractError(
                f"unknown preprocess requirement provider: {self.provider!r}"
            ) from None
        object.__setattr__(self, "provider", provider)
        _canonical_text(self.payload_type, field_name="payload_type")
        modalities = _canonical_string_tuple(
            self.required_modalities,
            field_name="required_modalities",
            allow_empty=False,
        )
        fields = _canonical_string_tuple(
            self.required_output_fields,
            field_name="required_output_fields",
            allow_empty=False,
        )
        negative_fields = _canonical_string_tuple(
            self.required_negative_condition_fields,
            field_name="required_negative_condition_fields",
            allow_empty=True,
        )
        if not set(negative_fields).issubset(fields):
            raise PreprocessContractError(
                "required_negative_condition_fields must be a subset of "
                "required_output_fields"
            )
        if type(self.requires_negative_condition) is not bool:
            raise TypeError("requires_negative_condition must be bool")
        if self.requires_negative_condition != bool(negative_fields):
            raise PreprocessContractError(
                "requires_negative_condition must exactly match whether negative "
                "condition fields are required"
            )
        if (
            provider is PreprocessRequirementProvider.CONDITIONER
            and self.requires_negative_condition
        ):
            raise PreprocessContractError(
                "Conditioner-provided requirements cannot request model negative condition"
            )
        object.__setattr__(self, "required_modalities", modalities)
        object.__setattr__(self, "required_output_fields", fields)
        object.__setattr__(
            self,
            "required_negative_condition_fields",
            negative_fields,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "consumer_identity": self.consumer_identity,
            "provider": self.provider.value,
            "payload_type": self.payload_type,
            "required_modalities": list(self.required_modalities),
            "required_output_fields": list(self.required_output_fields),
            "required_negative_condition_fields": list(
                self.required_negative_condition_fields
            ),
            "requires_negative_condition": self.requires_negative_condition,
        }


@dataclass(frozen=True, slots=True)
class PreprocessRequirementSet:
    """Canonical union of model, algorithm, and Conditioner consumers."""

    requirements: tuple[PreprocessConsumerRequirement, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise PreprocessContractError(
                "unsupported preprocess requirement-set schema_version"
            )
        if type(self.requirements) is not tuple or not self.requirements:
            raise PreprocessContractError("requirements must be a non-empty tuple")
        if any(
            not isinstance(item, PreprocessConsumerRequirement)
            for item in self.requirements
        ):
            raise TypeError(
                "requirements must contain PreprocessConsumerRequirement values"
            )
        canonical = tuple(
            sorted(
                self.requirements,
                key=lambda item: (item.provider.value, item.consumer_identity),
            )
        )
        identities = tuple(item.consumer_identity for item in canonical)
        if len(identities) != len(set(identities)):
            raise PreprocessContractError(
                "preprocess consumer identities must be globally unique"
            )
        model_payload_types = {
            item.payload_type
            for item in canonical
            if item.provider is PreprocessRequirementProvider.MODEL
        }
        if len(model_payload_types) != 1:
            raise PreprocessContractError(
                "model preprocess consumers must agree on one payload type"
            )
        object.__setattr__(self, "requirements", canonical)

    @property
    def required_modalities(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    modality
                    for item in self.requirements
                    for modality in item.required_modalities
                }
            )
        )

    @property
    def required_output_fields(self) -> tuple[str, ...]:
        """Fields the model preprocess producer, specifically, must emit."""

        return tuple(
            sorted(
                {
                    field
                    for item in self.requirements
                    if item.provider is PreprocessRequirementProvider.MODEL
                    for field in item.required_output_fields
                }
            )
        )

    @property
    def conditioner_output_fields(self) -> tuple[str, ...]:
        """Fields explicitly owned by Conditioner, never by model preprocess."""

        return tuple(
            sorted(
                {
                    field
                    for item in self.requirements
                    if item.provider is PreprocessRequirementProvider.CONDITIONER
                    for field in item.required_output_fields
                }
            )
        )

    @property
    def requires_negative_condition(self) -> bool:
        return any(
            item.requires_negative_condition
            for item in self.requirements
            if item.provider is PreprocessRequirementProvider.MODEL
        )

    @property
    def consumer_identities(self) -> tuple[str, ...]:
        return tuple(item.consumer_identity for item in self.requirements)

    @property
    def model_payload_type(self) -> str:
        return next(
            item.payload_type
            for item in self.requirements
            if item.provider is PreprocessRequirementProvider.MODEL
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "preprocess_requirement_set",
            "required_modalities": list(self.required_modalities),
            "required_output_fields": list(self.required_output_fields),
            "conditioner_output_fields": list(self.conditioner_output_fields),
            "requires_negative_condition": self.requires_negative_condition,
            "consumer_identities": list(self.consumer_identities),
            "requirements": [item.to_payload() for item in self.requirements],
        }

    @property
    def requirement_set_id(self) -> str:
        return _digest(self.to_payload())

    def validate_model_producer(self, port: PreprocessPortContract) -> None:
        """Fail closed unless the typed producer covers effective model demand."""

        if not isinstance(port, PreprocessPortContract):
            raise TypeError("port must be a PreprocessPortContract")
        if port.schema_version < 2:
            raise PreprocessContractError(
                "effective preprocess requirements require a schema-v2 producer port"
            )
        if self.model_payload_type != port.output_payload_type:
            raise PreprocessContractError(
                "model preprocess producer and consumer payload types differ"
            )
        required = set(self.required_output_fields)
        produced = set(port.producer_output_fields)
        if not required.issubset(produced):
            raise PreprocessContractError(
                "model preprocess producer does not cover effective requirements; "
                f"missing={sorted(required - produced)}"
            )
        if self.requires_negative_condition:
            negative = set(port.negative_condition_fields)
            required_negative = {
                field
                for item in self.requirements
                if item.provider is PreprocessRequirementProvider.MODEL
                and item.requires_negative_condition
                for field in item.required_negative_condition_fields
            }
            if not required_negative.issubset(negative):
                raise PreprocessContractError(
                    "model preprocess producer does not cover required negative "
                    f"condition fields; missing={sorted(required_negative - negative)}"
                )


@dataclass(frozen=True, slots=True)
class PreprocessCompatibilityReceipt:
    """Audit receipt kept separate from the payload cache identity.

    Consumer identities may include algorithm, Rollout, Conditioner, reward, or
    training declarations.  They prove that one producer covers the effective
    consumer graph, but they do not describe the bytes emitted by that producer
    and therefore must never enter :class:`PreprocessPlan` or its cache keys.
    """

    preprocess_plan_id: str
    producer_port: PreprocessPortContract
    requirements: PreprocessRequirementSet
    resolved_model_manifest_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _sha256(self.preprocess_plan_id, field_name="preprocess_plan_id")
        if not isinstance(self.producer_port, PreprocessPortContract):
            raise TypeError("producer_port must be a PreprocessPortContract")
        if not isinstance(self.requirements, PreprocessRequirementSet):
            raise TypeError("requirements must be a PreprocessRequirementSet")
        _sha256(
            self.resolved_model_manifest_sha256,
            field_name="resolved_model_manifest_sha256",
        )
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise PreprocessContractError(
                "unsupported preprocess compatibility receipt schema_version"
            )
        self.requirements.validate_model_producer(self.producer_port)

    @property
    def requirement_set_id(self) -> str:
        return self.requirements.requirement_set_id

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "preprocess_compatibility_receipt",
            "preprocess_plan_id": self.preprocess_plan_id,
            "producer_port": self.producer_port.to_payload(),
            "requirements": self.requirements.to_payload(),
            "requirement_set_id": self.requirement_set_id,
            "resolved_model_manifest_sha256": (self.resolved_model_manifest_sha256),
        }

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class PreprocessGeometry:
    """Output geometry and temporal sampling semantics used by preprocessing."""

    height: int
    width: int
    aspect_ratio_bucket: str
    frame_count: int = 1
    frame_rate_numerator: int | None = None
    frame_rate_denominator: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.height, field_name="height")
        _positive_int(self.width, field_name="width")
        _canonical_text(
            self.aspect_ratio_bucket,
            field_name="aspect_ratio_bucket",
        )
        _positive_int(self.frame_count, field_name="frame_count")
        numerator = self.frame_rate_numerator
        denominator = self.frame_rate_denominator
        if (numerator is None) != (denominator is None):
            raise PreprocessContractError(
                "frame rate numerator and denominator must be both set or both None"
            )
        if numerator is not None:
            _positive_int(numerator, field_name="frame_rate_numerator")
            _positive_int(denominator, field_name="frame_rate_denominator")
            divisor = math.gcd(numerator, denominator)
            object.__setattr__(self, "frame_rate_numerator", numerator // divisor)
            object.__setattr__(
                self,
                "frame_rate_denominator",
                denominator // divisor,
            )
        elif self.frame_count > 1:
            raise PreprocessContractError(
                "multi-frame preprocessing requires an explicit rational frame rate"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "height": self.height,
            "width": self.width,
            "aspect_ratio_bucket": self.aspect_ratio_bucket,
            "frame_count": self.frame_count,
            "frame_rate": (
                None
                if self.frame_rate_numerator is None
                else {
                    "numerator": self.frame_rate_numerator,
                    "denominator": self.frame_rate_denominator,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class PreprocessTransform:
    """One ordered, revision-pinned input transform stage."""

    stage_id: str
    implementation_id: str
    revision: str
    config: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        for name in ("stage_id", "implementation_id", "revision"):
            _canonical_text(getattr(self, name), field_name=name)
        object.__setattr__(
            self,
            "config",
            _frozen_config(
                self.config,
                field_name=f"transform[{self.stage_id}].config",
                allowed_identity_fields=frozenset({"algorithm"}),
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "implementation_id": self.implementation_id,
            "revision": self.revision,
            "config": to_plain_dict(self.config),
        }


@dataclass(frozen=True, slots=True)
class PreprocessProducerSpec:
    """Data-owned declaration of one model preprocessing producer.

    Models provide this immutable description, while the data plane owns its
    schema, identity inputs, and validation. Artifact content and the resolved
    registry manifest are bound later by the preprocessing plan factory.
    """

    implementation_id: str
    implementation_revision: str
    port: PreprocessPortContract
    geometry: PreprocessGeometry
    transforms: tuple[PreprocessTransform, ...] = ()
    preprocess_config: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        _canonical_text(self.implementation_id, field_name="implementation_id")
        _canonical_text(
            self.implementation_revision,
            field_name="implementation_revision",
        )
        if not isinstance(self.port, PreprocessPortContract):
            raise TypeError("port must be a PreprocessPortContract")
        if not isinstance(self.geometry, PreprocessGeometry):
            raise TypeError("geometry must be a PreprocessGeometry")
        if type(self.transforms) is not tuple or any(
            not isinstance(item, PreprocessTransform) for item in self.transforms
        ):
            raise TypeError("transforms must contain PreprocessTransform values")
        stage_ids = tuple(item.stage_id for item in self.transforms)
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("preprocess transform stage ids must be unique")
        if not isinstance(self.preprocess_config, FrozenMapping):
            raise TypeError("preprocess_config must be a FrozenMapping")
        if "mode" in self.preprocess_config:
            raise ValueError("preprocess execution mode is owned by the plan factory")
        if self.port.schema_version >= 2:
            guidance = self.preprocess_config.get("do_classifier_free_guidance")
            if type(guidance) is not bool:
                raise ValueError(
                    "schema-v2 preprocess specs require boolean "
                    "do_classifier_free_guidance"
                )
            if guidance != bool(self.port.negative_condition_fields):
                raise ValueError(
                    "do_classifier_free_guidance must exactly match producer "
                    "negative_condition_fields"
                )
        try:
            canonical_json_text(self.preprocess_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"preprocess_config is not canonical JSON: {exc}") from exc

    def to_payload(self) -> dict[str, object]:
        return {
            "implementation_id": self.implementation_id,
            "implementation_revision": self.implementation_revision,
            "port": self.port.to_payload(),
            "geometry": self.geometry.to_payload(),
            "transforms": [item.to_payload() for item in self.transforms],
            "preprocess_config": to_plain_dict(self.preprocess_config),
        }


@dataclass(frozen=True, slots=True)
class PreprocessPlan:
    """Complete immutable cache identity for one model preprocessing port."""

    port: PreprocessPortContract
    components: tuple[PreprocessComponentIdentity, ...]
    geometry: PreprocessGeometry
    transforms: tuple[PreprocessTransform, ...]
    preprocess_config: FrozenMapping = field(default_factory=FrozenMapping)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.port, PreprocessPortContract):
            raise TypeError("port must be a PreprocessPortContract")
        if self.schema_version != 2 or type(self.schema_version) is not int:
            raise PreprocessContractError(
                "unsupported preprocessing plan schema_version"
            )
        if type(self.components) is not tuple or not self.components:
            raise PreprocessContractError("components must be a non-empty tuple")
        if any(
            not isinstance(item, PreprocessComponentIdentity)
            for item in self.components
        ):
            raise TypeError(
                "components must contain PreprocessComponentIdentity values"
            )
        components = tuple(
            sorted(
                self.components,
                key=lambda item: (item.role.value, item.logical_name),
            )
        )
        component_keys = tuple(
            (item.role.value, item.logical_name) for item in components
        )
        if len(component_keys) != len(set(component_keys)):
            raise PreprocessContractError(
                "component role/logical_name pairs must be unique"
            )
        declared = set(self.port.dependency_keys)
        observed = set(component_keys)
        if observed != declared:
            missing = sorted(declared - observed)
            extra = sorted(observed - declared)
            raise PreprocessContractError(
                "preprocessing components do not exactly match the model port "
                f"dependencies; missing={missing}, extra={extra}"
            )
        object.__setattr__(self, "components", components)
        if not isinstance(self.geometry, PreprocessGeometry):
            raise TypeError("geometry must be a PreprocessGeometry")
        if type(self.transforms) is not tuple:
            raise TypeError("transforms must be a tuple")
        if any(not isinstance(item, PreprocessTransform) for item in self.transforms):
            raise TypeError("transforms must contain PreprocessTransform values")
        stage_ids = tuple(item.stage_id for item in self.transforms)
        if len(stage_ids) != len(set(stage_ids)):
            raise PreprocessContractError("transform stage_id values must be unique")
        object.__setattr__(
            self,
            "preprocess_config",
            _frozen_config(
                self.preprocess_config,
                field_name="preprocess_config",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "preprocess_payload_plan",
            "port": self.port.to_payload(),
            "components": [item.to_payload() for item in self.components],
            "geometry": self.geometry.to_payload(),
            "transforms": [item.to_payload() for item in self.transforms],
            "preprocess_config": to_plain_dict(self.preprocess_config),
        }

    @property
    def plan_id(self) -> str:
        return _digest(self.to_payload())

    @property
    def preprocess_port_id(self) -> str:
        return self.port.port_id

    @property
    def output_payload_type(self) -> str:
        return self.port.output_payload_type

    def cache_key_for(self, source: SourceItemContext) -> str:
        """Return a per-source key; iteration-local row state is unrepresentable."""

        if not isinstance(source, SourceItemContext):
            raise TypeError(
                "cache keys require SourceItemContext, not runtime row state"
            )
        source.validate()
        payload = {
            "schema_version": 2,
            "kind": "preprocess_payload_cache_key",
            "preprocess_plan_id": self.plan_id,
            "source_content": {
                "schema_version": source.schema_version,
                "dataset_revision": source.dataset_revision,
                "dataset_index": source.dataset_index,
            },
        }
        return _digest(payload)


@dataclass(frozen=True, slots=True)
class PreprocessedItem:
    """Integrity-checked descriptor for one model-specific cached payload."""

    source: SourceItemContext
    preprocess_plan_id: str
    cache_key: str
    payload_type: str
    payload_sha256: str
    payload_size_bytes: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceItemContext):
            raise TypeError("source must be a SourceItemContext")
        self.source.validate()
        _sha256(self.preprocess_plan_id, field_name="preprocess_plan_id")
        _sha256(self.cache_key, field_name="cache_key")
        _canonical_text(self.payload_type, field_name="payload_type")
        _sha256(self.payload_sha256, field_name="payload_sha256")
        if type(self.payload_size_bytes) is not int or self.payload_size_bytes < 0:
            raise PreprocessContractError(
                "payload_size_bytes must be a non-negative integer"
            )
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise PreprocessContractError(
                "unsupported preprocessed item schema_version"
            )

    @classmethod
    def from_bytes(
        cls,
        *,
        plan: PreprocessPlan,
        source: SourceItemContext,
        payload: bytes,
    ) -> PreprocessedItem:
        if not isinstance(plan, PreprocessPlan):
            raise TypeError("plan must be a PreprocessPlan")
        if not isinstance(payload, bytes):
            raise TypeError("preprocessed payload must be immutable bytes")
        return cls(
            source=source,
            preprocess_plan_id=plan.plan_id,
            cache_key=plan.cache_key_for(source),
            payload_type=plan.output_payload_type,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            payload_size_bytes=len(payload),
        )

    def validate_payload(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("preprocessed payload must be immutable bytes")
        if len(payload) != self.payload_size_bytes:
            raise PreprocessContractError(
                "cached payload size does not match descriptor"
            )
        if hashlib.sha256(payload).hexdigest() != self.payload_sha256:
            raise PreprocessContractError(
                "cached payload digest does not match descriptor"
            )

    def validate_against(self, plan: PreprocessPlan) -> None:
        if not isinstance(plan, PreprocessPlan):
            raise TypeError("plan must be a PreprocessPlan")
        if self.preprocess_plan_id != plan.plan_id:
            raise PreprocessContractError(
                "preprocessed item belongs to a different preprocessing plan"
            )
        if self.cache_key != plan.cache_key_for(self.source):
            raise PreprocessContractError(
                "preprocessed item cache key does not match source and plan"
            )
        if self.payload_type != plan.output_payload_type:
            raise PreprocessContractError(
                "preprocessed item payload type does not match plan"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.serialize(),
            "preprocess_plan_id": self.preprocess_plan_id,
            "cache_key": self.cache_key,
            "payload_type": self.payload_type,
            "payload_sha256": self.payload_sha256,
            "payload_size_bytes": self.payload_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PreprocessManifest:
    """Exact cache population published for one plan and source snapshot."""

    preprocess_plan_id: str
    expected_cache_keys: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _sha256(self.preprocess_plan_id, field_name="preprocess_plan_id")
        if type(self.expected_cache_keys) is not tuple or not self.expected_cache_keys:
            raise PreprocessContractError(
                "expected_cache_keys must be a non-empty tuple"
            )
        for value in self.expected_cache_keys:
            _sha256(value, field_name="expected cache key")
        keys = tuple(sorted(self.expected_cache_keys))
        if len(keys) != len(set(keys)):
            raise PreprocessContractError("expected cache keys must be unique")
        object.__setattr__(self, "expected_cache_keys", keys)
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise PreprocessContractError(
                "unsupported preprocess manifest schema_version"
            )

    @classmethod
    def for_sources(
        cls,
        *,
        plan: PreprocessPlan,
        sources: tuple[SourceItemContext, ...],
    ) -> PreprocessManifest:
        if not isinstance(plan, PreprocessPlan):
            raise TypeError("plan must be a PreprocessPlan")
        if type(sources) is not tuple or not sources:
            raise PreprocessContractError("sources must be a non-empty tuple")
        if any(not isinstance(item, SourceItemContext) for item in sources):
            raise TypeError("sources must contain SourceItemContext values")
        return cls(
            preprocess_plan_id=plan.plan_id,
            expected_cache_keys=tuple(plan.cache_key_for(item) for item in sources),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "preprocess_manifest",
            "preprocess_plan_id": self.preprocess_plan_id,
            "expected_cache_keys": list(self.expected_cache_keys),
        }

    @property
    def manifest_id(self) -> str:
        return _digest(self.to_payload())

    def validate_items(self, items: tuple[PreprocessedItem, ...]) -> None:
        if type(items) is not tuple or not items:
            raise PreprocessContractError("manifest items must be a non-empty tuple")
        if any(not isinstance(item, PreprocessedItem) for item in items):
            raise TypeError("manifest items must contain PreprocessedItem values")
        observed = tuple(sorted(item.cache_key for item in items))
        if observed != self.expected_cache_keys:
            raise PreprocessContractError(
                "preprocess manifest items are missing, duplicated, or unexpected"
            )
        if any(item.preprocess_plan_id != self.preprocess_plan_id for item in items):
            raise PreprocessContractError(
                "preprocess manifest item belongs to a different plan"
            )


@runtime_checkable
class PreprocessWriteLease(Protocol):
    """Exclusive writer lease for one exact manifest identity."""

    @property
    def manifest_id(self) -> str:
        """The one manifest identity exclusively owned by this lease."""

    def write_atomic(self, item: PreprocessedItem, payload: bytes) -> None:
        """Publish bytes and descriptor atomically, or leave neither visible."""

    def commit(self) -> None:
        """Publish the complete manifest after validating every expected key."""

    def abort(self) -> None:
        """Release ownership without publishing a complete manifest."""


@runtime_checkable
class PreprocessCacheWriter(Protocol):
    """Single-writer lease boundary for a content-addressed cache manifest."""

    def acquire_exclusive(
        self,
        manifest: PreprocessManifest,
    ) -> PreprocessWriteLease:
        """Acquire one exclusive lease or fail while another writer owns it."""


@runtime_checkable
class PreprocessCacheReader(Protocol):
    """Reader boundary which validates the complete manifest before use."""

    def load_manifest(
        self,
        manifest_id: str,
    ) -> PreprocessManifest:
        """Load an integrity-checked complete manifest by exact identity."""

    def read(self, item: PreprocessedItem) -> bytes:
        """Load bytes and verify them against the item descriptor."""


@runtime_checkable
class PreprocessBarrier(Protocol):
    """Synchronization boundary after writers publish a complete-plan manifest."""

    def wait(self, manifest_id: str) -> PreprocessManifest:
        """Return only when this exact source snapshot manifest is visible."""
