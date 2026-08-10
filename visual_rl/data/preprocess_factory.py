"""Bind model declarations to immutable inline preprocessing identities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.data.preprocess import (
    PreprocessCompatibilityReceipt,
    PreprocessComponentIdentity,
    PreprocessComponentRole,
    PreprocessContractError,
    PreprocessDependency,
    PreprocessPlan,
    PreprocessProducerSpec,
    PreprocessRequirementSet,
)

__all__ = (
    "InlinePreprocessPlanFactory",
    "InlinePreprocessPlanRequest",
    "InlinePreprocessPlanResolution",
)


_MODEL_DEPENDENCY = PreprocessDependency(
    role=PreprocessComponentRole.MODEL,
    logical_name="model_artifact",
)
_MODEL_MANIFEST_KEYS = frozenset(
    {
        "alias",
        "config",
        "config_type_path",
        "declaration_provider_abi",
        "declaration_provider_path",
        "declared_contract",
        "implementation_class_path",
        "interface_version",
        "kind",
        "optional_dependencies",
        "schema_version",
    }
)
_DECLARED_CONTRACT_KEYS = frozenset(
    {
        "algorithm",
        "component_id",
        "component_kind",
        "conditioner",
        "credit",
        "dynamics",
        "model",
        "pending_fields",
        "reward",
        "rollout",
        "trainer",
    }
)
_NON_MODEL_CONTRACT_KEYS = _DECLARED_CONTRACT_KEYS - {
    "component_id",
    "component_kind",
    "model",
    "pending_fields",
}
_NON_PAYLOAD_TRAINING_FIELDS = frozenset(
    {
        "beta",
        "group_size",
        "lora_alpha",
        "lora_rank",
        "lora_target_modules",
        "optimizer",
        "optimizer_id",
        "training",
        "training_params",
    }
)
_FORBIDDEN_IDENTITY_FIELDS = (
    frozenset(
        {
            "absolute_path",
            "accumulation_index",
            "algorithm",
            "algorithm_id",
            "artifact_location",
            "artifact_path",
            "batch_row_id",
            "bound_contract_id",
            "cache_dir",
            "checkpoint_path",
            "cwd",
            "device",
            "device_id",
            "device_map",
            "group_id",
            "group_rank",
            "group_world_size",
            "hostname",
            "launch_id",
            "local_process_index",
            "local_rank",
            "local_world_size",
            "logging_dir",
            "location",
            "master_addr",
            "master_port",
            "member_id",
            "node_rank",
            "num_processes",
            "object_id",
            "occurrence_id",
            "optimizer_step",
            "output_dir",
            "path",
            "phase",
            "pid",
            "process_index",
            "rank",
            "recipe",
            "recipe_id",
            "repr",
            "reward",
            "reward_id",
            "rollout",
            "rng",
            "rng_state",
            "row_id",
            "rollout_id",
            "run_dir",
            "runtime_context",
            "runtime_facts",
            "sample_id",
            "seed",
            "source_path",
            "trainer",
            "trainer_id",
            "trajectory_id",
            "workdir",
            "working_directory",
            "world_size",
        }
    )
    | _NON_PAYLOAD_TRAINING_FIELDS
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()


def _reject_forbidden_fields(
    value: object,
    *,
    location: str,
    allowed_field_names: frozenset[str] = frozenset(),
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.lower()
            if (
                normalized in _FORBIDDEN_IDENTITY_FIELDS
                and normalized not in allowed_field_names
            ):
                raise PreprocessContractError(
                    f"{location}.{key} cannot enter preprocessing identity"
                )
            _reject_forbidden_fields(
                item,
                location=f"{location}.{key}",
                allowed_field_names=allowed_field_names,
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden_fields(
                item,
                location=f"{location}[{index}]",
                allowed_field_names=allowed_field_names,
            )
    elif isinstance(value, Path):
        raise PreprocessContractError(
            f"{location} is a filesystem path and cannot enter preprocessing identity"
        )


def _validate_artifact_identity(identity: FrozenMapping) -> None:
    if not identity:
        raise PreprocessContractError("model_artifact_identity must not be empty")
    _reject_forbidden_fields(identity, location="model_artifact_identity")
    immutable_digests = 0

    def visit(value: object, *, location: str) -> None:
        nonlocal immutable_digests
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key.lower().endswith("_sha256"):
                    if not isinstance(item, str) or _SHA256.fullmatch(item) is None:
                        raise PreprocessContractError(
                            f"{location}.{key} must be a lowercase SHA-256 digest"
                        )
                    immutable_digests += 1
                visit(item, location=f"{location}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, location=f"{location}[{index}]")

    visit(identity, location="model_artifact_identity")
    if immutable_digests == 0:
        raise PreprocessContractError(
            "model_artifact_identity must contain an immutable *_sha256 digest"
        )
    try:
        canonical_json_text(identity)
    except (TypeError, ValueError) as exc:
        raise PreprocessContractError(
            f"model_artifact_identity is not canonical JSON: {exc}"
        ) from exc


def _canonical_model_manifest_projection(
    manifest: FrozenMapping,
    *,
    spec: PreprocessProducerSpec,
) -> dict[str, object]:
    if set(manifest) != _MODEL_MANIFEST_KEYS:
        missing = sorted(_MODEL_MANIFEST_KEYS - set(manifest))
        extra = sorted(set(manifest) - _MODEL_MANIFEST_KEYS)
        raise PreprocessContractError(
            "resolved_model_manifest has an invalid schema; "
            f"missing={missing}, extra={extra}"
        )
    if manifest["kind"] != "model":
        raise PreprocessContractError("resolved_model_manifest.kind must be model")
    if manifest["schema_version"] != 1:
        raise PreprocessContractError(
            "resolved_model_manifest.schema_version must be 1"
        )
    for key in (
        "alias",
        "implementation_class_path",
        "declaration_provider_path",
        "declaration_provider_abi",
        "config_type_path",
        "interface_version",
    ):
        value = manifest[key]
        if not isinstance(value, str) or not value or value.strip() != value:
            raise PreprocessContractError(
                f"resolved_model_manifest.{key} must be a canonical string"
            )
    if manifest["implementation_class_path"] != spec.implementation_id:
        raise PreprocessContractError(
            "preprocess implementation_id differs from the resolved model "
            "implementation_class_path"
        )
    alias = manifest["alias"]
    dependencies = manifest["optional_dependencies"]
    if type(dependencies) is not tuple or any(
        not isinstance(item, str) or not item for item in dependencies
    ):
        raise PreprocessContractError(
            "resolved_model_manifest.optional_dependencies must contain strings"
        )
    if len(dependencies) != len(set(dependencies)):
        raise PreprocessContractError(
            "resolved_model_manifest.optional_dependencies must be unique"
        )
    config = manifest["config"]
    declared = manifest["declared_contract"]
    if not isinstance(config, FrozenMapping):
        raise PreprocessContractError(
            "resolved_model_manifest.config must be a FrozenMapping"
        )
    if not isinstance(declared, FrozenMapping):
        raise PreprocessContractError(
            "resolved_model_manifest.declared_contract must be a FrozenMapping"
        )
    _reject_forbidden_fields(
        config,
        location="resolved_model_manifest.config",
        allowed_field_names=_NON_PAYLOAD_TRAINING_FIELDS,
    )
    if set(declared) != _DECLARED_CONTRACT_KEYS:
        missing = sorted(_DECLARED_CONTRACT_KEYS - set(declared))
        extra = sorted(set(declared) - _DECLARED_CONTRACT_KEYS)
        raise PreprocessContractError(
            "resolved model declared_contract has an invalid schema; "
            f"missing={missing}, extra={extra}"
        )
    if declared["component_kind"] != "model":
        raise PreprocessContractError(
            "resolved model declared_contract.component_kind must be model"
        )
    component_id = declared["component_id"]
    if not isinstance(component_id, str) or not component_id:
        raise PreprocessContractError(
            "resolved model declared_contract.component_id must be non-empty"
        )
    pending_fields = declared["pending_fields"]
    if type(pending_fields) is not tuple or any(
        not isinstance(item, str) or not item for item in pending_fields
    ):
        raise PreprocessContractError(
            "resolved model declared_contract.pending_fields must contain strings"
        )
    if pending_fields != tuple(sorted(set(pending_fields))):
        raise PreprocessContractError(
            "resolved model declared_contract.pending_fields must be sorted and unique"
        )
    if any(declared[key] is not None for key in _NON_MODEL_CONTRACT_KEYS):
        raise PreprocessContractError(
            "resolved model declared_contract contains a non-model contract"
        )
    model_contract = declared["model"]
    if not isinstance(model_contract, FrozenMapping):
        raise PreprocessContractError(
            "resolved model declared_contract.model must be a FrozenMapping"
        )
    payload_types = model_contract.get("condition_payload_types")
    if type(payload_types) is not tuple or payload_types != (
        spec.port.output_payload_type,
    ):
        raise PreprocessContractError(
            "preprocess output payload type differs from the model contract"
        )
    _reject_forbidden_fields(
        model_contract,
        location="resolved_model_manifest.declared_contract.model",
    )
    projection = {
        "schema_version": manifest["schema_version"],
        "kind": manifest["kind"],
        "alias": alias,
        "implementation_class_path": manifest["implementation_class_path"],
        "declaration_provider_path": manifest["declaration_provider_path"],
        "declaration_provider_abi": manifest["declaration_provider_abi"],
        "config_type_path": manifest["config_type_path"],
        "interface_version": manifest["interface_version"],
        "optional_dependencies": sorted(dependencies),
        "config": to_plain_dict(config),
        "declared_contract": {
            "component_kind": "model",
            "component_id": component_id,
            "model": to_plain_dict(model_contract),
            "pending_fields": list(pending_fields),
        },
    }
    try:
        canonical_json_text(projection)
    except (TypeError, ValueError) as exc:
        raise PreprocessContractError(
            f"resolved_model_manifest is not canonical JSON: {exc}"
        ) from exc
    return projection


@dataclass(frozen=True, slots=True)
class InlinePreprocessPlanRequest:
    """Strict immutable inputs for the V0 inline preprocessing identity."""

    spec: PreprocessProducerSpec
    model_artifact_identity: FrozenMapping
    resolved_model_manifest: FrozenMapping
    requirements: PreprocessRequirementSet | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, PreprocessProducerSpec):
            raise TypeError("spec must be a PreprocessProducerSpec")
        if not isinstance(self.model_artifact_identity, FrozenMapping):
            raise TypeError("model_artifact_identity must be a FrozenMapping")
        if not isinstance(self.resolved_model_manifest, FrozenMapping):
            raise TypeError("resolved_model_manifest must be a FrozenMapping")
        if self.requirements is not None and not isinstance(
            self.requirements,
            PreprocessRequirementSet,
        ):
            raise TypeError("requirements must be a PreprocessRequirementSet or None")


@dataclass(frozen=True, slots=True)
class InlinePreprocessPlanResolution:
    """Payload identity plus its optional, non-cache compatibility receipt."""

    plan: PreprocessPlan
    compatibility_receipt: PreprocessCompatibilityReceipt | None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PreprocessPlan):
            raise TypeError("plan must be a PreprocessPlan")
        if self.compatibility_receipt is not None:
            if not isinstance(
                self.compatibility_receipt,
                PreprocessCompatibilityReceipt,
            ):
                raise TypeError(
                    "compatibility_receipt must be a "
                    "PreprocessCompatibilityReceipt or None"
                )
            if self.compatibility_receipt.preprocess_plan_id != self.plan.plan_id:
                raise PreprocessContractError(
                    "compatibility receipt belongs to a different preprocess plan"
                )


class InlinePreprocessPlanFactory:
    """Create a canonical plan without loading or calling live model objects."""

    def create(self, request: InlinePreprocessPlanRequest) -> PreprocessPlan:
        """Compatibility API returning only the payload cache plan."""

        return self.resolve(request).plan

    def resolve(
        self,
        request: InlinePreprocessPlanRequest,
    ) -> InlinePreprocessPlanResolution:
        """Resolve byte identity and compatibility evidence as separate values."""

        if not isinstance(request, InlinePreprocessPlanRequest):
            raise TypeError("request must be an InlinePreprocessPlanRequest")
        spec = request.spec
        if request.requirements is not None:
            request.requirements.validate_model_producer(spec.port)
        if spec.port.dependencies != (_MODEL_DEPENDENCY,):
            raise PreprocessContractError(
                "V0 inline preprocessing requires only "
                "(model, model_artifact) dependency"
            )
        _validate_artifact_identity(request.model_artifact_identity)
        _reject_forbidden_fields(
            spec.preprocess_config,
            location="preprocess_spec.preprocess_config",
        )
        for transform in spec.transforms:
            _reject_forbidden_fields(
                transform.config,
                location=f"preprocess_spec.transforms.{transform.stage_id}.config",
                # A resize/interpolation algorithm is a byte-producing transform,
                # not the outer training algorithm declaration.
                allowed_field_names=frozenset({"algorithm"}),
            )
        canonical_manifest = _canonical_model_manifest_projection(
            request.resolved_model_manifest,
            spec=spec,
        )
        artifact_payload = to_plain_dict(request.model_artifact_identity)
        component_config = {
            "schema_version": 2,
            "output_schema": spec.port.to_payload(),
            "geometry": spec.geometry.to_payload(),
            "transforms": [item.to_payload() for item in spec.transforms],
            "preprocess_config": to_plain_dict(spec.preprocess_config),
        }
        component = PreprocessComponentIdentity(
            role=PreprocessComponentRole.MODEL,
            logical_name="model_artifact",
            implementation_id=spec.implementation_id,
            revision=spec.implementation_revision,
            content_sha256=_digest(artifact_payload),
            config_sha256=_digest(component_config),
        )
        plan = PreprocessPlan(
            port=spec.port,
            components=(component,),
            geometry=spec.geometry,
            transforms=spec.transforms,
            preprocess_config=FrozenMapping(
                {
                    "mode": "inline",
                    **to_plain_dict(spec.preprocess_config),
                }
            ),
        )
        receipt = (
            None
            if request.requirements is None
            else PreprocessCompatibilityReceipt(
                preprocess_plan_id=plan.plan_id,
                producer_port=spec.port,
                requirements=request.requirements,
                resolved_model_manifest_sha256=_digest(canonical_manifest),
            )
        )
        return InlinePreprocessPlanResolution(
            plan=plan,
            compatibility_receipt=receipt,
        )
