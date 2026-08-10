"""Adapter-declared preprocessing metadata and inline identity binding."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from visual_rl.core.contracts import ComputePrecision
from visual_rl.core.contracts import DECLARATION_PROVIDER_ABI
from visual_rl.core.identity import to_identity_value
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.data import (
    InlinePreprocessPlanFactory,
    InlinePreprocessPlanRequest,
    PreprocessComponentRole,
    PreprocessConsumerRequirement,
    PreprocessContractError,
    PreprocessDependency,
    PreprocessGeometry,
    PreprocessPlan,
    PreprocessPortContract,
    PreprocessProducerSpec,
    PreprocessRequirementProvider,
    PreprocessRequirementSet,
    PreprocessTransform,
)
from visual_rl.models import ModelAdapter, ModelPortError, ModelPreprocessSpec
from visual_rl.models.catalog import SD3Config, WanConfig
from visual_rl.models.implementations.sd3 import SD3Adapter
from visual_rl.models.implementations.wan import WanT2VAdapter


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact(*, salt: str = "model") -> FrozenMapping:
    return FrozenMapping(
        {
            "identity_schema": "filesystem-artifact.v1",
            "content_policy": "all-files.v1",
            "content_sha256": _hash(salt),
            "file_count": 4,
        }
    )


def _manifest(
    *,
    alias: str,
    class_path: str,
    config_type_path: str,
    config: object,
    declared_contract: object,
) -> FrozenMapping:
    return FrozenMapping(
        {
            "schema_version": 1,
            "kind": "model",
            "alias": alias,
            "implementation_class_path": class_path,
            "declaration_provider_path": (
                "tests.test_preprocess_factory:TestModelDeclarationProvider"
            ),
            "declaration_provider_abi": DECLARATION_PROVIDER_ABI,
            "config_type_path": config_type_path,
            "interface_version": "1.0",
            "optional_dependencies": (
                "diffusers",
                "peft",
                "torch",
                "transformers",
            ),
            "config": to_identity_value(config),
            "declared_contract": to_identity_value(declared_contract),
        }
    )


def _sd3(tmp_path: Path, *, precision: ComputePrecision = ComputePrecision.FP32):
    config = SD3Config(artifact_ref="main", resolution=64)
    adapter = SD3Adapter(
        config,
        artifact_path=tmp_path,
        precision=precision,
        model_loader=lambda *_args: (_ for _ in ()).throw(
            AssertionError("preprocess metadata must not load model components")
        ),
    )
    manifest = _manifest(
        alias="sd3",
        class_path="visual_rl.models.implementations.sd3:SD3Adapter",
        config_type_path="visual_rl.models.catalog:SD3Config",
        config=config,
        declared_contract=config.describe_contract(),
    )
    return adapter, manifest


def _wan(
    tmp_path: Path,
    *,
    precision: ComputePrecision = ComputePrecision.FP32,
    frame_rate_numerator: int = 16,
    frame_rate_denominator: int = 1,
):
    config = WanConfig(
        artifact_ref="main",
        height=64,
        width=96,
        frames=9,
        frame_rate_numerator=frame_rate_numerator,
        frame_rate_denominator=frame_rate_denominator,
    )
    adapter = WanT2VAdapter(
        config,
        artifact_path=tmp_path,
        precision=precision,
        model_loader=lambda *_args: (_ for _ in ()).throw(
            AssertionError("preprocess metadata must not load model components")
        ),
    )
    manifest = _manifest(
        alias="wan-t2v",
        class_path="visual_rl.models.implementations.wan:WanT2VAdapter",
        config_type_path="visual_rl.models.catalog:WanConfig",
        config=config,
        declared_contract=config.describe_contract(),
    )
    return adapter, manifest


def _plan(
    adapter: SD3Adapter | WanT2VAdapter,
    manifest: FrozenMapping,
    *,
    artifact: FrozenMapping | None = None,
) -> PreprocessPlan:
    return InlinePreprocessPlanFactory().create(
        InlinePreprocessPlanRequest(
            spec=adapter.describe_preprocess(),
            model_artifact_identity=_artifact() if artifact is None else artifact,
            resolved_model_manifest=manifest,
        )
    )


def _requirements(
    spec: PreprocessProducerSpec,
    *,
    consumer_identity: str = "algorithm-and-rollout:v1",
) -> PreprocessRequirementSet:
    return PreprocessRequirementSet(
        (
            PreprocessConsumerRequirement(
                consumer_identity=consumer_identity,
                provider=PreprocessRequirementProvider.MODEL,
                payload_type=spec.port.output_payload_type,
                required_modalities=("prompt_text",),
                required_output_fields=spec.port.producer_output_fields,
                required_negative_condition_fields=(
                    spec.port.negative_condition_fields
                ),
                requires_negative_condition=bool(spec.port.negative_condition_fields),
            ),
        )
    )


def _resolution(
    adapter: SD3Adapter | WanT2VAdapter,
    manifest: FrozenMapping,
    *,
    artifact: FrozenMapping | None = None,
):
    spec = adapter.describe_preprocess()
    return InlinePreprocessPlanFactory().resolve(
        InlinePreprocessPlanRequest(
            spec=spec,
            model_artifact_identity=_artifact() if artifact is None else artifact,
            resolved_model_manifest=manifest,
            requirements=_requirements(spec),
        )
    )


def test_preprocess_producer_spec_has_one_data_owned_runtime_identity(tmp_path) -> None:
    adapter, _manifest_value = _sd3(tmp_path)

    assert ModelPreprocessSpec is PreprocessProducerSpec
    assert PreprocessProducerSpec.__module__ == "visual_rl.data.preprocess"
    assert type(adapter.describe_preprocess()) is PreprocessProducerSpec


def test_data_preprocess_owner_does_not_import_the_models_domain() -> None:
    package_root = Path(__file__).parents[1] / "visual_rl" / "data"
    violations: list[tuple[str, int, str]] = []
    for filename in ("preprocess.py", "preprocess_factory.py"):
        path = package_root / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                targets = () if node.module is None else (node.module,)
            else:
                continue
            violations.extend(
                (filename, node.lineno, target)
                for target in targets
                if target == "visual_rl.models"
                or target.startswith("visual_rl.models.")
            )

    assert violations == []


def test_sd3_and_wan_declare_metadata_without_loading_or_encoding(
    tmp_path,
    monkeypatch,
) -> None:
    sd3, sd3_manifest = _sd3(tmp_path)
    wan, wan_manifest = _wan(tmp_path)
    encode_calls = []

    def forbidden_encode(*_args, **_kwargs):
        encode_calls.append("encode")
        raise AssertionError("identity construction must not call live encode")

    monkeypatch.setattr(sd3, "encode", forbidden_encode)
    monkeypatch.setattr(wan, "encode", forbidden_encode)

    sd3_spec = sd3.describe_preprocess()
    wan_spec = wan.describe_preprocess()
    assert sd3_spec.port.output_payload_type == "sd3_prompt_embeddings.v1"
    assert wan_spec.port.output_payload_type == "wan_prompt_embeddings.v1"
    for spec in (sd3_spec, wan_spec):
        assert spec.port.dependencies == (
            PreprocessDependency(
                role=PreprocessComponentRole.MODEL,
                logical_name="model_artifact",
            ),
        )
        assert spec.transforms == ()
        assert spec.preprocess_config["embedding_dtype"] == "fp32"

    assert sd3_spec.geometry == PreprocessGeometry(
        height=64,
        width=64,
        aspect_ratio_bucket="64x64",
    )
    assert wan_spec.geometry == PreprocessGeometry(
        height=64,
        width=96,
        aspect_ratio_bucket="64x96",
        frame_count=9,
        frame_rate_numerator=16,
        frame_rate_denominator=1,
    )
    assert _plan(sd3, sd3_manifest).output_payload_type == ("sd3_prompt_embeddings.v1")
    assert _plan(wan, wan_manifest).output_payload_type == ("wan_prompt_embeddings.v1")
    assert encode_calls == []


def test_wan_frame_rate_is_explicit_positive_and_canonical() -> None:
    config = WanConfig(
        artifact_ref="main",
        frame_rate_numerator=32,
        frame_rate_denominator=2,
    )
    assert (config.frame_rate_numerator, config.frame_rate_denominator) == (16, 1)
    resolved = WanConfig.from_mapping(
        {
            "artifact_ref": "main",
            "frame_rate_numerator": 24,
            "frame_rate_denominator": 1,
        },
        context=None,
    )
    assert (resolved.frame_rate_numerator, resolved.frame_rate_denominator) == (24, 1)
    with pytest.raises(ValueError, match="positive"):
        WanConfig(artifact_ref="main", frame_rate_denominator=0)
    with pytest.raises(ValueError, match="unknown wan-t2v params"):
        WanConfig.from_mapping(
            {"artifact_ref": "main", "fps": 16},
            context=None,
        )


def test_inline_plan_is_canonical_and_uses_only_existing_plan_id(tmp_path) -> None:
    adapter, manifest = _sd3(tmp_path)
    baseline = _plan(adapter, manifest)
    reordered_artifact = FrozenMapping(tuple(reversed(tuple(_artifact().items()))))
    manifest_payload = to_plain_dict(manifest)
    manifest_payload["optional_dependencies"].reverse()
    reordered_manifest = FrozenMapping(tuple(reversed(tuple(manifest_payload.items()))))
    equivalent = _plan(adapter, reordered_manifest, artifact=reordered_artifact)

    assert isinstance(baseline, PreprocessPlan)
    assert baseline.plan_id == equivalent.plan_id
    assert baseline.preprocess_config["mode"] == "inline"
    assert baseline.components[0].logical_name == "model_artifact"
    assert set(baseline.to_payload()) == {
        "schema_version",
        "kind",
        "port",
        "components",
        "geometry",
        "transforms",
        "preprocess_config",
    }


def test_payload_identity_changes_only_with_byte_affecting_inputs(
    tmp_path,
) -> None:
    adapter, manifest = _sd3(tmp_path)
    baseline = _plan(adapter, manifest)
    changed_artifact = _plan(adapter, manifest, artifact=_artifact(salt="changed"))

    manifest_payload = to_plain_dict(manifest)
    manifest_payload["interface_version"] = "1.1"
    changed_manifest = _plan(adapter, FrozenMapping(manifest_payload))

    bf16, bf16_manifest = _sd3(tmp_path, precision=ComputePrecision.BF16)
    changed_precision = _plan(bf16, bf16_manifest)

    changed_spec = replace(
        adapter.describe_preprocess(),
        geometry=PreprocessGeometry(
            height=128,
            width=128,
            aspect_ratio_bucket="128x128",
        ),
    )
    changed_geometry = InlinePreprocessPlanFactory().create(
        InlinePreprocessPlanRequest(
            spec=changed_spec,
            model_artifact_identity=_artifact(),
            resolved_model_manifest=manifest,
        )
    )

    assert changed_manifest.plan_id == baseline.plan_id
    assert (
        len(
            {
                baseline.plan_id,
                changed_artifact.plan_id,
                changed_precision.plan_id,
                changed_geometry.plan_id,
            }
        )
        == 4
    )

    baseline_resolution = _resolution(adapter, manifest)
    changed_manifest_resolution = _resolution(
        adapter,
        FrozenMapping(manifest_payload),
    )
    assert changed_manifest_resolution.plan.plan_id == (
        baseline_resolution.plan.plan_id
    )
    assert baseline_resolution.compatibility_receipt is not None
    assert changed_manifest_resolution.compatibility_receipt is not None
    assert changed_manifest_resolution.compatibility_receipt.receipt_id != (
        baseline_resolution.compatibility_receipt.receipt_id
    )


def test_pure_training_model_config_changes_only_compatibility_receipt(
    tmp_path: Path,
) -> None:
    adapter, manifest = _sd3(tmp_path)
    spec = adapter.describe_preprocess()
    requirements = _requirements(spec)
    request = InlinePreprocessPlanRequest(
        spec=spec,
        model_artifact_identity=_artifact(),
        resolved_model_manifest=manifest,
        requirements=requirements,
    )
    baseline = InlinePreprocessPlanFactory().resolve(request)

    changed_config = replace(
        adapter.config,
        lora_rank=64,
        lora_alpha=128,
        gradient_checkpointing=not adapter.config.gradient_checkpointing,
    )
    changed_manifest = _manifest(
        alias="sd3",
        class_path="visual_rl.models.implementations.sd3:SD3Adapter",
        config_type_path="visual_rl.models.catalog:SD3Config",
        config=changed_config,
        declared_contract=changed_config.describe_contract(),
    )
    changed = InlinePreprocessPlanFactory().resolve(
        replace(request, resolved_model_manifest=changed_manifest)
    )

    assert changed.plan.plan_id == baseline.plan.plan_id
    assert changed.plan.components[0].config_sha256 == (
        baseline.plan.components[0].config_sha256
    )
    assert baseline.compatibility_receipt is not None
    assert changed.compatibility_receipt is not None
    assert changed.compatibility_receipt.receipt_id != (
        baseline.compatibility_receipt.receipt_id
    )


def test_producer_output_transform_and_revision_change_payload_identity(
    tmp_path: Path,
) -> None:
    adapter, manifest = _sd3(tmp_path)
    spec = adapter.describe_preprocess()

    def plan_for(changed_spec: PreprocessProducerSpec) -> PreprocessPlan:
        return InlinePreprocessPlanFactory().create(
            InlinePreprocessPlanRequest(
                spec=changed_spec,
                model_artifact_identity=_artifact(),
                resolved_model_manifest=manifest,
            )
        )

    baseline = plan_for(spec)
    changed_output = plan_for(
        replace(
            spec,
            port=replace(
                spec.port,
                producer_output_fields=(
                    *spec.port.producer_output_fields,
                    "prompt_attention_mask",
                ),
            ),
        )
    )
    changed_transform = plan_for(
        replace(
            spec,
            transforms=(
                PreprocessTransform(
                    stage_id="normalize-prompt",
                    implementation_id="text.normalize",
                    revision="1",
                    config=FrozenMapping(
                        {"algorithm": "unicode-normalize", "unicode_form": "NFC"}
                    ),
                ),
            ),
        )
    )
    changed_revision = plan_for(
        replace(spec, implementation_revision="sd3-prompt-encode.v2")
    )

    assert (
        len(
            {
                baseline.plan_id,
                changed_output.plan_id,
                changed_transform.plan_id,
                changed_revision.plan_id,
            }
        )
        == 4
    )


@pytest.mark.parametrize(
    "training_field",
    (
        {"beta": 0.04},
        {"group_size": 8},
        {"lora_rank": 64},
        {"optimizer": "adamw"},
        {"training_params": {"learning_rate": 1e-4}},
    ),
)
def test_training_fields_cannot_masquerade_as_preprocess_byte_config(
    tmp_path: Path,
    training_field: dict[str, object],
) -> None:
    adapter, manifest = _sd3(tmp_path)
    spec = adapter.describe_preprocess()
    changed = replace(
        spec,
        preprocess_config=FrozenMapping(
            {**to_plain_dict(spec.preprocess_config), **training_field}
        ),
    )

    with pytest.raises(PreprocessContractError, match="cannot enter"):
        InlinePreprocessPlanFactory().create(
            InlinePreprocessPlanRequest(
                spec=changed,
                model_artifact_identity=_artifact(),
                resolved_model_manifest=manifest,
            )
        )


@pytest.mark.parametrize(
    ("family", "current_prefix", "legacy_prefix"),
    (
        (
            "sd3",
            "visual_rl.models.implementations.sd3",
            "visual_rl.models.sd3",
        ),
        (
            "wan",
            "visual_rl.models.implementations.wan",
            "visual_rl.models.wan",
        ),
    ),
)
def test_only_producer_paths_change_payload_identity(
    tmp_path: Path,
    family: str,
    current_prefix: str,
    legacy_prefix: str,
) -> None:
    if family == "sd3":
        adapter, current_manifest = _sd3(tmp_path)
        adapter_name = "SD3Adapter"
        config_name = "SD3Config"
    else:
        adapter, current_manifest = _wan(tmp_path)
        adapter_name = "WanT2VAdapter"
        config_name = "WanConfig"

    current_spec = adapter.describe_preprocess()
    current_resolution = _resolution(adapter, current_manifest)
    current = current_resolution.plan
    assert current_spec.implementation_id == f"{current_prefix}:{adapter_name}"
    assert current_spec.port.port_id == f"{current_prefix}:prompt-encode.v1"
    assert current_manifest["config_type_path"] == (
        f"visual_rl.models.catalog:{config_name}"
    )

    def resolution_with_config_type(config_type_path: str):
        payload = to_plain_dict(current_manifest)
        payload["config_type_path"] = config_type_path
        return InlinePreprocessPlanFactory().resolve(
            InlinePreprocessPlanRequest(
                spec=current_spec,
                model_artifact_identity=_artifact(),
                resolved_model_manifest=FrozenMapping(payload),
                requirements=_requirements(current_spec),
            )
        )

    intermediate_resolution = resolution_with_config_type(
        f"{current_prefix}:{config_name}"
    )
    flat_resolution = resolution_with_config_type(f"{legacy_prefix}:{config_name}")

    legacy_manifest_payload = to_plain_dict(current_manifest)
    legacy_manifest_payload["implementation_class_path"] = (
        f"{legacy_prefix}:{adapter_name}"
    )
    legacy_manifest_payload["config_type_path"] = f"{legacy_prefix}:{config_name}"
    legacy_manifest = FrozenMapping(legacy_manifest_payload)
    legacy_spec = replace(
        current_spec,
        implementation_id=f"{legacy_prefix}:{adapter_name}",
        port=replace(
            current_spec.port,
            port_id=f"{legacy_prefix}:prompt-encode.v1",
        ),
    )
    legacy_resolution = InlinePreprocessPlanFactory().resolve(
        InlinePreprocessPlanRequest(
            spec=legacy_spec,
            model_artifact_identity=_artifact(),
            resolved_model_manifest=legacy_manifest,
            requirements=_requirements(legacy_spec),
        )
    )
    intermediate_config = intermediate_resolution.plan
    flat_config = flat_resolution.plan
    legacy = legacy_resolution.plan

    assert current.plan_id == intermediate_config.plan_id == flat_config.plan_id
    assert legacy.plan_id != current.plan_id
    assert current.components[0].config_sha256 == (
        intermediate_config.components[0].config_sha256
    )
    assert current.components[0].config_sha256 == (
        flat_config.components[0].config_sha256
    )
    assert legacy.components[0].config_sha256 != (current.components[0].config_sha256)
    receipts = (
        current_resolution.compatibility_receipt,
        intermediate_resolution.compatibility_receipt,
        flat_resolution.compatibility_receipt,
        legacy_resolution.compatibility_receipt,
    )
    assert all(receipt is not None for receipt in receipts)
    assert len({receipt.receipt_id for receipt in receipts if receipt is not None}) == 4
    assert intermediate_config.port.contract_id == current.port.contract_id
    assert flat_config.port.contract_id == current.port.contract_id
    assert legacy.port.contract_id != current.port.contract_id
    assert legacy.components[0].implementation_id != (
        current.components[0].implementation_id
    )

    with pytest.raises(PreprocessContractError, match="class_path"):
        InlinePreprocessPlanFactory().create(
            InlinePreprocessPlanRequest(
                spec=legacy_spec,
                model_artifact_identity=_artifact(),
                resolved_model_manifest=current_manifest,
            )
        )


def test_wan_cfg_semantics_and_frame_rate_change_identity(tmp_path) -> None:
    adapter, manifest = _wan(tmp_path)
    baseline = _plan(adapter, manifest)
    no_cfg_config = replace(adapter.config, guidance_scale=1.0)
    no_cfg = WanT2VAdapter(
        no_cfg_config,
        artifact_path=tmp_path,
        precision=ComputePrecision.FP32,
        model_loader=lambda *_args: None,
    )
    no_cfg_manifest = _manifest(
        alias="wan-t2v",
        class_path="visual_rl.models.implementations.wan:WanT2VAdapter",
        config_type_path="visual_rl.models.catalog:WanConfig",
        config=no_cfg_config,
        declared_contract=no_cfg_config.describe_contract(),
    )
    at_24_fps, at_24_fps_manifest = _wan(
        tmp_path,
        frame_rate_numerator=24,
    )

    assert (
        no_cfg.describe_preprocess().preprocess_config["do_classifier_free_guidance"]
        is False
    )
    assert _plan(no_cfg, no_cfg_manifest).plan_id != baseline.plan_id
    assert _plan(at_24_fps, at_24_fps_manifest).plan_id != baseline.plan_id


def test_factory_rejects_dependency_payload_and_implementation_drift(tmp_path) -> None:
    adapter, manifest = _sd3(tmp_path)
    spec = adapter.describe_preprocess()
    wrong_dependency = replace(
        spec,
        port=PreprocessPortContract(
            port_id=spec.port.port_id,
            output_payload_type=spec.port.output_payload_type,
            dependencies=(
                PreprocessDependency(
                    role=PreprocessComponentRole.TEXT_ENCODER,
                    logical_name="text_encoder",
                ),
            ),
        ),
    )
    with pytest.raises(PreprocessContractError, match="model_artifact"):
        InlinePreprocessPlanFactory().create(
            InlinePreprocessPlanRequest(
                spec=wrong_dependency,
                model_artifact_identity=_artifact(),
                resolved_model_manifest=manifest,
            )
        )

    wrong_payload = replace(
        spec,
        port=replace(spec.port, output_payload_type="wrong.v1"),
    )
    with pytest.raises(PreprocessContractError, match="output payload"):
        InlinePreprocessPlanFactory().create(
            InlinePreprocessPlanRequest(
                spec=wrong_payload,
                model_artifact_identity=_artifact(),
                resolved_model_manifest=manifest,
            )
        )

    wrong_implementation = replace(spec, implementation_id="plugin:OtherAdapter")
    with pytest.raises(PreprocessContractError, match="class_path"):
        InlinePreprocessPlanFactory().create(
            InlinePreprocessPlanRequest(
                spec=wrong_implementation,
                model_artifact_identity=_artifact(),
                resolved_model_manifest=manifest,
            )
        )


@pytest.mark.parametrize(
    "forbidden",
    (
        {"absolute_path": "/tmp/model"},
        {"root": Path("/tmp/model")},
        {"runtime": {"device": "cuda:0"}},
        {"recipe_id": "recipe"},
        {"algorithm_id": "grpo"},
        {"rollout": "tempflow"},
        {"row": {"group_id": "group-0"}},
        {"seed": 7},
    ),
)
def test_artifact_identity_rejects_locations_runtime_recipe_and_rows(
    tmp_path,
    forbidden,
) -> None:
    adapter, manifest = _sd3(tmp_path)
    artifact = FrozenMapping(
        {
            "content_sha256": _hash("model"),
            **forbidden,
        }
    )
    with pytest.raises(PreprocessContractError, match="cannot enter"):
        _plan(adapter, manifest, artifact=artifact)


def test_factory_rejects_weak_artifacts_and_manifest_schema_drift(tmp_path) -> None:
    adapter, manifest = _sd3(tmp_path)
    with pytest.raises(PreprocessContractError, match="immutable"):
        _plan(
            adapter,
            manifest,
            artifact=FrozenMapping({"revision": "main"}),
        )

    payload = to_plain_dict(manifest)
    payload["recipe_id"] = _hash("recipe")
    with pytest.raises(PreprocessContractError, match="invalid schema"):
        _plan(adapter, FrozenMapping(payload))

    payload = to_plain_dict(manifest)
    payload["config"]["path"] = "/tmp/model"
    with pytest.raises(PreprocessContractError, match="cannot enter"):
        _plan(adapter, FrozenMapping(payload))


def test_base_adapter_and_factory_fail_closed_without_typed_metadata(tmp_path) -> None:
    class MissingMetadataAdapter(ModelAdapter):
        @classmethod
        def describe(cls, config):
            del config
            raise NotImplementedError

        @classmethod
        def from_config(cls, config, *, runtime_context):
            del config, runtime_context
            return cls()

        def load_components(self, session):
            del session
            raise NotImplementedError

        def encode(self, batch):
            del batch
            raise NotImplementedError

        def prepare_latents(self, latent_spec, *, generator):
            del latent_spec, generator
            raise NotImplementedError

        def predict(self, model_input):
            del model_input
            raise NotImplementedError

        def decode(self, latents, latent_spec):
            del latents, latent_spec
            raise NotImplementedError

    with pytest.raises(ModelPortError, match="describe_preprocess"):
        MissingMetadataAdapter().describe_preprocess()
    with pytest.raises(TypeError, match="FrozenMapping"):
        InlinePreprocessPlanRequest(
            spec=_sd3(tmp_path)[0].describe_preprocess(),
            model_artifact_identity={"content_sha256": _hash("model")},  # type: ignore[arg-type]
            resolved_model_manifest=_sd3(tmp_path)[1],
        )
