"""Model-owned rollout/Dynamics numerics independent of compute precision."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.contracts import ComputePrecision
from visual_rl.models import (
    MODEL_RUNTIME_NUMERICS_SCHEMA_VERSION,
    ModelAdapter,
    ModelPortError,
    ModelRuntimeNumerics,
)
from visual_rl.models.implementations.sd3 import SD3Adapter, SD3Config
from visual_rl.models.implementations.wan import WanConfig, WanT2VAdapter


def _spec(dtype: str = "float32") -> ModelRuntimeNumerics:
    return ModelRuntimeNumerics(
        rollout_latent_dtype=dtype,
        transition_latent_dtype=dtype,
    )


def test_runtime_numerics_has_canonical_payload_identity_and_safe_conversion() -> None:
    spec = _spec()
    payload = {
        "schema_version": MODEL_RUNTIME_NUMERICS_SCHEMA_VERSION,
        "kind": "model_runtime_numerics",
        "rollout_latent_dtype": "float32",
        "transition_latent_dtype": "float32",
    }

    assert spec.to_payload() == payload
    assert (
        spec.runtime_numerics_id
        == hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()
    )
    assert spec.runtime_numerics_id == _spec().runtime_numerics_id
    assert spec.runtime_numerics_id != _spec("float16").runtime_numerics_id
    assert spec.rollout_torch_dtype is torch.float32
    assert spec.transition_torch_dtype is torch.float32
    assert _spec("bfloat16").rollout_torch_dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    (
        ("rollout_latent_dtype", "fp32", ValueError),
        ("rollout_latent_dtype", "torch.float32", ValueError),
        ("rollout_latent_dtype", torch.float32, TypeError),
        ("transition_latent_dtype", "FP32", ValueError),
        ("transition_latent_dtype", 32, TypeError),
        ("schema_version", 2, ValueError),
        ("schema_version", True, ValueError),
    ),
)
def test_runtime_numerics_rejects_aliases_objects_and_schema_drift(
    field_name,
    value,
    error_type,
) -> None:
    with pytest.raises(error_type):
        replace(_spec(), **{field_name: value})


def test_runtime_numerics_v1_rejects_implicit_boundary_casts() -> None:
    with pytest.raises(ValueError, match="rollout and transition"):
        ModelRuntimeNumerics(
            rollout_latent_dtype="float16",
            transition_latent_dtype="float32",
        )


@pytest.mark.parametrize(
    "adapter",
    (
        pytest.param(
            lambda path: SD3Adapter(
                SD3Config(artifact_ref="model", resolution=24),
                artifact_path=path,
                precision=ComputePrecision.BF16,
                model_loader=None,
            ),
            id="sd3",
        ),
        pytest.param(
            lambda path: WanT2VAdapter(
                WanConfig(
                    artifact_ref="model",
                    height=24,
                    width=32,
                    frames=9,
                ),
                artifact_path=path,
                precision=ComputePrecision.BF16,
                model_loader=None,
            ),
            id="wan",
        ),
    ),
)
def test_concrete_adapters_keep_bf16_compute_and_fp32_transition_independent(
    tmp_path,
    adapter,
) -> None:
    instance = adapter(tmp_path)
    numerics = instance.describe_runtime_numerics()

    assert instance.precision is ComputePrecision.BF16
    assert instance.describe_preprocess().preprocess_config["embedding_dtype"] == (
        "bf16"
    )
    assert numerics.rollout_latent_dtype == "float32"
    assert numerics.transition_latent_dtype == "float32"
    assert numerics.rollout_torch_dtype is torch.float32
    assert numerics.transition_torch_dtype is torch.float32


def test_base_adapter_runtime_numerics_port_fails_closed() -> None:
    with pytest.raises(ModelPortError, match="describe_runtime_numerics"):
        ModelAdapter.describe_runtime_numerics(object())
