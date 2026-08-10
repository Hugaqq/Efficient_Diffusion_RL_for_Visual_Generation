"""Immutable model-owned rollout and Dynamics latent numerics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from visual_rl.core.serialization import canonical_json_text

__all__ = (
    "MODEL_RUNTIME_NUMERICS_SCHEMA_VERSION",
    "ModelRuntimeNumerics",
)


MODEL_RUNTIME_NUMERICS_SCHEMA_VERSION = 1
_KIND = "model_runtime_numerics"
_FLOATING_DTYPE_NAMES = frozenset(
    {
        "bfloat16",
        "float16",
        "float32",
        "float64",
    }
)


def _dtype_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical dtype name")
    if value not in _FLOATING_DTYPE_NAMES:
        raise ValueError(f"{field_name} must be one of {sorted(_FLOATING_DTYPE_NAMES)}")
    return value


def _torch_dtype(value: str) -> Any:
    """Resolve one validated name without eval/getattr or alias guessing."""

    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[value]
    except KeyError as exc:  # Defensive: construction already validates the name.
        raise ValueError(f"unsupported runtime latent dtype: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ModelRuntimeNumerics:
    """Durable latent dtype contract, independent of model compute precision.

    The v1 rollout path stores and transitions the same latent tensor, so the
    two declared dtypes must agree.  They remain separate payload fields to
    make the model/Dynamics boundary explicit and permit a future schema to
    introduce an owned conversion policy instead of an implicit cast.
    """

    rollout_latent_dtype: str
    transition_latent_dtype: str
    schema_version: int = MODEL_RUNTIME_NUMERICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != MODEL_RUNTIME_NUMERICS_SCHEMA_VERSION
        ):
            raise ValueError("unsupported model runtime numerics schema_version")
        rollout = _dtype_name(
            self.rollout_latent_dtype,
            field_name="rollout_latent_dtype",
        )
        transition = _dtype_name(
            self.transition_latent_dtype,
            field_name="transition_latent_dtype",
        )
        if rollout != transition:
            raise ValueError(
                "model runtime numerics v1 requires rollout and transition "
                "latent dtypes to match"
            )

    @property
    def rollout_torch_dtype(self) -> Any:
        return _torch_dtype(self.rollout_latent_dtype)

    @property
    def transition_torch_dtype(self) -> Any:
        return _torch_dtype(self.transition_latent_dtype)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": _KIND,
            "rollout_latent_dtype": self.rollout_latent_dtype,
            "transition_latent_dtype": self.transition_latent_dtype,
        }

    @property
    def runtime_numerics_id(self) -> str:
        return hashlib.sha256(
            canonical_json_text(self.to_payload()).encode("utf-8")
        ).hexdigest()
