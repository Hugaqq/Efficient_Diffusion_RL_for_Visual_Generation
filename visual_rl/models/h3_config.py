"""Import-safe, closed H3 T2VA declaration; not a generic Gaussian recipe."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import re

from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI, ComponentDeclaration, ComputePrecision, DeclaredContract,
    LatentLayout, MediaKind, ModelContract, PredictionType, TaskKind, TimeCoordinate, TrainingMode,
)
from visual_rl.models.catalog import positive_int, strict_values
from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry


@dataclass(frozen=True)
class H3Config:
    artifact_ref: str
    text_cache_ref: str
    max_tokens: int
    max_embedding_bytes: int
    max_media_bytes: int
    max_decode_bytes: int
    height: int = 256
    width: int = 384
    frames: int = 124
    fps: int = 24
    guidance_scale: float = 1.0
    lora_rank: int = 32
    lora_alpha: int = 64

    def __post_init__(self):
        for key in ("artifact_ref", "text_cache_ref"):
            value = getattr(self, key)
            if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", value) is None:
                raise ValueError(f"{key} must be a logical artifact reference, not a path")
        if self.artifact_ref == self.text_cache_ref:
            raise ValueError("model and text cache require distinct artifact refs")
        for key in ("max_tokens", "max_embedding_bytes", "max_media_bytes", "max_decode_bytes", "lora_rank", "lora_alpha"):
            positive_int(key, getattr(self, key))
        self.geometry  # Validate real family geometry without constructing tensors.

    @property
    def geometry(self) -> H3T2VAGeometry:
        return H3T2VAGeometry(self.height, self.width, self.frames, self.fps, guidance=self.guidance_scale)

    @classmethod
    def from_mapping(cls, values: Mapping, *, context=None):
        del context
        return cls(**strict_values(values, allowed=frozenset(f.name for f in fields(cls)),
            required=frozenset(("artifact_ref", "text_cache_ref", "max_tokens", "max_embedding_bytes",
                                "max_media_bytes", "max_decode_bytes")), label="h3-t2va"))

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(component_kind="model", component_id="h3-t2va", model=ModelContract(
            tasks=(TaskKind.T2VA,), output_media=(MediaKind.VIDEO, MediaKind.AUDIO),
            latent_layouts=(LatentLayout.JOINT_AV_FLAT,), latent_ranks=(2,), axis_semantics=(("batch", "joint"),),
            prediction_types=(PredictionType.FLOW,), time_coordinates=(TimeCoordinate.SIGMA,),
            training_modes=(TrainingMode.LORA,), supported_precisions=(ComputePrecision.BF16,),
            provides_reference_policy=False, condition_payload_types=("h3.text-cache.v1",),
            spatial_stride=(16, 16), packer_identity="h3.generated-av-flat.v1",
            # No fabricated scheduler blueprint/replay schema. H3 dual-sigma CPS
            # binding and mixed precision G3 evidence belong to the next binder.
        ))


class H3DeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.models.h3_config:H3Config"

    @classmethod
    def declare_component(cls, raw_params, *, context=None):
        config = H3Config.from_mapping(raw_params, context=context)
        return ComponentDeclaration(config=config, declared_contract=config.describe_contract())
