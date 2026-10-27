"""Registered H3 ModelAdapter, composing existing narrow model-domain helpers.

This is not a controller/binder: runtime must prepare meta with H3 FSDP2,
then bind its existing PreparedComponentHandle. No PG/optimizer is created
here. Generic Gaussian scheduler/G3 ports intentionally remain undeclared.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from visual_rl.core.contracts import LatentLayout, PredictionType
from visual_rl.models.h3_config import H3Config
from visual_rl.models.h3_interface import (
    H3Conditioning, H3Generators, H3ModelInput, H3PromptInput, H3_INITIAL_STATE_VERSION,
)
from visual_rl.models.interface import ModelAdapter, ModelLatentSpec, ModelPortError, ModelPrediction
from visual_rl.models.lifecycle.components import ComponentLoadSession, ComponentRole

if TYPE_CHECKING:
    from visual_rl.models.implementations.h3_decode import H3AVMedia
    from visual_rl.models.implementations.h3_loading import H3MetaModel


class _H3Aux:
    """One idempotent non-prepared owner; cache/CPU decoder are acquired lazily."""
    def __init__(self, config, model_path, cache_path, expected_identity):
        self.config, self.model_path, self.cache_path = config, model_path, cache_path
        self.expected_identity = MappingProxyType(dict(expected_identity))
        self._cache = self._decoder = None
        self.closed = False

    def require_open(self):
        if self.closed:
            raise ModelPortError("H3 auxiliary resources are closed")

    @property
    def cache(self):
        self.require_open()
        if self._cache is None:
            from visual_rl.models.implementations.h3_text_cache import H3TextCache
            self._cache = H3TextCache(self.cache_path, expected_identity=self.expected_identity,
                max_tokens=self.config.max_tokens, max_embedding_bytes=self.config.max_embedding_bytes)
        return self._cache

    @property
    def decoder(self):
        self.require_open()
        if self._decoder is None:
            from visual_rl.models.implementations.h3_decode import H3CPUDecoder
            self._decoder = H3CPUDecoder(self.model_path, max_media_bytes=self.config.max_media_bytes,
                                         max_decode_bytes=self.config.max_decode_bytes)
        return self._decoder

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._cache = self._decoder = None  # helpers own no persistent file/model handles


class H3Adapter(ModelAdapter):
    initial_state_version = H3_INITIAL_STATE_VERSION

    def __init__(self, config: H3Config, *, model_path: Path, cache_path: Path, expected_cache_identity: Mapping):
        self.config = config
        self.geometry = config.geometry
        self._aux = _H3Aux(config, model_path, cache_path, expected_cache_identity)
        self._meta = None
        self._load_attempted = False

    @classmethod
    def describe(cls, config):
        if not isinstance(config, H3Config):
            raise TypeError("config must be H3Config")
        return config.describe_contract()

    @classmethod
    def from_config(cls, config, *, runtime_context):
        cls.describe(config)
        if not isinstance(runtime_context, Mapping) or set(runtime_context) != {"artifact_paths", "expected_cache_identity"}:
            raise ValueError("H3 runtime_context requires exactly artifact_paths and expected_cache_identity")
        paths = runtime_context["artifact_paths"]
        expected = runtime_context["expected_cache_identity"]
        if not isinstance(paths, Mapping) or set(paths) != {config.artifact_ref, config.text_cache_ref}:
            raise ValueError("H3 artifact_paths must bind exactly the model and text cache refs")
        if not isinstance(expected, Mapping) or not expected:
            raise ValueError("H3 requires explicit expected cache identity")
        resolved = {}
        for ref, path in paths.items():
            if not isinstance(path, (str, Path)) or not Path(path).expanduser().is_dir():
                raise ValueError(f"H3 requires an existing local artifact directory for {ref}")
            resolved[ref] = Path(path).expanduser().resolve()
        return cls(config, model_path=resolved[config.artifact_ref], cache_path=resolved[config.text_cache_ref],
                   expected_cache_identity=expected)

    def _require_loaded(self):
        self._assert_open()
        if self._meta is None or self._aux is None:
            raise ModelPortError("H3 components have not been loaded")
        self._aux.require_open()

    @property
    def meta_model(self) -> H3MetaModel:
        """Loader result for future runtime prepare, never an alternate forward."""
        self._require_loaded()
        return self._meta

    def load_components(self, session):
        self._assert_open()
        if not isinstance(session, ComponentLoadSession):
            raise TypeError("session must be ComponentLoadSession")
        if self._load_attempted:
            raise ModelPortError("H3 load permits one attempt")
        self._load_attempted = True
        from visual_rl.models.implementations.h3_loading import build_h3_meta
        meta = build_h3_meta(self._aux.model_path, rank=self.config.lora_rank, alpha=self.config.lora_alpha)
        if meta.model.config.get("text_dim") != 5120:
            raise ModelPortError("H3 transformer must consume 5120-wide cache embeddings")
        session.register("transformer", meta.model, roles=(ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
                         managed_residency=False)  # meta belongs to future FSDP prepare, not .to(cpu)
        session.register("h3_aux", self._aux, roles=(ComponentRole.PREPROCESS, ComponentRole.DECODER),
                         managed_residency=False)
        components = session.freeze()
        self._meta = meta
        return components

    def verify_coverage(self, train_prompts, eval_prompts):
        self._require_loaded()
        self._aux.cache.verify_coverage(train_prompts, eval_prompts)

    def encode(self, batch: H3PromptInput) -> H3Conditioning:
        self._require_loaded()
        if not isinstance(batch, H3PromptInput):
            raise TypeError("H3 encode requires H3PromptInput, not a T2V batch")
        cache = self._aux.cache
        return H3Conditioning(batch.prompt, cache.identity_sha256, cache.get(batch.prompt))

    def latent_spec_for_batch(self, batch: H3PromptInput, *, device, dtype) -> ModelLatentSpec:
        import torch
        self._assert_open()
        if not isinstance(batch, H3PromptInput) or dtype != torch.float16 or torch.device(device).type not in ("cpu", "cuda"):
            raise ModelPortError("H3 latent spec requires B1 prompt, FP16, CPU/CUDA")
        return ModelLatentSpec(shape=(1, self.geometry.joint_layout.packed_width), layout=LatentLayout.JOINT_AV_FLAT,
                               axis_semantics=("batch", "joint"), device=device, dtype=dtype, spatial_stride=(16, 16))

    def _validate_spec(self, spec):
        import torch
        if (not isinstance(spec, ModelLatentSpec) or spec.layout is not LatentLayout.JOINT_AV_FLAT
                or spec.shape != (1, self.geometry.joint_layout.packed_width) or spec.dtype != torch.float16
                or spec.device.type not in ("cpu", "cuda")):
            raise ModelPortError("H3 latent spec differs from configured joint geometry/FP16/B1")

    def prepare_latents(self, latent_spec: ModelLatentSpec, *, generator: H3Generators):
        import torch
        self._require_loaded()
        self._validate_spec(latent_spec)
        if not isinstance(generator, H3Generators):
            raise TypeError("H3 prepare_latents requires H3Generators(video,audio)")
        if any(g.device != latent_spec.device for g in (generator.video, generator.audio)):
            raise ModelPortError("H3 generator device must match latent spec")
        layout = self.geometry.joint_layout
        video = torch.randn((1, layout.video_rows, layout.video_channels), dtype=torch.float32,
                            device=latent_spec.device, generator=generator.video).half()
        audio = torch.randn((1, layout.audio_rows, layout.audio_channels), dtype=torch.float32,
                            device=latent_spec.device, generator=generator.audio).half()
        return layout.pack(video, audio)

    def _prepared_forward(self, **kwargs):
        # Deliberately no root autocast: the H3 projections own FP32 alignment.
        return self._forward_prepared("transformer", **kwargs)

    def predict(self, model_input: H3ModelInput) -> ModelPrediction:
        self._require_loaded()
        if not isinstance(model_input, H3ModelInput):
            raise TypeError("H3 predict requires independent-sigma H3ModelInput")
        self._validate_spec(model_input.latent_spec)
        if model_input.conditioning.cache_identity != self._aux.cache.identity_sha256:
            raise ModelPortError("H3 conditioning cache identity differs from bound cache")
        self.prepared_components  # fail before any model call if not prepared
        from visual_rl.models.implementations.h3_forward import h3_t2va_forward
        value = h3_t2va_forward(self._prepared_forward, x=model_input.latents, geometry=self.geometry,
            text_embeddings=model_input.conditioning.embeddings.to(model_input.latents.device),
            video_sigma=model_input.timestep, audio_sigma=model_input.audio_sigma)
        result = ModelPrediction(value, PredictionType.FLOW, model_input.condition_identity, model_input.guidance_identity)
        result.validate_against(model_input)
        return result

    def decode(self, latents, latent_spec: ModelLatentSpec) -> AbstractContextManager[H3AVMedia]:
        self._require_loaded()
        self._validate_spec(latent_spec)
        if latent_spec.device.type != "cpu":
            raise ModelPortError("H3 decode requires a CPU terminal/spec")
        return self._decode_lease(latents)

    @contextmanager
    def _decode_lease(self, latents):
        self._require_loaded()  # A queued lease must not bypass adapter.close().
        with self._aux.decoder.decode_one(latents, self.geometry) as media:
            yield media

    def _release_runtime_parts(self):
        aux, self._aux = self._aux, None
        self._meta = None
        if aux is not None:
            aux.close()
