"""Single-forward Wan T2V adapter with algorithm-neutral model semantics."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from visual_rl.core.contracts import (
    ComputePrecision,
    DeclaredContract,
    LatentLayout,
    PredictionType,
)
from visual_rl.core.types import FrozenMapping
from visual_rl.data.media import DecodedMediaBatch
from visual_rl.data.preprocess import (
    PreprocessComponentRole,
    PreprocessDependency,
    PreprocessGeometry,
    PreprocessPortContract,
    PreprocessProducerSpec,
)
from visual_rl.models.catalog import (
    WAN_TEMPORAL_STRIDE,
    WanConfig,
    positive_int,
)
from visual_rl.models.implementations.common import (
    local_model_artifact,
    runtime_model_loader,
    runtime_precision,
)
from visual_rl.models.interface import (
    BatchRowProjection,
    ModelAdapter,
    ModelInput,
    ModelLatentSpec,
    ModelPortError,
    ModelPrediction,
)
from visual_rl.models.lifecycle.components import (
    ComponentLoadSession,
    ComponentRole,
    ModelComponents,
)
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.models.numerics.policy import (
    ParameterViewEvidence,
    ParameterViewMode,
)
from visual_rl.models.numerics.runtime import ModelRuntimeNumerics
from visual_rl.models.preprocessing import ModelPreprocessConsumerSpec
from visual_rl.models.scheduler import SchedulerArtifactBlueprint
from visual_rl.models.state.parameters import ParameterStateManager

if TYPE_CHECKING:
    from visual_rl.data.samples import StackedSampleBatch

__all__ = (
    "WanConditioning",
    "WanConfig",
    "WanRuntimeParts",
    "WanT2VAdapter",
)


@dataclass(frozen=True, slots=True)
class WanConditioning:
    prompt_embeds: Any
    negative_prompt_embeds: Any | None
    condition_identity: tuple[str, ...]

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.prompt_embeds, torch.Tensor):
            raise TypeError("Wan prompt_embeds must be a tensor")
        if not self.prompt_embeds.is_floating_point():
            raise TypeError("Wan prompt_embeds must be floating point")
        batch_size = int(self.prompt_embeds.shape[0])
        negative = self.negative_prompt_embeds
        if negative is not None:
            if not isinstance(negative, torch.Tensor):
                raise TypeError("Wan negative_prompt_embeds must be a tensor")
            if negative.shape[0] != batch_size:
                raise ModelPortError("Wan negative embeddings batch mismatch")
            if (
                negative.device != self.prompt_embeds.device
                or negative.dtype != self.prompt_embeds.dtype
            ):
                raise ModelPortError(
                    "Wan positive and negative embeddings must share device/dtype"
                )
        if (
            type(self.condition_identity) is not tuple
            or len(self.condition_identity) != batch_size
        ):
            raise ModelPortError(
                "Wan condition_identity must contain one value per row"
            )
        if any(
            not isinstance(identity, str) or not identity
            for identity in self.condition_identity
        ):
            raise ModelPortError(
                "Wan condition_identity values must be non-empty strings"
            )

    @property
    def batch_size(self) -> int:
        return len(self.condition_identity)

    def project_rows(self, projection: BatchRowProjection) -> WanConditioning:
        """Project or repeat encoded rows without exposing Wan field names."""

        if not isinstance(projection, BatchRowProjection):
            raise TypeError("Wan projection must be a BatchRowProjection")
        if projection.source_batch_size != self.batch_size:
            raise ModelPortError(
                "Wan projection source_batch_size does not match conditioning"
            )
        import torch

        index = torch.tensor(
            projection.row_indices,
            dtype=torch.int64,
            device=self.prompt_embeds.device,
        )
        negative = self.negative_prompt_embeds
        return WanConditioning(
            prompt_embeds=self.prompt_embeds.index_select(0, index),
            negative_prompt_embeds=(
                None if negative is None else negative.index_select(0, index)
            ),
            condition_identity=projection.project_tuple(self.condition_identity),
        )


@dataclass(frozen=True, slots=True)
class WanRuntimeParts:
    prompt_encoder: object
    transformer: object
    decoder: object
    latent_channels: int
    scheduler_artifact_blueprint: SchedulerArtifactBlueprint
    expand_timesteps: bool = False

    def __post_init__(self) -> None:
        for name, value, methods in (
            ("prompt_encoder", self.prompt_encoder, ("encode", "to")),
            ("decoder", self.decoder, ("decode", "to")),
        ):
            if any(not callable(getattr(value, method, None)) for method in methods):
                raise TypeError(f"Wan {name} must expose {methods}")
        if type(self.expand_timesteps) is not bool:
            raise TypeError("expand_timesteps must be bool")
        object.__setattr__(
            self,
            "latent_channels",
            positive_int("latent_channels", self.latent_channels),
        )
        if not isinstance(
            self.scheduler_artifact_blueprint,
            SchedulerArtifactBlueprint,
        ):
            raise TypeError(
                "Wan scheduler_artifact_blueprint must use the model-owned ABI"
            )


class WanT2VAdapter(ModelAdapter):
    """One Wan model port with no rollout-strategy branching."""

    CONFIG_TYPE = "visual_rl.models.catalog:WanConfig"

    def __init__(
        self,
        config: WanConfig,
        *,
        artifact_path: Path,
        precision: ComputePrecision,
        model_loader: object | None,
    ) -> None:
        if not isinstance(config, WanConfig):
            raise TypeError("config must be WanConfig")
        if not isinstance(artifact_path, Path) or not artifact_path.is_dir():
            raise ValueError("artifact_path must be an existing local directory")
        if not isinstance(precision, ComputePrecision):
            raise TypeError("precision must be ComputePrecision")
        if model_loader is not None and not callable(model_loader):
            raise TypeError("model_loader must be callable")
        self.config = config
        self.artifact_path = artifact_path
        self.precision = precision
        self._model_loader = model_loader
        self._prompt_encoder: object | None = None
        self._transformer: object | None = None
        self._decoder: object | None = None
        self._expand_timesteps = False
        self._latent_channels: int | None = None
        self._scheduler_artifact_blueprint: SchedulerArtifactBlueprint | None = None

    @classmethod
    def describe(cls, config: object) -> DeclaredContract:
        if not isinstance(config, WanConfig):
            raise TypeError("config must be WanConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> WanT2VAdapter:
        if not isinstance(config, WanConfig):
            raise TypeError("config must be WanConfig")
        precision = runtime_precision(runtime_context)
        if precision not in config.describe_contract().model.supported_precisions:
            raise ValueError(f"Wan does not support precision {precision.value!r}")
        return cls(
            config,
            artifact_path=local_model_artifact(
                runtime_context,
                config.artifact_ref,
            ),
            precision=precision,
            model_loader=runtime_model_loader(runtime_context),
        )

    def describe_preprocess(self) -> PreprocessProducerSpec:
        uses_negative_condition = self.config.guidance_scale > 1.0
        return PreprocessProducerSpec(
            implementation_id="visual_rl.models.implementations.wan:WanT2VAdapter",
            implementation_revision="wan-prompt-encode.v1",
            port=PreprocessPortContract(
                port_id="visual_rl.models.implementations.wan:prompt-encode.v1",
                output_payload_type="wan_prompt_embeddings.v1",
                dependencies=(
                    PreprocessDependency(
                        role=PreprocessComponentRole.MODEL,
                        logical_name="model_artifact",
                    ),
                ),
                producer_output_fields=(
                    ("prompt_embeds", "negative_prompt_embeds")
                    if uses_negative_condition
                    else ("prompt_embeds",)
                ),
                negative_condition_fields=(
                    ("negative_prompt_embeds",) if uses_negative_condition else ()
                ),
                schema_version=2,
            ),
            geometry=PreprocessGeometry(
                height=self.config.height,
                width=self.config.width,
                aspect_ratio_bucket=f"{self.config.height}x{self.config.width}",
                frame_count=self.config.frames,
                frame_rate_numerator=self.config.frame_rate_numerator,
                frame_rate_denominator=self.config.frame_rate_denominator,
            ),
            transforms=(),
            preprocess_config=FrozenMapping(
                {
                    "do_classifier_free_guidance": uses_negative_condition,
                    "embedding_dtype": self.precision.value,
                    "max_sequence_length": self.config.max_sequence_length,
                    "negative_prompt_policy": "empty_string",
                    "outputs_per_prompt": 1,
                }
            ),
        )

    def describe_preprocess_consumption(self) -> ModelPreprocessConsumerSpec:
        uses_negative_condition = self.config.guidance_scale > 1.0
        return ModelPreprocessConsumerSpec(
            implementation_revision="wan-conditioning-consumer.v1",
            payload_type="wan_prompt_embeddings.v1",
            required_modalities=("prompt_text",),
            positive_output_fields=("prompt_embeds",),
            negative_output_fields=("negative_prompt_embeds",),
            uses_negative_condition=uses_negative_condition,
        )

    def describe_runtime_numerics(self) -> ModelRuntimeNumerics:
        return ModelRuntimeNumerics(
            rollout_latent_dtype="float32",
            transition_latent_dtype="float32",
        )

    def describe_parameter_view_evidence(
        self,
        parameter_state: ParameterStateManager,
        *,
        distribution_mode: str,
    ) -> tuple[ParameterViewEvidence, ...]:
        self._assert_open()
        if not isinstance(parameter_state, ParameterStateManager):
            raise TypeError("parameter_state must be ParameterStateManager")
        if distribution_mode != "single":
            raise ModelPortError(
                "Wan LoRA parameter views are verified only for single process"
            )
        projection = parameter_state.state_projection
        owners = projection.standalone_saved_component_names
        states = projection.standalone_parameter_names
        if owners != ("transformer",) or not states:
            raise ModelPortError(
                "Wan LoRA state projection must be owned only by transformer"
            )
        return (
            ParameterViewEvidence(
                parameter_view=ParameterView.CURRENT,
                mode=ParameterViewMode.CURRENT,
                owner_component_names=owners,
                restorable_state_names=states,
                source_projection_id=projection.projection_id,
                mutates_parameters_in_place=False,
            ),
        )

    def load_components(self, session: ComponentLoadSession) -> ModelComponents:
        self._assert_open()
        if not isinstance(session, ComponentLoadSession):
            raise TypeError("session must be ComponentLoadSession")
        parts = self._load_runtime_parts()
        import torch

        if not isinstance(parts.transformer, torch.nn.Module):
            raise TypeError("Wan transformer must be torch.nn.Module")
        self._prompt_encoder = parts.prompt_encoder
        self._transformer = parts.transformer
        self._decoder = parts.decoder
        self._expand_timesteps = parts.expand_timesteps
        self._latent_channels = parts.latent_channels
        self._scheduler_artifact_blueprint = parts.scheduler_artifact_blueprint
        session.register(
            "prompt_encoder",
            parts.prompt_encoder,
            roles=(ComponentRole.PREPROCESS,),
            # Wan's frozen text encoder is substantially larger than the
            # trainable transformer.  Keep its weights on CPU while the
            # prepared transformer owns the execution device; only the small
            # encoded conditioning tensors cross the model port below.
            managed_residency=False,
        )
        session.register(
            "transformer",
            parts.transformer,
            roles=(ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
        )
        session.register(
            "decoder",
            parts.decoder,
            roles=(ComponentRole.DECODER,),
        )
        return session.freeze()

    @property
    def scheduler_artifact_blueprint(self) -> SchedulerArtifactBlueprint:
        self._assert_open()
        blueprint = self._scheduler_artifact_blueprint
        if blueprint is None:
            raise ModelPortError(
                "Wan components must be loaded before scheduler binding"
            )
        return blueprint

    def latent_spec_for_batch(
        self,
        batch: StackedSampleBatch,
        *,
        device: Any,
        dtype: Any,
    ) -> ModelLatentSpec:
        self._assert_open()
        from visual_rl.data.samples import StackedSampleBatch

        if not isinstance(batch, StackedSampleBatch):
            raise TypeError("Wan latent spec requires a StackedSampleBatch")
        batch.validate()
        if batch.task_type != "t2v":
            raise ModelPortError("Wan latent spec requires a T2V sample batch")
        channels = self._latent_channels
        if channels is None:
            raise ModelPortError(
                "Wan components must be loaded before latent geometry is available"
            )
        latent_frames = (self.config.frames - 1) // WAN_TEMPORAL_STRIDE + 1
        return ModelLatentSpec(
            shape=(
                batch.batch_size,
                channels,
                latent_frames,
                self.config.height // 8,
                self.config.width // 8,
            ),
            layout=LatentLayout.BCTHW,
            axis_semantics=("batch", "channel", "time", "height", "width"),
            device=device,
            dtype=dtype,
            spatial_stride=(8, 8),
            temporal_stride=WAN_TEMPORAL_STRIDE,
        )

    def encode(self, batch: object) -> WanConditioning:
        self._assert_open()
        from visual_rl.data.samples import StackedSampleBatch

        if not isinstance(batch, StackedSampleBatch) or batch.task_type != "t2v":
            raise TypeError("Wan encode requires a T2V StackedSampleBatch")
        encoder = self._require_part("prompt_encoder", self._prompt_encoder)
        encoded = encoder.encode(
            batch.prompts,
            self.config.max_sequence_length,
            self.config.guidance_scale,
        )
        if not isinstance(encoded, tuple) or len(encoded) != 2:
            raise ModelPortError("Wan prompt encoder must return exactly two values")
        uses_negative_condition = self.config.guidance_scale > 1.0
        if uses_negative_condition and encoded[1] is None:
            raise ModelPortError(
                "Wan CFG preprocess must produce negative_prompt_embeds"
            )
        if not uses_negative_condition and encoded[1] is not None:
            raise ModelPortError(
                "Wan non-CFG preprocess must not produce negative_prompt_embeds"
            )
        return WanConditioning(
            prompt_embeds=encoded[0],
            negative_prompt_embeds=encoded[1],
            condition_identity=tuple(row.identity for row in batch.rows),
        )

    def prepare_latents(
        self,
        latent_spec: ModelLatentSpec,
        *,
        generator: Any,
    ) -> Any:
        self._assert_open()
        self._validate_latent_spec(latent_spec)
        import torch

        return torch.randn(
            latent_spec.shape,
            device=latent_spec.device,
            dtype=latent_spec.dtype,
            generator=generator,
        )

    def predict(self, model_input: ModelInput) -> ModelPrediction:
        self._assert_open()
        if not isinstance(model_input, ModelInput):
            raise TypeError("model_input must be ModelInput")
        self._validate_latent_spec(model_input.latent_spec)
        conditioning = model_input.conditioning
        if not isinstance(conditioning, WanConditioning):
            raise TypeError("Wan predict requires WanConditioning")
        if conditioning.condition_identity != model_input.condition_identity:
            raise ModelPortError("Wan conditioning identity drift")
        batch_size = model_input.latent_spec.batch_size
        timestep = model_input.timestep
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)
        if self._expand_timesteps:
            _, _, frames, height, width = model_input.latent_spec.shape
            token_count = frames * ((height + 1) // 2) * ((width + 1) // 2)
            model_timestep = timestep.reshape(batch_size, 1).expand(
                batch_size,
                token_count,
            )
        else:
            model_timestep = timestep
        hidden_states = model_input.latents.to(dtype=conditioning.prompt_embeds.dtype)
        conditioning_device = hidden_states.device
        positive_embeddings = conditioning.prompt_embeds.to(
            device=conditioning_device
        )
        transformer = self._require_part("transformer", self._transformer)
        with _cache_context(transformer, "cond"):
            conditional = _first_output_tensor(
                self._forward_prepared(
                    "transformer",
                    hidden_states=hidden_states,
                    timestep=model_timestep,
                    encoder_hidden_states=positive_embeddings,
                    attention_kwargs=None,
                    return_dict=False,
                ),
                label="Wan transformer",
            )
        prediction = conditional
        if self.config.guidance_scale > 1.0:
            negative_embeddings = conditioning.negative_prompt_embeds
            if negative_embeddings is None:
                raise ModelPortError("Wan CFG requires negative prompt embeddings")
            negative_embeddings = negative_embeddings.to(device=conditioning_device)
            with _cache_context(transformer, "uncond"):
                negative = _first_output_tensor(
                    self._forward_prepared(
                        "transformer",
                        hidden_states=hidden_states,
                        timestep=model_timestep,
                        encoder_hidden_states=negative_embeddings,
                        attention_kwargs=None,
                        return_dict=False,
                    ),
                    label="Wan transformer",
                )
            prediction = negative + self.config.guidance_scale * (
                conditional - negative
            )
        prediction = prediction.to(dtype=model_input.latents.dtype)
        result = ModelPrediction(
            value=prediction,
            prediction_type=PredictionType.FLOW,
            condition_identity=model_input.condition_identity,
            guidance_identity=model_input.guidance_identity,
        )
        result.validate_against(model_input)
        return result

    def decode(self, latents: Any, latent_spec: ModelLatentSpec) -> DecodedMediaBatch:
        self._assert_open()
        self._validate_latent_spec(latent_spec)
        decoder = self._require_part("decoder", self._decoder)
        return DecodedMediaBatch(
            tensor=decoder.decode(latents, latent_spec),
            layout="BFCHW",
        )

    def _validate_latent_spec(self, latent_spec: ModelLatentSpec) -> None:
        if not isinstance(latent_spec, ModelLatentSpec):
            raise TypeError("latent_spec must be ModelLatentSpec")
        if latent_spec.layout is not LatentLayout.BCTHW or latent_spec.rank != 5:
            raise ModelPortError("Wan requires BCTHW rank-5 latents")
        expected_geometry = (
            (self.config.frames - 1) // WAN_TEMPORAL_STRIDE + 1,
            self.config.height // 8,
            self.config.width // 8,
        )
        if latent_spec.shape[2:] != expected_geometry:
            raise ModelPortError(
                "Wan latent geometry must match configured frames/height/width: "
                f"expected={expected_geometry}, got={latent_spec.shape[2:]}"
            )

    def _load_runtime_parts(self) -> WanRuntimeParts:
        self._assert_open()
        if self._model_loader is not None:
            parts = self._model_loader(
                "wan-t2v",
                self.artifact_path,
                self.config,
                self.precision,
            )
            if not isinstance(parts, WanRuntimeParts):
                raise TypeError("model_loader must return WanRuntimeParts")
            return parts
        return _load_diffusers_wan_parts(
            self.artifact_path,
            self.config,
            self.precision,
        )

    def _release_runtime_parts(self) -> None:
        prompt_encoder = self._prompt_encoder
        decoder = self._decoder
        self._prompt_encoder = None
        self._transformer = None
        self._decoder = None
        self._expand_timesteps = False
        self._latent_channels = None
        self._scheduler_artifact_blueprint = None
        self._model_loader = None
        errors: list[BaseException] = []
        for part in (decoder, prompt_encoder):
            close = getattr(part, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
        if errors:
            primary = errors[0]
            for cleanup_error in errors[1:]:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "additional Wan runtime-part cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise primary

    @staticmethod
    def _require_part(name: str, value: object | None) -> object:
        if value is None:
            raise ModelPortError(f"Wan {name} has not been loaded")
        return value


class _WanPromptEncoder:
    def __init__(self, pipeline: object, dtype: Any) -> None:
        self.pipeline = pipeline
        self.dtype = dtype
        self.device: object = "cpu"

    def to(self, device: object) -> _WanPromptEncoder:
        if self.pipeline is None:
            raise ModelPortError("Wan prompt encoder is closed")
        for name in ("text_encoder", "text_encoder_2"):
            module = getattr(self.pipeline, name, None)
            if module is not None:
                module.to(device=device, dtype=self.dtype)
        self.device = device
        return self

    def encode(
        self,
        prompts: tuple[str, ...],
        max_sequence_length: int,
        guidance_scale: float,
    ) -> tuple[Any, Any | None]:
        if self.pipeline is None:
            raise ModelPortError("Wan prompt encoder is closed")
        encoded = self.pipeline.encode_prompt(
            prompt=list(prompts),
            negative_prompt=[""] * len(prompts),
            do_classifier_free_guidance=guidance_scale > 1.0,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=self.device,
        )
        if not isinstance(encoded, tuple) or len(encoded) < 2:
            raise ModelPortError("Wan encode_prompt returned an invalid payload")
        positive, negative = encoded[:2]
        positive = positive.to(dtype=self.dtype)
        if negative is not None:
            negative = negative.to(dtype=self.dtype)
        return positive, negative

    def close(self) -> None:
        self.pipeline = None
        self.device = "cpu"


class _WanDecoder:
    def __init__(self, vae: object, video_processor: object) -> None:
        self.vae = vae
        self.video_processor = video_processor

    def to(self, device: object) -> _WanDecoder:
        import torch

        if self.vae is None:
            raise ModelPortError("Wan decoder is closed")
        self.vae.to(device=device, dtype=torch.float32)
        return self

    def decode(self, latents: Any, latent_spec: ModelLatentSpec) -> object:
        del latent_spec
        import torch

        vae = self.vae
        video_processor = self.video_processor
        if vae is None or video_processor is None:
            raise ModelPortError("Wan decoder is closed")
        parameter = next(iter(vae.parameters()), None)
        device = parameter.device if parameter is not None else latents.device
        dtype = parameter.dtype if parameter is not None else torch.float32
        value = latents.to(device=device, dtype=dtype)
        mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(value)
        )
        inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1
        ).to(value)
        decoded = vae.decode(
            value / inverse_std + mean,
            return_dict=False,
        )[0]
        media = video_processor.postprocess_video(decoded, output_type="pt")
        if not isinstance(media, torch.Tensor):
            raise ModelPortError("Wan video processor returned a non-tensor")
        return media.detach().to(device="cpu", dtype=torch.float32).contiguous()

    def close(self) -> None:
        self.vae = None
        self.video_processor = None


def _artifact_config_int(component: object, field_name: str, *, label: str) -> int:
    config = getattr(component, "config", None)
    if config is None:
        raise ModelPortError(f"{label} must expose config.{field_name}")
    value = (
        config.get(field_name)
        if isinstance(config, Mapping)
        else getattr(config, field_name, None)
    )
    if type(value) is not int or value < 1:
        raise ModelPortError(f"{label} config.{field_name} must be a positive integer")
    return value


def _load_diffusers_wan_parts(
    artifact_path: Path,
    config: WanConfig,
    precision: ComputePrecision,
) -> WanRuntimeParts:
    try:
        from diffusers import WanPipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install visual-rl[train] to load Wan") from exc
    from visual_rl.models.implementations.common_diffusers import (
        apply_peft_lora,
        configure_gradient_checkpointing,
        resolve_torch_dtype,
        verify_gradient_checkpointing,
    )

    dtype = resolve_torch_dtype(precision.value)
    pipeline = WanPipeline.from_pretrained(
        str(artifact_path),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    scheduler_blueprint = SchedulerArtifactBlueprint.from_scheduler(pipeline.scheduler)
    vae = pipeline.vae
    if vae is None:
        raise ModelPortError("Wan pipeline must expose a VAE")
    vae.requires_grad_(False)
    vae.eval()
    if config.vae_tiling:
        enable_tiling = getattr(vae, "enable_tiling", None)
        if not callable(enable_tiling):
            raise ModelPortError("Wan vae_tiling requires VAE.enable_tiling()")
        enable_tiling()
    for name in ("text_encoder", "text_encoder_2"):
        module = getattr(pipeline, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    transformer = pipeline.transformer
    if transformer is None:
        raise ModelPortError("Wan pipeline must expose a transformer")
    transformer_channels = _artifact_config_int(
        transformer,
        "in_channels",
        label="Wan transformer",
    )
    vae_channels = _artifact_config_int(
        vae,
        "z_dim",
        label="Wan VAE",
    )
    if transformer_channels != vae_channels:
        raise ModelPortError(
            "Wan transformer/VAE latent channel metadata does not match"
        )
    latent_channels = transformer_channels
    checkpointing_state = configure_gradient_checkpointing(
        transformer,
        config.gradient_checkpointing,
        context="Wan transformer",
    )
    transformer = apply_peft_lora(
        transformer,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
    )
    verify_gradient_checkpointing(
        transformer,
        checkpointing_state,
        context="Wan transformer",
    )
    expand_timesteps = bool(getattr(pipeline.config, "expand_timesteps", False))
    video_processor = pipeline.video_processor
    pipeline.transformer = None
    pipeline.vae = None
    pipeline.scheduler = None
    return WanRuntimeParts(
        prompt_encoder=_WanPromptEncoder(pipeline, dtype),
        transformer=transformer,
        decoder=_WanDecoder(vae, video_processor),
        latent_channels=latent_channels,
        scheduler_artifact_blueprint=scheduler_blueprint,
        expand_timesteps=expand_timesteps,
    )


def _first_output_tensor(output: object, *, label: str) -> Any:
    import torch

    if not isinstance(output, (tuple, list)) or not output:
        raise ModelPortError(f"{label} must return a non-empty tuple")
    value = output[0]
    if not isinstance(value, torch.Tensor):
        raise ModelPortError(f"{label} first output must be a tensor")
    return value


def _cache_context(module: object, name: str):
    factory = getattr(module, "cache_context", None)
    return factory(name) if callable(factory) else nullcontext()
