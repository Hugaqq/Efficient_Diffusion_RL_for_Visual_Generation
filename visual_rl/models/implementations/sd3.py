"""Single-forward SD3 adapter for image diffusion recipes."""

from __future__ import annotations

from collections.abc import Mapping
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
from visual_rl.models.catalog import SD3Config, positive_int
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
    "SD3Adapter",
    "SD3Conditioning",
    "SD3Config",
    "SD3RuntimeParts",
)


@dataclass(frozen=True, slots=True)
class SD3Conditioning:
    prompt_embeds: Any
    pooled_prompt_embeds: Any
    negative_prompt_embeds: Any
    negative_pooled_prompt_embeds: Any
    condition_identity: tuple[str, ...]

    def __post_init__(self) -> None:
        import torch

        tensors = (
            self.prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        )
        if any(not isinstance(item, torch.Tensor) for item in tensors):
            raise TypeError("SD3 conditioning values must be tensors")
        batch_size = int(self.prompt_embeds.shape[0])
        if batch_size < 1 or any(item.shape[0] != batch_size for item in tensors):
            raise ModelPortError("SD3 conditioning tensors must share a batch axis")
        if any(not item.is_floating_point() for item in tensors):
            raise TypeError("SD3 conditioning tensors must be floating point")
        if any(item.device != self.prompt_embeds.device for item in tensors):
            raise ModelPortError("SD3 conditioning tensors must share one device")
        if any(item.dtype != self.prompt_embeds.dtype for item in tensors):
            raise ModelPortError("SD3 conditioning tensors must share one dtype")
        if (
            type(self.condition_identity) is not tuple
            or len(self.condition_identity) != batch_size
        ):
            raise ModelPortError(
                "SD3 condition_identity must contain one value per row"
            )
        if any(
            not isinstance(identity, str) or not identity
            for identity in self.condition_identity
        ):
            raise ModelPortError(
                "SD3 condition_identity values must be non-empty strings"
            )

    @property
    def batch_size(self) -> int:
        return len(self.condition_identity)

    def project_rows(self, projection: BatchRowProjection) -> SD3Conditioning:
        """Project or repeat encoded rows without exposing SD3 field names."""

        if not isinstance(projection, BatchRowProjection):
            raise TypeError("SD3 projection must be a BatchRowProjection")
        if projection.source_batch_size != self.batch_size:
            raise ModelPortError(
                "SD3 projection source_batch_size does not match conditioning"
            )
        import torch

        cache: dict[object, Any] = {}

        def project(value: Any) -> Any:
            index = cache.get(value.device)
            if index is None:
                index = torch.tensor(
                    projection.row_indices,
                    dtype=torch.int64,
                    device=value.device,
                )
                cache[value.device] = index
            return value.index_select(0, index)

        return SD3Conditioning(
            prompt_embeds=project(self.prompt_embeds),
            pooled_prompt_embeds=project(self.pooled_prompt_embeds),
            negative_prompt_embeds=project(self.negative_prompt_embeds),
            negative_pooled_prompt_embeds=project(self.negative_pooled_prompt_embeds),
            condition_identity=projection.project_tuple(self.condition_identity),
        )


@dataclass(frozen=True, slots=True)
class SD3RuntimeParts:
    """Runtime-only injectable parts used by real and lightweight loaders."""

    prompt_encoder: object
    transformer: object
    decoder: object
    reference_context: object
    latent_channels: int
    scheduler_artifact_blueprint: SchedulerArtifactBlueprint
    transformer_patch_size: int | None = None

    def __post_init__(self) -> None:
        for name, value, methods in (
            ("prompt_encoder", self.prompt_encoder, ("encode", "to")),
            ("decoder", self.decoder, ("decode", "to")),
        ):
            if any(not callable(getattr(value, method, None)) for method in methods):
                raise TypeError(f"SD3 {name} must expose {methods}")
        if not callable(self.reference_context):
            raise TypeError("SD3 reference_context must be callable")
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
                "SD3 scheduler_artifact_blueprint must use the model-owned ABI"
            )
        if self.transformer_patch_size is not None:
            object.__setattr__(
                self,
                "transformer_patch_size",
                positive_int("transformer_patch_size", self.transformer_patch_size),
            )


class SD3Adapter(ModelAdapter):
    """Own SD3 components and one prepared transformer forward only."""

    CONFIG_TYPE = "visual_rl.models.catalog:SD3Config"

    def __init__(
        self,
        config: SD3Config,
        *,
        artifact_path: Path,
        precision: ComputePrecision,
        model_loader: object | None,
    ) -> None:
        if not isinstance(config, SD3Config):
            raise TypeError("config must be SD3Config")
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
        self._decoder: object | None = None
        self._reference_context: object | None = None
        self._latent_channels: int | None = None
        self._transformer_patch_size: int | None = None
        self._scheduler_artifact_blueprint: SchedulerArtifactBlueprint | None = None

    @classmethod
    def describe(cls, config: object) -> DeclaredContract:
        if not isinstance(config, SD3Config):
            raise TypeError("config must be SD3Config")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> SD3Adapter:
        if not isinstance(config, SD3Config):
            raise TypeError("config must be SD3Config")
        precision = runtime_precision(runtime_context)
        if precision not in config.describe_contract().model.supported_precisions:
            raise ValueError(f"SD3 does not support precision {precision.value!r}")
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
        resolution = self.config.resolution
        return PreprocessProducerSpec(
            implementation_id="visual_rl.models.implementations.sd3:SD3Adapter",
            implementation_revision="sd3-prompt-encode.v1",
            port=PreprocessPortContract(
                port_id="visual_rl.models.implementations.sd3:prompt-encode.v1",
                output_payload_type="sd3_prompt_embeddings.v1",
                dependencies=(
                    PreprocessDependency(
                        role=PreprocessComponentRole.MODEL,
                        logical_name="model_artifact",
                    ),
                ),
                producer_output_fields=(
                    "negative_pooled_prompt_embeds",
                    "negative_prompt_embeds",
                    "pooled_prompt_embeds",
                    "prompt_embeds",
                ),
                negative_condition_fields=(
                    "negative_pooled_prompt_embeds",
                    "negative_prompt_embeds",
                ),
                schema_version=2,
            ),
            geometry=PreprocessGeometry(
                height=resolution,
                width=resolution,
                aspect_ratio_bucket=f"{resolution}x{resolution}",
            ),
            transforms=(),
            preprocess_config=FrozenMapping(
                {
                    "do_classifier_free_guidance": True,
                    "embedding_dtype": self.precision.value,
                    "max_sequence_length": self.config.max_sequence_length,
                    "negative_prompt_policy": "empty_string",
                    "outputs_per_prompt": 1,
                }
            ),
        )

    def describe_preprocess_consumption(self) -> ModelPreprocessConsumerSpec:
        return ModelPreprocessConsumerSpec(
            implementation_revision="sd3-conditioning-consumer.v1",
            payload_type="sd3_prompt_embeddings.v1",
            required_modalities=("prompt_text",),
            positive_output_fields=("pooled_prompt_embeds", "prompt_embeds"),
            negative_output_fields=(
                "negative_pooled_prompt_embeds",
                "negative_prompt_embeds",
            ),
            uses_negative_condition=self.config.guidance_scale > 1.0,
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
                "SD3 LoRA parameter views are verified only for single process"
            )
        if not callable(self._reference_context):
            raise ModelPortError(
                "SD3 reference view requires a loaded LoRA disable context"
            )
        projection = parameter_state.state_projection
        owners = projection.standalone_saved_component_names
        states = projection.standalone_parameter_names
        if owners != ("transformer",) or not states:
            raise ModelPortError(
                "SD3 LoRA state projection must be owned only by transformer"
            )
        current = ParameterViewEvidence(
            parameter_view=ParameterView.CURRENT,
            mode=ParameterViewMode.CURRENT,
            owner_component_names=owners,
            restorable_state_names=states,
            source_projection_id=projection.projection_id,
            mutates_parameters_in_place=False,
        )
        reference = ParameterViewEvidence(
            parameter_view=ParameterView.REFERENCE,
            mode=ParameterViewMode.LORA_DISABLE,
            owner_component_names=owners,
            restorable_state_names=states,
            source_projection_id=projection.projection_id,
            mutates_parameters_in_place=False,
        )
        return (current, reference)

    def load_components(self, session: ComponentLoadSession) -> ModelComponents:
        self._assert_open()
        if not isinstance(session, ComponentLoadSession):
            raise TypeError("session must be ComponentLoadSession")
        parts = self._load_runtime_parts()
        import torch

        if not isinstance(parts.transformer, torch.nn.Module):
            raise TypeError("SD3 transformer must be torch.nn.Module")
        self._prompt_encoder = parts.prompt_encoder
        self._decoder = parts.decoder
        self._reference_context = parts.reference_context
        self._latent_channels = parts.latent_channels
        self._transformer_patch_size = parts.transformer_patch_size
        self._scheduler_artifact_blueprint = parts.scheduler_artifact_blueprint
        session.register(
            "prompt_encoder",
            parts.prompt_encoder,
            roles=(ComponentRole.PREPROCESS,),
            # The prepared transformer remains resident on the execution
            # device.  Moving all three frozen SD3 text encoders beside it is
            # not viable in FP32 on a 32 GiB device, and preprocessing does
            # not require those weights to share the transformer's device.
            # Keep model-specific placement here; algorithms receive only the
            # resulting conditioning tensors through the typed model port.
            managed_residency=False,
        )
        session.register(
            "transformer",
            parts.transformer,
            roles=(ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
            closer=self._close_transformer_binding,
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
                "SD3 components must be loaded before scheduler binding"
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
            raise TypeError("SD3 latent spec requires a StackedSampleBatch")
        batch.validate()
        if batch.task_type != "t2i":
            raise ModelPortError("SD3 latent spec requires a T2I sample batch")
        channels = self._latent_channels
        if channels is None:
            raise ModelPortError(
                "SD3 components must be loaded before latent geometry is available"
            )
        spatial = self.config.resolution // 8
        return ModelLatentSpec(
            shape=(batch.batch_size, channels, spatial, spatial),
            layout=LatentLayout.BCHW,
            axis_semantics=("batch", "channel", "height", "width"),
            device=device,
            dtype=dtype,
            spatial_stride=(8, 8),
            scheduler_patch_size=self._transformer_patch_size,
        )

    def encode(self, batch: object) -> SD3Conditioning:
        self._assert_open()
        from visual_rl.data.samples import StackedSampleBatch

        if not isinstance(batch, StackedSampleBatch) or batch.task_type != "t2i":
            raise TypeError("SD3 encode requires a T2I StackedSampleBatch")
        encoder = self._require_part("prompt_encoder", self._prompt_encoder)
        encoded = encoder.encode(
            batch.prompts,
            self.config.max_sequence_length,
            self.config.guidance_scale,
        )
        if not isinstance(encoded, tuple) or len(encoded) != 4:
            raise ModelPortError("SD3 prompt encoder must return exactly four tensors")
        prompt, negative, pooled, negative_pooled = encoded
        return SD3Conditioning(
            prompt_embeds=prompt,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative,
            negative_pooled_prompt_embeds=negative_pooled,
            condition_identity=tuple(row.identity for row in batch.rows),
        )

    def prepare_latents(
        self,
        latent_spec: ModelLatentSpec,
        *,
        generator: Any,
    ) -> Any:
        self._assert_open()
        if not isinstance(latent_spec, ModelLatentSpec):
            raise TypeError("latent_spec must be ModelLatentSpec")
        self._validate_latent_spec(latent_spec, operation="prepare")
        import torch

        return torch.randn(
            latent_spec.shape,
            device=latent_spec.device,
            dtype=latent_spec.dtype,
            generator=generator,
        )

    def predict(self, model_input: ModelInput) -> ModelPrediction:
        return self._predict(model_input, reference=False)

    def predict_reference(self, model_input: ModelInput) -> ModelPrediction:
        return self._predict(model_input, reference=True)

    def _predict(
        self,
        model_input: ModelInput,
        *,
        reference: bool,
    ) -> ModelPrediction:
        self._assert_open()
        if not isinstance(model_input, ModelInput):
            raise TypeError("model_input must be ModelInput")
        self._validate_latent_spec(model_input.latent_spec, operation="predict")
        conditioning = model_input.conditioning
        if not isinstance(conditioning, SD3Conditioning):
            raise TypeError("SD3 predict requires SD3Conditioning")
        if conditioning.condition_identity != model_input.condition_identity:
            raise ModelPortError("SD3 conditioning identity drift")
        import torch

        timestep = model_input.timestep
        if timestep.numel() == 1:
            timestep = timestep.expand(model_input.latent_spec.batch_size)
        hidden_states = model_input.latents.to(dtype=conditioning.prompt_embeds.dtype)
        conditioning_device = hidden_states.device
        if self.config.guidance_scale > 1.0:
            hidden_states = torch.cat((hidden_states, hidden_states))
            timestep = torch.cat((timestep, timestep))
            embeddings = torch.cat(
                (conditioning.negative_prompt_embeds, conditioning.prompt_embeds)
            ).to(device=conditioning_device)
            pooled = torch.cat(
                (
                    conditioning.negative_pooled_prompt_embeds,
                    conditioning.pooled_prompt_embeds,
                )
            ).to(device=conditioning_device)
        else:
            embeddings = conditioning.prompt_embeds.to(device=conditioning_device)
            pooled = conditioning.pooled_prompt_embeds.to(device=conditioning_device)

        def prepared_forward():
            return self._forward_prepared(
                "transformer",
                parameter_view=(
                    ParameterView.REFERENCE if reference else ParameterView.CURRENT
                ),
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=embeddings,
                pooled_projections=pooled,
                return_dict=False,
            )

        if reference:
            context_factory = self._require_part(
                "reference_context",
                self._reference_context,
            )
            context = context_factory()
            if not hasattr(context, "__enter__") or not hasattr(context, "__exit__"):
                raise TypeError("SD3 reference_context must return a context manager")
            with context:
                output = prepared_forward()
        else:
            output = prepared_forward()
        prediction = _first_output_tensor(output, label="SD3 transformer")
        if self.config.guidance_scale > 1.0:
            negative, positive = prediction.chunk(2)
            prediction = negative + self.config.guidance_scale * (positive - negative)
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
        if not isinstance(latent_spec, ModelLatentSpec):
            raise TypeError("latent_spec must be ModelLatentSpec")
        self._validate_latent_spec(latent_spec, operation="decode")
        decoder = self._require_part("decoder", self._decoder)
        return DecodedMediaBatch(
            tensor=decoder.decode(latents, latent_spec),
            layout="BCHW",
        )

    def _validate_latent_spec(
        self,
        latent_spec: ModelLatentSpec,
        *,
        operation: str,
    ) -> None:
        if latent_spec.layout is not LatentLayout.BCHW or latent_spec.rank != 4:
            raise ModelPortError(f"SD3 {operation} requires BCHW rank-4 latents")
        expected_spatial = self.config.resolution // 8
        if latent_spec.shape[2:] != (expected_spatial, expected_spatial):
            raise ModelPortError(
                f"SD3 {operation} expected latent spatial geometry "
                f"{(expected_spatial, expected_spatial)}, got {latent_spec.shape[2:]}"
            )

    def _load_runtime_parts(self) -> SD3RuntimeParts:
        self._assert_open()
        if self._model_loader is not None:
            parts = self._model_loader(
                "sd3",
                self.artifact_path,
                self.config,
                self.precision,
            )
            if not isinstance(parts, SD3RuntimeParts):
                raise TypeError("model_loader must return SD3RuntimeParts")
            return parts
        return _load_diffusers_sd3_parts(
            self.artifact_path,
            self.config,
            self.precision,
        )

    def _close_transformer_binding(self, _transformer: object) -> None:
        # ``disable_adapter`` is a bound method and otherwise keeps the full
        # transformer alive after the manager has released its prepared root.
        self._reference_context = None

    def _release_runtime_parts(self) -> None:
        prompt_encoder = self._prompt_encoder
        decoder = self._decoder
        self._prompt_encoder = None
        self._decoder = None
        self._reference_context = None
        self._latent_channels = None
        self._transformer_patch_size = None
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
                        "additional SD3 runtime-part cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise primary

    @staticmethod
    def _require_part(name: str, value: object | None) -> object:
        if value is None:
            raise ModelPortError(f"SD3 {name} has not been loaded")
        return value


class _SD3PromptEncoder:
    def __init__(self, pipeline: object, dtype: Any) -> None:
        self.pipeline = pipeline
        self.dtype = dtype
        self.device: object = "cpu"

    def to(self, device: object) -> _SD3PromptEncoder:
        if self.pipeline is None:
            raise ModelPortError("SD3 prompt encoder is closed")
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
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
    ) -> tuple[Any, Any, Any, Any]:
        if self.pipeline is None:
            raise ModelPortError("SD3 prompt encoder is closed")
        del guidance_scale
        encoded = self.pipeline.encode_prompt(
            prompt=list(prompts),
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[""] * len(prompts),
            negative_prompt_2=None,
            negative_prompt_3=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            clip_skip=None,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        if not isinstance(encoded, tuple) or len(encoded) != 4:
            raise ModelPortError("SD3 encode_prompt returned an invalid payload")
        return tuple(item.to(dtype=self.dtype) for item in encoded)

    def close(self) -> None:
        self.pipeline = None
        self.device = "cpu"


class _SD3Decoder:
    def __init__(self, vae: object, image_processor: object) -> None:
        self.vae = vae
        self.image_processor = image_processor

    def to(self, device: object) -> _SD3Decoder:
        import torch

        if self.vae is None:
            raise ModelPortError("SD3 decoder is closed")
        self.vae.to(device=device, dtype=torch.float32)
        return self

    def decode(self, latents: Any, latent_spec: ModelLatentSpec) -> object:
        del latent_spec
        vae = self.vae
        image_processor = self.image_processor
        if vae is None or image_processor is None:
            raise ModelPortError("SD3 decoder is closed")
        parameter = next(iter(vae.parameters()))
        scaling = float(vae.config.scaling_factor)
        shift = float(getattr(vae.config, "shift_factor", 0.0))
        value = (latents / scaling + shift).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        decoded = vae.decode(value, return_dict=False)[0]
        return image_processor.postprocess(decoded, output_type="pt")

    def close(self) -> None:
        self.vae = None
        self.image_processor = None


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


def _load_diffusers_sd3_parts(
    artifact_path: Path,
    config: SD3Config,
    precision: ComputePrecision,
) -> SD3RuntimeParts:
    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install visual-rl[train] to load SD3") from exc
    from visual_rl.models.implementations.common_diffusers import (
        apply_peft_lora,
        configure_gradient_checkpointing,
        resolve_torch_dtype,
        verify_gradient_checkpointing,
    )

    dtype = resolve_torch_dtype(precision.value)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        str(artifact_path),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    transformer = pipeline.transformer
    vae = pipeline.vae
    if transformer is None or vae is None:
        raise ModelPortError("SD3 pipeline must expose transformer and VAE components")
    transformer_channels = _artifact_config_int(
        transformer,
        "in_channels",
        label="SD3 transformer",
    )
    vae_channels = _artifact_config_int(
        vae,
        "latent_channels",
        label="SD3 VAE",
    )
    if transformer_channels != vae_channels:
        raise ModelPortError(
            "SD3 transformer/VAE latent channel metadata does not match"
        )
    latent_channels = transformer_channels
    transformer_patch_size = _artifact_config_int(
        transformer,
        "patch_size",
        label="SD3 transformer",
    )
    scheduler_blueprint = SchedulerArtifactBlueprint.from_scheduler(pipeline.scheduler)
    for name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipeline, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    checkpointing_state = configure_gradient_checkpointing(
        transformer,
        config.gradient_checkpointing,
        context="SD3 transformer",
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
        context="SD3 transformer",
    )
    image_processor = pipeline.image_processor
    pipeline.transformer = None
    pipeline.vae = None
    pipeline.scheduler = None
    reference_context = getattr(transformer, "disable_adapter", None)
    if not callable(reference_context):
        raise ModelPortError(
            "SD3 LoRA transformer must expose disable_adapter() for reference"
        )
    return SD3RuntimeParts(
        prompt_encoder=_SD3PromptEncoder(pipeline, dtype),
        transformer=transformer,
        decoder=_SD3Decoder(vae, image_processor),
        reference_context=reference_context,
        latent_channels=latent_channels,
        scheduler_artifact_blueprint=scheduler_blueprint,
        transformer_patch_size=transformer_patch_size,
    )


def _first_output_tensor(output: object, *, label: str) -> Any:
    import torch

    if not isinstance(output, (tuple, list)) or not output:
        raise ModelPortError(f"{label} must return a non-empty tuple")
    value = output[0]
    if not isinstance(value, torch.Tensor):
        raise ModelPortError(f"{label} first output must be a tensor")
    return value
