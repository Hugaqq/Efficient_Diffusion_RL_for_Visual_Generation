"""Import-safe typed port for one model forward, never a rollout loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from visual_rl.core.contracts import (
    DeclaredContract,
    LatentLayout,
    PredictionType,
)
from visual_rl.models.scheduler import (
    ModelScheduleContext,
    SchedulerArtifactBlueprint,
)

if TYPE_CHECKING:
    from visual_rl.data.media import DecodedMediaBatch
    from visual_rl.data.preprocess import PreprocessProducerSpec
    from visual_rl.models.lifecycle.components import (
        ComponentLoadSession,
        ModelComponents,
    )
    from visual_rl.models.lifecycle.prepared import PreparedComponentHandle
    from visual_rl.models.numerics.policy import (
        ModelExecutionNumericsEvidence,
        ParameterViewEvidence,
    )
    from visual_rl.models.numerics.runtime import ModelRuntimeNumerics
    from visual_rl.models.preprocessing import ModelPreprocessConsumerSpec
    from visual_rl.models.state.parameters import ParameterStateManager
    from visual_rl.data.samples import StackedSampleBatch

__all__ = (
    "BatchProjectableModelPayload",
    "BatchRowProjection",
    "ModelAdapter",
    "ModelInput",
    "ModelLatentSpec",
    "ModelPortError",
    "ModelPrediction",
)


_RowValue = TypeVar("_RowValue")


@dataclass(frozen=True, slots=True)
class BatchRowProjection:
    """One explicit target-row to source-row mapping for model payloads.

    ``row_indices[target_row]`` names the corresponding row in a payload with
    ``source_batch_size`` rows.  Unique indices express selection/reordering;
    repeated indices express expansion.  Keeping both operations in one value
    prevents rollout-specific notions such as exploration groups from leaking
    into a model adapter.
    """

    source_batch_size: int
    row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.source_batch_size) is not int or self.source_batch_size < 1:
            raise ModelPortError("projection source_batch_size must be positive")
        if type(self.row_indices) is not tuple or not self.row_indices:
            raise ModelPortError("projection row_indices must be a non-empty tuple")
        if any(type(index) is not int for index in self.row_indices):
            raise TypeError("projection row_indices must contain integers")
        if any(
            index < 0 or index >= self.source_batch_size for index in self.row_indices
        ):
            raise IndexError("projection row index is outside the source batch")

    @property
    def target_batch_size(self) -> int:
        return len(self.row_indices)

    def project_tuple(self, values: tuple[_RowValue, ...]) -> tuple[_RowValue, ...]:
        """Project one row-aligned tuple through the exact same mapping."""

        if type(values) is not tuple or len(values) != self.source_batch_size:
            raise ModelPortError(
                "projected tuple must match projection source_batch_size"
            )
        return tuple(values[index] for index in self.row_indices)


@runtime_checkable
class BatchProjectableModelPayload(Protocol):
    """Fail-closed model payload port for selection and repeated expansion."""

    @property
    def batch_size(self) -> int:
        """Return the current source batch size."""

        ...

    @property
    def condition_identity(self) -> tuple[str, ...]:
        """Return exactly one model-condition identity per current row."""

        ...

    def project_rows(self, projection: BatchRowProjection) -> object:
        """Return target rows in ``projection.row_indices`` order."""

        ...


class ModelPortError(ValueError):
    """Raised when a single-step model port violates its typed contract."""


def _identities(name: str, values: object, batch_size: int) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) != batch_size:
        raise ModelPortError(f"{name} must contain one identity per latent row")
    if any(not isinstance(item, str) or not item for item in values):
        raise ModelPortError(f"{name} values must be non-empty strings")
    return values


@dataclass(frozen=True, slots=True)
class ModelLatentSpec:
    """General latent geometry supporting images, videos, and packed tokens."""

    shape: tuple[int, ...]
    layout: LatentLayout
    axis_semantics: tuple[str, ...]
    device: Any
    dtype: Any
    spatial_stride: tuple[int, int] | None = None
    temporal_stride: int | None = None
    scheduler_patch_size: int | None = None

    def __post_init__(self) -> None:
        import torch

        if type(self.shape) is not tuple or len(self.shape) < 2:
            raise ModelPortError("latent shape must be a tuple with rank >= 2")
        if any(type(item) is not int or item < 1 for item in self.shape):
            raise ModelPortError("latent shape entries must be positive integers")
        try:
            layout = LatentLayout(self.layout)
        except (TypeError, ValueError):
            raise ModelPortError("invalid latent layout") from None
        object.__setattr__(self, "layout", layout)
        if type(self.axis_semantics) is not tuple or len(self.axis_semantics) != len(
            self.shape
        ):
            raise ModelPortError("axis_semantics must name every latent axis")
        if any(not isinstance(item, str) or not item for item in self.axis_semantics):
            raise ModelPortError("axis_semantics values must be non-empty strings")
        if len(set(self.axis_semantics)) != len(self.axis_semantics):
            raise ModelPortError("axis_semantics must not contain duplicates")
        if self.axis_semantics[0] != "batch":
            raise ModelPortError("the first latent axis must be batch")
        expected = {
            LatentLayout.BCHW: ("batch", "channel", "height", "width"),
            LatentLayout.BCTHW: (
                "batch",
                "channel",
                "time",
                "height",
                "width",
            ),
        }.get(layout)
        if expected is not None and self.axis_semantics != expected:
            raise ModelPortError(f"{layout.value} requires axis_semantics={expected!r}")
        if layout is LatentLayout.PACKED_SEQUENCE and len(self.shape) < 3:
            raise ModelPortError("packed_sequence latent rank must be at least 3")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("latent dtype must be a torch.dtype")
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError):
            raise TypeError("latent device must be torch.device-compatible") from None
        object.__setattr__(self, "device", device)
        if self.spatial_stride is not None and (
            type(self.spatial_stride) is not tuple
            or len(self.spatial_stride) != 2
            or any(type(item) is not int or item < 1 for item in self.spatial_stride)
        ):
            raise ModelPortError(
                "spatial_stride must contain two positive integers or be None"
            )
        for name in ("temporal_stride", "scheduler_patch_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ModelPortError(f"{name} must be a positive integer or None")

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True, slots=True)
class ModelInput:
    """One explicit model evaluation at one timestep per latent row."""

    latents: Any
    timestep: Any
    conditioning: object
    guidance: object | None
    latent_spec: ModelLatentSpec
    condition_identity: tuple[str, ...]
    guidance_identity: tuple[str, ...]

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.latent_spec, ModelLatentSpec):
            raise TypeError("latent_spec must be ModelLatentSpec")
        if not isinstance(self.latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if tuple(self.latents.shape) != self.latent_spec.shape:
            raise ModelPortError("latents shape does not match latent_spec")
        if not self.latents.is_floating_point():
            raise TypeError("latents must be floating point")
        if self.latents.dtype != self.latent_spec.dtype:
            raise ModelPortError("latents dtype does not match latent_spec")
        if self.latents.device != self.latent_spec.device:
            raise ModelPortError("latents device does not match latent_spec")
        if self.latents.requires_grad or self.latents.grad_fn is not None:
            raise ModelPortError("input latents must be detached")
        if not bool(torch.isfinite(self.latents).all()):
            raise ModelPortError("input latents must be finite")

        if not isinstance(self.timestep, torch.Tensor):
            raise TypeError("timestep must be a torch.Tensor")
        if self.timestep.ndim > 1 or self.timestep.numel() not in {
            1,
            self.latent_spec.batch_size,
        }:
            raise ModelPortError("timestep must be scalar or contain B values")
        if self.timestep.device != self.latents.device:
            raise ModelPortError("timestep must be on the latent device")
        if self.timestep.dtype == torch.bool or self.timestep.is_complex():
            raise TypeError("timestep must use a real numeric dtype")
        if self.timestep.requires_grad or self.timestep.grad_fn is not None:
            raise ModelPortError("timestep must be detached")
        if not bool(torch.isfinite(self.timestep).all()):
            raise ModelPortError("timestep must be finite")

        _identities(
            "condition_identity",
            self.condition_identity,
            self.latent_spec.batch_size,
        )
        _identities(
            "guidance_identity",
            self.guidance_identity,
            self.latent_spec.batch_size,
        )


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Typed single-step prediction; Dynamics owns every transition operation."""

    value: Any
    prediction_type: PredictionType
    condition_identity: tuple[str, ...]
    guidance_identity: tuple[str, ...]

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.value, torch.Tensor):
            raise TypeError("prediction value must be a torch.Tensor")
        if self.value.ndim < 2 or self.value.shape[0] < 1:
            raise ModelPortError("prediction value must have shape [B,...]")
        if not self.value.is_floating_point():
            raise TypeError("prediction value must be floating point")
        if not bool(torch.isfinite(self.value).all()):
            raise ModelPortError("prediction value must be finite")
        try:
            prediction_type = PredictionType(self.prediction_type)
        except (TypeError, ValueError):
            raise ModelPortError("invalid prediction_type") from None
        object.__setattr__(self, "prediction_type", prediction_type)
        batch_size = int(self.value.shape[0])
        _identities("condition_identity", self.condition_identity, batch_size)
        _identities("guidance_identity", self.guidance_identity, batch_size)

    def validate_against(self, model_input: ModelInput) -> None:
        if not isinstance(model_input, ModelInput):
            raise TypeError("model_input must be ModelInput")
        if tuple(self.value.shape) != tuple(model_input.latents.shape):
            raise ModelPortError("prediction shape must match input latents")
        if self.value.device != model_input.latents.device:
            raise ModelPortError("prediction device must match input latents")
        if self.condition_identity != model_input.condition_identity:
            raise ModelPortError("prediction condition identity drift")
        if self.guidance_identity != model_input.guidance_identity:
            raise ModelPortError("prediction guidance identity drift")


class ModelAdapter(ABC):
    """Model-specific computation with no scheduler or trajectory ownership."""

    INTERFACE_VERSION = "1.0"

    @classmethod
    @abstractmethod
    def describe(cls, config: object) -> DeclaredContract:
        """Return the typed static model contract used by the registry."""

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> ModelAdapter:
        raise NotImplementedError

    @property
    def closed(self) -> bool:
        return bool(getattr(self, "_visual_rl_closed", False))

    def _assert_open(self) -> None:
        if self.closed:
            raise ModelPortError("model adapter is closed")

    def close(self) -> None:
        """Invalidate runtime ports and release adapter-owned strong references."""

        if self.closed:
            return
        object.__setattr__(self, "_visual_rl_closed", True)
        errors: list[BaseException] = []
        handle = getattr(self, "_visual_rl_prepared_handle", None)
        if handle is not None:
            close_handle = getattr(handle, "close", None)
            if callable(close_handle):
                try:
                    close_handle()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
        self._clear_prepared_components()
        self._clear_model_execution_numerics()
        try:
            self._release_runtime_parts()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        if errors:
            primary = errors[0]
            for cleanup_error in errors[1:]:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "additional adapter cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise primary

    def _release_runtime_parts(self) -> None:
        """Concrete adapters clear loaded model wrappers in this close hook."""

    @property
    def prepared_components(self) -> PreparedComponentHandle:
        """The sole route to distributed/sharded model component forwards."""

        self._assert_open()
        handle = getattr(self, "_visual_rl_prepared_handle", None)
        if handle is None:
            raise ModelPortError("model components have not been prepared")
        return handle

    def _bind_prepared_components(self, handle: PreparedComponentHandle) -> None:
        from visual_rl.models.lifecycle.prepared import PreparedComponentHandle

        self._assert_open()
        if not isinstance(handle, PreparedComponentHandle):
            raise TypeError("handle must be PreparedComponentHandle")
        if getattr(self, "_visual_rl_prepared_handle", None) is not None:
            raise ModelPortError("a prepared component handle is already bound")
        object.__setattr__(self, "_visual_rl_prepared_handle", handle)

    def _clear_prepared_components(self) -> None:
        if hasattr(self, "_visual_rl_prepared_handle"):
            object.__delattr__(self, "_visual_rl_prepared_handle")

    @property
    def model_execution_numerics(self) -> ModelExecutionNumericsEvidence:
        self._assert_open()
        evidence = getattr(self, "_visual_rl_model_execution_numerics", None)
        if evidence is None:
            raise ModelPortError("model execution numerics have not been bound")
        return evidence

    def _bind_model_execution_numerics(
        self,
        evidence: ModelExecutionNumericsEvidence,
        *,
        execution_policy_provider: object,
    ) -> None:
        from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence

        self._assert_open()
        if not isinstance(evidence, ModelExecutionNumericsEvidence):
            raise TypeError("evidence must be ModelExecutionNumericsEvidence")
        if not callable(execution_policy_provider):
            raise TypeError("execution_policy_provider must be callable")
        if getattr(self, "_visual_rl_model_execution_numerics", None) is not None:
            raise ModelPortError("model execution numerics are already bound")
        object.__setattr__(self, "_visual_rl_model_execution_numerics", evidence)
        object.__setattr__(
            self,
            "_visual_rl_execution_policy_provider",
            execution_policy_provider,
        )

    def _clear_model_execution_numerics(self) -> None:
        for name in (
            "_visual_rl_model_execution_numerics",
            "_visual_rl_execution_policy_provider",
        ):
            if hasattr(self, name):
                object.__delattr__(self, name)

    def _forward_prepared(
        self,
        component_name: str,
        *args: Any,
        parameter_view: object | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route exactly one model forward through its narrow autocast scope."""

        from visual_rl.models.numerics.execution import ParameterView
        from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence

        view = (
            ParameterView.CURRENT
            if parameter_view is None
            else ParameterView(parameter_view)
        )
        evidence = getattr(self, "_visual_rl_model_execution_numerics", None)
        if evidence is None:
            # Lightweight port tests may prepare an adapter without the
            # production lifecycle. G3 rejects that path before a real run.
            return self.prepared_components.forward(
                component_name,
                *args,
                **kwargs,
            )
        if not isinstance(evidence, ModelExecutionNumericsEvidence):
            raise ModelPortError("bound model execution numerics are invalid")
        provider = getattr(self, "_visual_rl_execution_policy_provider", None)
        if not callable(provider):
            raise ModelPortError("model execution policy provider is not bound")
        active = provider()
        if active is None:
            raise ModelPortError(
                "prepared model forward requires an active stage execution policy"
            )
        if active.parameter_view is not view:
            raise ModelPortError(
                "model forward parameter view differs from the active stage policy"
            )
        evidence.view_evidence(view).assert_integrity()
        autocast = evidence.autocast_policy(active.stage, view)
        return autocast.run_forward(
            self.prepared_components.forward,
            component_name,
            *args,
            **kwargs,
        )

    def describe_preprocess(self) -> PreprocessProducerSpec:
        """Declare immutable preprocessing semantics without running encode().

        Lightweight adapters remain source-compatible, but production
        composition fails closed until a concrete adapter implements this
        typed metadata port.
        """

        raise ModelPortError(
            f"{type(self).__name__} does not implement describe_preprocess()"
        )

    def describe_preprocess_consumption(self) -> ModelPreprocessConsumerSpec:
        """Declare model-forward conditioning fields without running encode()."""

        raise ModelPortError(
            f"{type(self).__name__} does not implement "
            "describe_preprocess_consumption()"
        )

    def describe_runtime_numerics(self) -> ModelRuntimeNumerics:
        """Declare rollout/transition latent dtypes without alias inference."""

        raise ModelPortError(
            f"{type(self).__name__} does not implement describe_runtime_numerics()"
        )

    def describe_parameter_view_evidence(
        self,
        parameter_state: ParameterStateManager,
        *,
        distribution_mode: str,
    ) -> tuple[ParameterViewEvidence, ...]:
        """Describe concrete current/reference/EMA realization after dtype bind."""

        del parameter_state, distribution_mode
        raise ModelPortError(
            f"{type(self).__name__} does not implement parameter view evidence"
        )

    def latent_spec_for_batch(
        self,
        batch: StackedSampleBatch,
        *,
        device: Any,
        dtype: Any,
    ) -> ModelLatentSpec:
        """Describe model-native latent geometry after artifact loading.

        Concrete production adapters must implement this port from typed model
        configuration and loaded artifact metadata.  The hard failure keeps
        lightweight test adapters source-compatible without allowing a
        composition root to guess geometry from a registry alias.
        """

        del batch, device, dtype
        raise ModelPortError(
            f"{type(self).__name__} does not implement latent_spec_for_batch()"
        )

    @property
    def scheduler_artifact_blueprint(self) -> SchedulerArtifactBlueprint:
        """Return the immutable scheduler snapshot produced during model load."""

        raise ModelPortError(
            f"{type(self).__name__} does not provide a scheduler artifact blueprint"
        )

    def model_schedule_context(
        self,
        latent_spec: ModelLatentSpec,
    ) -> ModelScheduleContext:
        """Expose the existing latent geometry through the narrow binder ABI."""

        self._assert_open()
        if not isinstance(latent_spec, ModelLatentSpec):
            raise TypeError("latent_spec must be a ModelLatentSpec")
        if not isinstance(latent_spec, ModelScheduleContext):
            raise ModelPortError("latent spec does not implement ModelScheduleContext")
        return latent_spec

    @abstractmethod
    def load_components(
        self,
        session: ComponentLoadSession,
    ) -> ModelComponents:
        """Acquire every owned component through the manager's load session."""

        raise NotImplementedError

    @abstractmethod
    def encode(self, batch: object) -> object:
        """Encode model-specific input conditioning without sampling latents."""

        raise NotImplementedError

    @abstractmethod
    def prepare_latents(
        self,
        latent_spec: ModelLatentSpec,
        *,
        generator: Any,
    ) -> Any:
        """Create base model latents for later runtime-owned transformations."""

        raise NotImplementedError

    @abstractmethod
    def predict(self, model_input: ModelInput) -> ModelPrediction:
        """Run exactly one model forward and no diffusion transition."""

        raise NotImplementedError

    def predict_reference(self, model_input: ModelInput) -> ModelPrediction:
        """Run the typed frozen-reference port when the model declares one."""

        self._assert_open()
        del model_input
        contract = type(self).describe(getattr(self, "config", None))
        model = contract.model
        if model is None:
            raise ModelPortError("adapter describe() did not return a model contract")
        if model.provides_reference_policy is False:
            raise ModelPortError(
                "model contract explicitly provides no reference policy"
            )
        raise ModelPortError(
            "model declares a reference policy but the adapter has no typed "
            "reference implementation"
        )

    @abstractmethod
    def decode(self, latents: Any, latent_spec: ModelLatentSpec) -> DecodedMediaBatch:
        """Decode final latents and report the model-owned media layout."""

        raise NotImplementedError
