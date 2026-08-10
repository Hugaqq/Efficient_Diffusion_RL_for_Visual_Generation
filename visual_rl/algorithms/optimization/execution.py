"""Immutable optimizer-update ordering and commit semantics.

The plan is deliberately independent from the policy objective.  A caller gives
it one loss closure and the exact prepared model root; the plan owns the only
backward/optimizer transaction ordering used by the v0.8 training path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar


class UpdateDisposition(str, Enum):
    """Why an update transaction did or did not commit a logical step."""

    COMMITTED = "committed"
    ACCUMULATING = "accumulating"
    SCALER_SKIPPED = "scaler_skipped"


@dataclass(frozen=True, slots=True)
class OptimizerExecutionSpec:
    """Algorithm-owned optimizer and LR schedule values projected by runtime."""

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    amsgrad: bool
    schedule_kind: str
    warmup_steps: int
    min_lr_ratio: float
    max_optimizer_steps: int

    def __post_init__(self) -> None:
        for name in (
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "min_lr_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if type(self.amsgrad) is not bool:
            raise TypeError("amsgrad must be bool")
        if self.schedule_kind not in {"constant", "linear", "cosine"}:
            raise ValueError("schedule_kind must be constant, linear, or cosine")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if type(self.max_optimizer_steps) is not int or self.max_optimizer_steps < 1:
            raise ValueError("max_optimizer_steps must be positive")
        if self.warmup_steps > self.max_optimizer_steps:
            raise ValueError("warmup_steps cannot exceed max_optimizer_steps")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")


def build_adamw(
    parameters: Iterable[object],
    spec: OptimizerExecutionSpec,
) -> object:
    """Construct the one AdamW instance used by the canonical update path."""

    if not isinstance(spec, OptimizerExecutionSpec):
        raise TypeError("spec must be OptimizerExecutionSpec")
    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("AdamW requires at least one trainable parameter")
    import torch

    return torch.optim.AdamW(
        parameter_tuple,
        lr=spec.learning_rate,
        betas=(spec.beta1, spec.beta2),
        eps=spec.epsilon,
        weight_decay=spec.weight_decay,
        amsgrad=spec.amsgrad,
    )


def build_lr_scheduler(optimizer: object, spec: OptimizerExecutionSpec) -> object:
    """Construct the sole optimizer-step learning-rate scheduler."""

    if not isinstance(spec, OptimizerExecutionSpec):
        raise TypeError("spec must be OptimizerExecutionSpec")
    if not callable(getattr(optimizer, "step", None)):
        raise TypeError("optimizer must implement step()")
    import torch

    def multiplier(step: int) -> float:
        if spec.warmup_steps and step < spec.warmup_steps:
            return float(step + 1) / float(spec.warmup_steps)
        if spec.schedule_kind == "constant":
            return 1.0
        denominator = max(spec.max_optimizer_steps - spec.warmup_steps, 1)
        progress = min(
            max((step - spec.warmup_steps) / denominator, 0.0),
            1.0,
        )
        decay = (
            1.0 - progress
            if spec.schedule_kind == "linear"
            else 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        return spec.min_lr_ratio + (1.0 - spec.min_lr_ratio) * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@dataclass(frozen=True, slots=True)
class PreparedLoss:
    """Scalar loss plus caller-owned objective output produced in accumulate."""

    loss: Any
    payload: Any

    def __post_init__(self) -> None:
        if self.loss is None:
            raise ValueError("loss must not be None")


@dataclass(frozen=True, slots=True)
class UpdateTransactionResult:
    """Immutable outcome; only COMMITTED advances ``next_optimizer_step``."""

    optimizer_step: int
    disposition: UpdateDisposition
    payload: Any
    gradient_norm_pre_clip: float | None
    gradient_norm_post_clip: float | None
    trace: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if not isinstance(self.disposition, UpdateDisposition):
            raise TypeError("disposition must be an UpdateDisposition")
        if type(self.trace) is not tuple or not self.trace:
            raise ValueError("trace must be a non-empty tuple")
        norms = (self.gradient_norm_pre_clip, self.gradient_norm_post_clip)
        if self.disposition is UpdateDisposition.ACCUMULATING:
            if norms != (None, None):
                raise ValueError("an accumulating result cannot expose gradient norms")
        else:
            for name, value in zip(
                ("gradient_norm_pre_clip", "gradient_norm_post_clip"),
                norms,
                strict=True,
            ):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be a finite Python number")
                if not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite")

    @property
    def committed(self) -> bool:
        return self.disposition is UpdateDisposition.COMMITTED

    @property
    def skipped(self) -> bool:
        return self.disposition is UpdateDisposition.SCALER_SKIPPED

    @property
    def next_optimizer_step(self) -> int:
        return self.optimizer_step + int(self.committed)


class UpdateNotCommittedError(RuntimeError):
    """Raised by logical-update APIs that cannot return a pending transaction."""

    def __init__(self, result: UpdateTransactionResult) -> None:
        if not isinstance(result, UpdateTransactionResult):
            raise TypeError("result must be an UpdateTransactionResult")
        if result.committed:
            raise ValueError("a committed result is not an error")
        self.result = result
        super().__init__(f"optimizer update did not commit: {result.disposition.value}")


class UpdateTransactionPoisonedError(RuntimeError):
    """Fatal post-optimizer failure; retry requires restoring a safe checkpoint."""

    def __init__(
        self,
        *,
        optimizer_step: int,
        optimizer_step_applied: bool,
        failed_phase: str,
        trace: tuple[str, ...],
        cause: BaseException,
        cleanup_error: BaseException | None = None,
    ) -> None:
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if type(optimizer_step_applied) is not bool:
            raise TypeError("optimizer_step_applied must be bool")
        if not isinstance(failed_phase, str) or not failed_phase:
            raise ValueError("failed_phase must be non-empty")
        if type(trace) is not tuple or not trace:
            raise ValueError("trace must be a non-empty tuple")
        if not isinstance(cause, BaseException):
            raise TypeError("cause must be an exception")
        if cleanup_error is not None and not isinstance(cleanup_error, BaseException):
            raise TypeError("cleanup_error must be an exception or None")
        self.optimizer_step = optimizer_step
        self.optimizer_step_applied = optimizer_step_applied
        self.failed_phase = failed_phase
        self.trace = trace
        self.cause = cause
        self.cleanup_error = cleanup_error
        applied = (
            "was applied" if optimizer_step_applied else "may be partially applied"
        )
        message = (
            "fatal optimizer transaction failure: "
            f"step {optimizer_step} {applied}; failed_phase={failed_phase}; "
            f"cause={type(cause).__name__}: {cause}. "
            "Do not retry in-process; restore the last safe checkpoint."
        )
        if cleanup_error is not None:
            message += (
                " Gradient cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}."
            )
        super().__init__(message)

    @property
    def fatal(self) -> bool:
        return True

    @property
    def retryable(self) -> bool:
        return False


def _optional_positive_finite(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or None")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    return resolved


@dataclass(frozen=True, slots=True)
class UpdateExecutionPlan:
    """The one immutable accumulation/backward/update transaction definition."""

    require_finite_gradients: bool = True
    require_nonzero_gradients: bool = True
    max_grad_norm: float | None = None
    zero_grad_set_to_none: bool = True
    row_microbatch_size: int | None = None
    transition_window_size: int = 1

    SCHEMA_VERSION: ClassVar[int] = 2
    EXECUTION_ORDER: ClassVar[tuple[str, ...]] = (
        "accumulate",
        "objective",
        "backward",
        "unscale",
        "finite_check",
        "clip",
        "optimizer",
        "lr_scheduler",
        "ema",
        "reference",
        "zero_grad",
        "logical_commit",
    )

    def __post_init__(self) -> None:
        for name in (
            "require_finite_gradients",
            "require_nonzero_gradients",
            "zero_grad_set_to_none",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        object.__setattr__(
            self,
            "max_grad_norm",
            _optional_positive_finite("max_grad_norm", self.max_grad_norm),
        )
        if self.row_microbatch_size is not None and (
            type(self.row_microbatch_size) is not int or self.row_microbatch_size < 1
        ):
            raise ValueError("row_microbatch_size must be a positive integer or None")
        if (
            type(self.transition_window_size) is not int
            or self.transition_window_size < 1
        ):
            raise ValueError("transition_window_size must be a positive integer")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "execution_order": list(self.EXECUTION_ORDER),
            "require_finite_gradients": self.require_finite_gradients,
            "require_nonzero_gradients": self.require_nonzero_gradients,
            "max_grad_norm": self.max_grad_norm,
            "zero_grad_set_to_none": self.zero_grad_set_to_none,
            "row_microbatch_size": self.row_microbatch_size,
            "transition_window_size": self.transition_window_size,
        }

    @property
    def plan_id(self) -> str:
        encoded = json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def execute(
        self,
        *,
        loss_closure: Callable[[], PreparedLoss] | None = None,
        loss_closures: tuple[Callable[[], PreparedLoss], ...] | None = None,
        backward_context: Callable[[], object] | None = None,
        accelerator: object,
        prepared_root: object,
        optimizer: object,
        parameters: tuple[Any, ...],
        optimizer_step: int,
        scaler: object | None = None,
        lr_scheduler: object | None = None,
        ema_update: Callable[[], None] | None = None,
        reference_update: Callable[[], None] | None = None,
        logical_commit: Callable[[int], None] | None = None,
        strategy: object | None = None,
    ) -> UpdateTransactionResult:
        """Execute one or more backward slots and commit exactly once.

        The caller must clear gradients before the first microbatch of a logical
        update.  This method preserves gradients when Accelerator reports an
        accumulation-only microbatch, and clears them after every terminal
        outcome (commit, scaler skip, or exception).
        """

        closures = self._resolve_loss_closures(loss_closure, loss_closures)
        if backward_context is not None and not callable(backward_context):
            raise TypeError("backward_context must be callable or None")
        self._validate_runtime(
            loss_closure=closures[0],
            accelerator=accelerator,
            prepared_root=prepared_root,
            optimizer=optimizer,
            parameters=parameters,
            optimizer_step=optimizer_step,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            ema_update=ema_update,
            reference_update=reference_update,
            logical_commit=logical_commit,
            strategy=strategy,
        )
        trace: list[str] = []
        prepared_payloads: list[Any] = []
        optimizer_attempted = False
        optimizer_step_applied = False
        gradients_cleared = False
        active_phase = "accumulate"
        try:
            trace.append("accumulate")
            with accelerator.accumulate(prepared_root):
                context = (
                    nullcontext() if backward_context is None else backward_context()
                )
                if not callable(getattr(context, "__enter__", None)) or not callable(
                    getattr(context, "__exit__", None)
                ):
                    raise TypeError("backward_context must return a context manager")
                with context:
                    for index, closure in enumerate(closures):
                        active_phase = f"objective[{index}]"
                        trace.append("objective")
                        prepared_loss: PreparedLoss | None = None
                        try:
                            prepared_loss = closure()
                            if not isinstance(prepared_loss, PreparedLoss):
                                raise TypeError("loss_closure must return PreparedLoss")
                            prepared_payloads.append(prepared_loss.payload)
                            active_phase = f"backward[{index}]"
                            trace.append("backward")
                            accelerator.backward(prepared_loss.loss)
                        finally:
                            # Never let this frame retain a slot graph after
                            # backward succeeds or raises.
                            prepared_loss = None
                synchronize = self._synchronize_gradients(accelerator)

            payload = self._prepared_payload(prepared_payloads)

            if not synchronize:
                return UpdateTransactionResult(
                    optimizer_step=optimizer_step,
                    disposition=UpdateDisposition.ACCUMULATING,
                    payload=payload,
                    gradient_norm_pre_clip=None,
                    gradient_norm_post_clip=None,
                    trace=tuple(trace),
                )

            active_phase = "unscale"
            trace.append("unscale")
            self._unscale(
                accelerator=accelerator,
                scaler=scaler,
                optimizer=optimizer,
            )
            active_phase = "finite_check"
            trace.append("finite_check")
            pre_clip, post_clip = self._validate_and_clip(parameters, trace=trace)

            step_skipped = False

            def optimizer_operation() -> None:
                nonlocal optimizer_attempted, optimizer_step_applied, step_skipped
                optimizer_attempted = True
                trace.append("optimizer")
                step_skipped = self._optimizer_step_was_skipped(
                    accelerator=accelerator,
                    optimizer=optimizer,
                    scaler=scaler,
                )
                optimizer_step_applied = not step_skipped

            active_phase = "optimizer"
            if strategy is None:
                optimizer_operation()
            else:
                strategy.atomic_optimizer_step(
                    optimizer_operation,
                    parameters=parameters,
                    optimizer=optimizer,
                    scaler=scaler,
                )

            if step_skipped:
                active_phase = "zero_grad"
                self._zero_grad(optimizer, trace)
                gradients_cleared = True
                return UpdateTransactionResult(
                    optimizer_step=optimizer_step,
                    disposition=UpdateDisposition.SCALER_SKIPPED,
                    payload=payload,
                    gradient_norm_pre_clip=pre_clip,
                    gradient_norm_post_clip=post_clip,
                    trace=tuple(trace),
                )

            active_phase = "lr_scheduler"
            trace.append("lr_scheduler")
            if lr_scheduler is not None:
                lr_scheduler.step()
            active_phase = "ema"
            trace.append("ema")
            if ema_update is not None:
                ema_update()
            active_phase = "reference"
            trace.append("reference")
            if reference_update is not None:
                reference_update()
            active_phase = "zero_grad"
            self._zero_grad(optimizer, trace)
            gradients_cleared = True
            active_phase = "logical_commit"
            if logical_commit is not None:
                logical_commit(optimizer_step + 1)
            trace.append("logical_commit")
            return UpdateTransactionResult(
                optimizer_step=optimizer_step,
                disposition=UpdateDisposition.COMMITTED,
                payload=payload,
                gradient_norm_pre_clip=pre_clip,
                gradient_norm_post_clip=post_clip,
                trace=tuple(trace),
            )
        except BaseException as error:
            # Accumulation-only success is the sole result that intentionally
            # retains gradients.  Every exception aborts the logical update.
            cleanup_error: BaseException | None = None
            if not gradients_cleared:
                try:
                    self._zero_grad(optimizer, trace)
                    gradients_cleared = True
                except BaseException as cleanup:  # noqa: BLE001
                    cleanup_error = cleanup
            optimizer_outcome_unsafe = optimizer_step_applied or (
                optimizer_attempted and active_phase == "optimizer"
            )
            if optimizer_outcome_unsafe:
                raise UpdateTransactionPoisonedError(
                    optimizer_step=optimizer_step,
                    optimizer_step_applied=optimizer_step_applied,
                    failed_phase=active_phase,
                    trace=tuple(trace),
                    cause=error,
                    cleanup_error=cleanup_error,
                ) from error
            if cleanup_error is not None:
                try:
                    error.add_note(
                        "gradient cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                except AttributeError:
                    pass
            # Third-party backward frames commonly retain their ``loss`` local
            # through the propagated traceback.  Preserve the exception type
            # and message, but sever graph-bearing traceback/closure references
            # before handing the failure to a long-lived controller or logger.
            prepared_payloads.clear()
            closures = ()
            loss_closure = None
            loss_closures = None
            backward_context = None
            error.__traceback__ = None
            raise error from None

    @staticmethod
    def _resolve_loss_closures(
        loss_closure: object,
        loss_closures: object,
    ) -> tuple[Callable[[], PreparedLoss], ...]:
        if loss_closure is not None and loss_closures is not None:
            raise ValueError("provide loss_closure or loss_closures, not both")
        if loss_closures is None:
            if not callable(loss_closure):
                raise TypeError("loss_closure must be callable")
            return (loss_closure,)
        if type(loss_closures) is not tuple or not loss_closures:
            raise ValueError("loss_closures must be a non-empty tuple")
        if any(not callable(item) for item in loss_closures):
            raise TypeError("loss_closures must contain only callables")
        return loss_closures

    @staticmethod
    def _prepared_payload(prepared_payloads: list[Any]) -> Any:
        if not prepared_payloads:
            raise RuntimeError("update execution produced no prepared loss")
        payloads = tuple(prepared_payloads)
        return payloads[0] if len(payloads) == 1 else payloads

    def _validate_runtime(
        self,
        *,
        loss_closure: object,
        accelerator: object,
        prepared_root: object,
        optimizer: object,
        parameters: object,
        optimizer_step: object,
        scaler: object | None,
        lr_scheduler: object | None,
        ema_update: object | None,
        reference_update: object | None,
        logical_commit: object | None,
        strategy: object | None,
    ) -> None:
        if not callable(loss_closure):
            raise TypeError("loss_closure must be callable")
        if prepared_root is None:
            raise TypeError("prepared_root must not be None")
        for method in ("accumulate", "backward"):
            if not callable(getattr(accelerator, method, None)):
                raise TypeError(f"accelerator must implement {method}()")
        for method in ("step", "zero_grad"):
            if not callable(getattr(optimizer, method, None)):
                raise TypeError(f"optimizer must implement {method}()")
        if type(parameters) is not tuple or not parameters:
            raise ValueError("parameters must be a non-empty tuple")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameter identities must be unique")
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if scaler is not None:
            for method in ("unscale_", "step", "update"):
                if not callable(getattr(scaler, method, None)):
                    raise TypeError(f"scaler must implement {method}()")
        if lr_scheduler is not None and not callable(
            getattr(lr_scheduler, "step", None)
        ):
            raise TypeError("lr_scheduler must implement step()")
        for name, callback in (
            ("ema_update", ema_update),
            ("reference_update", reference_update),
            ("logical_commit", logical_commit),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")
        if strategy is not None and not callable(
            getattr(strategy, "atomic_optimizer_step", None)
        ):
            raise TypeError("strategy must implement atomic_optimizer_step()")

    @staticmethod
    def _synchronize_gradients(accelerator: object) -> bool:
        value = getattr(accelerator, "sync_gradients", True)
        if type(value) is not bool:
            raise TypeError("accelerator.sync_gradients must be bool")
        return value

    @staticmethod
    def _unscale(
        *,
        accelerator: object,
        scaler: object | None,
        optimizer: object,
    ) -> None:
        if scaler is not None:
            scaler.unscale_(optimizer)
            return
        unscale = getattr(accelerator, "unscale_gradients", None)
        if callable(unscale):
            unscale(optimizer)

    def _validate_and_clip(
        self,
        parameters: tuple[Any, ...],
        *,
        trace: list[str],
    ) -> tuple[float, float]:
        import torch

        gradients = tuple(
            parameter.grad for parameter in parameters if parameter.grad is not None
        )
        nonzero_gradient_count = sum(
            bool(torch.count_nonzero(gradient.detach())) for gradient in gradients
        )
        finite = all(
            bool(torch.isfinite(gradient.detach()).all()) for gradient in gradients
        )
        nonzero = nonzero_gradient_count > 0
        if self.require_finite_gradients and not finite:
            raise RuntimeError("gradient gate failed: non-finite gradient")
        if self.require_nonzero_gradients and not nonzero:
            raise RuntimeError(
                "gradient gate failed: all gradients are zero; "
                f"parameters_with_grad={len(gradients)}/{len(parameters)}; "
                f"parameters_with_nonzero_grad={nonzero_gradient_count}"
            )

        pre_clip_tensor = self._gradient_total_norm(gradients)
        if not bool(torch.isfinite(pre_clip_tensor)):
            raise RuntimeError("gradient norm is non-finite")
        trace.append("clip")
        if self.max_grad_norm is None:
            post_clip_tensor = pre_clip_tensor
        else:
            returned = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
            )
            if not bool(torch.isfinite(torch.as_tensor(returned))):
                raise RuntimeError("gradient clipping produced a non-finite norm")
            post_clip_tensor = self._gradient_total_norm(gradients)
            if not bool(torch.isfinite(post_clip_tensor)):
                raise RuntimeError("gradient clipping produced a non-finite gradient")
            if self.require_finite_gradients and not all(
                bool(torch.isfinite(gradient.detach()).all()) for gradient in gradients
            ):
                raise RuntimeError("gradient gate failed after clipping")
        return float(pre_clip_tensor.detach().cpu()), float(
            post_clip_tensor.detach().cpu()
        )

    @staticmethod
    def _gradient_total_norm(gradients: tuple[Any, ...]) -> Any:
        import torch

        if not gradients:
            return torch.zeros((), dtype=torch.float32)
        device = gradients[0].device
        norms = tuple(
            torch.linalg.vector_norm(gradient.detach(), ord=2).to(device=device)
            for gradient in gradients
        )
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    @staticmethod
    def _optimizer_step_was_skipped(
        *,
        accelerator: object,
        optimizer: object,
        scaler: object | None,
    ) -> bool:
        if scaler is None:
            optimizer.step()
            skipped = getattr(accelerator, "optimizer_step_was_skipped", False)
            if type(skipped) is not bool:
                raise TypeError("accelerator.optimizer_step_was_skipped must be bool")
            return skipped

        get_scale = getattr(scaler, "get_scale", None)
        scale_before = float(get_scale()) if callable(get_scale) else None
        scaler.step(optimizer)
        scaler.update()
        skipped = getattr(accelerator, "optimizer_step_was_skipped", None)
        if skipped is not None:
            if type(skipped) is not bool:
                raise TypeError("accelerator.optimizer_step_was_skipped must be bool")
            return skipped
        if scale_before is None:
            raise TypeError(
                "scaler skip detection requires scaler.get_scale() or "
                "accelerator.optimizer_step_was_skipped"
            )
        scale_after = float(get_scale())
        if not math.isfinite(scale_before) or not math.isfinite(scale_after):
            raise RuntimeError("gradient scaler scale must remain finite")
        return scale_after < scale_before

    def _zero_grad(self, optimizer: object, trace: list[str]) -> None:
        optimizer.zero_grad(set_to_none=self.zero_grad_set_to_none)
        trace.append("zero_grad")


__all__ = (
    "OptimizerExecutionSpec",
    "PreparedLoss",
    "UpdateDisposition",
    "UpdateExecutionPlan",
    "UpdateNotCommittedError",
    "UpdateTransactionPoisonedError",
    "UpdateTransactionResult",
    "build_adamw",
    "build_lr_scheduler",
)
