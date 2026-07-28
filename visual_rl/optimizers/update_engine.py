"""The sole ordered policy-update implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Any, TypeVar

from visual_rl.core.types import (
    MetricContribution,
    PolicyRecomputeStats,
    RewardBatch,
    RolloutBatch,
    StepContext,
)
from visual_rl.optimizers.advantages import AdvantageResult
from visual_rl.optimizers.objective import (
    ObjectiveOutput,
    PolicyLossInputs,
    PolicyObjective,
)


_T = TypeVar("_T")
_CORE_FIELDS = (
    "loss",
    "policy_loss",
    "reference_kl",
    "approx_kl",
    "clipfrac",
)


@dataclass(frozen=True)
class UpdateResult:
    """Globally reduced update scalars returned to the Runner."""

    loss: float
    policy_loss: float
    reference_kl: float
    approx_kl: float
    clipfrac: float
    active_transition_count: int
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in _CORE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite Python number")
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, resolved)
        if (
            type(self.active_transition_count) is not int
            or self.active_transition_count <= 0
        ):
            raise ValueError("active_transition_count must be a positive integer")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        frozen: dict[str, float] = {}
        for name, value in self.diagnostics.items():
            if (
                not isinstance(name, str)
                or not name.startswith(("advantage/", "algorithm/"))
            ):
                raise ValueError(
                    "diagnostics keys must use advantage/ or algorithm/"
                )
            if name in _CORE_FIELDS:
                raise ValueError("diagnostics cannot replace core update fields")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("diagnostics values must be finite numbers")
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError("diagnostics values must be finite")
            frozen[name] = resolved
        object.__setattr__(self, "diagnostics", MappingProxyType(frozen))


class UpdateEngine:
    """Prepare, recompute, reduce, backward, and mutate exactly once."""

    _PRECISIONS = frozenset({"fp32", "bf16", "fp16"})

    def __init__(
        self,
        *,
        algorithm: Any,
        advantage_function: Any,
        objective: PolicyObjective,
        max_initial_logprob_delta: float | None = None,
        require_initial_clipfrac_zero: bool = False,
        require_finite_gradients: bool = True,
        require_nonzero_gradients: bool = False,
        max_grad_norm: float | None = None,
        update_microbatch_size: int | None = None,
        precision: str = "fp32",
    ) -> None:
        if not callable(advantage_function):
            raise TypeError("advantage_function must be callable")
        for name in (
            "weight_normalization_request",
            "prepare_loss_inputs",
            "diagnostics",
        ):
            if not callable(getattr(algorithm, name, None)):
                raise TypeError(f"algorithm must define {name}()")
        if not isinstance(objective, PolicyObjective):
            raise TypeError("objective must be a PolicyObjective")
        self.algorithm = algorithm
        self.advantage_function = advantage_function
        self.objective = objective
        self.max_initial_logprob_delta = self._validated_threshold(
            "max_initial_logprob_delta",
            max_initial_logprob_delta,
            allow_zero=True,
        )
        self.require_initial_clipfrac_zero = bool(
            require_initial_clipfrac_zero
        )
        self.require_finite_gradients = bool(require_finite_gradients)
        self.require_nonzero_gradients = bool(require_nonzero_gradients)
        self.max_grad_norm = self._validated_threshold(
            "max_grad_norm",
            max_grad_norm,
            allow_zero=False,
        )
        self.update_microbatch_size = self._validated_microbatch_size(
            update_microbatch_size
        )
        if not isinstance(precision, str) or precision not in self._PRECISIONS:
            raise ValueError("precision must be one of: fp32, bf16, fp16")
        self.precision = precision

    @staticmethod
    def _validated_microbatch_size(value: Any) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value <= 0:
            raise ValueError(
                "update_microbatch_size must be a positive integer or None"
            )
        return value

    @staticmethod
    def _validated_threshold(
        name: str,
        value: Any,
        *,
        allow_zero: bool,
    ) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number or None")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be finite")
        invalid = resolved < 0.0 if allow_zero else resolved <= 0.0
        if invalid:
            condition = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {condition}")
        return resolved

    @staticmethod
    def _local_then_gate(
        strategy: Any,
        phase: str,
        operation: Callable[[], _T],
    ) -> _T:
        result: Any = None
        failure: BaseException | None = None
        try:
            result = operation()
        except BaseException as exc:
            failure = exc
        strategy.failure_gate(phase, failure)
        if failure is not None:
            raise AssertionError("failure_gate returned after a failure") from failure
        return result

    @staticmethod
    def _validate_step_contract(
        *,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        scaler: Any | None,
        context: StepContext,
        strategy: Any,
        precision: str,
    ) -> tuple[Any, ...]:
        import torch

        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(rewards, RewardBatch):
            raise TypeError("rewards must be a RewardBatch")
        if not isinstance(context, StepContext):
            raise TypeError("context must be a StepContext")
        if batch.context is not context:
            raise ValueError(
                "Optimizer StepContext must be the RolloutBatch.context object"
            )
        rewards.validate_against(batch)
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("optimizer must be torch.optim.AdamW")
        parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if not parameters:
            raise ValueError("optimizer must contain trainable parameters")
        if any(not isinstance(item, torch.nn.Parameter) for item in parameters):
            raise TypeError("optimizer parameters must be torch.nn.Parameter")
        identities = tuple(id(item) for item in parameters)
        if len(identities) != len(set(identities)):
            raise ValueError("optimizer parameters must have unique identities")
        if any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("optimizer parameters must all require gradients")
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError("optimizer parameters must share one device")
        device = next(iter(devices))
        if device != strategy.device:
            raise ValueError(
                "optimizer parameter device must match Strategy.device"
            )
        if precision == "fp32":
            if scaler is not None:
                raise ValueError("fp32 requires scaler=None")
        elif precision == "bf16":
            if device.type != "cuda":
                raise ValueError("bf16 requires a CUDA Strategy device")
            if scaler is not None:
                raise ValueError("bf16 requires scaler=None")
        else:
            if device.type != "cuda":
                raise ValueError("fp16 requires a CUDA Strategy device")
            required = (
                "scale",
                "unscale_",
                "step",
                "update",
                "state_dict",
                "load_state_dict",
            )
            if scaler is None or any(
                not callable(getattr(scaler, name, None))
                for name in required
            ):
                raise TypeError("fp16 requires one GradScaler")
        if (
            context.rank != strategy.rank
            or context.world_size != strategy.world_size
        ):
            raise ValueError("StepContext topology must match Strategy")
        return parameters

    @staticmethod
    def _validate_normalization_request(
        request: Any,
    ) -> tuple[Any, int] | None:
        import torch

        if request is None:
            return None
        if type(request) is not tuple or len(request) != 2:
            raise TypeError(
                "weight_normalization_request() must return "
                "(scalar_tensor, positive_count) or None"
            )
        value, count = request
        if not isinstance(value, torch.Tensor):
            raise TypeError("normalization mean must be a torch.Tensor")
        if (
            value.ndim != 0
            or not value.is_floating_point()
            or value.is_complex()
            or value.requires_grad
            or value.grad_fn is not None
        ):
            raise ValueError(
                "normalization mean must be a detached real floating scalar"
            )
        if not bool(torch.isfinite(value)):
            raise ValueError("normalization mean must be finite")
        if type(count) is not int or count <= 0:
            raise ValueError("normalization count must be a positive integer")
        return value, count

    @classmethod
    def _require_consistent_contract(
        cls,
        strategy: Any,
        *,
        phase: str,
        contract: Any,
    ) -> None:
        gathered = strategy.gather_object(contract, dst=0)
        mismatch: str | None = None
        if strategy.is_main_process:
            assert gathered is not None
            if any(item != gathered[0] for item in gathered[1:]):
                mismatch = f"{phase} differs across ranks: {gathered!r}"
        mismatch = strategy.broadcast_object(mismatch, src=0)
        failure = ValueError(mismatch) if mismatch is not None else None
        strategy.failure_gate(phase, failure)

    @staticmethod
    def _merge_diagnostic_contributions(
        advantage: Mapping[str, MetricContribution],
        algorithm: Mapping[str, MetricContribution],
    ) -> dict[str, MetricContribution]:
        if not isinstance(advantage, Mapping):
            raise TypeError("AdvantageResult.metrics must be a mapping")
        if not isinstance(algorithm, Mapping):
            raise TypeError("algorithm diagnostics must be a mapping")
        merged: dict[str, MetricContribution] = {}
        for namespace, values in (
            ("advantage/", advantage),
            ("algorithm/", algorithm),
        ):
            for name, contribution in values.items():
                if not isinstance(name, str) or not name.startswith(namespace):
                    raise ValueError(
                        f"diagnostic keys must use the {namespace} namespace"
                    )
                if name in merged or name in _CORE_FIELDS:
                    raise ValueError(f"duplicate diagnostic key: {name!r}")
                if not isinstance(contribution, MetricContribution):
                    raise TypeError(
                        "diagnostics must contain MetricContribution values"
                    )
                merged[name] = contribution
        return merged

    @staticmethod
    def _validate_loss_inputs_contract(
        inputs: Any,
        *,
        batch: RolloutBatch,
        algorithm: Any,
    ) -> PolicyLossInputs:
        import torch

        if not isinstance(inputs, PolicyLossInputs):
            raise TypeError("PolicyAlgorithm must return PolicyLossInputs")
        inputs.validate_against(batch)
        dtype_name = getattr(algorithm, "ADVANTAGE_DTYPE", None)
        expected_dtypes = {
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if dtype_name not in expected_dtypes:
            raise ValueError(
                "PolicyAlgorithm.ADVANTAGE_DTYPE must be float32 or float64"
            )
        if inputs.base_advantage.dtype != expected_dtypes[dtype_name]:
            raise TypeError(
                "PolicyLossInputs dtype must match "
                f"PolicyAlgorithm.ADVANTAGE_DTYPE={dtype_name}"
            )
        return inputs

    @staticmethod
    def _microbatches(
        batch: RolloutBatch,
        inputs: PolicyLossInputs,
        microbatch_size: int | None,
    ) -> tuple[tuple[RolloutBatch, PolicyLossInputs, int], ...]:
        import torch

        inputs.validate_against(batch)
        row_counts = inputs.active_mask.sum(dim=1)
        if not bool((row_counts > 0).all()):
            invalid = torch.nonzero(
                row_counts <= 0,
                as_tuple=False,
            ).reshape(-1)
            raise ValueError(
                "every sample must have at least one active transition; "
                f"inactive rows: {invalid.detach().cpu().tolist()}"
            )
        size = microbatch_size or batch.batch_size
        slots: list[tuple[RolloutBatch, PolicyLossInputs, int]] = []
        for start in range(0, batch.batch_size, size):
            indices = tuple(
                range(start, min(start + size, batch.batch_size))
            )
            micro_inputs = inputs.slice(indices)
            active_count = int(micro_inputs.active_mask.sum().item())
            if active_count <= 0:
                raise ValueError("every fixed microbatch slot must be active")
            slots.append(
                (
                    batch if len(indices) == batch.batch_size else batch.slice(indices),
                    inputs if len(indices) == batch.batch_size else micro_inputs,
                    active_count,
                )
            )
        if not slots:
            raise ValueError("update requires at least one microbatch slot")
        return tuple(slots)

    def _forward_context(self, device: Any):
        if self.precision == "fp32":
            return nullcontext()
        import torch

        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
        )

    @staticmethod
    def _aligned_objective_views(
        batch: RolloutBatch,
        inputs: PolicyLossInputs,
        stats: PolicyRecomputeStats,
    ) -> tuple[RolloutBatch, PolicyLossInputs, PolicyRecomputeStats]:
        import torch

        target_device = stats.new_log_probs.device
        target_dtype = inputs.base_advantage.dtype

        def align_optional(value: Any | None) -> Any | None:
            return (
                None
                if value is None
                else value.to(device=target_device, dtype=target_dtype)
            )

        objective_batch = replace(
            batch,
            old_log_probs=batch.old_log_probs.to(
                device=target_device,
                dtype=target_dtype,
            ),
        )
        objective_inputs = replace(
            inputs,
            base_advantage=inputs.base_advantage.to(
                device=target_device,
                dtype=target_dtype,
            ),
            algorithm_weight=inputs.algorithm_weight.to(
                device=target_device,
                dtype=target_dtype,
            ),
            active_mask=inputs.active_mask.to(
                device=target_device,
                dtype=torch.bool,
            ),
        )
        objective_stats = replace(
            stats,
            new_log_probs=stats.new_log_probs.to(
                device=target_device,
                dtype=target_dtype,
            ),
            current_transition_mean=align_optional(
                stats.current_transition_mean
            ),
            transition_std=align_optional(stats.transition_std),
            reference_transition_mean=align_optional(
                stats.reference_transition_mean
            ),
        )
        return objective_batch, objective_inputs, objective_stats

    @staticmethod
    def _validate_objective_output(
        output: Any,
        *,
        expected_active_count: int,
    ) -> ObjectiveOutput:
        import torch

        if not isinstance(output, ObjectiveOutput):
            raise TypeError("PolicyObjective must return ObjectiveOutput")
        if output.active_transition_count != expected_active_count:
            raise ValueError(
                "Objective active_transition_count does not match loss inputs"
            )
        for name in _CORE_FIELDS:
            value = getattr(output, name)
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.ndim != 0
            ):
                raise TypeError(f"ObjectiveOutput.{name} must be a float scalar")
            if not bool(torch.isfinite(value.detach())):
                raise ValueError(f"ObjectiveOutput.{name} must be finite")
        if not output.loss.requires_grad:
            raise ValueError("ObjectiveOutput.loss must require gradients")
        return output

    @staticmethod
    def _active_logprob_delta_max(
        batch: RolloutBatch,
        inputs: PolicyLossInputs,
        stats: PolicyRecomputeStats,
    ) -> float:
        import torch

        old = batch.old_log_probs.to(
            device=stats.new_log_probs.device,
            dtype=stats.new_log_probs.dtype,
        )
        mask = inputs.active_mask.to(
            device=stats.new_log_probs.device,
            dtype=torch.bool,
        )
        safe_new = torch.where(mask, stats.new_log_probs, 0.0)
        safe_old = torch.where(mask, old, 0.0)
        return float(
            (safe_new - safe_old)
            .abs()
            .masked_select(mask)
            .max()
            .detach()
            .cpu()
        )

    @staticmethod
    def _gradient_gate_and_clip(
        parameters: tuple[Any, ...],
        *,
        require_finite: bool,
        require_nonzero: bool,
        max_grad_norm: float | None,
    ) -> None:
        import torch

        gradients = tuple(
            parameter.grad for parameter in parameters if parameter.grad is not None
        )
        finite = all(
            bool(torch.isfinite(gradient.detach()).all())
            for gradient in gradients
        )
        nonzero = any(
            bool(torch.count_nonzero(gradient.detach()))
            for gradient in gradients
        )
        if require_finite and not finite:
            raise RuntimeError("Gradient gate failed: non-finite gradient")
        if require_nonzero and not nonzero:
            raise RuntimeError("Gradient gate failed: all gradients are zero")
        if max_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            if not bool(torch.isfinite(torch.as_tensor(norm))):
                raise RuntimeError("Gradient clipping produced a non-finite norm")
            if require_finite and not all(
                bool(torch.isfinite(gradient.detach()).all())
                for gradient in gradients
            ):
                raise RuntimeError(
                    "Gradient gate failed after gradient clipping"
                )

    @staticmethod
    def _validated_reduced_metrics(
        reduced: Mapping[str, Any],
        diagnostics: Mapping[str, MetricContribution],
        global_active_count: int,
    ) -> UpdateResult:
        if not isinstance(reduced, Mapping):
            raise TypeError("Strategy metric reduction must return a mapping")
        expected = {*_CORE_FIELDS, *diagnostics}
        if set(reduced) != expected:
            raise ValueError("Strategy metric reduction returned unexpected keys")
        values: dict[str, float] = {}
        for name, value in reduced.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("reduced metrics must be Python numbers")
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError("reduced metrics must be finite")
            values[name] = resolved
        return UpdateResult(
            loss=values["loss"],
            policy_loss=values["policy_loss"],
            reference_kl=values["reference_kl"],
            approx_kl=values["approx_kl"],
            clipfrac=values["clipfrac"],
            active_transition_count=global_active_count,
            diagnostics={
                name: values[name]
                for name in diagnostics
            },
        )

    def step(
        self,
        *,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        scaler: Any | None,
        context: StepContext,
        strategy: Any,
    ) -> UpdateResult:
        """Execute the only policy-update path for single process and DDP."""

        import torch

        parameters = self._local_then_gate(
            strategy,
            "update.contract",
            lambda: self._validate_step_contract(
                batch=batch,
                rewards=rewards,
                optimizer=optimizer,
                scaler=scaler,
                context=context,
                strategy=strategy,
                precision=self.precision,
            ),
        )
        self._local_then_gate(
            strategy,
            "update.zero_grad",
            lambda: optimizer.zero_grad(set_to_none=True),
        )
        advantages = self._local_then_gate(
            strategy,
            "update.advantage",
            lambda: self.advantage_function(batch, rewards),
        )
        if not isinstance(advantages, AdvantageResult):
            self._local_then_gate(
                strategy,
                "update.advantage_contract",
                lambda: (_ for _ in ()).throw(
                    TypeError(
                        "AdvantageComputer must return AdvantageResult"
                    )
                ),
            )

        request = self._local_then_gate(
            strategy,
            "update.normalization_request",
            lambda: self._validate_normalization_request(
                self.algorithm.weight_normalization_request(
                    batch,
                    advantages,
                )
            ),
        )
        self._require_consistent_contract(
            strategy,
            phase="update.normalization_presence",
            contract=request is not None,
        )
        normalization_mean = None
        if request is not None:
            normalization_mean = strategy.reduce_tensor_weighted_mean(*request)
            normalization_mean = self._local_then_gate(
                strategy,
                "update.normalization_result",
                lambda: self._validate_normalization_request(
                    (normalization_mean, 1)
                )[0],
            )

        inputs = self._local_then_gate(
            strategy,
            "update.loss_inputs",
            lambda: self.algorithm.prepare_loss_inputs(
                batch,
                advantages,
                normalization_mean=normalization_mean,
            ),
        )
        inputs = self._local_then_gate(
            strategy,
            "update.loss_inputs_contract",
            lambda: self._validate_loss_inputs_contract(
                inputs,
                batch=batch,
                algorithm=self.algorithm,
            ),
        )
        diagnostics = self._local_then_gate(
            strategy,
            "update.diagnostics",
            lambda: self._merge_diagnostic_contributions(
                advantages.metrics,
                self.algorithm.diagnostics(batch, inputs),
            ),
        )
        slots = self._local_then_gate(
            strategy,
            "update.slots",
            lambda: self._microbatches(
                batch,
                inputs,
                self.update_microbatch_size,
            ),
        )
        self._require_consistent_contract(
            strategy,
            phase="update.slot_contract",
            contract=(
                batch.batch_size,
                tuple(slot[0].batch_size for slot in slots),
            ),
        )
        local_active_count = sum(slot[2] for slot in slots)
        global_active_count = strategy.sum_active_transition_count(
            local_active_count
        )

        core_numerators = {
            name: torch.zeros(
                (),
                dtype=torch.float64,
                device=strategy.device,
            )
            for name in _CORE_FIELDS
        }
        local_delta_max = 0.0
        local_clip_numerator = 0.0

        for index, (micro_batch, micro_inputs, active_count) in enumerate(slots):
            is_last_slot = index == len(slots) - 1
            backward_failure: BaseException | None = None
            with strategy.gradient_sync_context(is_last_slot):
                require_reference = micro_inputs.reference_kl_weight > 0.0

                def recompute_and_objective() -> tuple[
                    ObjectiveOutput,
                    PolicyRecomputeStats,
                ]:
                    with self._forward_context(strategy.device):
                        stats = strategy.recompute_policy_stats(
                            micro_batch,
                            require_reference=require_reference,
                        )
                    if not isinstance(stats, PolicyRecomputeStats):
                        raise TypeError(
                            "Strategy recompute must return PolicyRecomputeStats"
                        )
                    stats.validate_against(
                        micro_batch,
                        require_reference=require_reference,
                    )
                    views = self._aligned_objective_views(
                        micro_batch,
                        micro_inputs,
                        stats,
                    )
                    objective_batch, objective_inputs, objective_stats = views
                    with torch.autocast(
                        device_type=strategy.device.type,
                        enabled=False,
                    ):
                        output = self.objective(
                            objective_batch,
                            objective_inputs,
                            objective_stats,
                        )
                    return (
                        self._validate_objective_output(
                            output,
                            expected_active_count=active_count,
                        ),
                        objective_stats,
                    )

                output, objective_stats = self._local_then_gate(
                    strategy,
                    f"update.microbatch.{index}",
                    recompute_and_objective,
                )
                for name in _CORE_FIELDS:
                    core_numerators[name] += (
                        getattr(output, name).detach().to(
                            device=strategy.device,
                            dtype=torch.float64,
                        )
                        * active_count
                    )
                local_delta_max = max(
                    local_delta_max,
                    self._active_logprob_delta_max(
                        micro_batch,
                        micro_inputs,
                        objective_stats,
                    ),
                )
                local_clip_numerator += (
                    float(output.clipfrac.detach().cpu()) * active_count
                )
                scaled_loss = (
                    output.loss
                    * strategy.world_size
                    * active_count
                    / global_active_count
                )
                try:
                    if scaler is None:
                        scaled_loss.backward()
                    else:
                        scaler.scale(scaled_loss).backward()
                except BaseException as exc:
                    if is_last_slot:
                        raise
                    backward_failure = exc
            if not is_last_slot:
                strategy.failure_gate(
                    f"update.backward.{index}",
                    backward_failure,
                )

        if scaler is not None:
            self._local_then_gate(
                strategy,
                "update.unscale",
                lambda: scaler.unscale_(optimizer),
            )

        def validate_gradients_and_initial_update() -> None:
            if context.step == 0:
                if (
                    self.max_initial_logprob_delta is not None
                    and local_delta_max > self.max_initial_logprob_delta
                ):
                    raise RuntimeError(
                        "Pre-update log-prob parity gate failed: "
                        f"{local_delta_max:.6g} exceeds "
                        f"{self.max_initial_logprob_delta:.6g}"
                    )
                local_clipfrac = (
                    local_clip_numerator / local_active_count
                )
                if (
                    self.require_initial_clipfrac_zero
                    and local_clipfrac != 0.0
                ):
                    raise RuntimeError(
                        "Pre-update clipfrac gate failed: "
                        f"expected 0, got {local_clipfrac:.6g}"
                    )
            self._gradient_gate_and_clip(
                parameters,
                require_finite=self.require_finite_gradients,
                require_nonzero=self.require_nonzero_gradients,
                max_grad_norm=self.max_grad_norm,
            )

        self._local_then_gate(
            strategy,
            "update.gradient_gate",
            validate_gradients_and_initial_update,
        )
        contributions = dict(diagnostics)
        contributions.update(
            {
                name: MetricContribution(
                    numerator=numerator.detach(),
                    denominator=local_active_count,
                )
                for name, numerator in core_numerators.items()
            }
        )
        reduced = strategy.reduce_metric_contributions(contributions)
        result = self._local_then_gate(
            strategy,
            "update.metric_result",
            lambda: self._validated_reduced_metrics(
                reduced,
                diagnostics,
                global_active_count,
            ),
        )

        def operation() -> None:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

        strategy.atomic_optimizer_step(
            operation,
            parameters=parameters,
            optimizer=optimizer,
            scaler=scaler,
        )
        return result
