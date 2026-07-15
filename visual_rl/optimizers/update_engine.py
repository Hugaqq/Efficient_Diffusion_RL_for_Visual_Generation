"""One ordered, validated policy update for algorithm-backed plugins."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
import math
from time import perf_counter
from typing import Any

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext


class UpdateEngine:
    """Own the shared mechanics around an algorithm-specific objective."""

    _PRECISIONS = frozenset({"fp32", "bf16", "fp16"})

    def __init__(
        self,
        advantage_function: Any,
        objective: Any,
        *,
        max_initial_logprob_delta: float | None = None,
        require_initial_clipfrac_zero: bool = False,
        require_finite_gradients: bool = True,
        require_nonzero_gradients: bool = False,
        max_grad_norm: float | None = None,
        update_microbatch_size: int | None = None,
        precision: str = "fp32",
    ) -> None:
        self.advantage_function = advantage_function
        self.objective = objective
        self.max_initial_logprob_delta = self._validated_threshold(
            "max_initial_logprob_delta",
            max_initial_logprob_delta,
            allow_zero=True,
        )
        self.require_initial_clipfrac_zero = bool(require_initial_clipfrac_zero)
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
        self._scaler: Any | None = None
        self._pending_scaler_state: dict[str, Any] | None = None

    @staticmethod
    def _validated_microbatch_size(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("update_microbatch_size must be a positive integer or None")
        if value <= 0:
            raise ValueError("update_microbatch_size must be positive")
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
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a finite number or None")
        try:
            resolved = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be a finite number or None") from exc
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be finite")
        if allow_zero and resolved < 0:
            raise ValueError(f"{name} must be non-negative")
        if not allow_zero and resolved <= 0:
            raise ValueError(f"{name} must be positive")
        return resolved

    @staticmethod
    def _logprob_metrics(batch: RolloutBatch, new_log_probs: Any) -> dict[str, float]:
        import torch

        new_log_probs = new_log_probs.detach().float()
        old_log_probs = (
            torch.as_tensor(
                batch.old_log_probs,
                device=new_log_probs.device,
            )
            .detach()
            .float()
        )
        delta = new_log_probs - old_log_probs
        metrics = {
            "old_logprob_mean": float(old_log_probs.mean().cpu()),
            "new_logprob_mean": float(new_log_probs.mean().cpu()),
            "logprob_delta_mean": float(delta.mean().cpu()),
            "logprob_delta_abs_max": float(delta.abs().max().cpu()),
        }
        if batch.kl is not None:
            kl = (
                torch.as_tensor(
                    batch.kl,
                    device=new_log_probs.device,
                )
                .detach()
                .float()
            )
            metrics["rollout_kl_mean"] = float(kl.mean().cpu())
            metrics["rollout_kl_abs_max"] = float(kl.abs().max().cpu())
        return metrics

    def _validate_pre_update(
        self,
        *,
        logprob_metrics: dict[str, float],
        objective_output: Any,
    ) -> None:
        max_delta = float(logprob_metrics["logprob_delta_abs_max"])
        if not math.isfinite(max_delta):
            raise RuntimeError("Pre-update log-prob parity produced a non-finite delta")
        if (
            self.max_initial_logprob_delta is not None
            and max_delta > self.max_initial_logprob_delta
        ):
            raise RuntimeError(
                "Pre-update log-prob parity gate failed: "
                f"max_abs_delta={max_delta:.6g} exceeds "
                f"{self.max_initial_logprob_delta:.6g}"
            )
        clipfrac = self._scalar_metric(
            "ObjectiveOutput.clipfrac",
            objective_output.clipfrac,
        )
        if self.require_initial_clipfrac_zero and clipfrac != 0.0:
            raise RuntimeError(
                f"Pre-update clipfrac gate failed: expected 0, got {clipfrac:.6g}"
            )

    @staticmethod
    def _validate_loss(loss: Any) -> float:
        import torch

        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                "ObjectiveOutput.loss must be a floating scalar torch.Tensor "
                "with backward()"
            )
        if not callable(getattr(loss, "backward", None)):
            raise TypeError("ObjectiveOutput.loss must define backward()")
        if not loss.is_floating_point():
            raise TypeError("ObjectiveOutput.loss must be a floating tensor")
        if loss.ndim != 0:
            raise ValueError(
                "ObjectiveOutput.loss must be a scalar tensor; "
                f"got shape {tuple(loss.shape)}"
            )
        if not loss.requires_grad:
            raise ValueError("ObjectiveOutput.loss must require gradients")
        detached = loss.detach()
        if not bool(torch.isfinite(detached).item()):
            raise ValueError("ObjectiveOutput.loss must be finite")
        return float(detached.cpu())

    @staticmethod
    def _scalar_metric(name: str, value: Any) -> float:
        import torch

        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool")
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bool:
                raise TypeError(f"{name} must not be bool")
            if value.is_complex():
                raise TypeError(f"{name} must be a real scalar")
            if value.ndim != 0:
                raise ValueError(
                    f"{name} must be a scalar tensor; got shape {tuple(value.shape)}"
                )
            value = value.detach().cpu().item()
        elif not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be a finite scalar tensor or finite int/float; "
                f"got {type(value).__name__}"
            )
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be convertible to a finite float") from exc
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return scalar

    @classmethod
    def _scalar_metrics(
        cls,
        namespace: str,
        metrics: Any,
        *,
        excluded: set[str] | None = None,
    ) -> dict[str, float]:
        if not isinstance(metrics, dict):
            raise TypeError(f"{namespace} metrics must be a dict")
        excluded = excluded or set()
        scalar_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            cls._validate_metric_key(namespace, key)
            if key in excluded:
                continue
            scalar_metrics[key] = cls._scalar_metric(
                f"{namespace}.{key}",
                value,
            )
        return scalar_metrics

    @staticmethod
    def _validate_metric_key(namespace: str, key: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"{namespace} metric keys must be non-empty str; "
                f"got {type(key).__name__}"
            )
        if not key:
            raise ValueError(f"{namespace} metric keys must be non-empty str")

    @classmethod
    def _merge_metric_sections(
        cls,
        *sections: tuple[str, dict[str, Any]],
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        owners: dict[str, str] = {}
        for section_name, metrics in sections:
            if not isinstance(metrics, dict):
                raise TypeError(f"{section_name} metrics must be a dict")
            for key, value in metrics.items():
                cls._validate_metric_key(section_name, key)
                if key in merged:
                    raise ValueError(
                        f"Metric key collision for {key!r}: "
                        f"{section_name} conflicts with {owners[key]}"
                    )
                merged[key] = value
                owners[key] = section_name
        return merged

    @classmethod
    def _objective_metrics(cls, objective_output: Any) -> dict[str, float]:
        standard = {
            name: cls._scalar_metric(
                f"ObjectiveOutput.{name}",
                getattr(objective_output, name),
            )
            for name in ("policy_loss", "approx_kl", "clipfrac")
        }
        extra = cls._scalar_metrics(
            "ObjectiveOutput.metrics",
            objective_output.metrics,
            excluded={"policy_loss", "approx_kl", "clipfrac"},
        )
        return cls._merge_metric_sections(
            ("objective_standard", standard),
            ("objective_extra", extra),
        )

    @staticmethod
    def _gradient_metrics(parameters: list[Any]) -> dict[str, float | int | bool]:
        import torch

        squared_norm = 0.0
        nonzero_count = 0
        tensor_count = 0
        finite = True
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            tensor_count += 1
            detached = gradient.detach().float()
            finite = finite and bool(torch.isfinite(detached).all())
            nonzero_count += int(torch.count_nonzero(detached).item())
            squared_norm += float(
                torch.sum(detached.double() * detached.double()).item()
            )
        return {
            "grad_norm": float(math.sqrt(squared_norm)),
            "grad_nonzero_count": nonzero_count,
            "grad_tensor_count": tensor_count,
            "gradients_finite": finite,
        }

    def _clip_gradients(self, parameters: list[Any]) -> dict[str, Any]:
        if self.max_grad_norm is None:
            return {}

        import torch

        preclip_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        return {
            "grad_preclip_norm": preclip_norm,
            "grad_clip_max_norm": self.max_grad_norm,
        }

    @staticmethod
    def _parameter_device(parameters: list[Any], batch: RolloutBatch) -> Any:
        import torch

        devices = {parameter.device for parameter in parameters}
        if len(devices) > 1:
            raise ValueError("Adapter parameters must be on one device")
        if devices:
            return next(iter(devices))
        if isinstance(batch.old_log_probs, torch.Tensor):
            return batch.old_log_probs.device
        return torch.device("cpu")

    def _validate_precision_device(self, device: Any) -> None:
        if self.precision == "fp16" and device.type != "cuda":
            raise RuntimeError("precision='fp16' requires a CUDA device")
        if self.precision == "bf16" and device.type not in {"cpu", "cuda"}:
            raise RuntimeError("precision='bf16' requires a CPU or CUDA device")

    def _autocast_context(self, device: Any) -> Any:
        if self.precision == "fp32":
            return nullcontext()

        import torch

        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=device.type, dtype=dtype)

    @staticmethod
    def _objective_inputs(advantages: Any, new_log_probs: Any) -> tuple[Any, Any]:
        import torch

        if not isinstance(advantages, torch.Tensor):
            advantages = torch.as_tensor(advantages)
        if not isinstance(new_log_probs, torch.Tensor):
            raise TypeError("recompute_log_probs must return a torch.Tensor")
        # Explicit float64 advantages are part of the TempFlow reference contract.
        dtype = torch.float64 if advantages.dtype == torch.float64 else torch.float32
        return (
            advantages.to(device=new_log_probs.device, dtype=dtype),
            new_log_probs.to(dtype=dtype),
        )

    @staticmethod
    def _validate_reward_mask(rewards: RewardBatch) -> None:
        import torch

        valid = torch.as_tensor(rewards.valid_mask, dtype=torch.bool).reshape(-1)
        invalid = torch.nonzero(~valid, as_tuple=False).reshape(-1).tolist()
        if invalid:
            raise ValueError(
                "Update requires every reward sample to be valid; invalid "
                f"valid_mask indices: {invalid}"
            )

    @staticmethod
    def _validated_reduction_weight(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must return a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must return a non-negative integer")
        return value

    @staticmethod
    def _logprob_element_count(batch: RolloutBatch) -> int:
        import torch

        old_log_probs = torch.as_tensor(batch.old_log_probs)
        if old_log_probs.ndim == 0 or old_log_probs.shape[0] != batch.batch_size:
            raise ValueError("old_log_probs must have a batch dimension")
        return old_log_probs.numel()

    @staticmethod
    def _default_reduction_weight(batch: RolloutBatch) -> int:
        import torch

        old_log_probs = torch.as_tensor(batch.old_log_probs)
        if old_log_probs.ndim == 0 or old_log_probs.shape[0] != batch.batch_size:
            raise ValueError("old_log_probs must have a batch dimension")
        if batch.transition_mask is None:
            return old_log_probs.numel()
        mask = torch.as_tensor(batch.transition_mask, dtype=torch.bool)
        if tuple(mask.shape) != tuple(old_log_probs.shape):
            raise ValueError(
                "transition_mask must have the same shape as old_log_probs"
            )
        return int(mask.sum().item())

    def _objective_reduction_weight(
        self,
        batch: RolloutBatch,
        advantages: Any,
    ) -> int:
        reduction_weight = getattr(self.objective, "reduction_weight", None)
        value = (
            self._default_reduction_weight(batch)
            if reduction_weight is None
            else reduction_weight(batch, advantages)
        )
        return self._validated_reduction_weight(
            "PolicyObjective.reduction_weight",
            value,
        )

    def _objective_metric_reduction_weight(
        self,
        batch: RolloutBatch,
        advantages: Any,
        metric_name: str,
    ) -> int:
        metric_reduction_weight = getattr(
            self.objective,
            "metric_reduction_weight",
            None,
        )
        value = (
            self._objective_reduction_weight(batch, advantages)
            if metric_reduction_weight is None
            else metric_reduction_weight(batch, advantages, metric_name)
        )
        return self._validated_reduction_weight(
            f"PolicyObjective.metric_reduction_weight({metric_name!r})",
            value,
        )

    def _microbatches(
        self,
        batch: RolloutBatch,
        rewards: RewardBatch,
        advantages: Any,
    ) -> list[tuple[RolloutBatch, RewardBatch, Any, int, int]]:
        import torch

        size = self.update_microbatch_size or batch.batch_size
        microbatches = []
        for start in range(0, batch.batch_size, size):
            indices = list(range(start, min(start + size, batch.batch_size)))
            if len(indices) == batch.batch_size:
                micro_batch = batch
                micro_rewards = rewards
                micro_advantages = advantages
            else:
                micro_batch = batch.slice(indices)
                micro_rewards = rewards.slice(indices)
                index = torch.tensor(
                    indices, device=advantages.device, dtype=torch.long
                )
                micro_advantages = advantages.index_select(0, index)
            reduction_weight = self._objective_reduction_weight(
                micro_batch,
                micro_advantages,
            )
            if reduction_weight:
                microbatches.append(
                    (
                        micro_batch,
                        micro_rewards,
                        micro_advantages,
                        reduction_weight,
                        self._logprob_element_count(micro_batch),
                    )
                )
        if not microbatches:
            raise ValueError("Update requires at least one objective reduction element")
        return microbatches

    def _accumulate_objective_metrics(
        self,
        numerators: dict[str, float],
        denominators: dict[str, int],
        values: dict[str, float],
        batch: RolloutBatch,
        advantages: Any,
    ) -> None:
        if numerators and set(numerators) != set(values):
            raise ValueError("Objective metric keys must match across microbatches")
        for key, value in values.items():
            weight = self._objective_metric_reduction_weight(
                batch,
                advantages,
                key,
            )
            if weight == 0:
                raise ValueError(
                    "Objective metric reduction weights must be positive for "
                    f"emitted metrics; got zero for {key!r}"
                )
            numerators[key] = numerators.get(key, 0.0) + value * weight
            denominators[key] = denominators.get(key, 0) + weight

    @staticmethod
    def _accumulate_logprob_metrics(
        accumulator: dict[str, float],
        values: dict[str, float],
        weight: float,
    ) -> None:
        if accumulator and set(accumulator) != set(values):
            raise ValueError("Log-prob metric keys must match across microbatches")
        for key, value in values.items():
            if key.endswith("_abs_max"):
                accumulator[key] = max(accumulator.get(key, -math.inf), value)
            else:
                accumulator[key] = accumulator.get(key, 0.0) + value * weight

    @staticmethod
    def _grad_scaler() -> Any:
        import torch

        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()

    def _get_grad_scaler(self) -> Any | None:
        if self.precision != "fp16":
            return None
        if self._scaler is None:
            scaler = self._grad_scaler()
            if self._pending_scaler_state is not None:
                scaler.load_state_dict(deepcopy(self._pending_scaler_state))
            self._scaler = scaler
            self._pending_scaler_state = None
        return self._scaler

    def scaler_state_dict(self) -> dict[str, Any] | None:
        if self._scaler is not None:
            return deepcopy(self._scaler.state_dict())
        if self._pending_scaler_state is not None:
            return deepcopy(self._pending_scaler_state)
        return None

    def load_scaler_state_dict(self, state: dict[str, Any] | None) -> None:
        if state is not None and not isinstance(state, dict):
            raise TypeError("GradScaler state must be a dict or None")
        if state is not None and self.precision != "fp16":
            raise ValueError("GradScaler state requires precision='fp16'")
        self._scaler = None
        self._pending_scaler_state = None if state is None else deepcopy(state)

    def step(
        self,
        adapter: Any,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        context: StepContext,
        *,
        recompute_log_probs: Callable[[RolloutBatch], Any] | None = None,
        gradient_sync_context: Callable[[bool], Any] | None = None,
        reduce_tensor_weighted_mean: Callable[[Any, int], Any] | None = None,
        synchronize_failure: Callable[[bool | BaseException | None], bool]
        | None = None,
        before_optimizer_step: Callable[[], Any] | None = None,
        optimizer_step: Callable[..., Any] | None = None,
    ) -> dict[str, float]:
        """Run one update while keeping GradScaler state retryable on failure."""

        scaler_state = self.scaler_state_dict()
        try:
            return self._step_impl(
                adapter,
                batch,
                rewards,
                optimizer,
                context,
                recompute_log_probs=recompute_log_probs,
                gradient_sync_context=gradient_sync_context,
                reduce_tensor_weighted_mean=reduce_tensor_weighted_mean,
                synchronize_failure=synchronize_failure,
                before_optimizer_step=before_optimizer_step,
                optimizer_step=optimizer_step,
            )
        except BaseException as exc:
            try:
                self.load_scaler_state_dict(scaler_state)
            except BaseException as restore_error:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "Failed to restore GradScaler after update failure: "
                        f"{type(restore_error).__name__}: {restore_error}"
                    )
            raise

    def _step_impl(
        self,
        adapter: Any,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        context: StepContext,
        *,
        recompute_log_probs: Callable[[RolloutBatch], Any] | None = None,
        gradient_sync_context: Callable[[bool], Any] | None = None,
        reduce_tensor_weighted_mean: Callable[[Any, int], Any] | None = None,
        synchronize_failure: Callable[[bool | BaseException | None], bool]
        | None = None,
        before_optimizer_step: Callable[[], Any] | None = None,
        optimizer_step: Callable[..., Any] | None = None,
    ) -> dict[str, float]:
        import torch

        if recompute_log_probs is None:
            recompute_log_probs = getattr(adapter, "recompute_log_probs", None)
        if not callable(recompute_log_probs):
            raise TypeError("recompute_log_probs must be callable")
        if gradient_sync_context is not None and not callable(
            gradient_sync_context
        ):
            raise TypeError("gradient_sync_context must be callable or None")
        if reduce_tensor_weighted_mean is not None and not callable(
            reduce_tensor_weighted_mean
        ):
            raise TypeError("reduce_tensor_weighted_mean must be callable or None")
        if synchronize_failure is not None and not callable(synchronize_failure):
            raise TypeError("synchronize_failure must be callable or None")
        if (reduce_tensor_weighted_mean is None) != (synchronize_failure is None):
            raise ValueError(
                "reduce_tensor_weighted_mean and synchronize_failure must be "
                "provided together"
            )
        if before_optimizer_step is not None and not callable(before_optimizer_step):
            raise TypeError("before_optimizer_step must be callable or None")
        if optimizer_step is not None and not callable(optimizer_step):
            raise TypeError("optimizer_step must be callable or None")
        if batch.context is not None and batch.context != context:
            raise ValueError("Optimizer StepContext must match RolloutBatch.context")
        rewards.validate_against(batch)
        self._validate_reward_mask(rewards)
        parameters = list(adapter.parameters())
        device = self._parameter_device(parameters, batch)
        self._validate_precision_device(device)

        advantage_result = self.advantage_function(batch, rewards)
        advantages = advantage_result.advantages
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.as_tensor(advantages)
        if advantages.ndim == 0 or advantages.shape[0] != batch.batch_size:
            raise ValueError("advantages must have a leading batch dimension")
        prepare_batch = getattr(self.objective, "prepare_batch", None)
        requires_global_reduction = getattr(
            self.objective,
            "requires_global_batch_reduction",
            None,
        )
        global_batch_reduction = getattr(
            self.objective,
            "global_batch_reduction",
            None,
        )
        apply_global_batch_reduction = getattr(
            self.objective,
            "apply_global_batch_reduction",
            None,
        )

        def validate_prepared_batch(candidate: Any) -> RolloutBatch:
            if not isinstance(candidate, RolloutBatch):
                raise TypeError(
                    "PolicyObjective.prepare_batch must return a RolloutBatch"
                )
            if candidate.batch_size != batch.batch_size:
                raise ValueError(
                    "PolicyObjective.prepare_batch must preserve batch size"
                )
            if list(candidate.sample_id) != list(batch.sample_id):
                raise ValueError(
                    "PolicyObjective.prepare_batch must preserve sample order"
                )
            return candidate

        use_global_reduction = False
        if reduce_tensor_weighted_mean is not None:
            if not callable(requires_global_reduction):
                raise TypeError(
                    "PolicyObjective.requires_global_batch_reduction must be callable"
                )
            use_global_reduction = requires_global_reduction()
            if not isinstance(use_global_reduction, bool):
                raise TypeError(
                    "PolicyObjective.requires_global_batch_reduction must return bool"
                )

        if use_global_reduction:
            if not callable(global_batch_reduction) or not callable(
                apply_global_batch_reduction
            ):
                raise TypeError(
                    "A globally reduced objective must define global reduction hooks"
                )
            prepared_batch: RolloutBatch | None = None
            reduction_request: tuple[Any, int] | None = None
            prepare_error: BaseException | None = None
            try:
                prepared_batch = validate_prepared_batch(
                    batch
                    if prepare_batch is None
                    else prepare_batch(batch, advantages)
                )
                request = global_batch_reduction(prepared_batch, advantages)
                if not isinstance(request, tuple) or len(request) != 2:
                    raise TypeError(
                        "PolicyObjective.global_batch_reduction must return "
                        "(local_mean, positive_count)"
                    )
                local_mean, local_count = request
                local_count = self._validated_reduction_weight(
                    "PolicyObjective.global_batch_reduction count",
                    local_count,
                )
                if local_count == 0:
                    raise ValueError(
                        "PolicyObjective.global_batch_reduction count must be positive"
                    )
                reduction_request = (local_mean, local_count)
            except BaseException as exc:
                prepare_error = exc
            assert synchronize_failure is not None
            synchronize_failure(prepare_error)
            if prepare_error is not None:
                raise prepare_error
            if prepared_batch is None or reduction_request is None:
                raise RuntimeError("global batch reduction preflight lost local state")

            global_mean = reduce_tensor_weighted_mean(*reduction_request)
            apply_error: BaseException | None = None
            globally_prepared: RolloutBatch | None = None
            try:
                globally_prepared = validate_prepared_batch(
                    apply_global_batch_reduction(
                        prepared_batch,
                        advantages,
                        global_mean,
                    )
                )
            except BaseException as exc:
                apply_error = exc
            synchronize_failure(apply_error)
            if apply_error is not None:
                raise apply_error
            if globally_prepared is None:
                raise RuntimeError("global batch reduction apply lost local state")
            prepared_batch = globally_prepared
        else:
            prepared_batch = validate_prepared_batch(
                batch if prepare_batch is None else prepare_batch(batch, advantages)
            )
        batch = prepared_batch
        microbatches = self._microbatches(batch, rewards, advantages)
        total_reduction_weight = sum(item[3] for item in microbatches)
        total_logprob_elements = sum(item[4] for item in microbatches)

        reward_metrics = self._scalar_metrics(
            "reward",
            {
                "reward_mean": rewards.weighted_total.mean(),
                "reward_std": rewards.weighted_total.std(unbiased=False),
            },
        )
        advantage_metrics = self._scalar_metrics(
            "advantage",
            advantage_result.metrics,
        )

        adapter.prepare_for_training()
        scaler = self._get_grad_scaler()
        gradients_initialized = False
        loss_metric = 0.0
        objective_metric_numerators: dict[str, float] = {}
        objective_metric_denominators: dict[str, int] = {}
        logprob_metrics: dict[str, float] = {}
        recompute_time_s = 0.0
        backward_time_s = 0.0

        for microbatch_index, (
            micro_batch,
            _micro_rewards,
            micro_advantages,
            reduction_weight,
            logprob_elements,
        ) in enumerate(microbatches):
            weight = reduction_weight / total_reduction_weight
            synchronize_gradients = microbatch_index == len(microbatches) - 1
            sync_context = (
                nullcontext()
                if gradient_sync_context is None
                else gradient_sync_context(synchronize_gradients)
            )
            if not all(
                callable(getattr(sync_context, name, None))
                for name in ("__enter__", "__exit__")
            ):
                raise TypeError(
                    "gradient_sync_context must return a context manager"
                )
            # DDP requires the forward pass to be inside no_sync(), not only the
            # backward call.  The last objective-bearing microbatch deliberately
            # uses the normal context so exactly one gradient reduction occurs.
            with sync_context:
                started = perf_counter()
                with self._autocast_context(device):
                    new_log_probs = recompute_log_probs(micro_batch)
                recompute_time_s += perf_counter() - started
                objective_advantages, objective_log_probs = self._objective_inputs(
                    micro_advantages, new_log_probs
                )
                objective_context = (
                    nullcontext()
                    if self.precision == "fp32"
                    else torch.autocast(device_type=device.type, enabled=False)
                )
                with objective_context:
                    objective_output = self.objective(
                        micro_batch,
                        objective_advantages,
                        objective_log_probs,
                    )
                loss_metric += self._validate_loss(objective_output.loss) * weight
                micro_objective_metrics = self._objective_metrics(objective_output)
                micro_logprob_metrics = self._scalar_metrics(
                    "logprob",
                    self._logprob_metrics(micro_batch, objective_log_probs),
                )
                self._validate_pre_update(
                    logprob_metrics=micro_logprob_metrics,
                    objective_output=objective_output,
                )
                self._accumulate_objective_metrics(
                    objective_metric_numerators,
                    objective_metric_denominators,
                    micro_objective_metrics,
                    micro_batch,
                    micro_advantages,
                )
                self._accumulate_logprob_metrics(
                    logprob_metrics,
                    micro_logprob_metrics,
                    logprob_elements / total_logprob_elements,
                )

                if not gradients_initialized:
                    optimizer.zero_grad(set_to_none=True)
                    gradients_initialized = True
                started = perf_counter()
                scaled_loss = objective_output.loss * weight
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                backward_time_s += perf_counter() - started

        if scaler is not None:
            scaler.unscale_(optimizer)
        gradient_metrics = self._gradient_metrics(parameters)
        if self.require_finite_gradients and not gradient_metrics["gradients_finite"]:
            raise RuntimeError("Gradient gate failed: non-finite gradient detected")
        if (
            self.require_nonzero_gradients
            and int(gradient_metrics["grad_nonzero_count"]) == 0
        ):
            raise RuntimeError("Gradient gate failed: all gradients are zero")
        scalar_gradient_metrics = self._scalar_metrics(
            "gradient",
            {
                key: value
                for key, value in gradient_metrics.items()
                if key != "gradients_finite"
            },
        )
        scalar_gradient_metrics["gradients_finite"] = gradient_metrics[
            "gradients_finite"
        ]
        raw_clip_metrics = self._clip_gradients(parameters)
        clip_metrics = self._scalar_metrics(
            "gradient_clip",
            raw_clip_metrics,
        )
        if raw_clip_metrics:
            postclip_gradient_metrics = self._gradient_metrics(parameters)
            if not postclip_gradient_metrics["gradients_finite"]:
                raise RuntimeError(
                    "Gradient clip gate failed: non-finite gradient detected"
                )
            self._scalar_metric(
                "gradient_clip.grad_postclip_norm",
                postclip_gradient_metrics["grad_norm"],
            )

        runtime_metrics = {
            "update_microbatches": len(microbatches),
            "recompute_time_s": recompute_time_s,
            "backward_time_s": backward_time_s,
            "optimizer_time_s": 0.0,
        }
        objective_metrics = {
            key: numerator / objective_metric_denominators[key]
            for key, numerator in objective_metric_numerators.items()
        }
        metrics = self._merge_metric_sections(
            ("canonical", {"loss": loss_metric}),
            ("reward", reward_metrics),
            ("objective", objective_metrics),
            ("logprob", logprob_metrics),
            ("gradient", scalar_gradient_metrics),
            ("gradient_clip", clip_metrics),
            ("advantage", advantage_metrics),
            ("runtime", runtime_metrics),
        )
        if before_optimizer_step is not None:
            before_optimizer_step()
        started = perf_counter()

        def apply_optimizer_step() -> None:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

        if optimizer_step is None:
            apply_optimizer_step()
        else:
            optimizer_step(
                apply_optimizer_step,
                parameters=parameters,
                optimizer=optimizer,
                scaler=scaler,
            )
        metrics["optimizer_time_s"] = perf_counter() - started
        return metrics
