"""Minimal native-PyTorch distributed strategies for policy recomputation."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import math
import os
import pickle
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from numbers import Real
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


_RANK_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
_NON_NEGATIVE_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_COUNT_METRIC_SUFFIXES = (
    "_attempts",
    "_cancelled",
    "_hits",
    "_microbatches",
    "_misses",
    "_retries",
    "_shards",
    "_timeouts",
)
DEFAULT_MAX_ROLLBACK_SNAPSHOT_TENSOR_BYTES = 1 << 30


class DistributedFailureError(RuntimeError):
    """Raised on every rank when any rank reports a local failure."""


class _SnapshotLimitError(RuntimeError):
    def __init__(self, message: str, metrics: Mapping[str, int | float]) -> None:
        super().__init__(message)
        self.metrics = dict(metrics)


@dataclass(slots=True)
class _OptimizerStepSnapshot:
    """Distributed-only rollback state for one catchable optimizer update."""

    parameters: list[tuple[torch.nn.Parameter, torch.Tensor]]
    optimizer_state: dict[str, Any]
    scaler_state: dict[str, Any] | None
    stateful: Any | None
    stateful_state: dict[str, Any] | None
    metrics: dict[str, int]

    @classmethod
    def capture(
        cls,
        parameters: list[torch.nn.Parameter],
        optimizer: Any,
        scaler: Any | None,
        stateful: Any | None,
        max_tensor_bytes: int | None,
    ) -> _OptimizerStepSnapshot:
        protected_parameters: list[torch.nn.Parameter] = []
        seen_parameters: set[int] = set()
        for parameter in parameters:
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError("distributed update parameters must be Parameters")
            if id(parameter) not in seen_parameters:
                protected_parameters.append(parameter)
                seen_parameters.add(id(parameter))
        if stateful is not None and not all(
            callable(getattr(stateful, name, None))
            for name in ("state_dict", "load_state_dict")
        ):
            raise TypeError(
                "distributed update stateful must define state_dict/load_state_dict"
            )
        optimizer_state = cls._mapping_state_dict(optimizer, "optimizer")
        scaler_state = (
            None
            if scaler is None
            else cls._mapping_state_dict(scaler, "gradient scaler")
        )
        stateful_state = (
            None
            if stateful is None
            else cls._mapping_state_dict(stateful, "distributed update stateful")
        )
        metrics = {
            "parameter_count": len(protected_parameters),
            "parameter_tensor_bytes": sum(
                _tensor_storage_bytes(parameter) for parameter in protected_parameters
            ),
            "optimizer_state_tensor_bytes": _state_tensor_bytes(optimizer_state),
            "scaler_state_tensor_bytes": _state_tensor_bytes(scaler_state),
            "stateful_state_tensor_bytes": _state_tensor_bytes(stateful_state),
            "snapshot_limit_tensor_bytes": (
                0 if max_tensor_bytes is None else max_tensor_bytes
            ),
            "snapshot_limit_enabled": int(max_tensor_bytes is not None),
        }
        metrics["total_tensor_bytes"] = sum(
            metrics[name]
            for name in (
                "parameter_tensor_bytes",
                "optimizer_state_tensor_bytes",
                "scaler_state_tensor_bytes",
                "stateful_state_tensor_bytes",
            )
        )
        if (
            max_tensor_bytes is not None
            and metrics["total_tensor_bytes"] > max_tensor_bytes
        ):
            raise _SnapshotLimitError(
                "Distributed optimizer rollback snapshot requires "
                f"{metrics['total_tensor_bytes']} tensor bytes, exceeding the "
                f"per-rank limit of {max_tensor_bytes}; reduce trainable state or "
                "explicitly configure a larger audited limit",
                metrics,
            )
        return cls(
            parameters=[
                (parameter, parameter.detach().clone())
                for parameter in protected_parameters
            ],
            optimizer_state=copy.deepcopy(optimizer_state),
            scaler_state=copy.deepcopy(scaler_state),
            stateful=stateful,
            stateful_state=copy.deepcopy(stateful_state),
            metrics=metrics,
        )

    @staticmethod
    def _mapping_state_dict(owner: Any, label: str) -> Mapping[str, Any]:
        state_dict = getattr(owner, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(f"{label} must define state_dict()")
        state = state_dict()
        if not isinstance(state, Mapping):
            raise TypeError(f"{label} state_dict() must return a mapping")
        return state

    def restore(self, optimizer: Any, scaler: Any | None) -> None:
        errors: list[tuple[str, BaseException]] = []

        with torch.no_grad():
            for index, (parameter, value) in enumerate(self.parameters):
                try:
                    parameter.copy_(value)
                except BaseException as exc:
                    errors.append((f"parameter[{index}]", exc))

        try:
            optimizer.load_state_dict(self.optimizer_state)
        except BaseException as exc:
            errors.append(("optimizer", exc))

        if self.scaler_state is not None:
            if scaler is None:
                errors.append(
                    (
                        "gradient scaler",
                        RuntimeError(
                            "optimizer rollback is missing its gradient scaler"
                        ),
                    )
                )
            else:
                try:
                    scaler.load_state_dict(self.scaler_state)
                    per_optimizer_states = getattr(
                        scaler,
                        "_per_optimizer_states",
                        None,
                    )
                    if hasattr(per_optimizer_states, "clear"):
                        per_optimizer_states.clear()
                except BaseException as exc:
                    errors.append(("gradient scaler", exc))

        if self.stateful_state is not None:
            try:
                self.stateful.load_state_dict(copy.deepcopy(self.stateful_state))
            except BaseException as exc:
                errors.append(("stateful plugin", exc))

        if errors:
            details = "; ".join(
                f"{stage}: {type(error).__name__}: {error}" for stage, error in errors
            )
            rollback_error = RuntimeError("optimizer rollback failed: " + details)
            for stage, error in errors[1:]:
                add_note = getattr(rollback_error, "add_note", None)
                if callable(add_note):
                    add_note(
                        f"additional rollback failure in {stage}: "
                        f"{type(error).__name__}: {error}"
                    )
            raise rollback_error from errors[0][1]


def _tensor_storage_bytes(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().nbytes())
    except (AttributeError, NotImplementedError, RuntimeError):
        return int(value.numel() * value.element_size())


def _state_tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    if value is None:
        return 0
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return _tensor_storage_bytes(value)
    if isinstance(value, Mapping):
        return sum(_state_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_state_tensor_bytes(item, seen) for item in value)
    return 0


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Validated rank, device, and backend information for one process."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def step_seed(self, base_seed: int, step: int) -> int:
        """Return the rank-specific seed for a non-negative logical step."""

        for name, value in (("base_seed", base_seed), ("step", step)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        return base_seed + step * self.world_size + self.rank

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        device: str | torch.device | None = None,
        backend: str | None = None,
    ) -> DistributedContext:
        """Build a context from the complete torchrun rank environment.

        An absent rank environment selects a single process. A partial or malformed
        rank environment is rejected instead of silently falling back.
        """

        source = os.environ if env is None else env
        present = [key for key in _RANK_ENV_KEYS if key in source]
        if present and len(present) != len(_RANK_ENV_KEYS):
            missing = [key for key in _RANK_ENV_KEYS if key not in source]
            raise ValueError(
                "Incomplete distributed environment: missing " + ", ".join(missing)
            )

        if present:
            rank = cls._parse_rank_value("RANK", source["RANK"])
            local_rank = cls._parse_rank_value("LOCAL_RANK", source["LOCAL_RANK"])
            world_size = cls._parse_rank_value("WORLD_SIZE", source["WORLD_SIZE"])
        else:
            rank = 0
            local_rank = 0
            world_size = 1

        if world_size < 1:
            raise ValueError("WORLD_SIZE must be at least 1")
        if rank >= world_size:
            raise ValueError("RANK must be smaller than WORLD_SIZE")
        if world_size == 1 and (rank != 0 or local_rank != 0):
            raise ValueError("Single-process rank values must all be zero")

        resolved_device = cls._resolve_device(device, local_rank)
        resolved_backend = cls._resolve_backend(backend, resolved_device)
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=resolved_device,
            backend=resolved_backend,
        )

    @staticmethod
    def _parse_rank_value(name: str, raw: Any) -> int:
        if not isinstance(raw, str) or _NON_NEGATIVE_INTEGER.fullmatch(raw) is None:
            raise ValueError(f"{name} must be a canonical non-negative integer")
        return int(raw)

    @staticmethod
    def _resolve_device(
        requested: str | torch.device | None,
        local_rank: int,
    ) -> torch.device:
        if requested is None:
            device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = torch.device(requested)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", local_rank)

        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Distributed device must be CPU or CUDA")
        if device.type == "cpu" and device.index is not None:
            raise ValueError("CPU device must not have an index")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is unavailable")
            if device.index != local_rank:
                raise ValueError("CUDA device index must match LOCAL_RANK")
            if local_rank >= torch.cuda.device_count():
                raise ValueError(
                    "LOCAL_RANK does not identify an available CUDA device"
                )
        return device

    @staticmethod
    def _resolve_backend(requested: str | None, device: torch.device) -> str:
        backend = (
            ("nccl" if device.type == "cuda" else "gloo")
            if requested is None
            else requested
        )
        if not isinstance(backend, str) or not backend or backend != backend.lower():
            raise ValueError("Distributed backend must be a lowercase non-empty string")
        if device.type == "cpu" and backend == "nccl":
            raise ValueError("NCCL cannot be used with a CPU device")
        if backend not in {"gloo", "nccl"}:
            raise ValueError("Distributed backend must be 'gloo' or 'nccl'")
        return backend


class _AdapterRecomputeFacade(torch.nn.Module):
    """Register adapter parameters without replacing adapter-owned module links."""

    def __init__(self, adapter: Any, train_module: torch.nn.Module) -> None:
        super().__init__()
        self.train_module = train_module
        object.__setattr__(self, "_adapter", adapter)

    def forward(self, batch: Any) -> Any:
        return self._adapter.recompute_log_probs(batch)


class SingleProcessStrategy:
    """No-collective strategy preserving the existing one-process execution path."""

    def __init__(self, context: DistributedContext | None = None) -> None:
        self.context = context or DistributedContext.from_env()
        if self.context.is_distributed:
            raise ValueError("SingleProcessStrategy requires WORLD_SIZE=1")
        self._adapter: Any | None = None
        self._setup = False
        self._closed = False

    @property
    def is_main_process(self) -> bool:
        return self.context.is_main_process

    @property
    def closed(self) -> bool:
        return self._closed

    def setup(self) -> SingleProcessStrategy:
        self._require_open()
        self._setup = True
        return self

    def prepare(self, adapter: Any) -> Any:
        self._require_open()
        self.setup()
        self._validate_adapter(adapter)
        if self._adapter is not None and self._adapter is not adapter:
            raise RuntimeError("Strategy is already prepared with another adapter")
        self._adapter = adapter
        return adapter

    def forward(self, batch: Any) -> Any:
        self._require_prepared()
        return self._adapter.recompute_log_probs(batch)

    def recompute_log_probs(self, batch: Any) -> Any:
        return self.forward(batch)

    def gradient_sync_context(self, synchronize_gradients: bool):
        """Return the accumulation context for one objective-bearing microbatch."""

        self._require_prepared()
        if not isinstance(synchronize_gradients, bool):
            raise TypeError("synchronize_gradients must be a bool")
        return nullcontext()

    def reduce_weighted_mean(self, value: Any, weight: Any) -> float:
        self._require_open()
        scalar, scalar_weight = self._validated_scalar_and_weight(value, weight)
        if scalar_weight == 0:
            raise ValueError("Global reduction weight must be positive")
        return scalar

    def reduce_tensor_weighted_mean(
        self,
        value: torch.Tensor,
        weight: int,
    ) -> torch.Tensor:
        """Reduce a scalar mean without leaving its tensor dtype or device."""

        self._require_open()
        scalar, _ = self._validated_tensor_mean_and_weight(value, weight)
        return scalar.detach().clone()

    def reduce_weighted_scalar(self, value: Any, weight: Any) -> float:
        return self.reduce_weighted_mean(value, weight)

    def reduce_weighted_scalars(
        self,
        values: Mapping[str, Any],
        weight: Any,
    ) -> dict[str, float]:
        self._require_open()
        scalar_weight = self._validated_weight(weight)
        if scalar_weight == 0:
            raise ValueError("Global reduction weight must be positive")
        return self._validated_scalar_mapping(values)

    def reduce_metrics(
        self,
        metrics: Mapping[str, Any],
        sample_count: int,
        reward_values: Any | None = None,
    ) -> dict[str, float | bool]:
        """Reduce one rank's metrics using the shared metric-name contract.

        ``count`` and ``*_count`` are sums. Timing metrics named ``time``,
        ``*_time``, or ``*_time_s`` and ``*_abs_max`` metrics are maxima. Boolean
        metrics use logical AND. All other metrics are sample-weighted means.
        Supplying reward values computes global ``reward_mean`` and
        ``reward_std`` from moments; those names are reserved for that path.
        """

        self._require_open()
        scalars, kinds, count, rewards = self._prepare_metrics(
            metrics,
            sample_count,
            reward_values,
        )
        if any(kind == "mean" for kind in kinds.values()) and count == 0:
            raise ValueError("Global sample_count must be positive for mean metrics")

        result: dict[str, float | bool] = {}
        for key in sorted(scalars):
            if key not in {"reward_mean", "reward_std"} or rewards is None:
                result[key] = (
                    bool(scalars[key]) if kinds[key] == "bool_and" else scalars[key]
                )
        if rewards is not None:
            result.update(self._reward_moments(rewards))
        return result

    def metric_contract(
        self,
        metrics: Mapping[str, Any],
        sample_count: int,
        reward_values: Any | None = None,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], bool]:
        """Validate local metrics and return a small canonical rank contract.

        This helper deliberately performs no collective.  It is suitable for an
        ``atomic_optimizer_step`` result validator, which owns the synchronization
        needed to compare the returned contract before committing an update.
        """

        self._require_open()
        scalars, kinds, _, rewards = self._prepare_metrics(
            metrics,
            sample_count,
            reward_values,
        )
        return self._metric_contract_from_prepared(scalars, kinds, rewards)

    def broadcast_object(self, value: Any, *, src: int = 0) -> Any:
        """Return ``value`` unchanged in a single-process strategy."""

        self._validate_rank(src, name="src")
        return value

    def gather_object(self, value: Any, *, dst: int = 0) -> list[Any] | None:
        """Gather small metadata objects; media remains in rank-local artifacts."""

        self._validate_rank(dst, name="dst")
        return [value]

    def barrier(self) -> None:
        self._require_open()

    def synchronize_failure(self, failure: bool | BaseException | None) -> bool:
        self._require_open()
        if isinstance(failure, bool):
            return failure
        if failure is None:
            return False
        if not isinstance(failure, BaseException):
            raise TypeError("failure must be bool, an exception, or None")
        raise DistributedFailureError(
            self._failure_message([(0, failure)])
        ) from failure

    def atomic_optimizer_step(
        self,
        operation: Any,
        *,
        parameters: list[torch.nn.Parameter],
        optimizer: Any,
        scaler: Any | None = None,
        stateful: Any | None = None,
        validate_result: Any | None = None,
    ) -> Any:
        """Run an optimizer operation directly without single-process snapshots."""

        self._require_open()
        del parameters, optimizer, scaler, stateful
        if not callable(operation):
            raise TypeError("optimizer step operation must be callable")
        if validate_result is not None and not callable(validate_result):
            raise TypeError("optimizer result validator must be callable or None")
        result = operation()
        if validate_result is not None:
            validate_result(result)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._adapter = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Distributed strategy is closed")

    def _require_prepared(self) -> None:
        self._require_open()
        if not self._setup or self._adapter is None:
            raise RuntimeError("Distributed strategy must be prepared before forward")

    @staticmethod
    def _validate_adapter(adapter: Any) -> torch.nn.Module:
        if adapter is None or not callable(
            getattr(adapter, "recompute_log_probs", None)
        ):
            raise TypeError("adapter must define recompute_log_probs(batch)")
        module = getattr(adapter, "train_module", None)
        if not isinstance(module, torch.nn.Module):
            raise TypeError("adapter.train_module must be a torch.nn.Module")
        return module

    @staticmethod
    def _validated_scalar_and_weight(value: Any, weight: Any) -> tuple[float, float]:
        return SingleProcessStrategy._validated_scalar(
            value
        ), SingleProcessStrategy._validated_weight(weight)

    @staticmethod
    def _validated_tensor_mean_and_weight(
        value: Any,
        weight: Any,
    ) -> tuple[torch.Tensor, int]:
        if not isinstance(value, torch.Tensor):
            raise TypeError("Reduced tensor mean must be a torch.Tensor")
        if value.ndim != 0:
            raise ValueError("Reduced tensor mean must be scalar")
        if not value.is_floating_point() or value.is_complex():
            raise TypeError("Reduced tensor mean must be a real floating tensor")
        if not bool(torch.isfinite(value.detach()).item()):
            raise ValueError("Reduced tensor mean must be finite")
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise TypeError("Tensor reduction weight must be a positive integer")
        if weight <= 0:
            raise ValueError("Tensor reduction weight must be positive")
        return value.detach(), weight

    @staticmethod
    def _validated_scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Reduced values must be scalar")
            value = value.detach().item()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Reduced values must be real scalars")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("Reduced values must be finite")
        return scalar

    @staticmethod
    def _validated_weight(weight: Any) -> float:
        scalar = SingleProcessStrategy._validated_scalar(weight)
        if scalar < 0:
            raise ValueError("Reduction weight must be non-negative")
        return scalar

    @classmethod
    def _validated_scalar_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")
        result: dict[str, float] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Reduced scalar names must be non-empty strings")
            result[key] = cls._validated_scalar(value)
        return result

    @classmethod
    def _prepare_metrics(
        cls,
        metrics: Mapping[str, Any],
        sample_count: int,
        reward_values: Any | None,
    ) -> tuple[dict[str, float], dict[str, str], int, list[float] | None]:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        scalars: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Metric names must be non-empty strings")
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().item()
            if isinstance(value, bool):
                if key in {"reward_mean", "reward_std"}:
                    raise TypeError(f"Metric {key!r} must be a real scalar")
                scalars[key] = float(value)
            else:
                scalars[key] = cls._validated_scalar(value)
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise TypeError("sample_count must be an integer")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")

        kinds: dict[str, str] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().item()
            if isinstance(value, bool):
                kinds[key] = "bool_and"
            elif (
                key == "count"
                or key.endswith("_count")
                or key.endswith(_COUNT_METRIC_SUFFIXES)
            ):
                if scalars[key] < 0:
                    raise ValueError(f"Count metric {key!r} must be non-negative")
                kinds[key] = "sum"
            elif (
                key == "time"
                or key.endswith("_time")
                or key.endswith("_time_s")
                or ("latency" in key and key.endswith("_s"))
                or key.endswith("_abs_max")
                or (key.startswith("peak_") and key.endswith("_bytes"))
            ):
                kinds[key] = "max"
            else:
                kinds[key] = "mean"

        rewards = cls._validated_reward_values(reward_values)
        if rewards:
            cls._reward_moments(rewards)
        reward_keys = {"reward_mean", "reward_std"}.intersection(scalars)
        if reward_keys and rewards is None:
            names = ", ".join(sorted(reward_keys))
            raise ValueError(f"{names} require reward_values for global reduction")
        return scalars, kinds, sample_count, rewards

    @staticmethod
    def _metric_contract_from_prepared(
        scalars: Mapping[str, float],
        kinds: Mapping[str, str],
        rewards: list[float] | None,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], bool]:
        return (
            tuple(sorted(scalars)),
            tuple(sorted(kinds.items())),
            rewards is not None,
        )

    @classmethod
    def _validated_reward_values(cls, values: Any | None) -> list[float] | None:
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.detach().reshape(-1).tolist()
        elif isinstance(values, (str, bytes, Mapping)):
            raise TypeError("reward_values must be an iterable of real scalars")
        else:
            try:
                values = list(values)
            except TypeError as error:
                raise TypeError(
                    "reward_values must be an iterable of real scalars"
                ) from error
        return [cls._validated_scalar(value) for value in values]

    @staticmethod
    def _reward_moments(values: list[float]) -> dict[str, float]:
        if not values:
            raise ValueError("Global reward_values must not be empty")
        total = math.fsum(values)
        total_squared = math.fsum(value * value for value in values)
        if not math.isfinite(total) or not math.isfinite(total_squared):
            raise ValueError("Reward moments must be finite")
        mean = total / len(values)
        variance = max(total_squared / len(values) - mean * mean, 0.0)
        return {"reward_mean": mean, "reward_std": math.sqrt(variance)}

    def _validate_rank(self, rank: int, *, name: str) -> None:
        self._require_open()
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"{name} must be an integer rank")
        if rank < 0 or rank >= self.context.world_size:
            raise ValueError(f"{name} must identify a rank in the process group")

    @staticmethod
    def _failure_message(failures: list[tuple[int, BaseException]]) -> str:
        details = "; ".join(
            f"rank {rank}: {type(error).__name__}: {error}" for rank, error in failures
        )
        return "Distributed step failed: " + details


class DDPStrategy(SingleProcessStrategy):
    """Native DDP strategy wrapping adapter recomputation through a facade."""

    def __init__(
        self,
        context: DistributedContext,
        *,
        timeout_s: float = 30.0,
        max_snapshot_tensor_bytes: int | None = (
            DEFAULT_MAX_ROLLBACK_SNAPSHOT_TENSOR_BYTES
        ),
    ) -> None:
        if not context.is_distributed:
            raise ValueError("DDPStrategy requires WORLD_SIZE greater than 1")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
            raise TypeError("timeout_s must be a positive number")
        if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
            raise ValueError("timeout_s must be a positive finite number")
        if max_snapshot_tensor_bytes is not None and (
            isinstance(max_snapshot_tensor_bytes, bool)
            or not isinstance(max_snapshot_tensor_bytes, int)
        ):
            raise TypeError(
                "max_snapshot_tensor_bytes must be a positive integer or None"
            )
        if max_snapshot_tensor_bytes is not None and max_snapshot_tensor_bytes <= 0:
            raise ValueError("max_snapshot_tensor_bytes must be positive")

        self.context = context
        self.timeout_s = float(timeout_s)
        self.max_snapshot_tensor_bytes = max_snapshot_tensor_bytes
        self._last_atomic_snapshot_metrics: dict[str, int | float] | None = None
        self._adapter: Any | None = None
        self._facade: _AdapterRecomputeFacade | None = None
        self._ddp: DistributedDataParallel | None = None
        self._setup = False
        self._closed = False
        self._owns_process_group = False

    @property
    def module(self) -> DistributedDataParallel | None:
        return self._ddp

    @property
    def last_atomic_snapshot_metrics(self) -> dict[str, int | float] | None:
        if self._last_atomic_snapshot_metrics is None:
            return None
        return dict(self._last_atomic_snapshot_metrics)

    def setup(self) -> DDPStrategy:
        self._require_open()
        if self._setup:
            return self
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if self.context.backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("PyTorch was built without the gloo backend")
        if self.context.backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("PyTorch was built without the NCCL backend")

        if self.context.device.type == "cuda":
            torch.cuda.set_device(self.context.device)

        if dist.is_initialized():
            self._validate_existing_process_group()
        else:
            dist.init_process_group(
                backend=self.context.backend,
                rank=self.context.rank,
                world_size=self.context.world_size,
                timeout=timedelta(seconds=self.timeout_s),
            )
            self._owns_process_group = True
        self._setup = True
        return self

    def prepare(self, adapter: Any) -> Any:
        self._require_open()
        self.setup()
        if self._adapter is not None:
            if self._adapter is not adapter:
                raise RuntimeError("Strategy is already prepared with another adapter")
            return adapter

        train_module: torch.nn.Module | None = None
        local_error: BaseException | None = None
        try:
            train_module = self._validate_adapter(adapter)
            self._validate_module_device(train_module)
        except BaseException as exc:
            local_error = exc
        self.synchronize_failure(local_error)
        if train_module is None:
            raise RuntimeError("distributed adapter preflight lost local state")

        facade = _AdapterRecomputeFacade(adapter, train_module)
        ddp_kwargs: dict[str, Any] = {}
        if self.context.device.type == "cuda":
            ddp_kwargs.update(
                device_ids=[self.context.local_rank],
                output_device=self.context.local_rank,
            )
        self._facade = facade
        self._ddp = DistributedDataParallel(facade, **ddp_kwargs)
        self._adapter = adapter
        return adapter

    def forward(self, batch: Any) -> Any:
        self._require_prepared()
        return self._ddp(batch)

    def gradient_sync_context(self, synchronize_gradients: bool):
        """Suppress DDP reductions until the final contributing microbatch."""

        self._require_prepared()
        if not isinstance(synchronize_gradients, bool):
            raise TypeError("synchronize_gradients must be a bool")
        if synchronize_gradients:
            return nullcontext()
        return self._ddp.no_sync()

    def reduce_weighted_mean(self, value: Any, weight: Any) -> float:
        self._require_collectives_ready()
        validated: tuple[float, float] | None = None
        local_error: BaseException | None = None
        try:
            validated = self._validated_scalar_and_weight(value, weight)
        except BaseException as exc:
            local_error = exc
        self.synchronize_failure(local_error)
        if validated is None:
            raise RuntimeError("distributed scalar preflight lost local state")
        scalar, scalar_weight = validated
        reduced = torch.tensor(
            [scalar * scalar_weight, scalar_weight],
            dtype=torch.float64,
            device=self.context.device,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        denominator = float(reduced[1].item())
        if denominator <= 0:
            raise ValueError("Global reduction weight must be positive")
        return float((reduced[0] / reduced[1]).item())

    def reduce_tensor_weighted_mean(
        self,
        value: torch.Tensor,
        weight: int,
    ) -> torch.Tensor:
        """Reduce a scalar tensor mean while preserving reference arithmetic."""

        self._require_collectives_ready()
        validated: tuple[torch.Tensor, int] | None = None
        local_error: BaseException | None = None
        try:
            validated = self._validated_tensor_mean_and_weight(value, weight)
            if validated[0].device != self.context.device:
                raise ValueError(
                    "Reduced tensor mean must be on the distributed context device"
                )
        except BaseException as exc:
            local_error = exc
        self.synchronize_failure(local_error)
        if validated is None:
            raise RuntimeError("distributed tensor reduction preflight lost local state")

        scalar, scalar_weight = validated
        local_contract = (
            str(scalar.dtype),
            scalar.device.type,
            scalar.device.index,
            scalar_weight,
        )
        contracts: list[tuple[str, str, int | None, int] | None] = [
            None
        ] * self.context.world_size
        dist.all_gather_object(contracts, local_contract)
        tensor_contracts = [contract[:3] for contract in contracts if contract]
        if len(tensor_contracts) != self.context.world_size or any(
            contract != tensor_contracts[0]
            for contract in tensor_contracts[1:]
        ):
            raise ValueError(
                "Reduced tensor dtype and device must match on every rank"
            )
        weights = [contract[3] for contract in contracts if contract]
        if len(weights) != self.context.world_size:
            raise RuntimeError("distributed tensor reduction contract was incomplete")

        reduced = scalar.clone()
        if all(item == weights[0] for item in weights[1:]):
            # This is the published reference path: reduce each rank-local mean,
            # then divide by world size in the coefficient tensor's dtype.
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced.div_(self.context.world_size)
        else:
            # Uneven rank batches are a VisualRL extension: preserve the tensor
            # dtype/device while using the true global sample-weighted mean.
            numerator_and_count = torch.stack(
                (
                    reduced * scalar_weight,
                    reduced.new_tensor(scalar_weight),
                )
            )
            dist.all_reduce(numerator_and_count, op=dist.ReduceOp.SUM)
            if not bool(torch.isfinite(numerator_and_count).all().item()):
                raise ValueError("Reduced tensor sum/count must be finite")
            if not bool((numerator_and_count[1] > 0).item()):
                raise ValueError("Global tensor reduction weight must be positive")
            reduced = numerator_and_count[0] / numerator_and_count[1]
        if not bool(torch.isfinite(reduced).item()):
            raise ValueError("Reduced tensor mean must be finite")
        return reduced.detach()

    def reduce_weighted_scalars(
        self,
        values: Mapping[str, Any],
        weight: Any,
    ) -> dict[str, float]:
        self._require_collectives_ready()
        validated: tuple[dict[str, float], float] | None = None
        local_error: BaseException | None = None
        try:
            validated = (
                self._validated_scalar_mapping(values),
                self._validated_weight(weight),
            )
        except BaseException as exc:
            local_error = exc
        self.synchronize_failure(local_error)
        if validated is None:
            raise RuntimeError("distributed scalar mapping preflight lost local state")
        scalars, scalar_weight = validated
        key_sets: list[list[str]] = [None] * self.context.world_size  # type: ignore[list-item]
        dist.all_gather_object(key_sets, sorted(scalars))
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError("Reduced scalar keys must match on every rank")
        keys = key_sets[0]
        reduced = torch.tensor(
            [*(scalars[key] * scalar_weight for key in keys), scalar_weight],
            dtype=torch.float64,
            device=self.context.device,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        denominator = float(reduced[-1].item())
        if denominator <= 0:
            raise ValueError("Global reduction weight must be positive")
        return {
            key: float((reduced[index] / reduced[-1]).item())
            for index, key in enumerate(keys)
        }

    def reduce_metrics(
        self,
        metrics: Mapping[str, Any],
        sample_count: int,
        reward_values: Any | None = None,
    ) -> dict[str, float | bool]:
        self._require_collectives_ready()
        prepared: (
            tuple[
                dict[str, float],
                dict[str, str],
                int,
                list[float] | None,
            ]
            | None
        ) = None
        local_error: tuple[str, str] | None = None
        try:
            prepared = self._prepare_metrics(metrics, sample_count, reward_values)
        except BaseException as error:
            local_error = (type(error).__name__, str(error))

        local_contract = None
        if prepared is not None:
            scalars, kinds, _, rewards = prepared
            local_contract = self._metric_contract_from_prepared(
                scalars,
                kinds,
                rewards,
            )
        local_state = (local_error, local_contract)
        gathered_states: list[Any] = [None] * self.context.world_size
        dist.all_gather_object(gathered_states, local_state)

        errors = [
            (rank, state[0])
            for rank, state in enumerate(gathered_states)
            if state[0] is not None
        ]
        if errors:
            details = "; ".join(
                f"rank {rank}: {name}: {message}" for rank, (name, message) in errors
            )
            raise ValueError("Invalid distributed metrics: " + details)

        contracts = [state[1] for state in gathered_states]
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise ValueError(
                "Metric keys, reduction kinds, and reward_values presence must "
                "match on every rank"
            )
        if prepared is None:
            raise RuntimeError("Distributed metric validation lost local state")

        scalars, kinds, count, rewards = prepared
        mean_keys = sorted(key for key, kind in kinds.items() if kind == "mean")
        sum_keys = sorted(key for key, kind in kinds.items() if kind == "sum")
        max_keys = sorted(key for key, kind in kinds.items() if kind == "max")
        bool_keys = sorted(key for key, kind in kinds.items() if kind == "bool_and")
        reserved = {"reward_mean", "reward_std"} if rewards is not None else set()
        mean_keys = [key for key in mean_keys if key not in reserved]

        sum_values = [scalars[key] * count for key in mean_keys]
        sum_values.extend(scalars[key] for key in sum_keys)
        if mean_keys:
            sum_values.append(float(count))
        if rewards is not None:
            try:
                reward_sum = math.fsum(rewards)
                reward_sum_squared = math.fsum(value * value for value in rewards)
            except OverflowError as error:
                raise ValueError("Reward moments must be finite") from error
            sum_values.extend((reward_sum, reward_sum_squared, float(len(rewards))))

        result: dict[str, float | bool] = {}
        if sum_values:
            reduced_sums = torch.tensor(
                sum_values,
                dtype=torch.float64,
                device=self.context.device,
            )
            dist.all_reduce(reduced_sums, op=dist.ReduceOp.SUM)
            if not bool(torch.isfinite(reduced_sums).all().item()):
                raise ValueError("Reduced metric sums must be finite")

            offset = 0
            mean_numerators = reduced_sums[offset : offset + len(mean_keys)]
            offset += len(mean_keys)
            for key, value in zip(
                sum_keys,
                reduced_sums[offset : offset + len(sum_keys)],
                strict=True,
            ):
                result[key] = float(value.item())
            offset += len(sum_keys)
            if mean_keys:
                total_count = float(reduced_sums[offset].item())
                offset += 1
                if total_count <= 0:
                    raise ValueError(
                        "Global sample_count must be positive for mean metrics"
                    )
                for key, value in zip(mean_keys, mean_numerators, strict=True):
                    result[key] = float(value.item() / total_count)
            if rewards is not None:
                reward_sum, reward_sum_squared, reward_count = (
                    float(value.item()) for value in reduced_sums[offset : offset + 3]
                )
                if reward_count <= 0:
                    raise ValueError("Global reward_values must not be empty")
                reward_mean = reward_sum / reward_count
                reward_variance = max(
                    reward_sum_squared / reward_count - reward_mean * reward_mean,
                    0.0,
                )
                result.update(
                    reward_mean=reward_mean,
                    reward_std=math.sqrt(reward_variance),
                )

        if max_keys:
            reduced_maxima = torch.tensor(
                [scalars[key] for key in max_keys],
                dtype=torch.float64,
                device=self.context.device,
            )
            dist.all_reduce(reduced_maxima, op=dist.ReduceOp.MAX)
            if not bool(torch.isfinite(reduced_maxima).all().item()):
                raise ValueError("Reduced metric maxima must be finite")
            result.update(
                (key, float(value.item()))
                for key, value in zip(max_keys, reduced_maxima, strict=True)
            )

        if bool_keys:
            reduced_booleans = torch.tensor(
                [int(bool(scalars[key])) for key in bool_keys],
                dtype=torch.int32,
                device=self.context.device,
            )
            dist.all_reduce(reduced_booleans, op=dist.ReduceOp.MIN)
            result.update(
                (key, bool(value.item()))
                for key, value in zip(bool_keys, reduced_booleans, strict=True)
            )
        return {key: result[key] for key in sorted(result)}

    def broadcast_object(self, value: Any, *, src: int = 0) -> Any:
        self._require_collectives_ready()
        self._synchronize_object_collective_contract(
            operation="broadcast",
            root=src,
            root_name="src",
        )
        self._preflight_object(
            value if self.context.rank == src else None,
            operation="broadcast",
        )
        payload = [value if self.context.rank == src else None]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]

    def gather_object(self, value: Any, *, dst: int = 0) -> list[Any] | None:
        """Gather small metadata objects; media remains in rank-local artifacts."""

        self._require_collectives_ready()
        self._synchronize_object_collective_contract(
            operation="gather",
            root=dst,
            root_name="dst",
        )
        self._preflight_object(value, operation="gather")
        gathered = (
            [None] * self.context.world_size if self.context.rank == dst else None
        )
        dist.gather_object(value, gathered, dst=dst)
        return gathered

    def barrier(self) -> None:
        self._require_collectives_ready()
        dist.barrier()

    def synchronize_failure(self, failure: bool | BaseException | None) -> bool:
        self._require_collectives_ready()
        if not isinstance(failure, (bool, BaseException)) and failure is not None:
            raise TypeError("failure must be bool, an exception, or None")

        if isinstance(failure, bool):
            failed = torch.tensor(
                int(failure),
                dtype=torch.int32,
                device=self.context.device,
            )
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            return bool(failed.item())

        failures = self._gather_failure_details(failure)
        if failures:
            error = DistributedFailureError(
                self._format_failure_details("Distributed step failed", failures)
            )
            if failure is not None:
                raise error from failure
            raise error
        return False

    def atomic_optimizer_step(
        self,
        operation: Any,
        *,
        parameters: list[torch.nn.Parameter],
        optimizer: Any,
        scaler: Any | None = None,
        stateful: Any | None = None,
        validate_result: Any | None = None,
    ) -> Any:
        """Run and coordinate a catchable optimizer update with rollback.

        Snapshots exist only on the distributed path. A hard process death or a
        failed process-group collective cannot be rolled back; recovery then resumes
        from the last authoritative artifact commit marker.  When supplied,
        ``validate_result`` must be local-only and return a small canonical,
        pickleable contract.  Validation failures and cross-rank contract mismatches
        are rolled back inside the same snapshot boundary as the optimizer call.
        """

        self._require_collectives_ready()
        callable_error: BaseException | None = None
        if not callable(operation):
            callable_error = TypeError("optimizer step operation must be callable")
        elif validate_result is not None and not callable(validate_result):
            callable_error = TypeError(
                "optimizer result validator must be callable or None"
            )
        self.synchronize_failure(callable_error)

        validator_presence: list[bool] = [False] * self.context.world_size
        dist.all_gather_object(validator_presence, validate_result is not None)
        if any(value != validator_presence[0] for value in validator_presence[1:]):
            self.synchronize_failure(
                ValueError(
                    "optimizer result validator presence must match on every rank"
                )
            )

        self._last_atomic_snapshot_metrics = None
        snapshot: _OptimizerStepSnapshot | None = None
        snapshot_error: BaseException | None = None
        snapshot_started = perf_counter()
        try:
            snapshot = _OptimizerStepSnapshot.capture(
                parameters,
                optimizer,
                scaler,
                stateful,
                self.max_snapshot_tensor_bytes,
            )
        except BaseException as exc:
            snapshot_error = exc
            if isinstance(exc, _SnapshotLimitError):
                self._last_atomic_snapshot_metrics = dict(exc.metrics)
        if snapshot is not None:
            self._last_atomic_snapshot_metrics = dict(snapshot.metrics)
        if self._last_atomic_snapshot_metrics is not None:
            self._last_atomic_snapshot_metrics["capture_time_s"] = (
                perf_counter() - snapshot_started
            )
        self.synchronize_failure(snapshot_error)
        if snapshot is None:
            raise RuntimeError("optimizer rollback snapshot lost local state")

        result: Any = None
        step_error: BaseException | None = None
        try:
            result = operation()
        except BaseException as exc:
            step_error = exc

        if self.synchronize_failure(step_error is not None):
            self._rollback_atomic_failure(
                snapshot,
                optimizer,
                scaler,
                step_error,
                failure_heading="Distributed step failed",
                original_heading="update failure",
            )

        if validate_result is None:
            return result

        validation_error: BaseException | None = None
        serialized_contract: bytes | None = None
        try:
            contract = validate_result(result)
            serialized_contract = pickle.dumps(
                contract,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except BaseException as exc:
            validation_error = exc

        if self.synchronize_failure(validation_error is not None):
            self._rollback_atomic_failure(
                snapshot,
                optimizer,
                scaler,
                validation_error,
                failure_heading="Distributed optimizer result validation failed",
                original_heading="result validation failure",
            )
        if serialized_contract is None:
            raise RuntimeError("optimizer result validation lost its local contract")

        contracts: list[bytes] = [b""] * self.context.world_size
        dist.all_gather_object(contracts, serialized_contract)
        if any(contract != contracts[0] for contract in contracts[1:]):
            contract_error = ValueError(
                "optimizer result contracts must match on every rank"
            )
            self._rollback_atomic_failure(
                snapshot,
                optimizer,
                scaler,
                contract_error,
                failure_heading="Distributed optimizer result validation failed",
                original_heading="result contract mismatch",
            )
        return result

    def _rollback_atomic_failure(
        self,
        snapshot: _OptimizerStepSnapshot,
        optimizer: Any,
        scaler: Any | None,
        failure: BaseException | None,
        *,
        failure_heading: str,
        original_heading: str,
    ) -> None:
        failures = self._gather_failure_details(failure)
        if not failures:
            raise RuntimeError("distributed optimizer failure lost its cause")

        restore_error: BaseException | None = None
        restore_started = perf_counter()
        try:
            snapshot.restore(optimizer, scaler)
        except BaseException as exc:
            restore_error = exc
        if self._last_atomic_snapshot_metrics is not None:
            self._last_atomic_snapshot_metrics["restore_time_s"] = (
                perf_counter() - restore_started
            )
        restore_failures = self._gather_failure_details(restore_error)
        if restore_failures:
            message = self._format_failure_details(
                "Distributed optimizer rollback failed",
                restore_failures,
            )
            message += "; original " + self._format_failure_details(
                original_heading,
                failures,
            )
            error = DistributedFailureError(message)
            cause = restore_error if restore_error is not None else failure
            if cause is not None:
                raise error from cause
            raise error

        error = DistributedFailureError(
            self._format_failure_details(failure_heading, failures)
        )
        if failure is not None:
            raise error from failure
        raise error

    def _gather_failure_details(
        self,
        failure: BaseException | None,
    ) -> list[tuple[int, str, str]]:
        local = None if failure is None else (type(failure).__name__, str(failure))
        gathered: list[tuple[str, str] | None] = [None] * self.context.world_size
        dist.all_gather_object(gathered, local)
        return [
            (rank, name, message)
            for rank, item in enumerate(gathered)
            if item is not None
            for name, message in [item]
        ]

    @staticmethod
    def _format_failure_details(
        prefix: str,
        failures: list[tuple[int, str, str]],
    ) -> str:
        details = "; ".join(
            f"rank {rank}: {name}: {message}" for rank, name, message in failures
        )
        return f"{prefix}: {details}"

    def _synchronize_object_collective_contract(
        self,
        *,
        operation: str,
        root: Any,
        root_name: str,
    ) -> None:
        local_error: BaseException | None = None
        try:
            self._validate_rank(root, name=root_name)
        except BaseException as exc:
            local_error = exc

        local_state = (
            operation,
            root_name,
            int(root) if local_error is None else None,
            (
                None
                if local_error is None
                else (type(local_error).__name__, str(local_error))
            ),
        )
        gathered_states: list[Any] = [None] * self.context.world_size
        dist.all_gather_object(gathered_states, local_state)

        invalid = [
            (rank, state[3])
            for rank, state in enumerate(gathered_states)
            if state[3] is not None
        ]
        if invalid:
            details = "; ".join(
                f"rank {rank}: {name}: {message}" for rank, (name, message) in invalid
            )
            error = DistributedFailureError(
                "Invalid distributed object collective root: " + details
            )
            if local_error is not None:
                raise error from local_error
            raise error

        contracts = [state[:3] for state in gathered_states]
        if any(contract != contracts[0] for contract in contracts[1:]):
            details = "; ".join(
                f"rank {rank}: {state[0]}({state[1]}={state[2]})"
                for rank, state in enumerate(gathered_states)
            )
            raise DistributedFailureError(
                "Distributed object collective operation and root rank must match "
                f"on every rank: {details}"
            )

    def _preflight_object(self, value: Any, *, operation: str) -> None:
        local_error: BaseException | None = None
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:
            local_error = TypeError(
                f"distributed {operation} object is not serializable: {exc}"
            )
        self.synchronize_failure(local_error)

    def close(self) -> None:
        if self._closed:
            return
        self._ddp = None
        self._facade = None
        self._adapter = None
        if self._owns_process_group and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_process_group = False
        self._closed = True

    def _validate_existing_process_group(self) -> None:
        backend = str(dist.get_backend()).lower()
        if backend != self.context.backend:
            raise RuntimeError(
                f"Existing process group backend is {backend!r}, expected "
                f"{self.context.backend!r}"
            )
        if dist.get_rank() != self.context.rank:
            raise RuntimeError("Existing process group rank does not match context")
        if dist.get_world_size() != self.context.world_size:
            raise RuntimeError(
                "Existing process group world size does not match context"
            )

    def _require_collectives_ready(self) -> None:
        self._require_open()
        if not self._setup or not dist.is_initialized():
            raise RuntimeError("DDP strategy must be set up before collectives")

    def _validate_module_device(self, module: torch.nn.Module) -> None:
        tensors = [*module.parameters(), *module.buffers()]
        mismatched = [
            tensor.device for tensor in tensors if tensor.device != self.context.device
        ]
        if mismatched:
            raise ValueError(
                f"adapter.train_module must already be on {self.context.device}; "
                f"found {mismatched[0]}"
            )
        if not any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("adapter.train_module has no trainable parameters")
