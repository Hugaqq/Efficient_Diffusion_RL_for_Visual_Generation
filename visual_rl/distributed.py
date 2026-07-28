"""The sole single-process/DDP strategy surface used by VisualRL training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import math
from numbers import Real
import pickle
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from visual_rl.core.types import (
    MetricContribution,
    PolicyRecomputeStats,
    RewardBatch,
    RolloutBatch,
    ValidatedRuntimeEnv,
)


DEFAULT_MAX_ROLLBACK_SNAPSHOT_TENSOR_BYTES = 1 << 30
_T = TypeVar("_T")

__all__ = (
    "DDPStrategy",
    "DistributedContext",
    "DistributedFailureError",
    "SingleProcessStrategy",
    "build_strategy",
)


class DistributedFailureError(RuntimeError):
    """An ordinary rank-local failure synchronized to every worker."""


class _ProcessGroupFatalError(DistributedFailureError):
    """A collective/reducer failure after which consensus is no longer safe."""


class _SnapshotLimitError(RuntimeError):
    """The configured DDP rollback-copy budget is insufficient."""


@dataclass(slots=True)
class _OptimizerStepSnapshot:
    """Rank-local rollback data captured before the sole mutable boundary."""

    parameters: tuple[tuple[torch.nn.Parameter, torch.Tensor], ...]
    optimizer_state: dict[str, Any]
    scaler_state: dict[str, Any] | None

    @classmethod
    def capture(
        cls,
        parameters: tuple[torch.nn.Parameter, ...],
        optimizer: torch.optim.AdamW,
        scaler: Any | None,
        *,
        max_tensor_bytes: int | None,
    ) -> _OptimizerStepSnapshot:
        optimizer_state = cls._mapping_state_dict(optimizer, "optimizer")
        scaler_state = (
            None
            if scaler is None
            else cls._mapping_state_dict(scaler, "gradient scaler")
        )
        tensor_bytes = (
            sum(_tensor_storage_bytes(parameter) for parameter in parameters)
            + _state_tensor_bytes(optimizer_state)
            + _state_tensor_bytes(scaler_state)
        )
        if max_tensor_bytes is not None and tensor_bytes > max_tensor_bytes:
            raise _SnapshotLimitError(
                "Distributed optimizer rollback snapshot requires "
                f"{tensor_bytes} tensor bytes, exceeding the per-rank limit "
                f"of {max_tensor_bytes}"
            )

        # The byte gate above must precede every clone/deepcopy.
        return cls(
            parameters=tuple(
                (parameter, parameter.detach().clone())
                for parameter in parameters
            ),
            optimizer_state=copy.deepcopy(dict(optimizer_state)),
            scaler_state=(
                None
                if scaler_state is None
                else copy.deepcopy(dict(scaler_state))
            ),
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

    def restore(
        self,
        optimizer: torch.optim.AdamW,
        scaler: Any | None,
    ) -> None:
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
                        RuntimeError("rollback is missing its gradient scaler"),
                    )
                )
            else:
                try:
                    scaler.load_state_dict(self.scaler_state)
                    per_optimizer = getattr(
                        scaler,
                        "_per_optimizer_states",
                        None,
                    )
                    if hasattr(per_optimizer, "clear"):
                        per_optimizer.clear()
                except BaseException as exc:
                    errors.append(("gradient scaler", exc))
        if errors:
            details = "; ".join(
                f"{stage}: {type(error).__name__}: {error}"
                for stage, error in errors
            )
            raise RuntimeError(
                "distributed optimizer rollback failed: " + details
            ) from errors[0][1]


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
        return sum(
            _state_tensor_bytes(key, seen) + _state_tensor_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(_state_tensor_bytes(item, seen) for item in value)
    return 0


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Rank-local topology projected only from validated Preflight state."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def build_strategy(
    distributed_config: Any,
    validated_env: ValidatedRuntimeEnv,
) -> SingleProcessStrategy | DDPStrategy:
    """Build the only Strategy from the cached validated environment."""

    if not isinstance(validated_env, ValidatedRuntimeEnv):
        raise TypeError("validated_env must be a ValidatedRuntimeEnv")
    mode = getattr(distributed_config, "mode", None)
    requested_device = getattr(distributed_config, "device", None)
    timeout_s = getattr(distributed_config, "timeout_s", None)
    snapshot_limit = getattr(
        distributed_config,
        "max_snapshot_tensor_bytes",
        None,
    )
    if mode not in {"single", "ddp"}:
        raise ValueError("distributed mode must be 'single' or 'ddp'")
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("distributed device must be 'cpu' or 'cuda'")
    if validated_env.mode != mode:
        raise ValueError("validated runtime mode does not match canonical config")
    if mode == "single" and snapshot_limit is not None:
        raise ValueError(
            "single mode requires max_snapshot_tensor_bytes to be None"
        )
    _validate_snapshot_shape(validated_env)

    from visual_rl.preflight import backend_for

    context = DistributedContext(
        rank=validated_env.rank,
        local_rank=validated_env.local_rank,
        world_size=validated_env.world_size,
        device=_rank_local_device(requested_device, validated_env),
        backend=backend_for(mode, requested_device),
    )
    if dist.is_available() and dist.is_initialized():
        raise RuntimeError(
            "VisualRL requires ownership of an uninitialized process group"
        )
    strategy: SingleProcessStrategy | DDPStrategy
    if mode == "single":
        strategy = SingleProcessStrategy(context)
    else:
        strategy = DDPStrategy(
            context,
            timeout_s=timeout_s,
            max_snapshot_tensor_bytes=snapshot_limit,
            master_addr=validated_env.master_addr,
            master_port=validated_env.master_port,
        )
    try:
        strategy._setup()
    except BaseException:
        try:
            strategy.close()
        except BaseException:
            pass
        raise
    return strategy


def _validate_snapshot_shape(env: ValidatedRuntimeEnv) -> None:
    if env.mode == "single":
        actual = (env.rank, env.local_rank, env.world_size, env.local_world_size)
        if actual != (0, 0, 1, 1):
            raise ValueError("single validated runtime snapshot has invalid ranks")
        if any(
            value is not None
            for value in (
                env.group_rank,
                env.group_world_size,
                env.master_addr,
                env.master_port,
            )
        ):
            raise ValueError(
                "single validated runtime snapshot must not contain launch metadata"
            )
        return
    if (env.world_size, env.local_world_size) != (2, 2):
        raise ValueError("ddp validated runtime snapshot must be single-node size 2")
    if not 0 <= env.rank < 2 or not 0 <= env.local_rank < 2:
        raise ValueError("ddp validated runtime snapshot has invalid ranks")
    if env.group_rank not in (None, 0) or env.group_world_size not in (None, 1):
        raise ValueError("ddp validated runtime snapshot is not single-node")
    if not isinstance(env.master_addr, str) or not env.master_addr:
        raise ValueError("ddp validated runtime snapshot requires master_addr")
    if type(env.master_port) is not int or not 1 <= env.master_port <= 65535:
        raise ValueError("ddp validated runtime snapshot requires valid master_port")


def _rank_local_device(
    requested: str,
    env: ValidatedRuntimeEnv,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    index = 0 if env.mode == "single" else env.local_rank
    if env.visible_gpu_count <= index:
        raise RuntimeError(
            "validated runtime snapshot does not expose the requested CUDA device"
        )
    return torch.device("cuda", index)


class _AdapterRecomputeFacade(torch.nn.Module):
    """Register trainable tensors with DDP while preserving Adapter ownership."""

    def __init__(self, adapter: Any, train_module: torch.nn.Module) -> None:
        super().__init__()
        self.train_module = train_module
        object.__setattr__(self, "_adapter", adapter)

    def forward(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        return self._adapter.recompute_policy_stats(
            batch,
            require_reference=require_reference,
        )


class _FinalGradientSyncContext:
    """Mark a last-slot DDP backward/reducer failure as process-group fatal."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, traceback
        if exc is None or isinstance(exc, DistributedFailureError):
            return False
        raise _ProcessGroupFatalError(
            "DDP final backward/reducer failed"
        ) from exc


class SingleProcessStrategy:
    """Identity/no-op implementation of the one training Strategy contract."""

    def __init__(self, context: DistributedContext) -> None:
        if not isinstance(context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if context.is_distributed or context.backend is not None:
            raise ValueError("SingleProcessStrategy requires WORLD_SIZE=1")
        self._context = context
        self._adapter: Any | None = None
        self._setup_complete = False
        self._closed = False

    @property
    def rank(self) -> int:
        return self._context.rank

    @property
    def local_rank(self) -> int:
        return self._context.local_rank

    @property
    def world_size(self) -> int:
        return self._context.world_size

    @property
    def device(self) -> torch.device:
        return self._context.device

    @property
    def backend(self) -> str | None:
        return self._context.backend

    @property
    def is_main_process(self) -> bool:
        return self._context.is_main_process

    def _setup(self) -> None:
        self._require_open()
        self._setup_complete = True

    def run_phase(
        self,
        name: str,
        operation: Callable[[], _T],
    ) -> _T:
        _validate_phase_name(name)
        if not callable(operation):
            raise TypeError("phase operation must be callable")
        try:
            return operation()
        except _ProcessGroupFatalError:
            raise
        except BaseException as exc:
            self.failure_gate(name, exc)
            raise AssertionError("failure_gate returned after a failure") from exc

    def dataset_start(self, step: int, batch_size: int) -> int:
        for name, value in (("step", step), ("batch_size", batch_size)):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer, not bool")
        if step < 0 or batch_size <= 0:
            raise ValueError("step must be non-negative and batch_size positive")
        return step * batch_size * self.world_size + self.rank * batch_size

    def prepare(self, adapter: Any) -> Any:
        self._require_open()
        if not self._setup_complete:
            raise RuntimeError("Strategy must be set up before prepare()")
        self._validate_adapter(adapter)
        if self._adapter is not None and self._adapter is not adapter:
            raise RuntimeError("Strategy is already prepared with another adapter")
        self._adapter = adapter
        return adapter

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        self._require_prepared()
        self._validate_recompute_request(batch, require_reference)
        stats = self._adapter.recompute_policy_stats(
            batch,
            require_reference=require_reference,
        )
        return self._validate_recompute_result(
            stats,
            batch,
            require_reference=require_reference,
        )

    def gradient_sync_context(self, synchronize_gradients: bool):
        self._require_prepared()
        if not isinstance(synchronize_gradients, bool):
            raise TypeError("synchronize_gradients must be a bool")
        return nullcontext()

    def sum_active_transition_count(self, local_count: int) -> int:
        return _positive_count(local_count, name="local active transition count")

    def reduce_tensor_weighted_mean(
        self,
        value: torch.Tensor,
        weight: int,
    ) -> torch.Tensor:
        scalar, _ = self._validated_tensor_mean_and_weight(value, weight)
        return scalar.detach().clone()

    def reduce_metric_contributions(
        self,
        contributions: Mapping[str, MetricContribution],
    ) -> dict[str, float]:
        prepared = self._prepare_metric_contributions(contributions)
        return {
            name: (
                numerator
                if denominator is None
                else numerator / denominator
            )
            for name, numerator, denominator in prepared
        }

    def reduce_reward_metrics(self, rewards: RewardBatch) -> dict[str, float]:
        values = self._reward_values(rewards)
        mean = math.fsum(values) / len(values)
        variance = max(
            math.fsum((value - mean) ** 2 for value in values) / len(values),
            0.0,
        )
        return {"reward_mean": mean, "reward_std": math.sqrt(variance)}

    def atomic_optimizer_step(
        self,
        operation: Callable[[], None],
        *,
        parameters: tuple[torch.nn.Parameter, ...],
        optimizer: torch.optim.AdamW,
        scaler: Any | None,
    ) -> None:
        self._validate_optimizer_boundary(
            operation,
            parameters=parameters,
            optimizer=optimizer,
            scaler=scaler,
        )
        operation()

    def gather_object(self, value: Any, *, dst: int = 0) -> list[Any] | None:
        self._validate_rank(dst, name="dst")
        return [value]

    def broadcast_object(self, value: Any, *, src: int = 0) -> Any:
        self._validate_rank(src, name="src")
        return value

    def failure_gate(
        self,
        phase: str,
        failure: BaseException | None,
    ) -> None:
        _validate_phase_name(phase)
        if failure is None:
            return
        if not isinstance(failure, BaseException):
            raise TypeError("failure must be an exception or None")
        raise failure

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
        if not self._setup_complete or self._adapter is None:
            raise RuntimeError(
                "Distributed strategy must be prepared before policy recompute"
            )

    @staticmethod
    def _validate_adapter(adapter: Any) -> torch.nn.Module:
        if adapter is None or not callable(
            getattr(adapter, "recompute_policy_stats", None)
        ):
            raise TypeError(
                "adapter must define recompute_policy_stats("
                "batch, *, require_reference=False)"
            )
        module = getattr(adapter, "train_module", None)
        if not isinstance(module, torch.nn.Module):
            raise TypeError("adapter.train_module must be a torch.nn.Module")
        return module

    @staticmethod
    def _validate_recompute_request(
        batch: RolloutBatch,
        require_reference: bool,
    ) -> None:
        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(require_reference, bool):
            raise TypeError("require_reference must be a bool")

    @staticmethod
    def _validate_recompute_result(
        stats: Any,
        batch: RolloutBatch,
        *,
        require_reference: bool,
    ) -> PolicyRecomputeStats:
        if not isinstance(stats, PolicyRecomputeStats):
            raise TypeError(
                "adapter.recompute_policy_stats() must return "
                "PolicyRecomputeStats"
            )
        stats.validate_against(batch, require_reference=require_reference)
        return stats

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
        return value.detach(), _positive_count(weight, name="reduction weight")

    @staticmethod
    def _prepare_metric_contributions(
        contributions: Mapping[str, MetricContribution],
    ) -> tuple[tuple[str, float, int | None], ...]:
        if not isinstance(contributions, Mapping):
            raise TypeError("metric contributions must be a mapping")
        prepared: list[tuple[str, float, int | None]] = []
        for name in sorted(contributions):
            value = contributions[name]
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if not isinstance(value, MetricContribution):
                raise TypeError(
                    "metric mappings must contain MetricContribution values"
                )
            numerator = float(value.numerator.detach().cpu())
            if not math.isfinite(numerator):
                raise ValueError("metric numerator must be finite")
            prepared.append((name, numerator, value.denominator))
        return tuple(prepared)

    @staticmethod
    def _reward_values(rewards: RewardBatch) -> tuple[float, ...]:
        if not isinstance(rewards, RewardBatch):
            raise TypeError("rewards must be a RewardBatch")
        values = tuple(
            float(value)
            for value in rewards.weighted_total.detach().cpu().tolist()
        )
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("reward metrics require finite non-empty values")
        return values

    @staticmethod
    def _validate_optimizer_boundary(
        operation: Any,
        *,
        parameters: Any,
        optimizer: Any,
        scaler: Any | None,
    ) -> None:
        if not callable(operation):
            raise TypeError("optimizer step operation must be callable")
        if type(parameters) is not tuple or not parameters:
            raise TypeError("parameters must be a non-empty tuple")
        if any(not isinstance(item, torch.nn.Parameter) for item in parameters):
            raise TypeError("parameters must contain only torch.nn.Parameter")
        identities = tuple(id(item) for item in parameters)
        if len(identities) != len(set(identities)):
            raise ValueError("parameters must have unique identities")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("optimizer must be torch.optim.AdamW")
        optimizer_ids = tuple(
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if optimizer_ids != identities:
            raise ValueError(
                "optimizer parameter identity/order must match parameters"
            )
        if scaler is not None:
            required = (
                "scale",
                "unscale_",
                "step",
                "update",
                "state_dict",
                "load_state_dict",
            )
            if any(not callable(getattr(scaler, name, None)) for name in required):
                raise TypeError("scaler must implement the GradScaler contract")

    def _validate_rank(self, rank: int, *, name: str) -> None:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError(
                f"{name} must satisfy 0 <= {name} < {self.world_size}"
            )


class DDPStrategy(SingleProcessStrategy):
    """Two-rank single-node DDP implementation of the same Strategy contract."""

    def __init__(
        self,
        context: DistributedContext,
        *,
        timeout_s: float,
        max_snapshot_tensor_bytes: int | None,
        master_addr: str | None,
        master_port: int | None,
    ) -> None:
        if not isinstance(context, DistributedContext) or not context.is_distributed:
            raise ValueError("DDPStrategy requires WORLD_SIZE greater than 1")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
            raise TypeError("timeout_s must be a positive number")
        if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
            raise ValueError("timeout_s must be a positive finite number")
        if max_snapshot_tensor_bytes is not None and (
            type(max_snapshot_tensor_bytes) is not int
            or max_snapshot_tensor_bytes <= 0
        ):
            raise ValueError(
                "max_snapshot_tensor_bytes must be a positive integer or None"
            )
        if not isinstance(master_addr, str) or not master_addr:
            raise ValueError("DDPStrategy requires a validated master_addr")
        if type(master_port) is not int or not 1 <= master_port <= 65535:
            raise ValueError("DDPStrategy requires a validated master_port")
        self._context = context
        self._timeout_s = float(timeout_s)
        self._max_snapshot_tensor_bytes = max_snapshot_tensor_bytes
        host = (
            f"[{master_addr}]"
            if ":" in master_addr and not master_addr.startswith("[")
            else master_addr
        )
        self._init_method = f"tcp://{host}:{master_port}"
        self._adapter: Any | None = None
        self._facade: _AdapterRecomputeFacade | None = None
        self._ddp: DistributedDataParallel | None = None
        self._setup_complete = False
        self._closed = False
        self._owns_process_group = False

    def _setup(self) -> None:
        self._require_open()
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if self.backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("PyTorch was built without the gloo backend")
        if self.backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("PyTorch was built without the NCCL backend")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if dist.is_initialized():
            raise RuntimeError(
                "VisualRL requires ownership of an uninitialized process group"
            )
        if self.backend is None:
            raise RuntimeError("DDP strategy requires a process-group backend")
        dist.init_process_group(
            backend=self.backend,
            init_method=self._init_method,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=self._timeout_s),
        )
        self._owns_process_group = True
        self._setup_complete = True

    def prepare(self, adapter: Any) -> Any:
        self._require_open()
        if not self._setup_complete:
            raise RuntimeError("Strategy must be set up before prepare()")
        if self._adapter is not None:
            if self._adapter is not adapter:
                raise RuntimeError("Strategy is already prepared with another adapter")
            return adapter
        train_module: torch.nn.Module | None = None
        error: BaseException | None = None
        try:
            train_module = self._validate_adapter(adapter)
            self._validate_module_device(train_module)
        except BaseException as exc:
            error = exc
        self.failure_gate("strategy.prepare", error)
        if train_module is None:
            raise RuntimeError("distributed adapter preflight lost local state")
        facade = _AdapterRecomputeFacade(adapter, train_module)
        # Model/base buffers are constructed from the same validated config on
        # every rank.  Disabling per-forward buffer broadcasts keeps ordinary
        # adapter forward/validation failures eligible for the pre-backward
        # failure gate; the only update-time reducer collective is backward.
        kwargs: dict[str, Any] = {"broadcast_buffers": False}
        if self.device.type == "cuda":
            kwargs.update(
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        try:
            self._facade = facade
            self._ddp = DistributedDataParallel(facade, **kwargs)
        except BaseException as exc:
            raise _ProcessGroupFatalError(
                "DDP construction failed"
            ) from exc
        self._adapter = adapter
        return adapter

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        self._require_prepared()
        self._validate_recompute_request(batch, require_reference)
        assert self._ddp is not None
        stats = self._ddp(
            batch,
            require_reference=require_reference,
        )
        return self._validate_recompute_result(
            stats,
            batch,
            require_reference=require_reference,
        )

    def gradient_sync_context(self, synchronize_gradients: bool):
        self._require_prepared()
        if not isinstance(synchronize_gradients, bool):
            raise TypeError("synchronize_gradients must be a bool")
        assert self._ddp is not None
        return (
            _FinalGradientSyncContext()
            if synchronize_gradients
            else self._ddp.no_sync()
        )

    def sum_active_transition_count(self, local_count: int) -> int:
        count = _positive_count(
            local_count,
            name="local active transition count",
        )
        reduced = torch.tensor(count, dtype=torch.int64, device=self.device)
        self._all_reduce(reduced, operation="active-count SUM")
        result = int(reduced.item())
        if result <= 0:
            raise DistributedFailureError(
                "global active transition count must be positive"
            )
        return result

    def reduce_tensor_weighted_mean(
        self,
        value: torch.Tensor,
        weight: int,
    ) -> torch.Tensor:
        prepared: tuple[torch.Tensor, int] | None = None
        error: BaseException | None = None
        try:
            prepared = self._validated_tensor_mean_and_weight(value, weight)
        except BaseException as exc:
            error = exc
        self.failure_gate("tensor_weighted_mean.validate", error)
        if prepared is None:
            raise RuntimeError("tensor reduction preflight lost local state")
        scalar, count = prepared
        reduced = torch.stack(
            (
                scalar.to(device=self.device) * count,
                torch.tensor(
                    float(count),
                    device=self.device,
                    dtype=scalar.dtype,
                ),
            )
        )
        self._all_reduce(reduced, operation="tensor weighted mean")
        if not bool(torch.isfinite(reduced).all()) or float(reduced[1]) <= 0:
            raise DistributedFailureError(
                "global tensor weighted mean is invalid"
            )
        return (reduced[0] / reduced[1]).to(
            device=value.device,
            dtype=value.dtype,
        )

    def reduce_metric_contributions(
        self,
        contributions: Mapping[str, MetricContribution],
    ) -> dict[str, float]:
        prepared: tuple[tuple[str, float, int | None], ...] | None = None
        error: BaseException | None = None
        try:
            prepared = self._prepare_metric_contributions(contributions)
        except BaseException as exc:
            error = exc
        self.failure_gate("metric_contributions.validate", error)
        if prepared is None:
            raise RuntimeError("metric contribution preflight lost local state")
        contract = tuple(
            (name, denominator is None)
            for name, _numerator, denominator in prepared
        )
        contracts = self._all_gather_object(
            contract,
            operation="metric contribution contract",
        )
        if any(item != contracts[0] for item in contracts[1:]):
            raise DistributedFailureError(
                "metric contribution keys/modes differ across ranks"
            )
        packed: list[float] = []
        for _name, numerator, denominator in prepared:
            packed.extend(
                (
                    numerator,
                    0.0 if denominator is None else float(denominator),
                )
            )
        reduced = torch.tensor(packed, dtype=torch.float64, device=self.device)
        self._all_reduce(reduced, operation="metric contribution SUM")
        if not bool(torch.isfinite(reduced).all()):
            raise DistributedFailureError(
                "reduced metric contributions must be finite"
            )
        result: dict[str, float] = {}
        for index, (name, _numerator, denominator) in enumerate(prepared):
            numerator_sum = float(reduced[2 * index].item())
            denominator_sum = float(reduced[2 * index + 1].item())
            if denominator is None:
                result[name] = numerator_sum
            else:
                if denominator_sum <= 0:
                    raise DistributedFailureError(
                        f"metric {name!r} has non-positive global denominator"
                    )
                result[name] = numerator_sum / denominator_sum
        return result

    def reduce_reward_metrics(self, rewards: RewardBatch) -> dict[str, float]:
        values: tuple[float, ...] | None = None
        error: BaseException | None = None
        try:
            values = self._reward_values(rewards)
        except BaseException as exc:
            error = exc
        self.failure_gate("reward_metrics.validate", error)
        if values is None:
            raise RuntimeError("reward metric preflight lost local state")
        try:
            packed = torch.tensor(
                (
                    math.fsum(values),
                    math.fsum(value * value for value in values),
                    float(len(values)),
                ),
                dtype=torch.float64,
                device=self.device,
            )
        except (OverflowError, RuntimeError, ValueError) as exc:
            self.failure_gate("reward_metrics.pack", exc)
            raise AssertionError("failure gate returned") from exc
        self._all_reduce(packed, operation="reward moment SUM")
        total, squared, count = (
            float(value.item()) for value in packed
        )
        if not all(math.isfinite(value) for value in (total, squared, count)):
            raise DistributedFailureError("global reward moments must be finite")
        if count <= 0:
            raise DistributedFailureError("global reward count must be positive")
        mean = total / count
        variance = max(squared / count - mean * mean, 0.0)
        return {"reward_mean": mean, "reward_std": math.sqrt(variance)}

    def atomic_optimizer_step(
        self,
        operation: Callable[[], None],
        *,
        parameters: tuple[torch.nn.Parameter, ...],
        optimizer: torch.optim.AdamW,
        scaler: Any | None,
    ) -> None:
        error: BaseException | None = None
        try:
            self._validate_optimizer_boundary(
                operation,
                parameters=parameters,
                optimizer=optimizer,
                scaler=scaler,
            )
        except BaseException as exc:
            error = exc
        self.failure_gate("optimizer_boundary.validate", error)

        snapshot: _OptimizerStepSnapshot | None = None
        error = None
        try:
            snapshot = _OptimizerStepSnapshot.capture(
                parameters,
                optimizer,
                scaler,
                max_tensor_bytes=self._max_snapshot_tensor_bytes,
            )
        except BaseException as exc:
            error = exc
        self.failure_gate("optimizer_snapshot", error)
        if snapshot is None:
            raise RuntimeError("optimizer snapshot preflight lost local state")

        operation_error: BaseException | None = None
        try:
            operation()
        except BaseException as exc:
            operation_error = exc
        failures = self._synchronize_failure_details(
            "optimizer_operation",
            operation_error,
        )
        if not failures:
            return

        restore_error: BaseException | None = None
        try:
            snapshot.restore(optimizer, scaler)
        except BaseException as exc:
            restore_error = exc
        message = self._format_failures(
            "Distributed optimizer operation failed",
            failures,
        )
        if restore_error is not None:
            message += (
                "; local rollback failed: "
                f"{type(restore_error).__name__}: {restore_error}"
            )
        error = DistributedFailureError(message)
        cause = restore_error if restore_error is not None else operation_error
        if cause is not None:
            raise error from cause
        raise error

    def gather_object(self, value: Any, *, dst: int = 0) -> list[Any] | None:
        self._require_collectives_ready()
        self._object_collective_contract("gather", dst, "dst")
        self._preflight_object(value, operation="gather")
        gathered = [None] * self.world_size if self.rank == dst else None
        try:
            dist.gather_object(value, gathered, dst=dst)
        except BaseException as exc:
            raise _ProcessGroupFatalError(
                "distributed gather_object failed"
            ) from exc
        return gathered

    def broadcast_object(self, value: Any, *, src: int = 0) -> Any:
        self._require_collectives_ready()
        self._object_collective_contract("broadcast", src, "src")
        self._preflight_object(
            value if self.rank == src else None,
            operation="broadcast",
        )
        payload = [value if self.rank == src else None]
        try:
            dist.broadcast_object_list(payload, src=src)
        except BaseException as exc:
            raise _ProcessGroupFatalError(
                "distributed broadcast_object failed"
            ) from exc
        return payload[0]

    def failure_gate(
        self,
        phase: str,
        failure: BaseException | None,
    ) -> None:
        failures = self._synchronize_failure_details(phase, failure)
        if not failures:
            return
        error = DistributedFailureError(
            self._format_failures(
                f"Distributed phase {phase!r} failed",
                failures,
            )
        )
        if failure is not None:
            raise error from failure
        raise error

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

    def _require_collectives_ready(self) -> None:
        self._require_open()
        if not self._setup_complete or not dist.is_initialized():
            raise RuntimeError("DDP strategy must be set up before collectives")

    def _validate_module_device(self, module: torch.nn.Module) -> None:
        tensors = (*module.parameters(), *module.buffers())
        mismatched = tuple(
            tensor.device for tensor in tensors if tensor.device != self.device
        )
        if mismatched:
            raise ValueError(
                f"adapter.train_module must already be on {self.device}; "
                f"found {mismatched[0]}"
            )
        if not any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("adapter.train_module has no trainable parameters")

    def _all_reduce(self, tensor: torch.Tensor, *, operation: str) -> None:
        self._require_collectives_ready()
        try:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        except BaseException as exc:
            raise _ProcessGroupFatalError(
                f"distributed {operation} failed"
            ) from exc

    def _all_gather_object(self, value: Any, *, operation: str) -> list[Any]:
        self._require_collectives_ready()
        gathered: list[Any] = [None] * self.world_size
        try:
            dist.all_gather_object(gathered, value)
        except BaseException as exc:
            raise _ProcessGroupFatalError(
                f"distributed {operation} failed"
            ) from exc
        return gathered

    def _synchronize_failure_details(
        self,
        phase: str,
        failure: BaseException | None,
    ) -> list[tuple[int, str, str]]:
        _validate_phase_name(phase)
        if failure is not None and not isinstance(failure, BaseException):
            raise TypeError("failure must be an exception or None")
        local = (
            None
            if failure is None
            else (type(failure).__name__, str(failure))
        )
        gathered = self._all_gather_object(
            local,
            operation=f"failure gate {phase!r}",
        )
        return [
            (rank, name, message)
            for rank, item in enumerate(gathered)
            if item is not None
            for name, message in [item]
        ]

    @staticmethod
    def _format_failures(
        prefix: str,
        failures: list[tuple[int, str, str]],
    ) -> str:
        details = "; ".join(
            f"rank {rank}: {name}: {message}"
            for rank, name, message in failures
        )
        return f"{prefix}: {details}"

    def _object_collective_contract(
        self,
        operation: str,
        root: Any,
        root_name: str,
    ) -> None:
        error: BaseException | None = None
        try:
            self._validate_rank(root, name=root_name)
        except BaseException as exc:
            error = exc
        state = (
            operation,
            root_name,
            root if error is None else None,
            None if error is None else (type(error).__name__, str(error)),
        )
        states = self._all_gather_object(
            state,
            operation=f"{operation} contract",
        )
        invalid = tuple(
            (rank, item[3])
            for rank, item in enumerate(states)
            if item[3] is not None
        )
        if invalid:
            details = "; ".join(
                f"rank {rank}: {name}: {message}"
                for rank, (name, message) in invalid
            )
            raise DistributedFailureError(
                "Invalid distributed object collective root: " + details
            )
        contracts = tuple(item[:3] for item in states)
        if any(item != contracts[0] for item in contracts[1:]):
            raise DistributedFailureError(
                "distributed object collective operation/root differ across ranks"
            )

    def _preflight_object(self, value: Any, *, operation: str) -> None:
        error: BaseException | None = None
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:
            error = TypeError(
                f"distributed {operation} object is not serializable: {exc}"
            )
        self.failure_gate(f"{operation}.object", error)


def _validate_phase_name(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("phase name must be a non-empty string")


def _positive_count(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
