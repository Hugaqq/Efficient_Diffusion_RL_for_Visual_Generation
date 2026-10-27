"""H3 WORLD8 cooperative gates, serialized CPU aux work, and shard-only norm.

Not a runner or a PG owner. A dead rank or failure inside model/transport
collectives requires caller timeout/whole-attempt teardown, not a rescue gate.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import math
from typing import TypeVar

import torch
import torch.distributed as dist

from visual_rl.algorithms.optimization.h3_world import (
    H3WorldPlan, WORLD_SIZE, _exchange, _identity, _world_rank,
)
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel, H3_REDUCER_DIVISOR, _cuda_device
from visual_rl.models.lifecycle.prepared import validate_optimizer_parameter_subset

T = TypeVar('T')


class H3CoordinationError(RuntimeError):
    """The attempt cannot continue after a failed coordination operation."""


def _dtensor_parts(value):
    """Actual DTensor observation only; never gather a full parameter/gradient."""
    from torch.distributed.tensor import DTensor
    if not isinstance(value, DTensor):
        raise ValueError('H3 norm requires sharded DTensor parameters AND gradients')
    return value.device_mesh, value.placements, tuple(value.shape), value.to_local().detach()


def _local_gradient(parameter, process_group, device, rank):
    from torch.distributed.tensor import Shard
    if not parameter.requires_grad or parameter.grad is None:
        raise ValueError('every trainable parameter must have a gradient, including empty shards')
    parts = [_dtensor_parts(value) for value in (parameter, parameter.grad)]
    for mesh, placements, shape, local in parts:
        if (mesh.device_type != 'cuda' or mesh.ndim != 1 or mesh.size() != WORLD_SIZE
                or mesh.mesh.tolist() != list(range(WORLD_SIZE)) or mesh.get_group(0) is not process_group):
            raise ValueError('gradient mesh must be the ordered one-dimensional CUDA WORLD8 mesh')
        if len(placements) != 1 or type(placements[0]) is not Shard or placements[0].dim != 0:
            raise ValueError('H3 norm only accepts the prepare backend Shard(0), not replicas/partial shards')
        if (not shape or local.device != device or local.dtype != torch.float32
                or local.layout != torch.strided or local.is_meta):
            raise ValueError('invalid FP32 local shard shape/device/layout')
        chunk = (shape[0] + WORLD_SIZE - 1) // WORLD_SIZE
        expected = (min(chunk, max(0, shape[0] - rank * chunk)), *shape[1:])
        if tuple(local.shape) != expected:
            raise ValueError('local shard must contain exactly its unique unpadded elements')
    if parts[0][2] != parts[1][2] or parts[0][3].shape != parts[1][3].shape:
        raise ValueError('parameter/gradient shard shape mismatch')
    return parts[1][3], parts[0][2]


def _trainables(prepared):
    if not isinstance(prepared, H3PreparedModel) or prepared.reducer_divisor != H3_REDUCER_DIVISOR:
        raise ValueError('norm requires the final H3 prepared backend with explicit divisor')
    named = prepared.trainable_named_parameters()
    expected = tuple((n, p) for n, p in prepared.prepared_root.named_parameters(remove_duplicate=False)
                     if p.requires_grad)
    if (not named or len({n for n, _ in named}) != len(named)
            or len({id(p) for _, p in named}) != len(named)
            or tuple((n, id(p)) for n, p in named) != tuple((n, id(p)) for n, p in expected)):
        raise ValueError('trainable parameter identity/coverage/uniqueness mismatch')
    validate_optimizer_parameter_subset(prepared.prepared_root, prepared.optimizer)
    return named


class H3Coordination:
    """One admitted collection's synchronous coordination capability.

    The caller uses rank-independent event names (local cell ordinals, never
    rank-local sample IDs). Every method participates in the same monotonic gate
    sequence; success advances it, any failure poisons this attempt permanently.
    """
    def __init__(self, plan: H3WorldPlan, *, process_group, device):
        self._poisoned = False
        self._closed = False
        self._sequence = 0
        self._in_aux = False
        self._group = process_group
        self._rank = _world_rank(process_group)  # no usable WORLD means no cooperative gate
        self._identity = None
        self._device = None
        error = None
        try:
            if not isinstance(plan, H3WorldPlan):
                raise TypeError('coordination requires an admitted H3WorldPlan')
            self._identity = (plan.plan_id, plan.collection_id, plan.behavior_id)
            requested = torch.device(device)
            if requested.type != 'cuda' or requested.index != self._rank:
                raise ValueError('coordination requires the explicit rank-local CUDA device')
            self._device = _cuda_device(requested)
        except Exception as exc:
            error = exc
        try:
            self.gate('coordination/init', local_error=error is not None)
        except H3CoordinationError as exc:
            if error is not None:
                raise exc from error
            raise

    def _require_live(self):
        if self._closed or self._poisoned:
            raise H3CoordinationError('coordination is closed or poisoned; start a fresh attempt')

    def gate(self, event: str, *, local_error: bool = False) -> None:
        """One bounded receipt exchange, including malformed local event/status."""
        self._require_live()
        valid = type(local_error) is bool and not local_error
        try:
            _identity(event)  # max 256 UTF-8 bytes, canonical nonempty string
            if type(self._sequence) is not int or not 0 <= self._sequence < 2**63:
                raise ValueError('invalid gate sequence')
        except (ValueError, TypeError):
            valid = False
            event = None
        # The common helper transmits only JSON bytes, includes rank/node/schema,
        # bounds serialized size, and rejects all finite local-error receipts.
        payload = [list(self._identity), self._sequence, event] if valid else None
        try:
            receipts = _exchange('coordination', self._rank, self._group, payload, valid)
            if any(receipt != payload for receipt in receipts):
                raise ValueError('plan/collection/behavior, sequence or event mismatch')
        except BaseException as exc:
            self._poisoned = True
            if not isinstance(exc, Exception):
                raise
            raise H3CoordinationError('H3 gate failed; attempt poisoned') from exc
        self._sequence += 1

    def aux_turns(self, work: Callable[[], T]) -> T:
        """Only rank 0..7's owner works between the common begin/end gates.

        Work must return its own small result; it must not enter model/PG gates.
        Results/media are never gathered or retained on this coordinator.
        """
        self._require_live()
        invalid = not callable(work) or self._in_aux
        self._in_aux = True
        result = None
        try:
            for owner in range(WORLD_SIZE):
                self.gate(f'aux/{owner}/begin', local_error=invalid)
                error = None
                if owner == self._rank:
                    try:
                        result = work()
                    except Exception as exc:
                        error = exc
                try:
                    self.gate(f'aux/{owner}/end', local_error=error is not None)
                except H3CoordinationError as exc:
                    if error is not None:
                        raise exc from error
                    raise
            return result
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._in_aux = False  # local only; never add a finally collective

    @torch.no_grad()
    def clip_grad_norm(self, prepared: H3PreparedModel, *, max_norm: float) -> float:
        """SUM local FP64 squared gradients once, then clip unique global norm.

        FSDP backward has already synchronized/averaged these gradients. Do not
        divide by WORLD again. Empty local shards and finite zero norm are legal;
        missing gradients are not. No optimizer/scheduler/zero_grad is performed.
        """
        self._require_live()
        error, gradients, descriptor = None, [], []
        event = 'grad/preflight'
        try:
            if (type(max_norm) not in (float, int) or not math.isfinite(max_norm) or max_norm <= 0):
                raise ValueError('max_norm must be finite and positive')
            total = torch.zeros((), dtype=torch.float64, device=self._device)
            for name, parameter in _trainables(prepared):
                local, shape = _local_gradient(parameter, self._group, self._device, self._rank)
                if not bool(torch.isfinite(local).all()):
                    raise ValueError('nonfinite local gradient')
                gradients.append(local)
                descriptor.append((name, shape))
                total.add_(local.double().square().sum())
            if not bool(torch.isfinite(total)):
                raise ValueError('local squared-gradient sum overflow')
            # Agree on full parameter coverage/shapes and clipping threshold,
            # without transmitting gradients or per-parameter tensors.
            digest = hashlib.sha256(json.dumps((descriptor, float(max_norm))).encode()).hexdigest()
            event += '/' + digest
        except Exception as exc:
            error = exc
        try:
            try:
                self.gate(event, local_error=error is not None)
            except H3CoordinationError as exc:
                if error is not None:
                    raise exc from error
                raise
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self._group)
            squared = total.item()
            if not math.isfinite(squared) or squared < 0:
                raise H3CoordinationError('nonfinite/invalid aggregate squared-gradient sum')
            norm = math.sqrt(squared)
            scale = 1. if norm == 0 else min(1., max_norm / norm)
            for local in gradients:
                local.mul_(scale)
            return norm
        except BaseException:
            self._poisoned = True
            raise

    def close(self) -> None:
        """Release local references only; the caller still owns WORLD."""
        self._closed = True
        self._group = None
        self._device = None
