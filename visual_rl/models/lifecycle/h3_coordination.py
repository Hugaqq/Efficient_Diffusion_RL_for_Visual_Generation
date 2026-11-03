"""H3 WORLD8 cooperative gates, serialized CPU aux work, and shard-only norm.

Not a runner or a PG owner. A dead rank or failure inside model/transport
collectives requires caller timeout/whole-attempt teardown, not a rescue gate.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import math
import threading
import time
import faulthandler
from contextlib import ExitStack
from typing import TypeVar

import torch
import torch.distributed as dist

from visual_rl.algorithms.optimization.h3_world import (
    H3WorldPlan, WORLD_SIZE, _exchange, _identity, _world_rank,
)
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel, H3_REDUCER_DIVISOR, _cuda_device
from visual_rl.models.lifecycle.prepared import validate_optimizer_parameter_subset

T = TypeVar('T')

# Opt-in concurrent auxiliary work (P2). Status codes per rank in a SUM-reduced
# heartbeat; every rank derives the same verdict from the same reduced tensor.
AUX_RUNNING, AUX_DONE, AUX_FAILED, AUX_TIMEOUT = 0, 1, 2, 3
AUX_BUDGET_SECONDS = 3600.
HOST_MEDIA_CAP = 16  # decoded media in flight across the single host
AUX_ABORT_JOIN_SECONDS = 900.  # fallback; lifecycle/launcher budgets HPS, retention and margin


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
        self._aux_worker = None  # retain the handle even after a timed-out join
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

    @property
    def aux_worker_alive(self):
        """Local cleanup guard, valid even when this coordinator is poisoned/closed."""
        return self._aux_worker is not None and self._aux_worker.is_alive()

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
                # Nonowners must not sit in the end gate throughout CPU VAE
                # decode / HTTP scoring. Keep main threads in short heartbeats
                # while only the owner worker runs auxiliary work (never PG work).
                done = threading.Event()
                worker = None
                started = time.monotonic()
                signal = torch.zeros((), dtype=torch.int32, device=self._device)
                if owner == self._rank:
                    producer = torch.cuda.current_stream(self._device)
                    work_stream = torch.cuda.Stream(device=self._device)
                    work_stream.wait_stream(producer)
                    grad_enabled = torch.is_grad_enabled()
                    def execute():
                        nonlocal result, error
                        try:
                            # CUDA selection and streams are thread-local. A
                            # separate work stream prevents long auxiliary CUDA
                            # kernels from holding up the main heartbeat stream.
                            with torch.cuda.device(self._device), torch.cuda.stream(work_stream), torch.set_grad_enabled(grad_enabled):
                                result = work()
                                work_stream.synchronize()
                        except BaseException as exc:
                            error = exc
                        finally:
                            done.set()
                    worker = threading.Thread(target=execute, name=f'h3-aux-{owner}', daemon=True)
                    self._aux_worker = worker
                    print(json.dumps(dict(event='h3_aux_begin', owner=owner,
                                          timeout_seconds=3600)), flush=True)
                    worker.start()
                tick = 0
                while True:
                    if owner == self._rank:
                        elapsed = time.monotonic() - started
                        signal.fill_(1 if done.is_set() else (2 if elapsed >= 3600 else 0))
                        if tick % 30 == 0:
                            print(json.dumps(dict(event='h3_aux_wait', owner=owner,
                                                  seconds=elapsed, done=done.is_set())), flush=True)
                        if tick and tick % 300 == 0 and not done.is_set():
                            faulthandler.dump_traceback(all_threads=True)
                    dist.broadcast(signal, src=owner, group=self._group)
                    code = int(signal.item())
                    if code == 2:
                        raise H3CoordinationError('auxiliary owner exceeded3600s; abort entire attempt')
                    if code == 1:
                        break
                    if code != 0:
                        raise H3CoordinationError('invalid auxiliary heartbeat status')
                    tick += 1
                    time.sleep(1)
                if worker is not None:
                    worker.join()
                    torch.cuda.current_stream(self._device).wait_stream(work_stream)
                    print(json.dumps(dict(event='h3_aux_end', owner=owner,
                                          seconds=time.monotonic()-started,
                                          error=None if error is None else repr(error))), flush=True)
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

    def aux_concurrent(self, work: Callable[[], T], *, cancel: threading.Event | None = None,
                       inflight: Callable[[], int] | None = None, media_cap: int = HOST_MEDIA_CAP,
                       budget_s: float = AUX_BUDGET_SECONDS, interval_s: float = 1.,
                       abort_join_s: float = AUX_ABORT_JOIN_SECONDS) -> T:
        """Opt-in: every rank runs its own auxiliary work at the same time.

        Collective sequence is timing-independent: begin gate, N heartbeat
        all_reduces, then either the end gate (all ranks done) or a common
        abort with no further collective. N is identical on all ranks because
        each verdict is a pure function of the SAME reduced status tensor. All
        local setup (streams, status tensor, worker thread) happens BEFORE the
        begin gate and reports failure through it; between gates every local
        step is caught and reported as a FAILED status, and logging can never
        raise, so no rank can leave peers waiting in a collective. A failure,
        local budget expiry, malformed status or host media count above
        media_cap on ANY rank aborts every rank at that heartbeat: each rank
        sets the cooperative cancel event and joins its own worker for at most
        abort_join_s (local only; in-flight HTTP/decode/encode cannot be killed)
        before raising, so caller cleanup does not close resources underneath
        a still-running owner in the common case. Results stay rank-local.
        """
        self._require_live()
        invalid = (not callable(work) or self._in_aux
                   or (cancel is not None and not isinstance(cancel, threading.Event))
                   or (inflight is not None and not callable(inflight))
                   or type(media_cap) is not int or media_cap < 1
                   or type(budget_s) not in (int, float) or not 0 < budget_s <= AUX_BUDGET_SECONDS
                   or type(interval_s) not in (int, float) or not 0 < interval_s <= 1
                   or type(abort_join_s) not in (int, float) or not 0 <= abort_join_s <= AUX_BUDGET_SECONDS)
        self._in_aux = True
        result = error = setup_error = None
        go, done = threading.Event(), threading.Event()
        launch = [False]  # set True only after a successful begin gate
        worker = work_stream = None
        def log(**fields):
            try:
                print(json.dumps(fields), flush=True)
            except Exception:
                pass  # diagnostics must never desynchronize the collective sequence
        try:
            try:
                if invalid:
                    raise ValueError('invalid concurrent auxiliary request')
                cuda = self._device.type == 'cuda'  # CPU only in the in-memory world simulation
                if cuda:
                    producer = torch.cuda.current_stream(self._device)
                    work_stream = torch.cuda.Stream(device=self._device)
                    work_stream.wait_stream(producer)
                grad_enabled = torch.is_grad_enabled()
                status = torch.zeros((2, WORLD_SIZE), dtype=torch.int32, device=self._device)
                def execute():
                    nonlocal result, error
                    try:
                        go.wait()
                        if not launch[0]:
                            return  # begin gate failed: never run the work
                        with ExitStack() as scope:
                            if cuda:  # thread-local CUDA selection + side stream, as in aux_turns
                                scope.enter_context(torch.cuda.device(self._device))
                                scope.enter_context(torch.cuda.stream(work_stream))
                            scope.enter_context(torch.set_grad_enabled(grad_enabled))
                            result = work()
                            if cuda:
                                work_stream.synchronize()
                    except BaseException as exc:
                        error = exc
                    finally:
                        done.set()
                worker = threading.Thread(target=execute, name=f'h3-aux-parallel-{self._rank}', daemon=True)
                self._aux_worker = worker
                worker.start()  # parked on `go` until the begin gate succeeds
            except Exception as exc:
                setup_error = exc
            try:
                self.gate(f'aux/parallel/{WORLD_SIZE}/media{media_cap if type(media_cap) is int else None}/begin',
                          local_error=setup_error is not None)
            except H3CoordinationError as exc:
                if setup_error is not None:
                    raise exc from setup_error
                raise
            launch[0] = True
            started = time.monotonic()
            go.set()
            log(event='h3_aux_parallel_begin', rank=self._rank, timeout_seconds=budget_s, media_cap=media_cap)
            tick, verdict = 0, None
            while verdict is None:
                try:
                    elapsed = time.monotonic() - started
                    held = 0
                    if done.is_set():
                        code = AUX_DONE if error is None else AUX_FAILED
                    elif elapsed >= budget_s:
                        code = AUX_TIMEOUT
                    else:
                        code = AUX_RUNNING
                        held = inflight() if inflight is not None else 0
                        if type(held) is not int or not 0 <= held <= media_cap:
                            raise ValueError('invalid local in-flight media count')
                except Exception as exc:
                    if error is None:
                        error = exc
                    code, held = AUX_FAILED, 0
                status.zero_()
                status[0, self._rank] = code
                status[1, self._rank] = held
                dist.all_reduce(status, op=dist.ReduceOp.SUM, group=self._group)
                codes, counts = status[0].tolist(), status[1].tolist()
                host_media = sum(counts)
                if any(c not in (AUX_RUNNING, AUX_DONE, AUX_FAILED, AUX_TIMEOUT) for c in codes) or min(counts) < 0:
                    verdict = 'invalid auxiliary heartbeat status'
                elif any(c in (AUX_FAILED, AUX_TIMEOUT) for c in codes):
                    failed = [r for r, c in enumerate(codes) if c == AUX_FAILED]
                    late = [r for r, c in enumerate(codes) if c == AUX_TIMEOUT]
                    verdict = f'auxiliary work failed on ranks {failed}, exceeded {budget_s:g}s on ranks {late}'
                elif host_media > media_cap:
                    verdict = f'host in-flight media {host_media} exceeds cap {media_cap}'
                elif all(c == AUX_DONE for c in codes):
                    verdict = 'done'
                if self._rank == 0 and (tick % 30 == 0 or verdict is not None):
                    log(event='h3_aux_parallel_wait', seconds=elapsed, codes=codes,
                        host_media=host_media, verdict=verdict)
                if verdict is None:
                    if tick and tick % 300 == 0 and not done.is_set():
                        try:
                            faulthandler.dump_traceback(all_threads=True)
                        except Exception:
                            pass
                    tick += 1
                    time.sleep(interval_s)
            if verdict != 'done':
                raise H3CoordinationError('parallel auxiliary phase aborted on all ranks: ' + verdict) from error
            end_error = None
            try:
                worker.join()  # already done
                if work_stream is not None:
                    torch.cuda.current_stream(self._device).wait_stream(work_stream)
            except Exception as exc:
                end_error = exc
            log(event='h3_aux_parallel_end', rank=self._rank, seconds=time.monotonic() - started)
            try:
                self.gate('aux/parallel/end', local_error=end_error is not None)
            except H3CoordinationError as exc:
                if end_error is not None:
                    raise exc from end_error
                raise
            return result
        except BaseException as exc:
            self._poisoned = True
            if launch[0]:
                if cancel is not None:
                    cancel.set()  # no new decode, HPS request or retention encode
                if isinstance(exc, Exception):
                    worker.join(abort_join_s)  # also on transport failure; local only, exactly one bounded join
                # KeyboardInterrupt/SystemExit: cancel only, do not delay the interrupt by a bounded join.
                log(event='h3_aux_parallel_abort', rank=self._rank, verdict=str(exc),
                    worker_alive=self.aux_worker_alive)
            raise
        finally:
            go.set()  # a parked worker whose gate failed exits without running work
            if worker is not None and not launch[0]:
                worker.join(1.)  # local; it only returns
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
