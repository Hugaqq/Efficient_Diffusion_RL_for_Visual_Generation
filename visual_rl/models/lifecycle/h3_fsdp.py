"""H3 WORLD8 block-FSDP2 initialization; not a runner or a PG owner.

Public PyTorch APIs, independently composed after reviewing UniRL's Apache-2.0
fsdp/wrap.py and state.py (8cfbac17765004afb68fef08a4e011a4c8daf555).
No CPU fallback: CPU tests replace only unavailable distributed/CUDA leaves.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import socket
import threading
import time

import torch
import torch.distributed as dist

from visual_rl.models.implementations.h3_loading import H3MetaModel, load_h3_cpu_state
from visual_rl.models.lifecycle.components import ComponentBinding, ComponentRole, ModelComponents
from visual_rl.models.lifecycle.prepared import (
    PreparedComponentHandle, PreparedModelBundle, validate_optimizer_parameter_subset,
)


# Explicitly installed on every block; never inferred from a default AVG op.
H3_REDUCER_DIVISOR = 8.0


def _require_apis():
    # Resolve before weight reads, wrapping, or heavy collectives. Do not fall
    # back to FSDP1/Accelerator or relax strict state loading on older versions.
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl

    divide = getattr(FSDPModule, 'set_gradient_divide_factor', None)
    if not callable(divide):
        raise RuntimeError('H3 requires FSDP2.set_gradient_divide_factor; no default-reduction guess')
    required = ((divide, {'factor'}),
                (fully_shard, {'mesh', 'reshard_after_forward', 'mp_policy'}),
                (MixedPrecisionPolicy, {'param_dtype', 'reduce_dtype', 'cast_forward_inputs'}),
                (StateDictOptions, {'full_state_dict', 'broadcast_from_rank0', 'strict'}),
                (set_model_state_dict, {'model', 'model_state_dict', 'options'}),
                (DeviceMesh.from_group, {'group', 'device_type', 'mesh_dim_names'}),
                (checkpoint_wrapper, {'checkpoint_impl'}))
    for function, names in required:
        if not names <= set(inspect.signature(function).parameters):
            raise RuntimeError(f"unsupported H3 FSDP2 API: {function.__qualname__}")
    if not hasattr(CheckpointImpl, 'NO_REENTRANT'):
        raise RuntimeError("H3 requires nonreentrant activation checkpointing")


def _cuda_device(device: torch.device) -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.current_device() != device.index:
        raise ValueError("caller must select the current authorized CUDA device before prepare")
    return device


def _admit_world(process_group, device, rank, local_rank, local_world_size, sp, tp):
    device = torch.device(device)
    if (not dist.is_available() or not dist.is_initialized()
            or process_group is not dist.group.WORLD
            or dist.get_backend(process_group) != 'nccl'):
        raise ValueError("H3 prepare requires the caller's initialized NCCL WORLD group")
    if (any(type(n) is not int for n in (rank, local_rank, local_world_size, sp, tp))
            or dist.get_world_size(process_group) != 8 or dist.get_rank(process_group) != rank
            or not 0 <= rank < 8 or local_rank != rank or local_world_size != 8
            or sp != 1 or tp != 1):
        raise ValueError("H3 requires single-node WORLD8, local_rank==rank, SP1/TP1")
    if device.type != 'cuda' or device.index != local_rank:
        raise ValueError("H3 requires an explicit rank-local CUDA device")
    return _cuda_device(device)


def _validate_meta(meta):
    if not isinstance(meta, H3MetaModel):
        raise TypeError("meta must be the approved H3MetaModel")
    model = meta.model
    if getattr(model, '_h3_prepare_started', False):
        raise ValueError("H3 model already prepared or prepare attempted; create a fresh model")
    params = dict(model.named_parameters())
    if not params or not all(p.is_meta for p in params.values()):
        raise ValueError("H3 prepare requires all parameters on meta")
    if not meta.block_names or len(set(meta.block_names)) != len(meta.block_names):
        raise ValueError("invalid main block names")
    expected_blocks = tuple(f'transformer_blocks.{i}' for i in range(len(model.transformer_blocks)))
    if meta.block_names != expected_blocks:
        raise ValueError("main block names differ from loader")
    if {n for n, p in params.items() if p.requires_grad} != set(meta.lora_names):
        raise ValueError("trainable scope must exactly equal loader LoRA names")
    if any(not any(n.startswith(b + '.') for b in meta.block_names) for n in meta.lora_names):
        raise ValueError("stray trainable outside sharded main blocks")
    if any(getattr(m, 'gradient_checkpointing', False) for m in model.modules()):
        raise ValueError("internal gradient checkpointing must remain disabled")
    state = model.state_dict()
    if set(state) != set(meta.state_specs):
        raise ValueError("model state differs from loader specifications")
    for name, value in state.items():
        spec = meta.state_specs[name]
        if tuple(value.shape) != spec.shape or value.dtype != spec.dtype:
            raise ValueError(f"model shape/precision differs from loader: {name}")
    buffers = dict(model.named_buffers())
    nonpersistent = set(buffers) - set(state)
    if nonpersistent != {'rope.inv_freq'} or set(meta.nonpersistent_buffers) != nonpersistent:
        raise ValueError("missing or unsupported nonpersistent buffer")
    saved = meta.nonpersistent_buffers['rope.inv_freq']
    if (saved.device.type != 'cpu' or saved.dtype != torch.float32 or saved.numel() == 0
            or saved.requires_grad or not torch.isfinite(saved).all()
            or not torch.equal(buffers['rope.inv_freq'], saved)):
        raise ValueError("rope.inv_freq must retain its real CPU FP32 constructor value")


def _metadata_digest(meta, init_seed, host_limit, optimizer_config, constant_scheduler):
    # Only a fixed-size digest is gathered, never state tensors. Include actual
    # adapter scaling: equal tensor shapes alone do not establish equal policy.
    adapters = []
    for name in meta.target_names:
        layer = meta.model.get_submodule(name)
        adapters.append((name, layer.r['default'], layer.lora_alpha['default'], layer.scaling['default']))
    buffer = meta.nonpersistent_buffers['rope.inv_freq']
    payload = dict(config=meta.config_sha256, blocks=meta.block_names, targets=meta.target_names,
                   lora=meta.lora_names, adapters=adapters, mapping=sorted(meta.base_to_model.items()),
                   specs=sorted((n, s.shape, str(s.dtype)) for n, s in meta.state_specs.items()),
                   rope=(tuple(buffer.shape), hashlib.sha256(buffer.contiguous().numpy().tobytes()).hexdigest()),
                   init_seed=init_seed, host_limit=host_limit, optimizer=optimizer_config,
                   constant_scheduler=constant_scheduler)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _agree_metadata(digest, rank, process_group):
    host = hashlib.sha256(socket.gethostname().encode()).hexdigest()
    receipt = (rank, host, digest)
    receipts = [None] * 8
    dist.all_gather_object(receipts, receipt, group=process_group)
    if receipts != [(r, host, digest) for r in range(8)]:
        raise ValueError("H3 ranks disagree on node, rank layout, initialization or model metadata")


def _world_mesh(process_group):
    from torch.distributed.device_mesh import DeviceMesh
    # Reuse WORLD; no subgroup/new process group, no UUID assumptions.
    return DeviceMesh.from_group(process_group, 'cuda', mesh_dim_names=('dp_shard',))


def _shard_block(block, mesh):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
    from torch.distributed.tensor import DTensor
    fully_shard(block, mesh=mesh, reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                               reduce_dtype=torch.float32,
                                               cast_forward_inputs=False))
    if not isinstance(block, FSDPModule) or any(not isinstance(p, DTensor) for p in block.parameters()):
        raise RuntimeError("FSDP2 did not shard every main-block parameter")
    # Public replacement for deprecated set_reduce_scatter_divide_factor.
    # One-dimensional WORLD sharding: sum of rank gradients divided by 8.
    block.set_gradient_divide_factor(H3_REDUCER_DIVISOR)


def _set_full_state(model, state):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    return set_model_state_dict(model, model_state_dict=state,
                                options=StateDictOptions(full_state_dict=True,
                                                         broadcast_from_rank0=True, strict=True))


def _canonical_parameter_names(model, block_names):
    # CheckpointWrapper state_dict hooks hide this prefix, whereas raw
    # named_parameters exposes it. Map only the exact wrappers we installed.
    result = {}
    for name, parameter in model.named_parameters():
        canonical = name
        for block in block_names:
            prefix = block + '._checkpoint_wrapped_module.'
            if name.startswith(prefix):
                canonical = block + '.' + name[len(prefix):]
                break
        if canonical in result:
            raise ValueError("aliased canonical parameter name")
        result[canonical] = parameter
    return result


class _SynchronousBackward:
    """Only the handle's accumulate capability, not an Accelerator adapter.

    Each backward synchronizes through FSDP2; no no_sync, implicit loss scaling,
    optimizer wrapping, or accumulation-step bookkeeping is provided here.
    """
    @contextmanager
    def accumulate(self, root):
        yield root


class H3PreparedModel(PreparedComponentHandle):
    """Single guarded H3 callable, with the existing handle's optimizer/close."""
    def __call__(self, *args, **kwargs):
        return self.forward('transformer', *args, **kwargs)

    def trainable_named_parameters(self):
        return self.prepared_root.trainable_named_parameters()

    @property
    def reducer_divisor(self) -> float:
        self._require_open()
        return H3_REDUCER_DIVISOR


def prepare_h3_fsdp(
    meta: H3MetaModel, *, process_group, device, rank: int, local_rank: int,
    local_world_size: int, host_weight_limit_bytes: int, init_seed: int,
    sp: int = 1, tp: int = 1, lr: float = 1e-4, betas: tuple[float, float] = (.9, .999),
    eps: float = 1e-12, weight_decay: float = 0., constant_scheduler: bool = False,
) -> H3PreparedModel:
    """All ranks call once: meta/AC/shard -> materialize -> bind final optimizer.

    Rank zero alone reads the complete CPU checkpoint. Root/refiner are frozen
    replicas; only main blocks are sharded. A failed attempt is not recoverable
    here: the caller owns PG/attempt teardown, with no cleanup collectives.
    """
    device = _admit_world(process_group, device, rank, local_rank, local_world_size, sp, tp)
    _require_apis()
    _validate_meta(meta)
    if type(host_weight_limit_bytes) is not int or host_weight_limit_bytes < 1:
        raise ValueError("explicit positive host_weight_limit_bytes required")
    if type(init_seed) is not int or not 0 <= init_seed < 2**63:
        raise ValueError("init_seed must be in [0,2**63)")
    if (type(constant_scheduler) is not bool or type(betas) is not tuple or len(betas) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in (lr, eps, weight_decay, *betas))
            or lr <= 0 or eps <= 0 or weight_decay < 0 or not all(0 <= v < 1 for v in betas)):
        raise ValueError("invalid H3 AdamW/constant scheduler configuration")
    optimizer_config = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    _agree_metadata(_metadata_digest(meta, init_seed, host_weight_limit_bytes,
                                    optimizer_config, constant_scheduler), rank, process_group)
    model = meta.model
    model._h3_prepare_started = True
    state = {}
    owner_error = None
    # Loading the real 66GB CPU state takes longer than WORLD's 180s
    # transport deadline. Keep the owner main thread in bounded heartbeats,
    # rather than leaving peers blocked in a single NCCL collective throughout
    # the entire disk/hash/finite-check phase. Only the worker reads weights;
    # only main threads use CUDA/distributed APIs. Failure aborts this attempt.
    done = threading.Event()
    def load_owner():
        nonlocal state, owner_error
        try:
            loaded = load_h3_cpu_state(meta, is_init_owner=True,
                                      host_weight_limit_bytes=host_weight_limit_bytes, init_seed=init_seed)
            state = loaded.state
            del loaded
            if set(state) != set(meta.state_specs):
                raise ValueError("incomplete owner initialization state")
        except BaseException as error:
            owner_error = error
            state = {}
        finally:
            done.set()

    started = time.monotonic()
    worker = None
    if rank == 0:
        print(json.dumps(dict(event='h3_owner_load_begin', rank=rank,
                              timeout_seconds=3600)), flush=True)
        worker = threading.Thread(target=load_owner, name='h3-cpu-weight-loader', daemon=True)
        worker.start()
    signal = torch.zeros((), dtype=torch.int32, device=device)
    tick = 0
    while True:
        if rank == 0:
            elapsed = time.monotonic() - started
            status_code = 1 if done.is_set() else (2 if elapsed >= 3600 else 0)
            signal.fill_(status_code)
            if tick % 30 == 0:
                print(json.dumps(dict(event='h3_owner_load_wait', seconds=elapsed,
                                      done=done.is_set())), flush=True)
        dist.broadcast(signal, src=0, group=process_group)
        status_code = int(signal.item())
        if status_code == 2:
            raise TimeoutError('H3 CPU owner initialization exceeded3600s; abort entire attempt')
        if status_code == 1:
            break
        if status_code != 0:
            raise RuntimeError('invalid H3 owner-load heartbeat status')
        tick += 1
        time.sleep(1)
    if worker is not None:
        worker.join()
        print(json.dumps(dict(event='h3_owner_load_end', seconds=time.monotonic()-started,
                              ok=owner_error is None,
                              error=None if owner_error is None else repr(owner_error))), flush=True)
    # Every live rank reaches this gate before any mesh/wrapping/state collective.
    status = hashlib.sha256(b"h3-init-ok" if owner_error is None else b"h3-init-failed").hexdigest()
    try:
        _agree_metadata(status, rank, process_group)
    except BaseException as error:
        if owner_error is not None:
            raise owner_error from error
        raise
    if owner_error is not None:
        raise owner_error

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
    mesh = _world_mesh(process_group)
    for name in meta.block_names:
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        block = checkpoint_wrapper(model.get_submodule(name), checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        setattr(parent, child_name, block)
        # FSDP outside AC: one gather/cast region per wrapped forward. There is
        # exactly one nonreentrant checkpoint path; internal AC stays disabled.
        _shard_block(block, mesh)
    if set(model.state_dict()) != set(meta.state_specs):
        raise ValueError("checkpoint/FSDP wrapping changed canonical state FQNs")
    model.to_empty(device=device)  # DTensors allocate local shards, never full GPU base
    incompatible = _set_full_state(model, state)  # ALL ranks, including empty-state peers
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("strict FSDP initialization returned incompatible keys")
    del state
    for name, saved in meta.nonpersistent_buffers.items():
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        current = getattr(parent, leaf)
        if current.shape != saved.shape or current.dtype != saved.dtype:
            raise ValueError(f"constructor buffer shape/precision changed: {name}")
        with torch.no_grad():
            current.copy_(saved.to(device=device))
        if current.device != device or not torch.equal(current.cpu(), saved):
            raise ValueError(f"constructor buffer restoration failed: {name}")

    params = _canonical_parameter_names(model, meta.block_names)
    if {n for n, p in params.items() if p.requires_grad} != set(meta.lora_names):
        raise ValueError("materialized trainable scope differs from loader")
    final_state = model.state_dict()
    if set(final_state) != set(meta.state_specs):
        raise ValueError("materialized state FQNs differ from loader")
    for name, value in final_state.items():
        spec = meta.state_specs[name]
        if (value.is_meta or value.device != device or tuple(value.shape) != spec.shape
                or value.dtype != spec.dtype):
            raise ValueError(f"materialized shape/device/precision mismatch: {name}")
    del final_state
    if any(getattr(m, 'gradient_checkpointing', False) for m in model.modules()):
        raise ValueError("materialization enabled internal gradient checkpointing")
    model.eval()  # eval is intentional; autograd remains caller-controlled
    bundle = PreparedModelBundle(ModelComponents((ComponentBinding(
        'transformer', model, (ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
    ),)))
    try:
        optimizer = torch.optim.AdamW(bundle.trainable_parameters(), **optimizer_config)
        validate_optimizer_parameter_subset(bundle, optimizer)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.) if constant_scheduler else None
        return H3PreparedModel(source_bundle=bundle, prepared_root=bundle, optimizer=optimizer,
                               scheduler=scheduler, accelerator=_SynchronousBackward())
    except BaseException:
        bundle.release_forward_guards()
        raise
