"""Prepare CPU control tests: real loader/PEFT/AC/Adam/guard/bridge.

Only CUDA admission, WORLD observations/metadata exchange, mesh, block sharding,
and the collective setter are substituted. No process group or fake DTensor.
"""
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_h3_mvp_cpu import _loading_fixture, _load_cpu, _forward_case
from visual_rl.models.implementations.h3_forward import h3_t2va_forward
from visual_rl.models.lifecycle import h3_fsdp
from visual_rl.models.lifecycle.prepared import PreparedBundleError


def test_h3_fsdp2_real_api_and_mixed_storage_dtype_admission():
    assert Path(h3_fsdp.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/models/lifecycle/h3_fsdp.py'
    h3_fsdp._require_apis()
    # Exercise the installed dtype methods without mesh/PG construction. Real
    # FSDPParam instances carry ordinary CPU params solely for dtype admission;
    # they are never represented as actual shards or used for a forward.
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    params = []
    for dtype in (torch.bfloat16, torch.float32):
        parameter = FSDPParam.__new__(FSDPParam)
        parameter.sharded_param = torch.nn.Parameter(torch.empty(2, dtype=dtype),
                                                     requires_grad=dtype == torch.float32)
        params.append(parameter)
    group = SimpleNamespace(
        fsdp_params=params,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.float32,
                                       cast_forward_inputs=False),
    )
    FSDPParamGroup._init_mp_dtypes(group)
    assert [p.orig_dtype for p in group.fsdp_params] == [torch.bfloat16, torch.float32]
    assert params[1].param_dtype == torch.bfloat16
    assert params[1].reduce_dtype in (None, torch.float32)


def _cpu_leaves(meta, monkeypatch, *, rank=0, broadcast_state=None):
    import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as ac

    events, meshes, setter_keys, optimizers = [], [], [], []
    pg, mesh = object(), object()
    monkeypatch.setattr(h3_fsdp.dist, 'group', SimpleNamespace(WORLD=pg))
    monkeypatch.setattr(h3_fsdp.dist, 'is_available', lambda: True)
    monkeypatch.setattr(h3_fsdp.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(h3_fsdp.dist, 'get_backend', lambda group: 'nccl')
    monkeypatch.setattr(h3_fsdp.dist, 'get_rank', lambda group: rank)
    monkeypatch.setattr(h3_fsdp.dist, 'get_world_size', lambda group: 8)

    def cuda(device):
        assert device == torch.device(f'cuda:{rank}')
        events.append('cuda')
        return torch.device('cpu')

    def gather(output, receipt, *, group):
        assert group is pg and len(output) == 8 and len(receipt[2]) == 64
        events.append('metadata')
        output[:] = [(r, receipt[1], receipt[2]) for r in range(8)]

    def make_mesh(group):
        assert group is pg
        events.append('mesh')
        meshes.append(mesh)
        return mesh

    real_ac = ac.checkpoint_wrapper
    @wraps(real_ac)
    def checkpoint(block, **kwargs):
        assert kwargs == {'checkpoint_impl': ac.CheckpointImpl.NO_REENTRANT}
        assert all(p.is_meta for p in block.parameters())
        events.append('ac')
        return real_ac(block, **kwargs)

    def shard(block, received_mesh):
        assert received_mesh is mesh and isinstance(block, ac.CheckpointWrapper)
        assert all(p.is_meta for p in block.parameters())
        assert {p.dtype for p in block.parameters()} == {torch.bfloat16, torch.float32}
        assert not any(getattr(m, 'gradient_checkpointing', False) for m in block.modules())
        events.append('shard')

    real_load = h3_fsdp.load_h3_cpu_state
    def load(*args, **kwargs):
        assert rank == 0 and kwargs['is_init_owner'] is True
        events.append('read')
        return real_load(*args, **kwargs)

    def setter(model, state):
        assert all(not p.is_meta and p.device.type == 'cpu' for p in model.parameters())
        events.append('materialized/setter')
        setter_keys.append(set(state))
        # Force a visible invalid to_empty buffer: prepare must restore it.
        model.rope.inv_freq.fill_(float('nan'))
        # Explicit CPU leaf substitute for the collective only. assign=True
        # replaces parameter objects, unlike the production DCP setter.
        return model.load_state_dict(state if rank == 0 else broadcast_state, strict=True, assign=True)

    real_adam = torch.optim.AdamW
    def adam(params, **kwargs):
        events.append('optimizer')
        assert torch.equal(meta.model.rope.inv_freq, meta.nonpersistent_buffers['rope.inv_freq'])
        assert not meta.model.training
        optimizer = real_adam(params, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(h3_fsdp, '_cuda_device', cuda)
    monkeypatch.setattr(h3_fsdp.dist, 'all_gather_object', gather)
    monkeypatch.setattr(h3_fsdp, '_world_mesh', make_mesh)
    monkeypatch.setattr(ac, 'checkpoint_wrapper', checkpoint)
    monkeypatch.setattr(h3_fsdp, '_shard_block', shard)
    monkeypatch.setattr(h3_fsdp, '_set_full_state', setter)
    monkeypatch.setattr(h3_fsdp, 'load_h3_cpu_state', load)
    monkeypatch.setattr(torch.optim, 'AdamW', adam)
    args = dict(process_group=pg, device=f'cuda:{rank}', rank=rank, local_rank=rank,
                local_world_size=8, host_weight_limit_bytes=10**8, init_seed=42)
    return SimpleNamespace(events=events, meshes=meshes, setter_keys=setter_keys,
                           optimizers=optimizers, args=args, pg=pg)


def test_h3_prepare_owner_order_final_optimizer_guard_and_real_bridge_step(tmp_path, monkeypatch):
    meta, _, cls = _loading_fixture(tmp_path, monkeypatch)
    old_params = tuple(meta.model.parameters())  # retain objects to prevent id reuse
    raw_forward = cls.forward
    calls = []
    def forward(self, *, hidden_states, audio_hidden_states, **kwargs):
        calls.append(kwargs)
        return raw_forward(self, hidden_states, audio_hidden_states)
    monkeypatch.setattr(cls, 'forward', forward)
    leaves = _cpu_leaves(meta, monkeypatch)
    prepared = h3_fsdp.prepare_h3_fsdp(meta, **leaves.args, constant_scheduler=True)
    assert leaves.events == ['cuda', 'metadata', 'read', 'mesh', 'ac', 'shard', 'ac', 'shard',
                             'materialized/setter', 'optimizer']
    assert leaves.setter_keys == [set(meta.state_specs)] and len(leaves.meshes) == 1
    assert prepared.reducer_divisor == h3_fsdp.H3_REDUCER_DIVISOR == 8.0
    final_params = dict(prepared.trainable_named_parameters())
    assert len(final_params) == len(meta.lora_names)
    assert all('._checkpoint_wrapped_module.' in n for n in final_params)
    optimizer = prepared.optimizer
    assert {id(p) for p in optimizer.param_groups[0]['params']} == {id(p) for p in final_params.values()}
    assert not {id(p) for p in old_params} & {id(p) for p in optimizer.param_groups[0]['params']}
    assert all(p.dtype == torch.float32 and not p.is_meta for p in final_params.values())
    assert optimizer.defaults['lr'] == 1e-4 and optimizer.defaults['eps'] == 1e-12
    assert optimizer.defaults['betas'] == (.9, .999) and optimizer.defaults['weight_decay'] == 0
    with pytest.raises(PreparedBundleError, match='bypassed'):
        meta.model(hidden_states=torch.zeros(1, 1, 96), audio_hidden_states=torch.zeros(1, 1, 32))
    before = {n: p.detach().clone() for n, p in final_params.items()}
    with prepared.accumulate():
        output = h3_t2va_forward(prepared, **_forward_case())
        video, audio = _forward_case()['geometry'].joint_layout.unpack(output)
        # Independently prove that both modalities drive real LoRA gradients.
        for value in (video, audio):
            optimizer.zero_grad(set_to_none=True)
            value.square().mean().backward(retain_graph=True)
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in final_params.values())
        optimizer.zero_grad(set_to_none=True)
        output.square().mean().backward()
    assert len(calls) == 1 and calls[0]['return_dict'] is False
    assert all(p.grad is None for p in meta.model.parameters() if not p.requires_grad)
    optimizer.step()
    prepared.scheduler.step()
    assert any(not torch.equal(before[n], p) for n, p in final_params.items())
    assert optimizer.param_groups[0]['lr'] == 1e-4
    with pytest.raises(ValueError, match='already prepared'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    prepared.close()
    prepared.close()
    assert not meta.model._forward_pre_hooks
    with pytest.raises(PreparedBundleError, match='closed'):
        prepared(hidden_states=torch.zeros(1, 1, 96), audio_hidden_states=torch.zeros(1, 1, 32))
    assert leaves.events.count('optimizer') == 1


def test_h3_prepare_nonowner_empty_setter_and_same_buffer(tmp_path, monkeypatch):
    meta, _, _ = _loading_fixture(tmp_path, monkeypatch)
    # Simulated rank-zero broadcast payload created before entering prepare.
    # This is not a multi-rank run; production rank-one cannot read weights.
    owner = _load_cpu(meta)
    leaves = _cpu_leaves(meta, monkeypatch, rank=1, broadcast_state=owner.state)
    prepared = h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    assert leaves.events == ['cuda', 'metadata', 'mesh', 'ac', 'shard', 'ac', 'shard',
                             'materialized/setter', 'optimizer']
    assert leaves.setter_keys == [set()] and prepared.scheduler is None
    assert torch.equal(meta.model.rope.inv_freq, owner.nonpersistent_buffers['rope.inv_freq'])
    actual = meta.model.state_dict()
    assert set(actual) == set(owner.state)
    assert all(torch.equal(actual[n], value) for n, value in owner.state.items())
    prepared.close()


def test_h3_prepare_topology_and_metadata_early_rejections(tmp_path, monkeypatch):
    meta, _, _ = _loading_fixture(tmp_path, monkeypatch)
    leaves = _cpu_leaves(meta, monkeypatch)
    for change in (dict(process_group=object()), dict(device='cpu'), dict(device='cuda'),
                   dict(device='cuda:1'), dict(rank=1), dict(local_rank=1), dict(local_world_size=4),
                   dict(sp=2), dict(tp=2), dict(rank=False), dict(local_rank=False)):
        with pytest.raises(ValueError):
            h3_fsdp.prepare_h3_fsdp(meta, **(leaves.args | change))
        assert leaves.events == []
    monkeypatch.setattr(h3_fsdp.dist, 'get_world_size', lambda group: 4)
    with pytest.raises(ValueError, match='WORLD8'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    monkeypatch.setattr(h3_fsdp.dist, 'get_world_size', lambda group: 8)
    for mismatch in ('host', 'digest', 'rank'):
        def gather(output, receipt, *, group):
            output[:] = [(r, receipt[1], receipt[2]) for r in range(8)]
            r, host, digest = output[7]
            output[7] = (0 if mismatch == 'rank' else r,
                         'other' if mismatch == 'host' else host,
                         'other' if mismatch == 'digest' else digest)
        monkeypatch.setattr(h3_fsdp.dist, 'all_gather_object', gather)
        with pytest.raises(ValueError, match='ranks disagree'):
            h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    assert not leaves.meshes and not leaves.optimizers and 'read' not in leaves.events
    assert all(p.is_meta for p in meta.model.parameters())


def test_h3_prepare_scope_buffers_and_hyperparameters_reject_before_read(tmp_path, monkeypatch):
    meta, _, _ = _loading_fixture(tmp_path, monkeypatch)
    leaves = _cpu_leaves(meta, monkeypatch)
    meta.model.proj_in.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='trainable scope'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    # Even altered metadata must not admit root trainables to replicated params.
    altered = replace(meta, lora_names=meta.lora_names + ('proj_in.weight',))
    with pytest.raises(ValueError, match='stray trainable'):
        h3_fsdp.prepare_h3_fsdp(altered, **leaves.args)
    meta.model.proj_in.weight.requires_grad_(False)
    with pytest.raises(ValueError, match='buffer'):
        h3_fsdp.prepare_h3_fsdp(replace(meta, nonpersistent_buffers={}), **leaves.args)
    saved = meta.model.rope.inv_freq.clone()
    meta.model.rope.inv_freq.zero_()
    with pytest.raises(ValueError, match='constructor'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    meta.model.rope.inv_freq.copy_(saved)
    meta.model.gradient_checkpointing = True
    with pytest.raises(ValueError, match='internal gradient'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    meta.model.gradient_checkpointing = False
    for change in (dict(init_seed=-1), dict(host_weight_limit_bytes=0), dict(lr=0), dict(eps=0),
                   dict(betas=(.9, 1.)), dict(weight_decay=-1.), dict(lr=float('nan')),
                   dict(constant_scheduler=1)):
        with pytest.raises(ValueError):
            h3_fsdp.prepare_h3_fsdp(meta, **(leaves.args | change))
    assert all(e == 'cuda' for e in leaves.events) and not leaves.optimizers
    meta.model.load_state_dict(_load_cpu(meta).state, strict=True, assign=True)
    with pytest.raises(ValueError, match='all parameters on meta'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)


def test_h3_prepare_failed_setter_has_no_optimizer_or_cleanup_collective(tmp_path, monkeypatch):
    meta, _, _ = _loading_fixture(tmp_path, monkeypatch)
    leaves = _cpu_leaves(meta, monkeypatch)
    def broken(model, state):
        leaves.events.append('setter-failure')
        state.pop(next(iter(state)))
        # Real strict CPU setter failure, not a mocked prepare failure.
        return model.load_state_dict(state, strict=True, assign=True)
    monkeypatch.setattr(h3_fsdp, '_set_full_state', broken)
    with pytest.raises(RuntimeError, match='Missing key'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    assert leaves.events[-1] == 'setter-failure' and not leaves.optimizers
    assert not meta.model._forward_pre_hooks
    with pytest.raises(ValueError, match='already prepared'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    assert leaves.events.count('metadata') == 1


def test_h3_prepare_owner_missing_state_fails_before_wrap(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    meta, source, _ = _loading_fixture(tmp_path, monkeypatch)
    leaves = _cpu_leaves(meta, monkeypatch)
    source.pop(next(iter(source)))
    save_file(source, str(meta.transformer_dir / 'diffusion_pytorch_model.safetensors'))
    with pytest.raises(ValueError, match='exactly cover'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    assert leaves.events == ['cuda', 'metadata', 'read']
    assert not leaves.optimizers and all(p.is_meta for p in meta.model.parameters())


def test_h3_prepare_metadata_includes_seed_precision_and_adapter_scaling(tmp_path, monkeypatch):
    meta, _, _ = _loading_fixture(tmp_path, monkeypatch)
    options = dict(lr=1e-4, betas=(.9, .999), eps=1e-12, weight_decay=0.)
    digest = h3_fsdp._metadata_digest(meta, 42, 10**8, options, False)
    assert digest != h3_fsdp._metadata_digest(meta, 43, 10**8, options, False)
    assert digest != h3_fsdp._metadata_digest(meta, 42, 10**8, options | {'eps': 1e-8}, False)
    layer = meta.model.get_submodule(meta.target_names[0])
    layer.scaling['default'] *= 2
    assert digest != h3_fsdp._metadata_digest(meta, 42, 10**8, options, False)


def test_h3_fsdp_public_reducer_divisor_setter(monkeypatch):
    import torch.distributed.fsdp as fsdp
    from torch.distributed.fsdp import FSDPModule
    assert hasattr(FSDPModule, 'set_gradient_divide_factor'), torch.__version__
    group, other_group = SimpleNamespace(), SimpleNamespace()
    state = SimpleNamespace(_fsdp_param_groups=[group, other_group])
    carrier = SimpleNamespace(_get_fsdp_state=lambda: state)
    # The real public setter, with only its non-collective state leaf replaced.
    FSDPModule.set_gradient_divide_factor(carrier, 8.0)
    assert group.gradient_divide_factor == 8.0
    assert other_group.gradient_divide_factor == 8.0

    class StateOnlyBlock(FSDPModule, torch.nn.Module):
        def _get_fsdp_state(self):
            return state
    # Empty CPU state carrier, not a fake DTensor or actual sharded model.
    # Exercise the production block leaf calling that same public setter.
    group.gradient_divide_factor = other_group.gradient_divide_factor = None
    with monkeypatch.context() as patch:
        patch.setattr(fsdp, 'fully_shard', lambda *args, **kwargs: None)
        block = object.__new__(StateOnlyBlock)  # bypass FSDPModule's dynamic-class factory
        torch.nn.Module.__init__(block)
        h3_fsdp._shard_block(block, object())
    assert group.gradient_divide_factor == other_group.gradient_divide_factor == h3_fsdp.H3_REDUCER_DIVISOR
    with monkeypatch.context() as patch:
        patch.setattr(FSDPModule, 'set_gradient_divide_factor', None)
        with pytest.raises(RuntimeError, match='no default-reduction guess'):
            h3_fsdp._require_apis()


def test_h3_prepare_real_leaf_options_and_fail_closed_sharding(monkeypatch):
    import torch.distributed.fsdp as fsdp
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.checkpoint import state_dict as dcp

    group, mesh = object(), object()
    calls = []
    def from_group(group_arg, device_type, *, mesh_dim_names):
        assert group_arg is group and device_type == 'cuda' and mesh_dim_names == ('dp_shard',)
        calls.append('mesh')
        return mesh
    monkeypatch.setattr(DeviceMesh, 'from_group', from_group)
    assert h3_fsdp._world_mesh(group) is mesh

    def fully_shard(block, **kwargs):
        assert kwargs['mesh'] is mesh and kwargs['reshard_after_forward'] is True
        policy = kwargs['mp_policy']
        assert policy.param_dtype == torch.bfloat16 and policy.reduce_dtype == torch.float32
        assert policy.cast_forward_inputs is False
        calls.append('shard')
    monkeypatch.setattr(fsdp, 'fully_shard', fully_shard)
    # A no-op shard API must not be accepted by the production leaf.
    with pytest.raises(RuntimeError, match='did not shard'):
        h3_fsdp._shard_block(torch.nn.Linear(2, 2), mesh)

    model, state = torch.nn.Linear(2, 2), {}
    incompatible = SimpleNamespace(missing_keys=[], unexpected_keys=[])
    def set_model_state_dict(model_arg, *, model_state_dict, options):
        assert model_arg is model and model_state_dict is state
        assert options.full_state_dict and options.broadcast_from_rank0 and options.strict
        calls.append('setter')
        return incompatible
    monkeypatch.setattr(dcp, 'set_model_state_dict', set_model_state_dict)
    assert h3_fsdp._set_full_state(model, state) is incompatible
    assert calls == ['mesh', 'shard', 'setter']


def test_h3_prepare_rejects_unsupported_api_and_materialized_precision(tmp_path, monkeypatch):
    import torch.distributed.fsdp as fsdp
    meta, _, _ = _loading_fixture(tmp_path, monkeypatch)
    leaves = _cpu_leaves(meta, monkeypatch)
    real_shard = fsdp.fully_shard
    monkeypatch.setattr(fsdp, 'fully_shard', lambda module: None)
    with pytest.raises(RuntimeError, match='unsupported H3 FSDP2 API'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    assert leaves.events == ['cuda'] and not leaves.optimizers
    monkeypatch.setattr(fsdp, 'fully_shard', real_shard)
    setter = h3_fsdp._set_full_state
    def wrong_precision(model, state):
        result = setter(model, state)
        model.proj_in.weight.data = model.proj_in.weight.data.bfloat16()
        return result
    monkeypatch.setattr(h3_fsdp, '_set_full_state', wrong_precision)
    with pytest.raises(ValueError, match='materialized shape/device/precision'):
        h3_fsdp.prepare_h3_fsdp(meta, **leaves.args)
    assert not leaves.optimizers and not meta.model._forward_pre_hooks
