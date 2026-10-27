"""H3 observed runtime binding, not default controller/CUDA validation.

Real files/profile/cache/source, meta/PEFT/AC/prepare/Adam/guards and HPS
factory/client execute. Only heavy models, HTTP and CUDA/FSDP/collective
observations are test leaves. No publisher is invoked.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.test_h3_mvp_cpu import _tiny_loading_class
from tests.test_h3_mvp_cpu_profile import _fixture as _profile_files, _save
from tests.test_h3_mvp_cpu_text_cache import _metadata
from tests.test_h3_mvp_cpu_hps import _health, _request, _Transport
from tests.test_h3_mvp_cpu_prepare import _cpu_leaves
from tests.test_h3_mvp_cpu_decode import _leaf_setup
from tests.test_h3_mvp_cpu_coordination import MeshObservation
from tests.test_h3_mvp_cpu_stages import _world_leaf
from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.trainer.h3_stages import H3IterationStages
from visual_rl.algorithms.trainer.interface import PrepareRunContext
from visual_rl.algorithms.rewards.clients import hpsv3
from visual_rl.algorithms.rewards.h3_hps import H3HPSReward
from visual_rl.artifacts.checkpoint.protocol import CheckpointContract
from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
from visual_rl.composition.config.h3_profile import compile_h3_profile, preflight_h3_profile
from visual_rl.composition.preflight.artifacts import _hash_artifact
from visual_rl.core.protocols.flash_hpsv3 import VideoContract, canonical_json
from visual_rl.data.h3_source import H3PromptSource
from visual_rl.models.h3_interface import H3ModelInput, H3PromptInput, H3_GUIDANCE_ID
from visual_rl.models.implementations import h3_loading
from visual_rl.models.implementations.h3 import H3Adapter
from visual_rl.models.interface import ModelPortError
from visual_rl.models.lifecycle.components import ComponentManager, ExecutionMode
from visual_rl.models.lifecycle import h3_fsdp, h3_coordination
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.runtime import h3_binding as binding_module
from visual_rl.runtime.h3_binding import H3RuntimeBinding, verify_h3_content


def _fixture(tmp_path, monkeypatch, *, bind=True):
    path, raw = _profile_files(tmp_path)
    raw['overrides'] = dict(model=dict(params=dict(height=32, width=32, lora_rank=2, lora_alpha=4)),
                            training=dict(group_size=2, num_steps=2))
    raw['launch']['hps']['timeout_s'] = 2.5
    base = _tiny_loading_class()
    class HeavyLeaf(base):
        calls = []
        def forward(self, *, hidden_states, audio_hidden_states, encoder_hidden_states, **kw):
            assert not torch.is_autocast_enabled('cpu') and not self.training
            type(self).calls.append((torch.is_grad_enabled(), kw['timestep'].clone()))
            v, a = super().forward(hidden_states, audio_hidden_states)
            condition = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype)).mean().float()
            return v + condition, a + condition
    monkeypatch.setattr(h3_loading, '_resolve_h3_class', lambda: HeavyLeaf)
    # Replace the profile fixture's header-only target subset with a real full
    # tiny checkpoint, then bind the NEW all-files content identity into cache.
    meta = h3_loading.build_h3_meta(tmp_path / 'model', rank=2, alpha=4)
    generator = torch.Generator().manual_seed(91)
    full = {key: torch.randn(meta.state_specs[name].shape, generator=generator)*.02
            for key, name in meta.base_to_model.items()}
    save_file(full, str(meta.transformer_dir / 'diffusion_pytorch_model.safetensors'))
    model_hash = _hash_artifact(tmp_path / 'model', content_policy='all-files.v1')['content_sha256']
    identity = raw['launch']['expected']['cache_identity'] | {'model_sha256': model_hash}
    raw['launch']['expected'].update(model_sha256=model_hash, cache_identity=identity)
    cache = tmp_path / 'cache'
    tensors = load_file(str(cache / 'embeddings.safetensors'))
    save_file(tensors, str(cache / 'embeddings.safetensors'), metadata=_metadata(identity))
    index = json.loads((cache / 'index.json').read_text())
    index['identity'] = identity
    import hashlib
    index['shards']['embeddings.safetensors'] = hashlib.sha256((cache / 'embeddings.safetensors').read_bytes()).hexdigest()
    (cache / 'index.json').write_text(json.dumps(index))
    contract = VideoContract(124, 32, 32)
    health = _health(contract)
    (tmp_path / 'hps.json').write_bytes(canonical_json(health))
    raw['launch']['expected']['hps_service_identity'] = health['service_identity']
    profile = compile_h3_profile(_save(tmp_path, raw))
    facts = preflight_h3_profile(profile)
    adapter = H3Adapter.from_config(profile.model.config, runtime_context=dict(
        artifact_paths={'h3-model': tmp_path / 'model', 'h3-text': cache}, expected_cache_identity=identity))
    manager = ComponentManager(adapter, execution_device='cpu')
    manager.load()
    real_adam = torch.optim.AdamW
    leaves = _cpu_leaves(adapter.meta_model, monkeypatch)
    prepared = h3_fsdp.prepare_h3_fsdp(adapter.meta_model, **leaves.args, constant_scheduler=True)
    monkeypatch.setattr(torch.optim, 'AdamW', real_adam)  # _cpu_leaves uses an optimizer construction spy
    events = []
    pg = _world_leaf(monkeypatch, events)
    old_gather = h3_coordination.dist.all_gather_object
    def gather(outputs, receipt, *, group):
        entry = json.loads(receipt)
        if entry[1] == 'coordination':
            outputs[:] = [json.dumps(entry[:2] + [r] + entry[3:]).encode() for r in range(8)]
        else:
            old_gather(outputs, receipt, group=group)
    monkeypatch.setattr(h3_coordination.dist, 'all_gather_object', gather)
    monkeypatch.setattr(h3_coordination.dist, 'all_reduce', lambda total, **kw: total.mul_(8))
    monkeypatch.setattr(h3_coordination, '_cuda_device', lambda device: torch.device('cpu'))
    monkeypatch.setattr(binding_module, '_cuda_device', lambda device: torch.device('cpu'))
    def block_observation(block):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
        assert isinstance(block, CheckpointWrapper)
        assert {p.dtype for p in block.parameters()} == {torch.float32, torch.bfloat16}
        return dict(param_dtype='bfloat16', reduce_dtype='float32', cast_forward_inputs=False, divisor=8)
    monkeypatch.setattr(binding_module, '_block_policy', block_observation)
    def dtensor_observation(value):
        for _, parameter in prepared.trainable_named_parameters():
            if value is parameter or value is parameter.grad:
                # Explicit virtual-shard observation only. Physical tensors CPU;
                # real _trainables/_local_gradient/norm checks/math still run.
                return MeshObservation(pg), (torch.distributed.tensor.Shard(0),), (8*value.shape[0], *value.shape[1:]), value.detach()
        raise AssertionError('unexpected DTensor observation')
    monkeypatch.setattr(h3_coordination, '_dtensor_parts', dtensor_observation)
    dynamics = H3CPSDynamics(adapter.geometry.joint_layout, H3ScheduleSnapshot.build(2))
    source_args = dict(train_path=profile.launch.train, eval_path=profile.launch.eval,
        train_sha256=facts.train_sha256, eval_sha256=facts.eval_sha256, adapter=adapter, dynamics=dynamics,
        rank=0, world_size=8, group_size=2, base_seed=profile.launch.seed, device='cpu',
        max_samples=2, max_cells=4, max_trajectory_bytes=profile.launch.budgets.trajectory_bytes)
    source = H3PromptSource(**source_args)
    request, _ = _request(tmp_path / 'reward', contract)
    transport = _Transport(health)
    monkeypatch.setattr(hpsv3, 'requests_session', lambda: transport)
    scorer = H3HPSReward(request, video_contract=contract).acquire()
    _leaf_setup(monkeypatch, adapter.geometry)
    code_root = tmp_path / 'code'
    code_root.mkdir()
    (code_root / 'implementation.py').write_text('def forward(x):\n    return x\n')
    code_hash = _hash_artifact(code_root, content_policy='python-code.v1')['content_sha256']
    kwargs = dict(profile=profile, facts=facts, adapter=adapter, prepared=prepared, source=source, scorer=scorer,
                  process_group=pg, rank=0, device='cuda:0', gpu_uuids=profile.launch.gpu_uuids,
                  code_root=code_root, expected_code_sha256=code_hash)
    binding = H3RuntimeBinding(**kwargs) if bind else None
    return SimpleNamespace(**locals())


def _predict(h, request):
    sid = request.samples[0][0]
    return h.binding.predict(h.prepared, sample_id=sid, x=request.initial_states[sid], index=0,
                             condition=request.conditions[sid], dynamics=h.dynamics)


def test_h3_binding_observed_numerics_real_scoped_predict_decode_and_ownership(tmp_path, monkeypatch):
    h = _fixture(tmp_path, monkeypatch)
    assert Path(binding_module.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/runtime/h3_binding.py'
    b = h.binding
    assert h.facts.model_weights_verified is False  # headers never promoted to a receipt
    assert b.content['model'] == _hash_artifact(h.profile.launch.model, content_policy='all-files.v1')
    assert b.contract.immutable_model_revision == h.profile.expected['model_sha256']
    assert b.contract.code_identity == h.code_hash
    assert b.numerics.source_projection_id == b.parameter_state.state_projection.projection_id
    assert b.numerics.parameter_dtype_policy.frozen_parameter_policy.value == 'preserve_loaded'
    assert [v.parameter_view for v in b.numerics.parameter_view_evidence] == [ParameterView.CURRENT]
    assert all(not p.enabled and p.compute_dtype == 'float32' for p in b.numerics.forward_autocast_policies)
    assert set(b.contract.model_state_projection.standalone_parameter_names) == {p.name for p in b.parameter_state.named_trainable_parameters()}
    assert {id(p.parameter) for p in b.parameter_state.named_trainable_parameters()} == {id(p) for p in h.prepared.optimizer.param_groups[0]['params']}
    assert h.adapter.model_execution_numerics is b.numerics
    request = h.source.reserve(0)
    sid = request.samples[0][0]
    typed = h.source.conditioning_for(sid)
    spec = h.adapter.latent_spec_for_batch(H3PromptInput(typed.prompt), device='cpu', dtype=torch.float16)
    model_input = H3ModelInput(request.initial_states[sid], torch.tensor(1.), typed, None, spec,
                              typed.condition_identity, (H3_GUIDANCE_ID,), audio_sigma=torch.tensor(1.))
    with pytest.raises(ModelPortError, match='active stage'):
        h.adapter.predict(model_input)
    prediction = _predict(h, request)
    params = tuple(b.parameter_state.parameters())
    for stream in h.dynamics.layout.unpack(prediction):
        gradients = torch.autograd.grad(stream.square().mean(), params, retain_graph=True)
        assert any(bool((g != 0).any()) for g in gradients)
    prediction.square().mean().backward()
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in params)
    with torch.no_grad():
        other = _predict(h, request)
    assert torch.equal(prediction.detach(), other)
    with b.decode(request.initial_states[sid]) as media:
        value = b.score_one(sample_id=sid, prompt=typed.prompt, media=media)
        assert type(value) is float and len(h.transport.frame_scores[-1]) == 124
    assert [c[0] for c in h.transport.calls] == ['GET', 'POST']  # bind does not acquire
    with pytest.raises(ValueError, match='CURRENT'):
        with b.scope(ExecutionMode.TRAIN, parameter_view=ParameterView.REFERENCE):
            pass
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='autocast'):
            _predict(h, request)
    with pytest.raises(ValueError, match='condition'):
        b.predict(h.prepared, sample_id=sid, x=request.initial_states[sid], index=0,
                  condition=request.conditions[sid] + 1, dynamics=h.dynamics)
    b.close()
    with pytest.raises((ValueError, RuntimeError), match='closed|CURRENT'):
        _predict(h, request)
    assert not h.scorer._closed and h.prepared.optimizer is not None
    h.scorer.close(); h.adapter.close(); h.manager.close()


def test_h3_binding_full_content_scope_precision_and_parameter_drift_fail_closed(tmp_path, monkeypatch):
    h = _fixture(tmp_path, monkeypatch, bind=False)
    with pytest.raises(ValueError, match='code content'):
        H3RuntimeBinding(**(h.kwargs | dict(expected_code_sha256='0'*64)))
    with pytest.raises(ValueError, match='topology'):
        H3RuntimeBinding(**(h.kwargs | dict(gpu_uuids=tuple(reversed(h.profile.launch.gpu_uuids)))))
    with pytest.raises(ValueError, match='local facts'):
        H3RuntimeBinding(**(h.kwargs | dict(facts=replace(h.facts, model_weights_verified=True))))
    scale = h.adapter.meta_model.model.time_embedder.scale
    h.adapter.meta_model.model.time_embedder.scale = scale.half()
    with pytest.raises(ValueError, match='persistent buffer'):
        H3RuntimeBinding(**h.kwargs)
    h.adapter.meta_model.model.time_embedder.scale = scale
    h.binding = H3RuntimeBinding(**h.kwargs)
    original_live = h.binding.validate_live()
    group = h.prepared.optimizer.param_groups[0]
    group['maximize'] = True
    with pytest.raises(ValueError, match='numerical mode'):
        h.binding.validate_live()
    group['maximize'] = False
    previous_foreach = group['foreach']
    group['foreach'] = True
    with pytest.raises(ValueError, match='hyperparameter'):
        h.binding.validate_live()
    group['foreach'] = previous_foreach
    p = next(iter(h.prepared.trainable_named_parameters()))[1]
    original_dtype = p.dtype
    p.data = p.data.to(torch.bfloat16)
    with pytest.raises(ValueError, match='precision'):
        h.binding.validate_live()
    p.data = p.data.to(original_dtype)
    model = h.adapter.meta_model.model
    name, original = next(iter(model.named_parameters()))
    parent, _, leaf = name.rpartition('.')
    module = model.get_submodule(parent) if parent else model
    setattr(module, leaf, torch.nn.Parameter(original.detach().clone(), requires_grad=original.requires_grad))
    with pytest.raises(ValueError, match='identity|parameter'):
        h.binding.validate_live()
    setattr(module, leaf, original)
    rope = model.rope.inv_freq.clone()
    model.rope.inv_freq.add_(1)
    with pytest.raises(ValueError, match='rope'):
        h.binding.validate_live()
    model.rope.inv_freq.copy_(rope)
    assert h.binding.validate_live() == original_live
    for path in (h.profile.launch.model / 'vae/diffusion_pytorch_model.safetensors',
                 h.code_root / 'implementation.py', h.profile.launch.text_cache / 'index.json',
                 h.scorer._request.artifact_location, h.profile.launch.train):
        raw = path.read_bytes()
        path.write_bytes(raw + b' ')
        with pytest.raises(ValueError):
            h.binding.validate_live()
        path.write_bytes(raw)
    assert h.binding.validate_live() == original_live
    h.scorer.close()
    with pytest.raises(ValueError, match='acquired'):
        h.binding.validate_live()


def test_h3_binding_source_schedule_contract_and_post_tail_progress(tmp_path, monkeypatch):
    h = _fixture(tmp_path, monkeypatch, bind=False)
    foreign = H3PromptSource(**(h.source_args | dict(base_seed=43)))
    with pytest.raises(ValueError, match='identity mismatch'):
        H3RuntimeBinding(**(h.kwargs | dict(source=foreign)))
    foreign = H3PromptSource(**(h.source_args | dict(dynamics=H3CPSDynamics(h.dynamics.layout, h.dynamics.schedule, eta=.5))))
    with pytest.raises(ValueError, match='identity mismatch'):
        H3RuntimeBinding(**(h.kwargs | dict(source=foreign)))
    foreign = H3PromptSource(**(h.source_args | dict(max_cells=5)))
    with pytest.raises(ValueError, match='identity mismatch'):
        H3RuntimeBinding(**(h.kwargs | dict(source=foreign)))
    b = H3RuntimeBinding(**h.kwargs)
    assert isinstance(b.contract, CheckpointContract)
    from visual_rl.artifacts.checkpoint.protocol import CHECKPOINT_PROGRESS_SCHEMA_VERSION
    assert b.contract.progress_state_schema == f'checkpoint-progress.v{CHECKPOINT_PROGRESS_SCHEMA_VERSION}'
    assert b.contract.prepared_projection().model_execution_numerics_id == b.numerics.execution_numerics_id
    assert b.state_identity.model_id == b.validate_live()['model']
    stages = H3IterationStages(data=h.source, prepared=h.prepared, predict=b.predict, decode=b.decode,
                               score_one=b.score_one, coordination_factory=b.coordination_factory,
                               process_group=h.pg, rank=0, device='cpu', read_behavior_id=h.source.read_behavior_id)
    trainer = stages.trainer()
    trainer.prepare_run(PrepareRunContext('binding-test', 'h3', 0))
    with pytest.raises(RuntimeError, match='tail'):
        b.checkpoint_boundary(stages, trainer)
    original_optimize = trainer._stages['optimize']
    observed = []
    def optimize(value):
        result = original_optimize(value)
        with pytest.raises(RuntimeError, match='tail'):
            b.checkpoint_boundary(stages, trainer)  # optimizer done, data/BaseTrainer tail NOT done
        observed.append('rejected-before-tail')
        return result
    trainer._stages['optimize'] = optimize
    result = trainer.run_iteration(0)
    assert result.value.payload.local_cells == 4 and observed == ['rejected-before-tail']
    before_rng = RankRNGSnapshot.capture_current(0)
    progress, proof, data, rng = b.checkpoint_boundary(stages, trainer)
    assert rng == before_rng == RankRNGSnapshot.capture_current(0)
    assert progress.next_optimizer_step == proof.committed_optimizer_step == data['next_optimizer_step'] == 1
    assert data['cursor'] == 8 and data['behavior_id'] == h.source.read_behavior_id()
    assert data['payload'] == h.source.state_dict()
    assert data['content_id'] == b.validate_live()['data']
    assert progress.next_prompt_batch_id == data['next_global_batch_id']
    assert progress.rng_state_id == rng.state_identity
    assert progress.dynamics_selection_policy.seed_derivation_schema == 'h3.no-selection-rng'
    proof.assert_ready(progress)
    for namespace in ('h3-data', 'h3-batch'):
        with pytest.raises(ValueError):
            binding_module._unpack_id('other:' + 'a'*64, namespace)
    p = b.parameter_state.parameters()[0]
    p.grad = torch.zeros_like(p)
    with pytest.raises(RuntimeError, match='gradients'):
        b.checkpoint_boundary(stages, trainer)
    p.grad = None
    trainer.run_iteration(1)
    progress2, proof2, data2, _ = b.checkpoint_boundary(stages, trainer)
    assert trainer.next_optimizer_step == progress2.next_optimizer_step == proof2.committed_optimizer_step == 2
    assert data2['cursor'] == 16 and data2['next_optimizer_step'] == 2
    assert h.prepared.scheduler.last_epoch == 2 and h.prepared.scheduler._step_count == 3
    assert h.prepared.scheduler.base_lrs == h.prepared.scheduler.get_last_lr() == [1e-4]
    assert h.prepared.optimizer.param_groups[0]['lr'] == 1e-4
    assert all(state['step'].item() == 2 for state in h.prepared.optimizer.state.values())
    assert observed == ['rejected-before-tail', 'rejected-before-tail']
    request = h.source.reserve(2)
    with pytest.raises(RuntimeError, match='boundary'):
        b.checkpoint_boundary(stages, trainer)
    h.source.abort(request)
    with pytest.raises(RuntimeError, match='boundary'):
        b.checkpoint_boundary(stages, trainer)
    trainer.close(); b.close(); h.scorer.close(); h.adapter.close(); h.manager.close()


def test_h3_binding_initial_scheduler_ownership_and_constant_state_no_probe_steps(tmp_path, monkeypatch):
    h = _fixture(tmp_path, monkeypatch, bind=False)
    optimizer, scheduler = h.prepared.optimizer, h.prepared.scheduler
    # Same exact parameter set is not ownership of the same optimizer object.
    external = torch.optim.AdamW(optimizer.param_groups[0]['params'], lr=1e-4)
    foreign = torch.optim.lr_scheduler.LambdaLR(external, lambda _: 1.)
    calls = len(h.HeavyLeaf.calls)
    def forbidden(*args, **kwargs):
        pytest.fail('binding must not probe scheduler/optimizer.step')
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, 'step', forbidden)
        patch.setattr(torch.optim.lr_scheduler.LambdaLR, 'step', forbidden)
        patch.setattr(h.prepared, '_scheduler', foreign)
        with pytest.raises(ValueError, match='scheduler.*prepared optimizer'):
            H3RuntimeBinding(**h.kwargs)
    for field, bad in (('base_lrs', [2e-4]), ('_last_lr', [2e-4]),
                       ('last_epoch', 1), ('_step_count', 2)):
        with monkeypatch.context() as patch:
            patch.setattr(torch.optim.AdamW, 'step', forbidden)
            patch.setattr(torch.optim.lr_scheduler.LambdaLR, 'step', forbidden)
            patch.setattr(scheduler, field, bad)
            with pytest.raises(ValueError, match='scheduler'):
                H3RuntimeBinding(**h.kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, 'step', forbidden)
        patch.setattr(torch.optim.lr_scheduler.LambdaLR, 'step', forbidden)
        h.binding = H3RuntimeBinding(**h.kwargs)
        h.binding.validate_live()
    assert len(h.HeavyLeaf.calls) == calls and not optimizer.state and not external.state
    assert scheduler.last_epoch == 0 and scheduler._step_count == 1
    h.binding.close(); h.scorer.close(); h.adapter.close(); h.manager.close()


def test_h3_binding_scheduler_drift_rejected_live_forward_and_post_tail(tmp_path, monkeypatch):
    h = _fixture(tmp_path, monkeypatch)
    b, optimizer, scheduler = h.binding, h.prepared.optimizer, h.prepared.scheduler
    stages = H3IterationStages(data=h.source, prepared=h.prepared, predict=b.predict, decode=b.decode,
        score_one=b.score_one, coordination_factory=b.coordination_factory,
        process_group=h.pg, rank=0, device='cpu', read_behavior_id=h.source.read_behavior_id)
    trainer = stages.trainer()
    trainer.prepare_run(PrepareRunContext('scheduler-drift', 'h3', 0))
    trainer.run_iteration(0)
    boundary = b.checkpoint_boundary(stages, trainer)
    external = torch.optim.AdamW(optimizer.param_groups[0]['params'], lr=1e-4)
    replacement = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    replacement.load_state_dict(scheduler.state_dict())
    # Even a replacement with identical state and correct optimizer must fail.
    assert replacement.state_dict() == scheduler.state_dict()
    mutations = [
        (scheduler, 'optimizer', external), (h.prepared, '_scheduler', replacement),
        (scheduler, 'base_lrs', [2e-4]), (scheduler, '_last_lr', [2e-4]),
        (scheduler, 'last_epoch', 0), (scheduler, '_step_count', 1),
        (optimizer.param_groups[0], 'initial_lr', 2e-4),
        (optimizer.param_groups[0], 'lr', 2e-4),
        (next(iter(optimizer.state.values())), 'step', torch.tensor(2.)),
    ]
    calls = len(h.HeavyLeaf.calls)
    def forbidden(*args, **kwargs):
        pytest.fail('validation/failed forward must not update optimizer or scheduler')
    def mutate(patch, target, field, bad):
        if isinstance(target, dict):
            patch.setitem(target, field, bad)
        else:
            patch.setattr(target, field, bad)
        patch.setattr(torch.optim.AdamW, 'step', forbidden)
        patch.setattr(torch.optim.lr_scheduler.LambdaLR, 'step', forbidden)
    for target, field, bad in mutations:
        with monkeypatch.context() as patch:
            mutate(patch, target, field, bad)
            with pytest.raises(ValueError):
                b.validate_live()
            with pytest.raises(ValueError):
                b.checkpoint_boundary(stages, trainer)
            with pytest.raises(ValueError):
                with b.scope(ExecutionMode.TRAIN):
                    pytest.fail('invalid scheduler admitted into forward scope')
        assert b.checkpoint_boundary(stages, trainer) == boundary
    request = h.source.reserve(1)
    for target, field, bad in mutations:
        with monkeypatch.context() as patch:
            mutate(patch, target, field, bad)
            with pytest.raises(ValueError):
                _predict(h, request)
    assert len(h.HeavyLeaf.calls) == calls and not external.state
    assert scheduler.last_epoch == 1 and scheduler._step_count == 2
    assert optimizer.param_groups[0]['lr'] == 1e-4
    assert all(state['step'].item() == 1 for state in optimizer.state.values())
    h.source.abort(request)
    trainer.close(); b.close(); h.scorer.close(); h.adapter.close(); h.manager.close()


def test_h3_binding_real_block_policy_observer_rejects_wrong_precision():
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    state = FSDPState()
    assert hasattr(state, '_fsdp_param_groups')
    class ObservedBlock(FSDPModule, torch.nn.Module):
        def _get_fsdp_state(self):
            return state
    # No sharding/mesh/PG: the FSDP state leaf holds real policy objects only.
    # Mixed parameter dtypes can occupy multiple groups; check every group.
    state._fsdp_param_groups = [SimpleNamespace(mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False), gradient_divide_factor=8)
        for _ in range(2)]
    block = object.__new__(ObservedBlock)
    torch.nn.Module.__init__(block)
    expected = dict(param_dtype='bfloat16', reduce_dtype='float32', cast_forward_inputs=False, divisor=8)
    assert binding_module._block_policy(block) == expected
    for changes in (dict(param_dtype=torch.float16), dict(reduce_dtype=torch.bfloat16), dict(cast_forward_inputs=True)):
        state._fsdp_param_groups[-1].mp_policy = MixedPrecisionPolicy(**(
            dict(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False) | changes))
        with pytest.raises(ValueError, match='mixed precision'):
            binding_module._block_policy(block)
    with pytest.raises(ValueError, match='FSDP2'):
        binding_module._block_policy(torch.nn.Linear(1, 1))
