"""Real CPU controller/PEFT/Adam/DCP/EMA path; only model size, device and transport leaves."""
from dataclasses import replace
import json
import math

import pytest
import torch
import yaml

from tests.test_h3_mvp_cpu_entry import _controller_case, _config, _run
from tests.test_h3_mvp_cpu_checkpoint_state import _equal
from visual_rl.algorithms.trainer import h3_ema
from visual_rl.artifacts.checkpoint import h3_transaction


def test_extensions_full_controller_checkpoint_resume_and_changed_config_rejection(tmp_path, monkeypatch):
    case = _controller_case(tmp_path, monkeypatch)
    def cpu_shard(parameter, *, rank, process_group):
        assert process_group is case.world.group
        shape = list(parameter.shape); chunk = math.ceil(shape[0]/8); offset = min(rank*chunk, shape[0])
        local = case.world.shard(parameter, rank).detach()
        return local, dict(shape=shape, local_shape=list(local.shape), offset=offset, dtype='torch.float32')
    monkeypatch.setattr(h3_ema, 'local_parameter', cpu_shard)
    def config(name, output, resume=None, **changes):
        path = _config(case, name, output=output, resume=resume)
        raw = yaml.safe_load(path.read_text())
        raw['overrides']['training'].update(algorithm='dual_flow', kl_beta=.1,
            ema=dict(enabled=True, update_interval=1), advantage_std_domain='group')
        raw['overrides']['training'].update(changes)
        raw['launch']['hps']['origins'] = [raw['launch']['hps']['origin'], 'http://127.0.0.1:18889']
        path.write_text(yaml.safe_dump(raw))
        return path
    continuous = tmp_path/'continuous'
    results,_ = _run(case, config('extensions', continuous))
    for result in results:
        if isinstance(result, Exception):
            raise result
    expected = case.saved.copy()
    metrics = json.loads((continuous/'h3-last-metrics.json').read_text())
    assert metrics['ema_committed_steps'] == metrics['ema_updates'] == 2
    assert all(math.isfinite(metrics[k]) and metrics[k] >= 0 for k in ('kl_video','kl_audio','kl','lora_norm','ema_live_distance'))
    assert metrics['lora_norm'] > 0 and metrics['ema_live_distance'] > 0
    assert metrics['policy_loss'] + metrics['kl_loss'] == pytest.approx(metrics['loss'], abs=1e-8)
    assert metrics['kl_loss'] == pytest.approx(.1*metrics['kl'], rel=1e-6, abs=1e-10)
    interrupted = tmp_path/'interrupted'
    path = config('extensions-interrupted', interrupted)
    original = h3_transaction.H3CheckpointTransaction.checkpoint
    def interrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result.step == 1:
            raise RuntimeError('stop after durable EMA checkpoint')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(h3_transaction.H3CheckpointTransaction, 'checkpoint', interrupt)
        stopped,_ = _run(case, path)
    assert all(isinstance(r, RuntimeError) and 'durable EMA' in str(r) for r in stopped), stopped
    first = json.loads((interrupted/'h3-last-metrics.json').read_text())
    assert first['kl_video'] == first['kl_audio'] == first['kl'] == 0.
    checkpoint = interrupted/'checkpoints'/'step-1'
    # Same code/topology, different science: reject at owner admission, before model load.
    for change in (dict(kl_beta=.2), dict(ema=dict(enabled=False)), dict(ema=dict(enabled=True, decay=.8)),
                   dict(ema=dict(enabled=True, update_interval=2)), dict(advantage_std_domain='batch')):
        changed,_ = _run(case, config('changed', interrupted, resume=checkpoint, **change))
        assert all(isinstance(r, Exception) for r in changed), changed
        assert any(isinstance(r, ValueError) and 'different H3' in str(r) for r in changed), changed
    different_origins = config('changed-origins', interrupted, resume=checkpoint)
    raw = yaml.safe_load(different_origins.read_text())
    raw['launch']['hps']['origins'].pop()
    different_origins.write_text(yaml.safe_dump(raw))
    changed,_ = _run(case, different_origins)
    assert all(isinstance(r, Exception) for r in changed)
    assert any('different H3' in str(r) for r in changed)
    resumed,_ = _run(case, config('extensions-resume', interrupted, resume=checkpoint))
    for result in resumed:
        if isinstance(result, Exception):
            raise result
    for rank in range(8):
        _equal(case.saved[rank], expected[rank])
        before = torch.load(continuous/'checkpoints'/'step-2'/'h3-ema'/f'rank-{rank}.pt', weights_only=True)
        after = torch.load(interrupted/'checkpoints'/'step-2'/'h3-ema'/f'rank-{rank}.pt', weights_only=True)
        _equal(before, after)
    assert json.loads((interrupted/'h3-last-metrics.json').read_text()) == metrics
    # The original checkpoint and live weights are not replaced by EMA.
    assert (checkpoint/'complete.json').is_file()


def test_disabled_flash_full_cpu_path_matches_7c85751(tmp_path, monkeypatch):
    from copy import deepcopy
    from tests.test_h3_training_extensions import legacy_module
    from visual_rl.algorithms.trainer import h3_stages, h3_dual_stages
    from visual_rl.algorithms.optimization import h3_world
    from visual_rl.models.lifecycle import h3_fsdp
    from visual_rl.runtime import h3_binding
    case = _controller_case(tmp_path, monkeypatch)
    captures = [{}, {}]; phase = [0]
    prepare = h3_fsdp.prepare_h3_fsdp
    def observed_prepare(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        rank = case.world.tls.rank
        def gradients(opt, args, kwargs):
            captures[phase[0]].setdefault(rank, {}).setdefault('gradients', []).append(
                [p.grad.clone() for _,p in prepared.trainable_named_parameters()])
        prepared.optimizer.register_step_pre_hook(gradients)
        return prepared
    monkeypatch.setattr(h3_fsdp, 'prepare_h3_fsdp', observed_prepare)
    rollout = h3_dual_stages.H3DualIterationStages.rollout
    advantage = h3_stages.H3IterationStages.advantage
    def observed_rollout(self, value):
        result = rollout(self, value)
        ctx = result.payload
        captures[phase[0]].setdefault(self.rank, {}).setdefault('rollouts', []).append(
            dict(records=[dict(x=r.x.clone(), action=r.action.clone(), index=r.index) for r in ctx.dual_records],
                 terminals={sid:v.clone() for sid,v in ctx.dual_terminals.items()}))
        return result
    def observed_advantage(self, value):
        result = advantage(self, value)
        captures[phase[0]].setdefault(self.rank, {}).setdefault('credit', []).append(
            dict(rewards=deepcopy(result.payload.rewards), advantages=deepcopy(result.payload.advantages)))
        return result
    monkeypatch.setattr(h3_dual_stages.H3DualIterationStages, 'rollout', observed_rollout)
    monkeypatch.setattr(h3_stages.H3IterationStages, 'advantage', observed_advantage)
    def config(name):
        path = _config(case, name, output=tmp_path/name)
        raw = yaml.safe_load(path.read_text())
        raw['overrides']['training'].update(algorithm='dual_flash', num_steps=10)
        path.write_text(yaml.safe_dump(raw))
        return path
    current,_ = _run(case, config('current-default'))
    assert all(not isinstance(r, Exception) for r in current), current
    current_state = case.saved.copy()
    phase[0] = 1
    old_stage = legacy_module('visual_rl/algorithms/trainer/h3_stages.py')
    old_dual = legacy_module('visual_rl/algorithms/trainer/h3_dual_stages.py')
    old_loss = legacy_module('visual_rl/algorithms/optimization/h3_dual_credit.py')
    old_world = legacy_module('visual_rl/algorithms/optimization/h3_world.py')
    old_binding = legacy_module('visual_rl/runtime/h3_binding.py')
    old_world.H3WorldPlan = h3_world.H3WorldPlan
    old_world._world_rank = h3_world._world_rank
    old_stage.H3IterationRequest = h3_stages.H3IterationRequest
    old_stage.admit_h3_world = old_world.admit_h3_world
    old_stage.gather_h3_credit = old_world.gather_h3_credit
    old_dual.policy_loss = old_loss.policy_loss
    advantage = old_stage.H3IterationStages.advantage
    monkeypatch.setattr(h3_stages.H3IterationStages, 'build', old_stage.H3IterationStages.build)
    monkeypatch.setattr(h3_stages.H3IterationStages, 'commit_iteration', old_stage.H3IterationStages.commit_iteration)
    monkeypatch.setattr(h3_dual_stages.H3DualIterationStages, 'optimize', old_dual.H3DualIterationStages.optimize)
    monkeypatch.setattr(h3_binding.H3RuntimeBinding, '_contract', old_binding.H3RuntimeBinding._contract)
    legacy,_ = _run(case, config('legacy-default'))
    assert all(not isinstance(r, Exception) for r in legacy), legacy
    for rank in range(8):
        _equal(case.saved[rank], current_state[rank])
        _equal(captures[0][rank], captures[1][rank])
    roots = [tmp_path/name/'checkpoints'/'step-2' for name in ('current-default','legacy-default')]
    for relative in ('h3-state/manifest.json', 'h3_transaction.json'):
        assert (roots[0]/relative).read_bytes() == (roots[1]/relative).read_bytes()
    for rank in range(8):
        _equal(*(torch.load(root/'h3-ranks'/f'rank-{rank}.pt', weights_only=True) for root in roots))
    assert all(not (root/'h3-ema').exists() for root in roots)
    assert current[0].last_metrics == legacy[0].last_metrics
