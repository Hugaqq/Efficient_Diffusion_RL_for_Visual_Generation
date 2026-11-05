from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from visual_rl.algorithms.trainer import h3_ema
from visual_rl.composition.config.h3_profile import H3EMAConfig
from visual_rl.artifacts.checkpoint.h3_ema import save_ema, load_ema


def cpu_local(parameter, *, rank, process_group):
    # Only the CUDA/DTensor observation is replaced: eight equal virtual local shards.
    shape = list(parameter.shape)
    rows = shape[0] if shape else 1
    return parameter.detach(), dict(shape=[rows*8, *shape[1:]], local_shape=shape,
                                    offset=rank*rows, dtype='torch.float32')


def make_ema(monkeypatch, rank=0, config=None):
    monkeypatch.setattr(h3_ema, 'local_parameter', cpu_local)
    p = torch.nn.Parameter(torch.arange(6).reshape(2,3).float() + rank)
    ema = h3_ema.H3LoRAEMA(lambda: (('transformer_blocks.0.attn.to_q.lora_A.default.weight',p),),
        config=config or H3EMAConfig(enabled=True), rank=rank, process_group=None)
    return p, ema


def test_ema_committed_schedule_actual_adam_and_exception_swap(monkeypatch):
    p, ema = make_ema(monkeypatch)
    optimizer = torch.optim.AdamW([p], lr=.1)
    initial = p.detach().clone()
    expected = initial.clone()
    for k in range(1, 18):
        p.square().sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        ema.commit(k)
        if k % 8 == 0:
            expected.lerp_(p.detach(), 1-min(k/(9+k), .9))
        assert torch.equal(next(iter(ema.shadow.values())), expected)
        assert ema.committed_steps == k and ema.updates == k//8
    saved = p.detach().clone()
    state = deepcopy(optimizer.state_dict())
    with pytest.raises(RuntimeError, match='evaluation failure'):
        with ema.evaluation_weights():
            assert not torch.is_grad_enabled() and torch.equal(p, expected)
            with pytest.raises(RuntimeError, match='swapped'):
                ema.commit(18)
            with pytest.raises(RuntimeError, match='swapped'):
                ema.state_dict()
            raise RuntimeError('evaluation failure')
    assert not ema.swapped and torch.equal(p, saved)
    for key in state['state'][0]:
        assert torch.equal(state['state'][0][key], optimizer.state_dict()['state'][0][key])
    assert p.requires_grad and p.grad is None
    with pytest.raises(ValueError, match='exactly once'):
        ema.commit(17)
    p.data.fill_(float('inf'))
    with pytest.raises(ValueError, match='nonfinite'):
        ema.commit(18)
    assert ema.committed_steps == 17


def test_ema_eight_rank_files_roundtrip_resume_and_strict_schema(tmp_path, monkeypatch):
    owners = [make_ema(monkeypatch, rank=r, config=H3EMAConfig(enabled=True, update_interval=2)) for r in range(8)]
    for p, ema in owners:
        for step in range(1,4):
            p.data.add_(1); ema.commit(step)
        save_ema(ema, tmp_path, step=3)
    for rank, (p, ema) in enumerate(owners):
        new_p, restored = make_ema(monkeypatch, rank=rank, config=ema.config)
        new_p.data.copy_(p)
        load_ema(restored, tmp_path, step=3)
        assert restored.committed_steps == 3 and restored.updates == 1
        p.data.add_(2); new_p.data.add_(2)
        ema.commit(4); restored.commit(4)
        assert all(torch.equal(ema.shadow[n], restored.shadow[n]) for n in ema.shadow)
        with pytest.raises(ValueError, match='fresh'):
            load_ema(restored, tmp_path, step=3)
    source = owners[0][1].state_dict()
    for key,value in (('rank',1), ('world_size',4), ('schema','unknown'), ('updates',True),
                      ('committed_steps',3), ('extra',1), ('config',{}), ('specs',{})):
        _, ema = make_ema(monkeypatch, config=owners[0][1].config)
        bad = deepcopy(source); bad[key] = value
        with pytest.raises(ValueError, match='mismatch'):
            ema.load_state_dict(bad, step=4)
        assert ema.committed_steps == 0
    for mutate in (lambda t: t.half(), lambda t: t[:, :1], lambda t: t*float('nan'), lambda t: t.requires_grad_()):
        _, ema = make_ema(monkeypatch, config=owners[0][1].config)
        bad = deepcopy(source); name = next(iter(bad['shadow'])); bad['shadow'][name] = mutate(bad['shadow'][name])
        with pytest.raises(ValueError, match='shadow'):
            ema.load_state_dict(bad, step=4)


def test_stage_ema_only_after_successful_commit_tail(tmp_path, monkeypatch):
    from tests.test_h3_mvp_cpu_stages import _setup
    from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
    from visual_rl.algorithms.trainer.interface import PrepareRunContext
    h = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda tensor, **kw: tensor.mul_(8))
    monkeypatch.setattr(h3_ema, 'local_parameter', cpu_local)
    def named():
        return tuple((f'transformer_blocks.{i}.attn.to_q.lora_A.default.weight', p)
                     for i, (_, p) in enumerate(h.prepared.trainable_named_parameters()))
    ema = h3_ema.H3LoRAEMA(named,
        config=H3EMAConfig(enabled=True, update_interval=1), rank=0, process_group=None)
    def commit(request):
        if request.identity.optimizer_step == 1:
            raise RuntimeError('injected tail failure')
        h.data.commits += 1; h.data.position += 2; h.data.behavior = request.next_behavior_id
    monkeypatch.setattr(h.data, 'commit', commit)
    stage = H3DualIterationStages(mode='dual_flow', seed=42, data=h.data, prepared=h.prepared,
        predict=h.predict, decode=h.decode, score_one=h.score_one, coordination_factory=h.coordination_factory,
        process_group=h.group, rank=0, device='cpu', read_behavior_id=lambda:h.data.behavior, ema=ema)
    trainer = stage.trainer(); trainer.prepare_run(PrepareRunContext('ema-test', 'h3', 0))
    trainer.run_iteration(0)
    assert ema.committed_steps == ema.updates == 1
    saved = ema.state_dict()
    with pytest.raises(ValueError, match='cooperative rejection'):
        trainer.run_iteration(1)
    assert ema.committed_steps == 1
    assert all(torch.equal(saved['shadow'][n], ema.shadow[n]) for n in ema.shadow)
    with pytest.raises(RuntimeError, match='poisoned'):
        trainer.run_iteration(1)
