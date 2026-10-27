"""Actual default controller lifecycle with CPU model/transport/collective leaves."""
from pathlib import Path
from types import SimpleNamespace
import json
import threading

import pytest
import torch
import yaml

from tests.test_h3_mvp_cpu_bound_checkpoint import _case, _snapshot, _equal
from tests.test_h3_mvp_cpu_hps import _Transport
from visual_rl.runtime.composition import create_default_run_controller
from visual_rl.runtime import h3_lifecycle, h3_session
from visual_rl.models.lifecycle import h3_fsdp
from visual_rl.models.implementations import h3_loading
from visual_rl.algorithms.rewards.clients import hpsv3
from visual_rl.artifacts.checkpoint import h3_transaction
from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot


def _controller_case(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    world, assets = case.world, case.assets
    monkeypatch.setattr(h3_loading, '_resolve_h3_class', lambda: assets.HeavyLeaf)
    # Accelerate's meta construction patches Module globally. Real ranks are
    # separate processes; serialize only this construction in our thread model.
    construction_lock = threading.Lock()
    build_meta = h3_loading.build_h3_meta
    def locked_build(*args, **kwargs):
        with construction_lock:
            return build_meta(*args, **kwargs)
    monkeypatch.setattr(h3_loading, 'build_h3_meta', locked_build)
    monkeypatch.setattr(hpsv3, 'requests_session', lambda: _Transport(assets.transport.health))
    monkeypatch.setattr(h3_lifecycle, '_execution_device', lambda session: torch.device('cpu'))
    world.owners = [None] * 8
    case.saved, case.closed, case.restored = {}, [], {}

    class SessionLeaf:
        local_world_size = 8
        def __init__(self, profile):
            self.rank = self.local_rank = world.tls.rank
            self.process_group, self.device = world.group, torch.device('cuda', self.rank)
            self.gpu_uuids = profile.launch.gpu_uuids
        def close(self):
            case.closed.append(self.rank)
    monkeypatch.setattr(h3_session.H3RuntimeSession, 'create', classmethod(lambda cls, profile, facts: SessionLeaf(profile)))
    # Only WORLD/CUDA admission and parameter communication are substituted.
    monkeypatch.setattr(h3_fsdp, '_admit_world', lambda *args: torch.device('cpu'))
    real_gather = world.gather
    def gather(output, receipt, *, group):
        if isinstance(receipt, tuple):
            assert group is world.group
            output[:] = world.exchange(world.tls.rank, 'prepare-metadata', receipt)
        else:
            real_gather(output, receipt, group=group)
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    monkeypatch.setattr(h3_fsdp, '_world_mesh', lambda group: world.group)
    def shard(block, mesh):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
        assert mesh is world.group and isinstance(block, CheckpointWrapper)
        assert all(p.is_meta for p in block.parameters())
    monkeypatch.setattr(h3_fsdp, '_shard_block', shard)
    def setter(model, state):
        states = world.exchange(world.tls.rank, 'initial-weights', state)
        return model.load_state_dict({k:v.clone() for k,v in states[0].items()}, strict=True, assign=True)
    monkeypatch.setattr(h3_fsdp, '_set_full_state', setter)
    real_prepare = h3_fsdp.prepare_h3_fsdp
    def prepare(meta, **kwargs):
        prepared = real_prepare(meta, **kwargs)
        rank = world.tls.rank
        world.owners[rank] = SimpleNamespace(rank=rank, prepared=prepared)
        hooks = []
        for name, p in prepared.trainable_named_parameters():
            def average(g, name=name):
                return world.exchange(rank, 'mean:'+name, g.detach().clone()).clone()
            hooks.append(p.register_hook(average))
        def clip_shards(optimizer, args, kwargs):
            for name, p in prepared.trainable_named_parameters():
                local = world.shard(p.grad, rank).detach().clone()
                p.grad.copy_(world.exchange(rank, 'clip-shards:'+name, local))
        hooks.append(prepared.optimizer.register_step_pre_hook(clip_shards))
        real_close = prepared.close
        def close():
            if rank not in case.saved:
                case.saved[rank] = _snapshot(prepared)
                for hook in hooks: hook.remove()
            real_close()
        monkeypatch.setattr(prepared, 'close', close)
        return prepared
    monkeypatch.setattr(h3_fsdp, 'prepare_h3_fsdp', prepare)  # observer, delegates the complete real prepare
    monkeypatch.setattr(RankRNGSnapshot, 'restore_current', lambda snapshot: case.restored.update({snapshot.rank:snapshot}))
    return case


def _config(case, name, *, output, resume=None):
    # Existing real profile fixture; no mocked compile or preflight.
    source = next(p for p in case.path.glob('*.yaml'))
    raw = yaml.safe_load(source.read_text())
    raw['launch'].update(output_dir=str(output), resume_from=None if resume is None else str(resume),
                         max_optimizer_steps=2, checkpoint_every_optimizer_steps=1)
    path = case.path / (name+'.yaml')
    path.write_text(yaml.safe_dump(raw))
    return path


def _run(case, path):
    controllers = [create_default_run_controller(code_root=case.assets.kwargs['code_root']) for _ in range(8)]
    case.saved = {}
    results = case.world.run(lambda r: controllers[r].run(path))
    return results, controllers


@pytest.mark.parametrize('stop_step', [1, 2])
def test_h3_default_controller_full_run_and_fresh_resume(tmp_path, monkeypatch, stop_step):
    case = _controller_case(tmp_path, monkeypatch)
    continuous = tmp_path / 'continuous'
    interrupted = tmp_path / 'interrupted'
    first = _config(case, 'continuous-config', output=continuous)
    results, controllers = _run(case, first)
    for result in results:
        if isinstance(result, Exception):
            raise result
    assert all(r.committed_steps == 2 for r in results)
    assert sorted(case.closed) == list(range(8))
    expected = case.saved.copy()
    marker = json.loads((continuous/'SUCCESS').read_text())
    assert marker['kind'] == 'visual_rl_h3_run_success' and marker['committed_steps'] == 2
    assert all(c.completed_stages[-1].value == 'checkpoint' for c in controllers)

    path = _config(case, 'interrupted-config', output=interrupted)
    original = h3_transaction.H3CheckpointTransaction.checkpoint
    def interrupt(self, *args, **kwargs):
        checkpoint = original(self, *args, **kwargs)
        if checkpoint.step == stop_step:
            raise RuntimeError('stop after durable checkpoint')
        return checkpoint
    with monkeypatch.context() as paused:
        paused.setattr(h3_transaction.H3CheckpointTransaction, 'checkpoint', interrupt)
        failed, _ = _run(case, path)
    assert all(isinstance(r, RuntimeError) and 'durable checkpoint' in str(r) for r in failed), failed
    assert not (interrupted/'SUCCESS').exists()
    checkpoint = interrupted / 'checkpoints' / f'step-{stop_step}'
    assert (checkpoint/'complete.json').is_file()
    resume = _config(case, 'resume-config', output=interrupted, resume=checkpoint)
    resumed, _ = _run(case, resume)
    assert not any(isinstance(r, Exception) for r in resumed), resumed
    assert set(case.restored) == set(range(8))
    for rank in range(8):
        _equal(case.saved[rank], expected[rank])
    assert (interrupted/'SUCCESS').is_file()
    assert all(r.committed_steps == 2 for r in resumed)


def test_h3_default_compile_rejects_invalid_profile_before_runtime(tmp_path, monkeypatch):
    path = tmp_path/'invalid.yaml'
    path.write_text('schema_version: 2\nrecipe: h3_t2va_mvp_v1\n')
    from visual_rl.errors import ConfigError
    def forbidden(*args):
        raise AssertionError('runtime entered before compile succeeded')
    monkeypatch.setattr(h3_session.H3RuntimeSession, 'create', forbidden)
    with pytest.raises(ConfigError):
        create_default_run_controller().run(path)


def test_h3_cli_uses_default_controller_and_returns_actual_checkpoint(tmp_path, monkeypatch):
    import io
    from visual_rl.train import main
    case = _controller_case(tmp_path, monkeypatch)
    path = _config(case, 'cli-config', output=tmp_path/'cli-output')
    raw = yaml.safe_load(path.read_text())
    raw['launch']['max_optimizer_steps'] = 1
    path.write_text(yaml.safe_dump(raw))
    outputs, errors = [io.StringIO() for _ in range(8)], [io.StringIO() for _ in range(8)]
    results = case.world.run(lambda rank: main([str(path)], stdout=outputs[rank], stderr=errors[rank]))
    assert results == [0]*8, results
    rows = [json.loads(stream.getvalue()) for stream in outputs]
    assert all(row['status'] == 'ok' and row['committed_steps'] == 1 for row in rows)
    assert len({row['run_id'] for row in rows}) == 1
    assert all(Path(row['authoritative_checkpoint']).joinpath('complete.json').is_file() for row in rows)
    assert all(stream.getvalue() == '' for stream in errors)


def test_h3_default_preflight_refuses_output_inside_input_artifact(tmp_path, monkeypatch):
    case = _controller_case(tmp_path, monkeypatch)
    path = _config(case, 'bad-output-config', output=case.assets.profile.launch.model/'run')
    from visual_rl.errors import ConfigError
    def forbidden(*args):
        raise AssertionError('runtime started for invalid output')
    monkeypatch.setattr(h3_session.H3RuntimeSession, 'create', forbidden)
    with pytest.raises(ConfigError, match='inside model/cache/source'):
        create_default_run_controller().run(path)
    assert not (case.assets.profile.launch.model/'run').exists()
