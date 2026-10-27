"""Real session control, only CUDA/PG observations and communication are leaves.

No actual CUDA call, process group, socket or service is created by these tests.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.test_h3_mvp_cpu_profile import _fixture, _fail_heavy
from visual_rl.composition.config.h3_profile import compile_h3_profile, preflight_h3_profile
from visual_rl.runtime import h3_session
from visual_rl.runtime.h3_session import H3RuntimeSession
from visual_rl.algorithms.optimization import h3_world


def _profile(tmp_path, monkeypatch):
    path, _ = _fixture(tmp_path)
    _fail_heavy(monkeypatch)
    profile = compile_h3_profile(path)
    facts = preflight_h3_profile(profile)
    def no_optimizer(*args, **kwargs):
        pytest.fail('session must never construct an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', no_optimizer)
    return profile, facts


def _environment(monkeypatch, profile, rank):
    for key, value in dict(WORLD_SIZE='8', LOCAL_WORLD_SIZE='8', RANK=str(rank), LOCAL_RANK=str(rank),
                           GROUP_WORLD_SIZE='1', GROUP_RANK='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29555',
                           CUDA_VISIBLE_DEVICES=','.join(profile.launch.gpu_uuids)).items():
        monkeypatch.setenv(key, value)


class _Leaves:
    def __init__(self, monkeypatch, profile, rank=0):
        _environment(monkeypatch, profile, rank)
        self.rank = rank
        self.world_size, self.backend = 8, 'nccl'
        self.initialized = False
        self.world = SimpleNamespace(WORLD=None)
        self.owned = object()
        self.current = 7
        self.count = 8
        self.properties = [SimpleNamespace(uuid=u if i % 2 == 0 else u[4:])
                           for i, u in enumerate(profile.launch.gpu_uuids)]
        self.events = []
        self.init_error = self.gather_error = None
        self.post_init = self.mutate_peer = None
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(torch.cuda, 'device_count', lambda: self.count)
        monkeypatch.setattr(torch.cuda, 'get_device_properties', self.device_properties)
        monkeypatch.setattr(torch.cuda, 'set_device', self.set_device)
        monkeypatch.setattr(torch.cuda, 'current_device', lambda: self.current)
        monkeypatch.setattr(torch.cuda, 'is_bf16_supported', self.bf16)
        monkeypatch.setattr(dist, 'is_available', lambda: True)
        monkeypatch.setattr(dist, 'is_nccl_available', lambda: True)
        monkeypatch.setattr(dist, 'is_initialized', lambda: self.initialized)
        monkeypatch.setattr(dist, 'group', self.world)
        monkeypatch.setattr(dist, 'init_process_group', self.init)
        monkeypatch.setattr(dist, 'destroy_process_group', self.destroy)
        monkeypatch.setattr(dist, 'get_backend', lambda group: self.backend)
        monkeypatch.setattr(dist, 'get_world_size', lambda group: self.world_size)
        monkeypatch.setattr(dist, 'get_rank', lambda group: self.rank)
        monkeypatch.setattr(dist, 'all_gather_object', self.gather)
        def forbidden(*args, **kwargs):
            pytest.fail('no barrier, subgroup, mesh or rescue collective is allowed')
        monkeypatch.setattr(dist, 'barrier', forbidden)
        monkeypatch.setattr(dist, 'new_group', forbidden)

    def device_properties(self, index):
        self.events.append(('uuid', index))
        return self.properties[index]

    def set_device(self, device):
        assert device == torch.device('cuda', self.rank)
        self.events.append(('select', device))
        self.current = device.index

    def bf16(self, *, including_emulation):
        assert including_emulation is False and self.current == self.rank
        self.events.append(('bf16', self.current))
        return True

    def init(self, **kwargs):
        self.events.append(('init', kwargs))
        assert kwargs == dict(backend='nccl', init_method='env://', timeout=timedelta(seconds=180))
        assert self.current == self.rank and ('bf16', self.rank) in self.events
        if self.init_error is not None:
            raise self.init_error
        self.initialized = True
        self.world.WORLD = self.owned
        if self.post_init:
            self.post_init()

    def gather(self, output, encoded, *, group):
        self.events.append(('gather', group))
        assert group is self.owned and len(output) == 8
        assert isinstance(encoded, bytes) and len(encoded) < 2048
        if self.gather_error is not None:
            raise self.gather_error
        original = json.loads(encoded)
        assert original[:3] == [1, 'session/admission', self.rank]
        for rank in range(8):
            peer = deepcopy(original)
            peer[2] = rank
            if peer[4] is not None:
                peer[4][2] = rank
            if self.mutate_peer:
                peer = self.mutate_peer(rank, peer)
            output[rank] = peer if isinstance(peer, bytes) else json.dumps(peer).encode()

    def destroy(self, group):
        assert group is self.owned and self.world.WORLD is self.owned
        self.events.append(('destroy', group))
        self.initialized = False
        self.world.WORLD = None

    def count_event(self, name):
        return sum(event[0] == name for event in self.events)


def test_h3_session_eight_rank_observations_public_api_order_and_owned_close(tmp_path, monkeypatch):
    assert Path(h3_session.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/runtime/h3_session.py'
    profile, facts = _profile(tmp_path, monkeypatch)
    rng = torch.random.get_rng_state().clone()
    for rank in range(8):
        leaves = _Leaves(monkeypatch, profile, rank)
        environment = dict(os.environ)
        session = H3RuntimeSession.create(profile, facts)
        assert session.process_group is leaves.owned
        assert session.rank == session.local_rank == rank and session.local_world_size == 8
        assert session.device == torch.device('cuda', rank)
        assert session.gpu_uuids == profile.launch.gpu_uuids
        assert leaves.events[:8] == [('uuid', i) for i in range(8)]
        assert leaves.count_event('init') == leaves.count_event('gather') == 1
        assert leaves.count_event('destroy') == 0
        for attr in ('process_group', 'rank', 'local_rank', 'local_world_size', 'device', 'gpu_uuids'):
            with pytest.raises(AttributeError):
                setattr(session, attr, None)
        with pytest.raises(RuntimeError, match='foreign process group'):
            H3RuntimeSession.create(profile, facts)
        assert leaves.count_event('init') == 1 and leaves.count_event('destroy') == 0
        session.close(); session.close()
        assert leaves.count_event('destroy') == 1 and not leaves.initialized
        assert dict(os.environ) == environment
        assert torch.equal(torch.random.get_rng_state(), rng)
        with pytest.raises(RuntimeError, match='closed'):
            _ = session.process_group
    with pytest.raises(TypeError, match='use H3RuntimeSession.create'):
        H3RuntimeSession()
    with pytest.raises(TypeError):
        H3RuntimeSession.create(profile, facts, verified=True)


def test_h3_session_profile_facts_and_torchrun_env_fail_before_cuda_or_pg(tmp_path, monkeypatch):
    profile, facts = _profile(tmp_path, monkeypatch)
    leaves = _Leaves(monkeypatch, profile)
    for bad_profile, bad_facts in (
        (replace(profile, launch=replace(profile.launch, output_dir=tmp_path/'different')), facts),
        (replace(profile, training=replace(profile.training, group_size=2)), facts),
        (profile, replace(facts, semantic_id='h3.semantic.v1:' + '0'*64)),
        (profile, replace(facts, binding_id='other')), (None, facts),
    ):
        with pytest.raises((ValueError, TypeError)):
            H3RuntimeSession.create(bad_profile, bad_facts)
    from visual_rl.core.immutable import FrozenMapping
    from visual_rl.core.identity import canonical_identity
    payload = FrozenMapping(dict(facts.semantic_payload) | {'data': dict(
        train_sha256='f'*64, eval_sha256=facts.eval_sha256, parse='h3.utf8-lf-crlf.v1')})
    inconsistent = replace(facts, semantic_payload=payload, semantic_id=canonical_identity('h3.semantic.v1', payload))
    with pytest.raises(ValueError, match='do not describe'):
        H3RuntimeSession.create(profile, inconsistent)
    assert leaves.events == []
    for key, value in (('RANK', None), ('LOCAL_RANK', '1'), ('RANK', '-1'), ('RANK', '8'),
                       ('WORLD_SIZE', '16'), ('LOCAL_WORLD_SIZE', '4'), ('GROUP_WORLD_SIZE', '2'),
                       ('GROUP_RANK', '1'), ('LOCAL_RANK', 'False'), ('MASTER_ADDR', None),
                       ('MASTER_PORT', '0'), ('RANK', '00')):
        _environment(monkeypatch, profile, 0)
        if value is None:
            monkeypatch.delenv(key)
        else:
            monkeypatch.setenv(key, value)
        with pytest.raises(ValueError):
            H3RuntimeSession.create(profile, facts)
    assert leaves.events == []


def test_h3_session_uuid_request_and_actual_mapping_admission(tmp_path, monkeypatch):
    profile, facts = _profile(tmp_path, monkeypatch)
    for visible in ('', '0,1,2,3,4,5,6,7', ','.join(reversed(profile.launch.gpu_uuids)),
                    ','.join(profile.launch.gpu_uuids[:7]), ','.join([profile.launch.gpu_uuids[0]]*8),
                    'MIG-' + ','.join(profile.launch.gpu_uuids)):
        leaves = _Leaves(monkeypatch, profile)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
        with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
            H3RuntimeSession.create(profile, facts)
        assert leaves.events == []
    for defect in ('ten-visible', 'wrong-order', 'duplicate', 'missing-uuid', 'partial-uuid'):
        leaves = _Leaves(monkeypatch, profile)
        if defect == 'ten-visible':
            leaves.count = 10
        elif defect == 'wrong-order':
            leaves.properties.reverse()
        elif defect == 'duplicate':
            leaves.properties[1] = leaves.properties[0]
        elif defect == 'missing-uuid':
            leaves.properties[1] = SimpleNamespace(name='CMP170HX', total_memory=64*2**30)
        else:
            leaves.properties[1] = SimpleNamespace(uuid='GPU-1234')
        with pytest.raises((ValueError, RuntimeError), match='UUID|eight requested'):
            H3RuntimeSession.create(profile, facts)
        assert leaves.count_event('init') == leaves.count_event('destroy') == 0


def test_h3_session_cuda_nccl_bf16_and_foreign_group_fail_closed(tmp_path, monkeypatch):
    profile, facts = _profile(tmp_path, monkeypatch)
    for defect in ('no-cuda', 'no-dist', 'no-nccl', 'no-bf16', 'old-bf16-api', 'foreign'):
        leaves = _Leaves(monkeypatch, profile)
        if defect == 'no-cuda':
            monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        elif defect == 'no-dist':
            monkeypatch.setattr(dist, 'is_available', lambda: False)
        elif defect == 'no-nccl':
            monkeypatch.setattr(dist, 'is_nccl_available', lambda: False)
        elif defect == 'no-bf16':
            monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda **kw: False)
        elif defect == 'old-bf16-api':
            monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
        else:
            leaves.initialized = True
            leaves.world.WORLD = object()
        with pytest.raises(RuntimeError):
            H3RuntimeSession.create(profile, facts)
        assert leaves.count_event('init') == leaves.count_event('destroy') == 0
        if defect == 'foreign':
            assert leaves.count_event('select') == 0 and leaves.initialized


def test_h3_session_bounded_peer_consensus_and_cooperative_rejection(tmp_path, monkeypatch):
    profile, facts = _profile(tmp_path, monkeypatch)
    for defect in ('host', 'rank', 'uuid', 'facts', 'local-world', 'peer-error', 'oversized'):
        leaves = _Leaves(monkeypatch, profile, rank=2)
        def mutate(rank, peer, defect=defect):
            if rank != 3:
                return peer
            if defect == 'host':
                peer[3] = 'foreign-host'
            elif defect == 'rank':
                peer[2] = 0
            elif defect == 'uuid':
                peer[4][1].reverse()
            elif defect == 'facts':
                peer[4][0][2] = 'different-binding'
            elif defect == 'local-world':
                peer[4][3] = 4
            elif defect == 'peer-error':
                peer[5] = False
            else:
                return b'x' * (h3_world.MAX_RECEIPT_BYTES + 1)
            return peer
        leaves.mutate_peer = mutate
        with pytest.raises(ValueError):
            H3RuntimeSession.create(profile, facts)
        assert leaves.count_event('init') == leaves.count_event('gather') == leaves.count_event('destroy') == 1
        assert not leaves.initialized
    leaves = _Leaves(monkeypatch, profile, rank=2)
    leaves.post_init = lambda: setattr(leaves, 'current', 5)
    with pytest.raises(ValueError, match='rejected local input') as failure:
        H3RuntimeSession.create(profile, facts)
    assert isinstance(failure.value.__cause__, ValueError)
    assert 'changed during initialization' in str(failure.value.__cause__)
    assert leaves.count_event('gather') == leaves.count_event('destroy') == 1


def test_h3_session_post_init_failures_transport_timeout_and_no_rescue_retry(tmp_path, monkeypatch):
    profile, facts = _profile(tmp_path, monkeypatch)
    for name, value in (('backend', 'gloo'), ('world_size', 7), ('rank', 4)):
        leaves = _Leaves(monkeypatch, profile)
        leaves.post_init = lambda name=name, value=value: setattr(leaves, name, value)
        with pytest.raises(ValueError, match='initialized group'):
            H3RuntimeSession.create(profile, facts)
        assert leaves.count_event('init') == leaves.count_event('destroy') == 1
        assert leaves.count_event('gather') == 0
    leaves = _Leaves(monkeypatch, profile)
    leaves.init_error = TimeoutError('rendezvous timed out')
    with pytest.raises(TimeoutError, match='rendezvous'):
        H3RuntimeSession.create(profile, facts)
    assert leaves.count_event('init') == 1 and leaves.count_event('gather') == leaves.count_event('destroy') == 0
    leaves = _Leaves(monkeypatch, profile)
    leaves.gather_error = TimeoutError('NCCL transport failed')
    with pytest.raises(TimeoutError, match='NCCL'):
        H3RuntimeSession.create(profile, facts)
    assert leaves.count_event('init') == leaves.count_event('gather') == leaves.count_event('destroy') == 1
    leaves = _Leaves(monkeypatch, profile)
    leaves.gather_error = TimeoutError('primary NCCL failure')
    def cleanup_failure(group):
        leaves.events.append(('destroy', group))
        raise RuntimeError('cleanup failure')
    monkeypatch.setattr(dist, 'destroy_process_group', cleanup_failure)
    with pytest.raises(TimeoutError, match='primary NCCL failure') as failure:
        H3RuntimeSession.create(profile, facts)
    assert any('cleanup also failed' in note for note in failure.value.__notes__)
    assert leaves.count_event('gather') == leaves.count_event('destroy') == 1


def test_h3_session_close_does_not_destroy_replaced_world_and_cleanup_is_once(tmp_path, monkeypatch):
    profile, facts = _profile(tmp_path, monkeypatch)
    leaves = _Leaves(monkeypatch, profile)
    session = H3RuntimeSession.create(profile, facts)
    replacement = object()
    leaves.world.WORLD = replacement
    session.close(); session.close()
    assert leaves.initialized and leaves.world.WORLD is replacement and leaves.count_event('destroy') == 0
    leaves = _Leaves(monkeypatch, profile)
    session = H3RuntimeSession.create(profile, facts)
    def failing_destroy(group):
        assert group is leaves.owned
        leaves.events.append(('destroy', group))
        raise RuntimeError('owned PG teardown failed')
    monkeypatch.setattr(dist, 'destroy_process_group', failing_destroy)
    with pytest.raises(RuntimeError, match='teardown'):
        session.close()
    session.close()
    assert leaves.count_event('destroy') == 1
