from dataclasses import replace
import json
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from tests.test_h3_mvp_cpu_hps import _request, _health, _Transport
from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient, Hpsv3MultiOriginClient, Hpsv3VideoResourceFactory, jpeg_frames
from visual_rl.algorithms.rewards.clients.hps_dispatch import HpsDispatcher
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.protocols.flash_hpsv3 import VideoContract

ORIGINS = ('http://127.0.0.1:18888', 'http://127.0.0.1:18889')


def test_multi_origin_scores_identity_and_failure_no_fallback(tmp_path):
    contract = VideoContract(124, 32, 32)
    health = _health(contract)
    frames = jpeg_frames(torch.linspace(0,1,124)[:,None,None,None].expand(124,3,32,32), 'BFCHW', contract)
    options = dict(service_identity=health['service_identity'], contract=contract, timeout_s=2.5, max_response_bytes=65536)
    transports = []
    fail = [None]
    def transport(origin):
        item = _Transport(health)
        if origin == fail[0]:
            item.timeout = True
        transports.append((origin,item))
        return item
    single = Hpsv3VideoClient(origin=ORIGINS[0], transport=transport(ORIGINS[0]), **options)
    pool = Hpsv3MultiOriginClient(origins=ORIGINS, transport_factory=transport, **options)
    assert pool.healthcheck() == health
    assert sum(len(t.calls) for _,t in transports) == 2
    args = dict(sample_id='sample', prompt='prompt', frames_base64=frames)
    expected = single.score_video(**args)
    assert pool.score_video(**args) == pool.score_video(**args) == expected
    assert [origin for origin,t in transports if t.calls and t.calls[0][0]=='POST'] == [ORIGINS[0],*ORIGINS]
    fail[0] = ORIGINS[0]
    with pytest.raises(TimeoutError):
        pool.score_video(**args)
    count = len(transports)
    with pytest.raises(RuntimeError, match='poisoned'):
        pool.score_video(**args)
    assert len(transports) == count  # never tries the healthy instance
    pool.close(); single.close()
    assert all(t.close_count == 1 for _,t in transports)


def test_all_origins_live_admitted_and_single_factory_unchanged(tmp_path, monkeypatch):
    from visual_rl.algorithms.rewards.clients import hpsv3
    contract = VideoContract(124,32,32)
    request, health = _request(tmp_path, contract)
    made = []
    def transport():
        t = _Transport(health); made.append(t); return t
    monkeypatch.setattr(hpsv3, 'requests_session', transport)
    single = Hpsv3VideoResourceFactory()(request)
    before = single.binding_facts.endpoint_identity
    assert type(single.resource) is Hpsv3VideoClient and len(made) == 1
    single.resource.close()
    semantic = dict(service_identity=health['service_identity'], video_contract=contract.payload(), origins=list(ORIGINS))
    multi_request = replace(request, descriptor=replace(request.descriptor, semantic_factory_config=FrozenMapping(semantic)))
    pool = Hpsv3VideoResourceFactory()(multi_request)
    assert isinstance(pool.resource, Hpsv3MultiOriginClient)
    assert pool.binding_facts.endpoint_identity != before
    assert [t.calls[0][1] for t in made[1:]] == [o+'/healthz' for o in ORIGINS]
    pool.resource.close()
    def bad_transport():
        t = transport()
        if len(made) % 2:
            t.health = dict(health, service_identity='f'*64)
        return t
    monkeypatch.setattr(hpsv3, 'requests_session', bad_transport)
    with pytest.raises(ValueError):
        Hpsv3VideoResourceFactory()(multi_request)
    assert all(t.close_count == 1 for t in made)


def test_cross_rank_least_inflight_file_leases_and_fail_closed(tmp_path):
    path = tmp_path/'dispatch.json'
    HpsDispatcher.initialize(path, ORIGINS)
    owners = [HpsDispatcher(ORIGINS, path=path) for _ in range(8)]
    barrier = threading.Barrier(8)
    def work(rank):
        with owners[rank].lease() as index:
            barrier.wait(timeout=5)
            counts = json.loads(path.read_text())['counts']
            # All 8 leases have been granted; a separate barrier prevents release.
            barrier.wait(timeout=5)
            return index, counts
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(work, range(8)))
    assert [index for index,_ in results].count(0) == 4
    assert all(counts == [4,4] for _,counts in results)
    assert json.loads(path.read_text())['counts'] == [0,0]
    with pytest.raises(RuntimeError, match='server failure'):
        with owners[0].lease():
            raise RuntimeError('server failure')
    for owner in owners:
        with pytest.raises(RuntimeError, match='poisoned'):
            with owner.lease():
                pytest.fail('failed instance was silently dropped')
    # A new attempt cannot reset the state beneath existing owners.
    HpsDispatcher.initialize(path, ORIGINS)
    with pytest.raises(ValueError, match='generation'):
        with owners[0].lease():
            pass
