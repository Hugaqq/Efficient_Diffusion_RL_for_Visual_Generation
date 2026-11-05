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
from visual_rl.core.protocols.flash_hpsv3 import VideoContract, digest, scoring_identity

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
    pool = Hpsv3MultiOriginClient(origins=ORIGINS, expected_scoring_identity=scoring_identity(health, contract),
                                transport_factory=transport, **options)
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


def _resident_health(contract, uuid):
    health = _health(contract)
    health['identity_facts']['gpu_residency_policy'] = dict(
        version='resident.v1', enabled=True, expected_gpu_uuid=uuid, ttl=60)
    return _rehash(health)


def _rehash(health):
    health['service_identity'] = digest(dict(identity_facts=health['identity_facts'],
                                            video_contract=health['video_contract']))
    return health


def test_reported_health_structure_accepts_distinct_gpu_uuids():
    from copy import deepcopy
    contract = VideoContract(124, 256, 384)
    health = _health(contract)
    health['identity_facts'] = dict(
        checkpoint_file_sha256='1'*64, config_file_sha256='2'*64,
        loaded_model_state_sha256='3'*64, loaded_processor_sha256='4'*64,
        hps_package_source_sha256='5'*64, service_source_sha256='6'*64,
        wire_source_sha256='7'*64, torch_version='test', transformers_version='test',
        use_special_tokens=True,
        data_config=dict(confidence_threshold=.95, max_pixels=200704, min_pixels=200704,
                         soft_label=False, test_json_list=[['Valid Set 1', ['example_set_1_part1.json', 'example_set_1_part2.json']],
                                                          ['Valid Set 2', ['example_set_2_part1.json']]],
                         tied_threshold=None, train_json_list=['example_train.json'],
                         val_json_list=['/path/to/dataset/meta_data.json'], with_instruction=True),
        gpu_residency_policy=dict(version='hps.dedicated-idle-residency.v1', enabled=True,
                                  expected_gpu_uuid='GPU-test-a', idle_ttl_seconds=60.0),
        request_queue_policy=dict(version='hps.bounded-fifo-queue.v1', full='503-retry-after-before-scoring',
                                  max_queue=8, order='admission', scoring_concurrency=1))
    _rehash(health)
    other = deepcopy(health)
    other['identity_facts']['gpu_residency_policy']['expected_gpu_uuid'] = 'GPU-test-b'
    _rehash(other)
    def make_pool(second):
        return Hpsv3MultiOriginClient(origins=ORIGINS, expected_scoring_identity=scoring_identity(health, contract),
            transport_factory=lambda origin: _Transport(health if origin == ORIGINS[0] else second),
            service_identity=health['service_identity'], contract=contract, timeout_s=2.5, max_response_bytes=65536)
    pool = make_pool(other)
    try:
        pool.healthcheck()
        assert pool.clients[0].service_identity != pool.clients[1].service_identity
    finally:
        pool.close()
    other['identity_facts']['request_queue_policy']['max_queue'] = 9
    _rehash(other)
    pool = make_pool(other)
    try:
        with pytest.raises(ValueError):
            pool.healthcheck()
    finally:
        pool.close()


def test_placement_independent_pool_pins_each_endpoint_and_run_identity(tmp_path, monkeypatch):
    from visual_rl.algorithms.rewards.clients import hpsv3
    from visual_rl.algorithms.rewards.h3_hps import H3HPSReward
    from visual_rl.core.protocols.flash_hpsv3 import canonical_json
    contract = VideoContract(124, 32, 32)
    healths = [_resident_health(contract, uuid) for uuid in ('GPU-a', 'GPU-b')]
    original = canonical_json(healths)
    assert scoring_identity(healths[0], contract) == scoring_identity(healths[1], contract)
    assert canonical_json(healths) == original  # 不原地删 UUID
    request, _ = _request(tmp_path, contract, manifest=healths[0])
    request = replace(request, descriptor=replace(request.descriptor, semantic_factory_config=FrozenMapping(
        dict(service_identity=healths[0]['service_identity'], video_contract=contract.payload(), origins=list(ORIGINS)))))
    calls = []
    class Transport(_Transport):
        def request(self, method, url, **kwargs):
            self.health = healths[ORIGINS.index(url.rsplit('/', 1)[0])]
            if method == 'POST':
                assert json.loads(kwargs['data'])['service_identity'] == self.health['service_identity']
                calls.append(url)
            return super().request(method, url, **kwargs)
    monkeypatch.setattr(hpsv3, 'requests_session', lambda: Transport(healths[0]))
    owner = H3HPSReward(request, video_contract=contract).acquire()
    identity = owner.identity_payload()
    assert 'service_identity' not in identity and 'manifest_sha256' not in identity
    assert identity['scoring_identity'] == scoring_identity(healths[0], contract)
    pool = owner._client
    frames = jpeg_frames(torch.zeros(124,3,32,32), 'BFCHW', contract)
    args = dict(sample_id='s', prompt='p', frames_base64=frames)
    assert pool.score_video(**args) == pool.score_video(**args)
    assert calls == [o+'/score' for o in ORIGINS]
    # 错端点的完整响应 identity 即使 scoring identity 相同也必须拒绝。
    original_request = Transport.request
    def wrong_score(self, method, url, **kwargs):
        self.score_mutation = lambda p: dict(p, service_identity=healths[1]['service_identity'])
        return original_request(self, method, url, **kwargs)
    monkeypatch.setattr(Transport, 'request', wrong_score)
    with pytest.raises(ValueError, match='identity/order'):
        pool.score_video(**args)
    owner.close()
    monkeypatch.setattr(Transport, 'request', original_request)
    # artifact 与线上卡 UUID 都换了，运行身份仍相同。
    for i, health in enumerate(healths):
        health['identity_facts']['gpu_residency_policy']['expected_gpu_uuid'] = f'GPU-new-{i}'
        _rehash(health)
    request.artifact_location.write_bytes(canonical_json(healths[0]))
    request = replace(request, descriptor=replace(request.descriptor, semantic_factory_config=FrozenMapping(
        dict(service_identity=healths[0]['service_identity'], video_contract=contract.payload(), origins=list(ORIGINS)))))
    owner = H3HPSReward(request, video_contract=contract).acquire()
    assert owner.identity_payload() == identity
    owner.close()


def test_scoring_identity_rejects_every_other_leaf():
    from copy import deepcopy
    contract = VideoContract(124, 32, 32)
    health = _resident_health(contract, 'GPU-a')
    expected = scoring_identity(health, contract)
    # 遍历全部其余叶字段：包括 residency 的 version/enabled/TTL，协议与精度。
    def paths(value, prefix=()):
        for key, item in value.items():
            if isinstance(item, dict):
                yield from paths(item, (*prefix, key))
            else:
                yield (*prefix, key)
    for path in paths(health):
        if path in [('service_identity',), ('identity_facts','gpu_residency_policy','expected_gpu_uuid')]:
            continue
        changed = deepcopy(health)
        target = changed
        for key in path[:-1]:
            target = target[key]
        old = target[path[-1]]
        target[path[-1]] = not old if type(old) is bool else old+1 if type(old) is int else old+'-changed'
        _rehash(changed)
        def transport(origin):
            return _Transport(health if origin == ORIGINS[0] else changed)
        pool = Hpsv3MultiOriginClient(origins=ORIGINS, expected_scoring_identity=expected,
            transport_factory=transport, service_identity=health['service_identity'], contract=contract,
            timeout_s=2.5, max_response_bytes=65536)
        try:
            with pytest.raises(ValueError):
                pool.healthcheck()
        finally:
            pool.close()


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


def test_dispatch_shared_across_eight_real_cpu_processes(tmp_path):
    import subprocess
    import sys
    path = tmp_path/'dispatch.json'
    HpsDispatcher.initialize(path, ORIGINS)
    script = '''
import json, sys
from visual_rl.algorithms.rewards.clients.hps_dispatch import HpsDispatcher
with HpsDispatcher(json.loads(sys.argv[2]), path=sys.argv[1]).lease() as index:
    print(index, flush=True)
    sys.stdin.readline()
'''
    processes = [subprocess.Popen([sys.executable, '-c', script, str(path), json.dumps(ORIGINS)],
                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(8)]
    try:
        # All children hold their leases until the parent releases them together.
        import selectors
        selector = selectors.DefaultSelector()
        for process in processes:
            selector.register(process.stdout, selectors.EVENT_READ)
        selected = []
        while selector.get_map():
            events = selector.select(timeout=15)
            assert events, 'dispatcher child did not acquire a lease'
            for key,_ in events:
                selected.append(int(key.fileobj.readline()))
                selector.unregister(key.fileobj)
        selector.close()
        assert selected.count(0) == selected.count(1) == 4
        assert json.loads(path.read_text())['counts'] == [4,4]
        for process in processes:
            process.stdin.write('\n'); process.stdin.flush()
        for process in processes:
            out,err = process.communicate(timeout=15)
            assert process.returncode == 0, err
        assert json.loads(path.read_text())['counts'] == [0,0]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=15)
