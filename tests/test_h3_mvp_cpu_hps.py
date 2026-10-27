"""H3 HPS adapter tests: real factory/client/JPEG/validators, HTTP leaf only."""

import base64
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import weakref

from PIL import Image, ImageStat
import pytest
import torch

from visual_rl.algorithms.rewards import h3_hps
from visual_rl.algorithms.rewards.clients import hpsv3
from visual_rl.algorithms.rewards.h3_hps import H3HPSReward
from visual_rl.algorithms.rewards.resource_descriptor import RewardResourceDescriptor, RewardRuntimePolicy
from visual_rl.composition.config.specs import RewardRuntimeBindingSpec
from visual_rl.composition.preflight.types import RuntimeFacts
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.filesystem_identity import filesystem_file_identity_from_snapshot
from visual_rl.core.protocols.flash_hpsv3 import (
    PROTOCOL, VERSION, MAX_REQUEST_BYTES, VideoContract, canonical_json, digest,
)
from visual_rl.models.implementations.h3_decode import H3AVMedia
from visual_rl.runtime.reward_resources import RewardResourceAcquireRequest


def _fp32(value):
    return struct.unpack('!f', struct.pack('!f', value))[0]


def _health(contract):
    facts = dict(model_sha256='1'*64, processor_sha256='2'*64, code_sha256='3'*64)
    identity = digest(dict(identity_facts=facts, video_contract=contract.payload()))
    return dict(status='ready', reward='hpsv3', protocol=PROTOCOL, protocol_version=VERSION,
                service_identity=identity, identity_facts=facts, video_contract=contract.payload(),
                result_device='cpu', result_dtype='fp32')


def _request(root, contract, *, manifest=None, binding_changes=None):
    root.mkdir(exist_ok=True, parents=True)
    health = _health(contract)
    raw = canonical_json(health if manifest is None else manifest)
    path = root / 'hpsv3-service.json'
    path.write_bytes(raw)
    descriptor = RewardResourceDescriptor(
        schema_version=1, factory_class='visual_rl.algorithms.rewards.clients.hpsv3:Hpsv3VideoResourceFactory',
        artifact_ref='h3-hps', protocol=PROTOCOL, protocol_version=VERSION,
        semantic_factory_config=FrozenMapping(dict(service_identity=health['service_identity'], video_contract=contract.payload())),
        allowed_runtime_policy=RewardRuntimePolicy(('cpu',), ('fp32',), ('remote',)),
    )
    binding = RewardRuntimeBindingSpec(**(dict(artifact_ref='h3-hps', execution_domain='remote', device='cpu',
        dtype='fp32', endpoint='http://127.0.0.1:18888', timeout_s=2.5, trusted_hosts=('127.0.0.1',),
        ca_bundle=None, max_response_bytes=65536) | (binding_changes or {})))
    request = RewardResourceAcquireRequest(
        reward_resource_spec_id='reward-resource-spec.v1:' + '4'*64, descriptor=descriptor,
        immutable_artifact_identity=FrozenMapping(filesystem_file_identity_from_snapshot(raw)),
        artifact_location=path.resolve(), runtime_facts=RuntimeFacts('single', 0, 0, 1, 'cpu', 'fp32', None),
        runtime_binding=binding,
    )
    return request, health


class _Response:
    def __init__(self, payload, *, status=200, content_type='application/json', raw=None):
        self.body = canonical_json(payload) if raw is None else raw
        self.status_code = status
        self.headers = {'Content-Type': content_type, 'Content-Length': str(len(self.body))}
        self.closed = False

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset+chunk_size]

    def close(self):
        self.closed = True


class _Transport:
    """No sockets. Fake service leaf scores the actual submitted JPEG pixels."""
    def __init__(self, health):
        self.health = health
        self.calls, self.responses, self.frame_scores = [], [], []
        self.close_count = 0
        self.score_mutation = None
        self.status = 200
        self.content_type = 'application/json'
        self.raw = None
        self.timeout = False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        assert kwargs['allow_redirects'] is False and kwargs['stream'] is True and kwargs['timeout'] == 2.5
        if self.timeout:
            raise TimeoutError('transport leaf timeout')
        if method == 'GET':
            assert url.endswith('/healthz')
            payload = self.health
        else:
            assert method == 'POST' and url.endswith('/score')
            body = json.loads(kwargs['data'])
            assert set(body) == {'protocol', 'protocol_version', 'service_identity', 'sample_id', 'prompt', 'jpeg_frames'}
            scores = []
            for encoded in body['jpeg_frames']:
                with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                    assert image.size == (self.health['video_contract']['width'], self.health['video_contract']['height'])
                    scores.append(_fp32(sum(ImageStat.Stat(image).mean) / (3*255)))
            self.frame_scores.append(scores)
            # Independent fake-service calculation, not a mocked validator.
            top = sorted(scores, reverse=True)[:37]
            payload = dict(protocol=PROTOCOL, protocol_version=VERSION, service_identity=body['service_identity'],
                           sample_id=body['sample_id'], frame_scores=scores, video_score=_fp32(sum(top)/37))
            if self.score_mutation is not None:
                payload = self.score_mutation(payload)
        response = _Response(payload, status=self.status, content_type=self.content_type, raw=self.raw)
        self.responses.append(response)
        return response

    def close(self):
        self.close_count += 1


def _scorer(root, monkeypatch, contract=None, **kwargs):
    contract = contract or VideoContract(124, 32, 32)
    request, health = _request(root, contract, **kwargs)
    transport = _Transport(health)
    monkeypatch.setattr(hpsv3, 'requests_session', lambda: transport)
    scorer = H3HPSReward(request, video_contract=contract)
    assert transport.calls == []  # construction/declaration is offline
    return scorer, transport, request


def _media(contract, offset=0.):
    values = .05 + .6 * (torch.arange(124) % 31) / 30 + offset
    video = values[:, None, None, None].expand(124, 3, contract.height, contract.width).contiguous()
    audio = torch.linspace(-2, 2, 165600*2).reshape(165600, 2)  # amplitude is not clipped/scored
    return H3AVMedia(video, audio, 24, 32000)


def test_h3_hps_real_factory_all124_jpeg_top37_and_immutable_identity(tmp_path, monkeypatch):
    assert Path(h3_hps.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/algorithms/rewards/h3_hps.py'
    for i, contract in enumerate((VideoContract(124, 32, 64), VideoContract(124, 64, 96))):
        scorer, transport, request = _scorer(tmp_path / str(i), monkeypatch, contract)
        with pytest.raises(RuntimeError, match='acquired'):
            scorer.score_one(sample_id='s0', prompt=' exact prompt\n', media=_media(contract))
        with pytest.raises(RuntimeError, match='not live-admitted'):
            _ = scorer.identity
        assert scorer.acquire() is scorer
        assert isinstance(scorer._client, hpsv3.Hpsv3VideoClient)
        identity = scorer.identity
        assert identity.profile == 'h3_hpsv3_video_only_v1' and identity.weight == 1 and identity.version == 1
        assert identity.video_contract.top_k == 37 and identity.service_identity == transport.health['service_identity']
        assert identity.endpoint_identity == digest(dict(origin='http://127.0.0.1:18888', observed=transport.health))
        with pytest.raises(FrozenInstanceError):
            identity.weight = 2
        scores = []
        for j, offset in enumerate((0., .1)):
            media = _media(contract, offset)
            vref, aref = weakref.ref(media.video), weakref.ref(media.audio)
            score = scorer.score_one(sample_id=f's{j}', prompt=' exact prompt\n', media=media)
            scores.append(score)
            assert type(score) is float
            values = transport.frame_scores[-1]
            assert len(values) == 124
            expected = _fp32(sum(sorted(values, reverse=True)[:37])/37)
            assert score == expected
            sent = json.loads(transport.calls[-1][2]['data'])
            assert sent['sample_id'] == f's{j}' and sent['prompt'] == ' exact prompt\n'
            assert len(sent['jpeg_frames']) == 124 and 'audio' not in sent
            assert len(set(sent['jpeg_frames'])) > 1
            media._release()
            assert vref() is None and aref() is None
        assert scores[1] > scores[0]
        assert [call[0] for call in transport.calls] == ['GET', 'POST', 'POST']
        assert all(response.closed for response in transport.responses)
        assert scorer.identity == identity
        with pytest.raises(RuntimeError, match='already attempted'):
            scorer.acquire()
        scorer.close(); scorer.close()
        assert transport.close_count == 1
        with pytest.raises(RuntimeError, match='acquired and open'):
            scorer.score_one(sample_id='s', prompt='p', media=_media(contract))


def test_h3_hps_health_artifact_legacy_contract_and_binding_rejections(tmp_path, monkeypatch):
    contract = VideoContract(124, 32, 32)
    request, health = _request(tmp_path / 'base', contract)
    attempts = []
    monkeypatch.setattr(hpsv3, 'requests_session', lambda: attempts.append(True))
    with pytest.raises(ValueError, match='resource factory'):
        H3HPSReward(replace(request, descriptor=replace(request.descriptor, factory_class='mock')), video_contract=contract)
    disallowed = RewardRuntimePolicy(('cpu',), ('bf16',), ('remote',))
    with pytest.raises(ValueError, match='runtime policy'):
        H3HPSReward(replace(request, descriptor=replace(request.descriptor, allowed_runtime_policy=disallowed)), video_contract=contract)
    with pytest.raises(ValueError, match='124 frames'):
        H3HPSReward(request, video_contract=VideoContract(81, 32, 32))
    with pytest.raises(ValueError, match='invalid video frames'):
        VideoContract(141, 32, 32)
    old, _ = _request(tmp_path / 'legacy', VideoContract(81, 32, 32))
    with pytest.raises(ValueError, match='contract mismatch'):
        H3HPSReward(old, video_contract=contract)
    for i, changed in enumerate((health | {'status': 'not-ready'}, health | {'service_identity': 'f'*64},
                                 health | {'video_contract': VideoContract(81, 32, 32).payload()})):
        bad, _ = _request(tmp_path / f'bad{i}', contract, manifest=changed)
        with pytest.raises(ValueError):
            H3HPSReward(bad, video_contract=contract)
    for i, changes in enumerate((dict(endpoint='http://example.com:18888'), dict(trusted_hosts=('localhost',)),
                                 dict(device='cuda:0'), dict(dtype='bf16'))):
        bad, _ = _request(tmp_path / f'binding{i}', contract, binding_changes=changes)
        scorer = H3HPSReward(bad, video_contract=contract)
        with pytest.raises(ValueError):
            scorer.acquire()
    assert attempts == []
    for i, changed in enumerate((health | {'status': 'not-ready'}, health | {'service_identity': 'f'*64},
                                 health | {'video_contract': VideoContract(81, 32, 32).payload()})):
        scorer, transport, _ = _scorer(tmp_path / f'live{i}', monkeypatch)
        transport.health = changed
        with pytest.raises(ValueError):
            scorer.acquire()
        assert len(transport.calls) == 1 and transport.close_count == 1
        with pytest.raises(RuntimeError, match='already attempted'):
            scorer.acquire()


def test_h3_hps_rejects_invalid_response_identity_frames_topk_and_nonfinite(tmp_path, monkeypatch):
    scorer, transport, _ = _scorer(tmp_path, monkeypatch)
    scorer.acquire()
    media = _media(VideoContract(124, 32, 32))
    changes = [lambda p: p | {'sample_id': 'wrong'}, lambda p: p | {'service_identity': 'f'*64},
               lambda p: p | {'frame_scores': p['frame_scores'][:-1]},
               lambda p: p | {'frame_scores': [True] * 124},
               lambda p: p | {'frame_scores': [1e100] * 124},
               lambda p: p | {'video_score': p['video_score'] + .1},
               lambda p: p | {'selected_frame_indices': [123] * 37}]
    for change in changes:
        transport.score_mutation = change
        before = len(transport.calls)
        with pytest.raises(ValueError):
            scorer.score_one(sample_id='sample', prompt='text', media=media)
        assert len(transport.calls) == before+1 and transport.responses[-1].closed
    transport.score_mutation = None
    transport.raw = b'{"frame_scores":[NaN]}'
    with pytest.raises(ValueError, match='non-finite'):
        scorer.score_one(sample_id='sample', prompt='text', media=media)
    scorer.close()


def test_h3_hps_transport_bounds_http_failures_and_no_post_retry(tmp_path, monkeypatch):
    scorer, transport, _ = _scorer(tmp_path, monkeypatch)
    scorer.acquire()
    media = _media(VideoContract(124, 32, 32))
    for status in (302, 503):
        transport.status = status
        before = len(transport.calls)
        with pytest.raises(RuntimeError, match=f'HTTP {status}.*no automatic POST retry'):
            scorer.score_one(sample_id='s', prompt='p', media=media)
        assert len(transport.calls) == before+1 and transport.responses[-1].closed
    transport.status = 200
    transport.content_type = 'text/plain'
    with pytest.raises(ValueError, match='must be JSON'):
        scorer.score_one(sample_id='s', prompt='p', media=media)
    transport.content_type = 'application/json'
    transport.raw = b' ' * 65537
    with pytest.raises(RuntimeError, match='exceeds max_response_bytes'):
        scorer.score_one(sample_id='s', prompt='p', media=media)
    assert transport.responses[-1].closed
    transport.raw = None
    transport.timeout = True
    before = len(transport.calls)
    with pytest.raises(TimeoutError):
        scorer.score_one(sample_id='s', prompt='p', media=media)
    assert len(transport.calls) == before+1
    transport.timeout = False
    before = len(transport.calls)
    # Exercise the unchanged client's actual 64MiB request ceiling without
    # allocating a multi-GB decoded video or mocking JPEG/validation logic.
    oversized = 'A' * (MAX_REQUEST_BYTES // 124 + 1)
    with pytest.raises(ValueError, match='request exceeds bounded transport size'):
        scorer._client.score_video(sample_id='s', prompt='p', frames_base64=[oversized] * 124)
    assert len(transport.calls) == before
    scorer.close()


def test_h3_hps_av_validation_expired_lease_and_manifest_drift(tmp_path, monkeypatch):
    scorer, transport, request = _scorer(tmp_path, monkeypatch)
    scorer.acquire()
    contract = VideoContract(124, 32, 32)
    media = _media(contract)
    metadata = media.metadata
    for bad in (SimpleNamespace(video=media.video[:81], audio=media.audio, metadata=metadata),
                SimpleNamespace(video=media.video.half(), audio=media.audio, metadata=metadata),
                SimpleNamespace(video=media.video, audio=media.audio[:, :1], metadata=metadata),
                SimpleNamespace(video=media.video, audio=media.audio.clone().requires_grad_(), metadata=metadata),
                SimpleNamespace(video=media.video, audio=torch.full_like(media.audio, float('inf')), metadata=metadata),
                SimpleNamespace(video=torch.full_like(media.video, 1.1), audio=media.audio, metadata=metadata),
                SimpleNamespace(video=torch.full_like(media.video, float('nan')), audio=media.audio, metadata=metadata)):
        with pytest.raises(ValueError):
            scorer.score_one(sample_id='s', prompt='p', media=bad)
    for change in (dict(sample_rate=16000), dict(fps=25), dict(sample_count=165599), dict(frame_count=81)):
        bad = SimpleNamespace(video=media.video, audio=media.audio, metadata=replace(metadata, **change))
        with pytest.raises(ValueError, match='metadata mismatch'):
            scorer.score_one(sample_id='s', prompt='p', media=bad)
    media._release()
    with pytest.raises(RuntimeError, match='expired'):
        scorer.score_one(sample_id='s', prompt='p', media=media)
    assert len(transport.calls) == 1  # health only; all media errors before POST
    with request.artifact_location.open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='manifest changed'):
        scorer.score_one(sample_id='s', prompt='p', media=_media(contract))
    assert len(transport.calls) == 1
    scorer.close()


def test_h3_hps_local_manifest_preflight_is_bounded_and_no_implicit_acquire(tmp_path, monkeypatch):
    contract = VideoContract(124, 32, 32)
    request, _ = _request(tmp_path, contract)
    attempts = []
    monkeypatch.setattr(hpsv3, 'requests_session', lambda: attempts.append(True))
    location = request.artifact_location
    raw = location.read_bytes()
    location.unlink()
    with pytest.raises(ValueError, match='existing local manifest'):
        H3HPSReward(request, video_contract=contract)
    other = tmp_path / 'other.json'
    other.write_bytes(raw)
    location.symlink_to(other)
    with pytest.raises(ValueError, match='symlink'):
        H3HPSReward(request, video_contract=contract)
    location.unlink()
    location.write_bytes(b' ' * (h3_hps.MAX_HEALTH_BYTES + 1))
    with pytest.raises(ValueError, match='manifest too large'):
        H3HPSReward(request, video_contract=contract)
    location.write_bytes(raw)
    scorer = H3HPSReward(request, video_contract=contract)
    location.write_bytes(raw + b' ')
    with pytest.raises(ValueError, match='manifest changed'):
        scorer.acquire()
    scorer.close(); scorer.close()
    assert attempts == []
