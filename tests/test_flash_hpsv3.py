"""CPU unit/transport tests. Fake backend here is NOT real reward evidence."""
import base64
from dataclasses import replace
from io import BytesIO
import threading
from http.server import HTTPServer

import pytest
import torch
from PIL import Image

from visual_rl.core.protocols.flash_hpsv3 import (
    PROTOCOL, VERSION, VideoContract, aggregate_frame_scores, canonical_json,
    decode_json, digest, scalar_fp32, validate_health, validate_score,
)
from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient, jpeg_frames, loopback_origin
from scripts.serve_hpsv3_video import make_handler, loaded_tensor_sha
from scripts.prepare_flash_workload import build_schedule


class UnitBackend:
    contract = VideoContract(3, 2, 3)
    def __init__(self):
        facts = {'unit_test_fake_backend': True}
        self.health = {'status': 'ready', 'reward': 'hpsv3', 'protocol': PROTOCOL,
            'protocol_version': VERSION, 'identity_facts': facts,
            'video_contract': self.contract.payload(),
            'service_identity': digest({'identity_facts': facts, 'video_contract': self.contract.payload()}),
            'result_device': 'cpu', 'result_dtype': 'fp32'}
        self.calls = 0
    def score(self, sample_id, prompt, images):
        assert len(images) == 3
        self.calls += 1
        return [1., 4., 2.]


@pytest.fixture
def server():
    backend = UnitBackend()
    http = HTTPServer(('127.0.0.1', 0), make_handler(backend))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield backend, f'http://127.0.0.1:{http.server_port}'
    finally:
        http.shutdown(); http.server_close(); thread.join(3)


def client_for(backend, origin):
    return Hpsv3VideoClient(origin=origin, contract=backend.contract,
        service_identity=backend.health['service_identity'], timeout_s=3, max_response_bytes=262144)


def test_all81_top24_author_mean():
    score, indices = aggregate_frame_scores(list(range(81)), VideoContract(81, 480, 832))
    assert score == 68.5 and indices == tuple(range(80, 56, -1))


@pytest.mark.parametrize('bad', [[1., 2.], [True, 1., 2.], [float('nan'), 1., 2.], [float('inf'), 1., 2.], [1e300, 1., 2.]])
def test_invalid_frame_scores_rejected(bad):
    with pytest.raises(ValueError):
        aggregate_frame_scores(bad, VideoContract(3, 2, 3))


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        decode_json(raw)


@pytest.mark.parametrize('url', ['http://example.com:18820', 'http://127.0.0.1:18820/?token=x', 'http://user:pw@localhost:18820', 'http://127.0.0.1:18820/x', 'http://localhost'])
def test_noncanonical_or_external_origins_rejected(url):
    with pytest.raises(ValueError):
        loopback_origin(url)


def test_jpeg_matches_author_and_uint8_path():
    contract = VideoContract(3, 2, 3)
    video = torch.linspace(-.1, 1.1, 54).reshape(3, 3, 2, 3)
    encoded = jpeg_frames(video, 'BFCHW', contract)
    u8 = (video * 255).round().clamp(0, 255).to(torch.uint8)
    assert encoded == jpeg_frames(u8, 'BFCHW', contract)
    assert encoded == jpeg_frames(u8.permute(0, 2, 3, 1), 'BFHWC', contract)
    for idx, frame in enumerate(u8.permute(0, 2, 3, 1).numpy()):
        output = BytesIO(); Image.fromarray(frame).save(output, format='JPEG')
        assert base64.b64decode(encoded[idx]) == output.getvalue()


def test_live_json_transport_all_frames(server):
    backend, origin = server
    client = client_for(backend, origin)
    try:
        assert client.healthcheck() == backend.health
        frames = jpeg_frames(torch.zeros(3, 3, 2, 3), 'BFCHW', backend.contract)
        value, metadata = client.score_video(sample_id='unit-only', prompt='red cube', frames_base64=frames)
        assert value == 4 and metadata['scored_frame_indices'] == [0, 1, 2]
        assert metadata['selected_frame_indices'] == [1] and backend.calls == 1
        with pytest.raises(ValueError):
            client.score_video(sample_id='unit-only', prompt='x', frames_base64=frames[:1])
        assert backend.calls == 1
    finally:
        client.close()
    with pytest.raises(RuntimeError):
        client.healthcheck()


def test_identity_and_aggregate_drift():
    b = UnitBackend()
    with pytest.raises(ValueError):
        validate_health(b.health, '0'*64, b.contract)
    bad = dict(b.health); bad['video_contract'] = VideoContract(81, 480, 832).payload()
    with pytest.raises(ValueError):
        validate_health(bad, b.health['service_identity'], b.contract)
    packet = {'protocol': PROTOCOL, 'protocol_version': VERSION, 'sample_id': 's',
        'service_identity': b.health['service_identity'], 'frame_scores': [1., 4., 2.], 'video_score': 3.}
    with pytest.raises(ValueError):
        validate_score(packet, sample_id='s', identity=b.health['service_identity'], contract=b.contract)
    packet['video_score'] = 4.
    with pytest.raises(ValueError):
        validate_score(packet, sample_id='other', identity=b.health['service_identity'], contract=b.contract)


def test_default_factory_plugin_acquires_live_identity(server, tmp_path):
    from visual_rl.core.immutable import FrozenMapping
    from visual_rl.algorithms.rewards.resource_descriptor import RewardResourceDescriptor, RewardRuntimePolicy
    from visual_rl.runtime.reward_resources import DefaultRewardResourceFactory, RewardResourceAcquireRequest
    from visual_rl.composition.preflight.types import RuntimeFacts
    from visual_rl.composition.config.specs import RewardRuntimeBindingSpec
    backend, origin = server
    artifact = tmp_path/'hpsv3-service.json'; artifact.write_bytes(canonical_json(backend.health))
    descriptor = RewardResourceDescriptor(schema_version=1,
        factory_class='visual_rl.algorithms.rewards.clients.hpsv3:Hpsv3VideoResourceFactory',
        artifact_ref='hpsv3-video', protocol=PROTOCOL, protocol_version=VERSION,
        semantic_factory_config=FrozenMapping({'service_identity': backend.health['service_identity'], 'video_contract': backend.contract.payload()}),
        allowed_runtime_policy=RewardRuntimePolicy(('cpu',), ('fp32',), ('remote',)))
    request = RewardResourceAcquireRequest(reward_resource_spec_id='reward-resource-spec.v1:'+'a'*64,
        descriptor=descriptor, immutable_artifact_identity=FrozenMapping({'unit_test': True}),
        artifact_location=artifact,
        runtime_facts=RuntimeFacts('single', 0, 0, 1, 'cpu', 'fp32', None),
        runtime_binding=RewardRuntimeBindingSpec(artifact_ref='hpsv3-video', execution_domain='remote',
            device='cpu', dtype='fp32', endpoint=origin, timeout_s=3, trusted_hosts=('127.0.0.1',), max_response_bytes=262144))
    acquired = DefaultRewardResourceFactory()(request)
    try:
        assert acquired.binding_facts.protocol == PROTOCOL
        assert acquired.binding_facts.worker_domain == 'remote'
        assert acquired.resource.healthcheck() == backend.health
    finally:
        acquired.resource.close()
    # Stale materialized model identity must fail, not become a made-up attestation.
    artifact.write_bytes(canonical_json(dict(backend.health, service_identity='0'*64)))
    with pytest.raises(ValueError):
        DefaultRewardResourceFactory()(request)


def test_loaded_weight_hash_is_actual_and_supports_bf16_scalar_buffers():
    model = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
    model.register_buffer('scalar', torch.tensor(1., dtype=torch.bfloat16))
    before = loaded_tensor_sha(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert loaded_tensor_sha(model) != before


def test_explicit_complete_resource_replacement_does_not_leak_world_identity():
    from visual_rl.composition.recipes.builtins import get_recipe_definition, apply_recipe_overrides
    from visual_rl.errors import ConfigError
    base = get_recipe_definition('flash_grpo_v1')
    resource = {'unit_test_complete_resource': True}
    changed = apply_recipe_overrides(base, {'rewards': {'reward_general': {
        'replace_params': True, 'params': {'resource': resource}}}})
    assert dict(changed.rewards[0].component.params['resource']) == resource
    assert 'server_revision_expectation' not in changed.rewards[0].component.params['resource']
    for invalid in [1, 'yes', None]:
        with pytest.raises(ConfigError):
            apply_recipe_overrides(base, {'model': {'replace_params': invalid, 'params': {}}})
    with pytest.raises(ConfigError):
        apply_recipe_overrides(base, {'model': {'replace_params': True}})


def test140_quantity_and_original_eval_cadence():
    result = build_schedule(140)
    assert result['training_videos_target'] == 26880
    assert result['evaluation_videos_target'] == 1200
    assert result['logical_hps_frame_scores_target'] == 2274480
    commits = [n for row in result['collections'] for n in row['expected_commit_numbers']]
    assert commits == list(range(1, 141))
    assert result['collections'][-1]['source_epoch'] == 71
    assert [r['after_optimizer_commits'] for r in result['evaluations']] == [0, 36, 76, 116]
    assert not result['sampling_plan_materialized']
    with pytest.raises(ValueError):
        build_schedule(139)
