"""Closed H3 profile compile/local preflight, no weight construction or service."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest
import torch
import yaml
from safetensors.torch import save_file, load_file

from tests.test_h3_mvp_cpu_decode import _artifacts
from tests.test_h3_mvp_cpu_hps import _health
from tests.test_h3_mvp_cpu_text_cache import _identity, _metadata, _key
from visual_rl.composition.config import h3_profile
from visual_rl.composition.config.h3_profile import compile_h3_profile, preflight_h3_profile
from visual_rl.composition.config.compiler import compile_recipe_v2
from visual_rl.composition.config.source import load_source_recipe
from visual_rl.composition.config.errors import ConfigSourceError
from visual_rl.composition.registry import RegistryError
from visual_rl.core.protocols.flash_hpsv3 import VideoContract, canonical_json


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _save(root, raw):
    path = root / 'profile.yaml'
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    model = root / 'model'
    _artifacts(model)
    transformer = model / 'transformer'
    transformer.mkdir()
    (transformer / 'config.json').write_text(json.dumps(dict(_class_name='MiniMaxH3Transformer3DModel',
        in_channels=24, audio_in_channels=32, patch_size=[1, 2, 2], text_dim=5120, num_layers=1, hidden_size=4)))
    targets = ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0', 'ff.net.0.proj', 'ff.net.2')
    tensors = {f'transformer_blocks.0.{target}.weight': torch.arange(16).reshape(4, 4).float() / 16 for target in targets}
    save_file(tensors, str(transformer / 'diffusion_pytorch_model.safetensors'))
    for component in ('vae', 'audio_vae'):
        save_file({'decoder.weight': torch.ones(2, 2)}, str(model / component / 'diffusion_pytorch_model.safetensors'))
    # Fixture's declared model hash comes from actual tiny file bytes. Production
    # preflight explicitly does NOT claim to verify the full model-weight hash.
    model_sha = _sha(b''.join(p.read_bytes() for p in sorted(model.rglob('*')) if p.is_file()))
    cache = root / 'cache'
    cache.mkdir()
    identity = _identity() | {'model_sha256': model_sha}
    cache_tensors = {_key(prompt): torch.arange(length * 5120).reshape(length, 5120).float() / 10000
                     for prompt, length in (('cat', 2), (' cat', 3), ('eval', 1))}
    save_file(cache_tensors, str(cache / 'embeddings.safetensors'), metadata=_metadata(identity))
    (cache / 'index.json').write_text(json.dumps(dict(schema_version='visual_rl.h3-text-cache.v1', identity=identity,
        shards={'embeddings.safetensors': _sha((cache / 'embeddings.safetensors').read_bytes())},
        entries={key: dict(shard='embeddings.safetensors', tensor=key, tokens=t.shape[0]) for key, t in cache_tensors.items()})))
    (root / 'train.txt').write_bytes(('cat\n cat\r\n' * 4).encode())
    (root / 'eval.txt').write_bytes(b'eval\n')
    health = _health(VideoContract(124, 256, 384))
    (root / 'hps.json').write_bytes(canonical_json(health))
    raw = dict(schema_version=2, recipe='h3_t2va_mvp_v1', overrides={}, launch=dict(
        output_dir='not-created/run', resume_from=None, resume_policy='complete_collection_same_topology',
        artifacts=dict(model='model', text_cache='cache', train='train.txt', eval='eval.txt', hps_manifest='hps.json'),
        expected=dict(model_sha256=model_sha, train_sha256=_sha((root/'train.txt').read_bytes()),
                      eval_sha256=_sha((root/'eval.txt').read_bytes()), cache_identity=identity,
                      hps_service_identity=health['service_identity']),
        # Synthetic fixture UUIDs are CPU schema inputs, NOT device authorization.
        execution=dict(world_size=8, sp=1, tp=1,
            gpu_uuids=['GPU-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, f'h3-fixture-{i}')) for i in range(8)]),
        seed=42, max_optimizer_steps=2, checkpoint_every_optimizer_steps=1,
        budgets=dict(host_weight_bytes=10**8, trajectory_bytes=10**9, media_bytes=10**9, decode_bytes=10**9,
                     condition_bytes=10**7, max_tokens=8, embedding_bytes=10**6, prompt_file_bytes=10**6),
        hps=dict(origin='http://127.0.0.1:18888', timeout_s=10., max_response_bytes=65536)))
    return _save(root, raw), raw


def _fail_heavy(monkeypatch):
    from visual_rl.models.implementations import h3_loading, h3_decode
    from visual_rl.algorithms.rewards.clients import hpsv3
    def fail(*args, **kwargs):
        pytest.fail('profile compile/preflight must not load heavy models, acquire HTTP, or initialize PG')
    monkeypatch.setattr(h3_loading, 'build_h3_meta', fail)
    monkeypatch.setattr(h3_loading, '_resolve_h3_class', fail)
    monkeypatch.setattr(h3_decode, '_resolve_vae_class', fail)
    monkeypatch.setattr(hpsv3, 'requests_session', fail)
    monkeypatch.setattr(hpsv3.Hpsv3VideoClient, 'healthcheck', fail)
    monkeypatch.setattr(torch.distributed, 'init_process_group', fail)


def test_h3_profile_real_compile_preflight_default_math_and_no_side_effects(tmp_path, monkeypatch):
    assert Path(h3_profile.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/composition/config/h3_profile.py'
    path, raw = _fixture(tmp_path)
    _fail_heavy(monkeypatch)
    profile = compile_h3_profile(path)
    assert profile.model.alias == 'h3-t2va'
    assert profile.model.implementation_class_path == 'visual_rl.models.implementations.h3:H3Adapter'
    assert profile.model.config.geometry.latent_shape == (24, 37, 16, 24)
    assert profile.launch.model == tmp_path / 'model'
    assert not profile.launch.output_dir.exists()
    math = profile.training.payload()
    assert (math['group_size'], math['num_steps'], math['world_size'], math['sp'], math['tp']) == (8, 10, 8, 1, 1)
    assert math['video_shift'] == 12 and math['audio_shift'] == 3 and math['eta'] == .7
    assert math['score'] == 'cps_proxy_negative_mse' and math['av_weighting'] == 'generated_element_mean'
    assert math['action_dtype'] == 'fp16' and math['score_dtype'] == 'fp32'
    assert math['old_policy'] == 'frozen_replay' and math['policy_steps'] == 'all_including_terminal_q0'
    assert (math['lr'], math['betas'], math['eps'], math['weight_decay'], math['clip_range'], math['max_grad_norm']) == (1e-4, (.9, .999), 1e-12, 0, .005, 1)
    assert math['reward'] == 'h3_hpsv3_video_only_v1' and not math['audio_quality_scored']
    facts = preflight_h3_profile(profile)
    assert facts.semantic_id.startswith('h3.semantic.v1:') and facts.binding_id == profile.launch.binding_id
    assert facts.train_sha256 == raw['launch']['expected']['train_sha256']
    assert facts.eval_sha256 == raw['launch']['expected']['eval_sha256']
    assert facts.cache_identity == _sha((tmp_path / 'cache/index.json').read_bytes())
    assert facts.model_weights_verified is False
    assert facts.minimum_trajectory_bytes == 8*10*(4*354240+4) + 8*2*354240
    assert facts.minimum_condition_bytes == 8*3*5120*4
    assert len(facts.semantic_payload['video_sigmas']) == 11
    assert facts.semantic_payload['video_sigmas'][-1] == facts.semantic_payload['audio_sigmas'][-1] == 0
    assert facts.semantic_payload['video_sigmas'][5] != facts.semantic_payload['audio_sigmas'][5]
    assert not profile.launch.output_dir.exists()
    with pytest.raises(FrozenInstanceError):
        profile.training.num_steps = 1
    with pytest.raises(TypeError):
        facts.semantic_payload['seed'] = 4


def test_h3_profile_content_semantics_are_path_budget_endpoint_and_uuid_independent(tmp_path):
    path, raw = _fixture(tmp_path / 'original')
    baseline = preflight_h3_profile(compile_h3_profile(path))
    shutil.copytree(path.parent, tmp_path / 'moved')
    moved = preflight_h3_profile(compile_h3_profile(tmp_path / 'moved/profile.yaml'))
    assert moved.semantic_id == baseline.semantic_id and moved.binding_id != baseline.binding_id
    changed = deepcopy(raw)
    changed['launch']['output_dir'] = 'different-output'
    changed['launch']['hps']['origin'] = 'http://localhost:19999'
    changed['launch']['budgets']['embedding_bytes'] *= 2
    changed['launch']['execution']['gpu_uuids'].reverse()
    changed['launch']['checkpoint_every_optimizer_steps'] = 2
    rebinding = preflight_h3_profile(compile_h3_profile(_save(path.parent, changed)))
    assert rebinding.semantic_id == baseline.semantic_id and rebinding.binding_id != baseline.binding_id
    for change in ({'group_size': 2}, {'num_steps': 2}):
        new = deepcopy(raw)
        new['overrides']['training'] = change
        assert preflight_h3_profile(compile_h3_profile(_save(path.parent, new))).semantic_id != baseline.semantic_id
    new = deepcopy(raw)
    new['launch']['seed'] = 9
    assert preflight_h3_profile(compile_h3_profile(_save(path.parent, new))).semantic_id != baseline.semantic_id
    # Exact prompt bytes, even just ordering of existing covered strings, matter.
    (path.parent / 'train.txt').write_text(' cat\ncat\n' * 4)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        preflight_h3_profile(compile_h3_profile(_save(path.parent, raw)))
    new = deepcopy(raw)
    new['launch']['expected']['train_sha256'] = _sha((path.parent / 'train.txt').read_bytes())
    assert preflight_h3_profile(compile_h3_profile(_save(path.parent, new))).semantic_id != baseline.semantic_id


def test_h3_profile_rejects_unknown_duplicate_and_unsupported_math(tmp_path):
    path, raw = _fixture(tmp_path)
    for group, change in (('training', {'eta': 0}), ('training', {'video_shift': 11}),
                          ('training', {'group_size': True}), ('training', {'num_steps': 0}),
                          ('model', {'class': 'evil:Factory'}), ('model', {'params': {'guidance_scale': 4.5}}),
                          ('model', {'params': {'frames': 141}}), ('model', {'params': {'height': 31}})):
        new = deepcopy(raw); new['overrides'][group] = change
        with pytest.raises((ValueError, TypeError, RegistryError)):
            compile_h3_profile(_save(tmp_path, new))
    new = deepcopy(raw); new['schema_version'] = True
    with pytest.raises(ValueError):
        compile_h3_profile(_save(tmp_path, new))
    new = deepcopy(raw); new['launch']['unknown'] = 4
    with pytest.raises(ValueError, match='unknown'):
        compile_h3_profile(_save(tmp_path, new))
    for key in ('world_size', 'sp', 'tp'):
        new = deepcopy(raw); new['launch']['execution'][key] = 4
        with pytest.raises(ValueError, match='WORLD8'):
            compile_h3_profile(_save(tmp_path, new))
    for values in ([], list(range(8)), ['GPU-00000000-0000-0000-0000-000000000000']*8,
                   raw['launch']['execution']['gpu_uuids'][:7], [raw['launch']['execution']['gpu_uuids'][0]]*8):
        new = deepcopy(raw); new['launch']['execution']['gpu_uuids'] = values
        with pytest.raises(ValueError, match='UUID'):
            compile_h3_profile(_save(tmp_path, new))
    for value in (float('nan'), float('inf'), True, 0):
        new = deepcopy(raw); new['launch']['hps']['timeout_s'] = value
        with pytest.raises((ValueError, ConfigSourceError), match='finite'):
            compile_h3_profile(_save(tmp_path, new))
    _save(tmp_path, raw)
    path.write_text(path.read_text() + '\nschema_version: 2\n')
    with pytest.raises(ConfigSourceError) as error:
        compile_h3_profile(path)
    assert error.value.code == 'config.duplicate_key'
    # Real source-loader path, including a nested duplicate.
    _save(tmp_path, raw)
    path.write_text(path.read_text().replace('seed: 42', 'seed: 42\n  seed: 43'))
    with pytest.raises(ConfigSourceError) as error:
        compile_h3_profile(path)
    assert error.value.code == 'config.duplicate_key'


def test_h3_profile_local_missing_coverage_hps_and_content_drift(tmp_path):
    path, raw = _fixture(tmp_path)
    baseline = preflight_h3_profile(compile_h3_profile(path))
    for key, content in (('train', b'cat\n'), ('eval', b''), ('eval', b'not-cached\n'), ('eval', b'\xff\n'), ('eval', b'\n'), ('eval', b'eval\r')):
        file = tmp_path / raw['launch']['artifacts'][key]
        old = file.read_bytes(); file.write_bytes(content)
        new = deepcopy(raw); new['launch']['expected'][f'{key}_sha256'] = _sha(content)
        with pytest.raises((ValueError, UnicodeError)):
            preflight_h3_profile(compile_h3_profile(_save(tmp_path, new)))
        file.write_bytes(old)
    new = deepcopy(raw); new['launch']['expected']['cache_identity']['target_layer'] = 49
    with pytest.raises(ValueError, match='layer50'):
        compile_h3_profile(_save(tmp_path, new))
    _save(tmp_path, raw)
    manifest = tmp_path / 'hps.json'
    old_manifest = manifest.read_bytes()
    manifest.write_bytes(canonical_json(_health(VideoContract(81, 256, 384))))
    with pytest.raises(ValueError, match='contract mismatch'):
        preflight_h3_profile(compile_h3_profile(path))
    health = _health(VideoContract(124, 256, 384))
    health['identity_facts']['code_sha256'] = '9' * 64
    from visual_rl.core.protocols.flash_hpsv3 import digest
    health['service_identity'] = digest(dict(identity_facts=health['identity_facts'], video_contract=health['video_contract']))
    manifest.write_bytes(canonical_json(health))
    new = deepcopy(raw); new['launch']['expected']['hps_service_identity'] = health['service_identity']
    assert preflight_h3_profile(compile_h3_profile(_save(tmp_path, new))).semantic_id != baseline.semantic_id
    manifest.write_bytes(old_manifest); _save(tmp_path, raw)
    weights = tmp_path / 'model/vae/diffusion_pytorch_model.safetensors'
    saved = weights.read_bytes(); weights.unlink()
    with pytest.raises(ValueError, match='safetensors'):
        preflight_h3_profile(compile_h3_profile(path))
    weights.write_bytes(saved)
    shard = tmp_path / 'cache/embeddings.safetensors'
    saved_shard = shard.read_bytes()
    corrupted = bytearray(saved_shard); corrupted[-1] ^= 1; shard.write_bytes(corrupted)
    with pytest.raises(ValueError, match='content/hash changed'):
        preflight_h3_profile(compile_h3_profile(path))
    shard.write_bytes(saved_shard)
    updated = load_file(str(shard))
    updated[_key('cat')].add_(.125)
    save_file(updated, str(shard), metadata=_metadata(raw['launch']['expected']['cache_identity']))
    index_path = tmp_path / 'cache/index.json'
    index = json.loads(index_path.read_text())
    index['shards'][shard.name] = _sha(shard.read_bytes())
    index_path.write_text(json.dumps(index))
    assert preflight_h3_profile(compile_h3_profile(path)).semantic_id != baseline.semantic_id


def test_h3_profile_capacity_resume_constraints_and_pending_template(tmp_path):
    path, raw = _fixture(tmp_path)
    for key in raw['launch']['budgets']:
        new = deepcopy(raw); new['launch']['budgets'][key] = True
        with pytest.raises(ValueError, match='positive integer'):
            compile_h3_profile(_save(tmp_path, new))
    for key in ('host_weight_bytes', 'trajectory_bytes', 'condition_bytes', 'media_bytes', 'decode_bytes'):
        new = deepcopy(raw); new['launch']['budgets'][key] = 1
        with pytest.raises(ValueError, match='budget'):
            preflight_h3_profile(compile_h3_profile(_save(tmp_path, new)))
    for key, value in (('max_optimizer_steps', True), ('max_optimizer_steps', 0),
                       ('checkpoint_every_optimizer_steps', 3), ('resume_policy', 'elastic'), ('resume_policy', 'mid_collection')):
        new = deepcopy(raw); new['launch'][key] = value
        with pytest.raises(ValueError):
            compile_h3_profile(_save(tmp_path, new))
    new = deepcopy(raw); new['launch']['resume_from'] = 'missing-checkpoint'
    with pytest.raises(ValueError, match='checkpoint directory'):
        preflight_h3_profile(compile_h3_profile(_save(tmp_path, new)))
    # Directory existence is not a completed checkpoint attestation; actual
    # manager/progress/topology restore admission remains runtime responsibility.
    (tmp_path / 'checkpoint').mkdir()
    new['launch']['resume_from'] = 'checkpoint'
    facts = preflight_h3_profile(compile_h3_profile(_save(tmp_path, new)))
    assert facts.model_weights_verified is False
    sample = Path(__file__).resolve().parents[1] / 'configs/v2/h3_t2va_mvp.yaml'
    with pytest.raises((TypeError, ValueError)):
        compile_h3_profile(sample)  # null identities/budget and [] UUIDs are intentionally not runnable


def test_h3_profile_existing_sd3_wan_flash_static_compilers_unchanged():
    root = Path(__file__).resolve().parents[1]
    for name, model, definition in (('flow_grpo_sd3.yaml', 'sd3', 'flow_grpo_v1'),
                                     ('flow_grpo_wan.yaml', 'wan-t2v', 'flow_grpo_v1'),
                                     ('flash_wan.yaml', 'wan-t2v', 'flash_grpo_v1')):
        source = load_source_recipe(root / 'configs/v2' / name)
        first = compile_recipe_v2(source)
        second = compile_recipe_v2(source)
        assert first.definition_id == definition and first.model.declaration.alias == model
        assert first.resolved_fingerprint == second.resolved_fingerprint
        assert first.model.declaration.declared_contract.model.declares_scheduler_binding


_CROSS_TRAIN = (' cat ', '猫', 'cat\t', 'dog', ' ', 'é', 'e\u0301', 'tail  ')
_CROSS_EVAL = ('  测试  ', '猫 ')


def _profile_source_fixture(root, monkeypatch):
    from tests.test_h3_mvp_cpu import _tiny_loading_class
    from visual_rl.models.implementations import h3_loading
    from visual_rl.models.implementations.h3 import H3Adapter
    from visual_rl.models.lifecycle.components import ComponentManager

    _, raw = _fixture(root)
    raw['overrides'] = dict(model=dict(params=dict(height=32, width=32, lora_rank=2, lora_alpha=4)),
                            training=dict(group_size=2, num_steps=2))
    health = _health(VideoContract(124, 32, 32))
    (root / 'hps.json').write_bytes(canonical_json(health))
    raw['launch']['expected']['hps_service_identity'] = health['service_identity']
    # Cache even the BOM-prefixed prompt: BOM rejection must be the common
    # parser's decision, not an incidental cache miss masking the old mismatch.
    texts = (*_CROSS_TRAIN, *_CROSS_EVAL, '\ufeff' + _CROSS_TRAIN[0])
    tensors = {_key(text): torch.full((2, 5120), (i+1)/100) for i, text in enumerate(texts)}
    cache = root / 'cache'
    identity = raw['launch']['expected']['cache_identity']
    shard = cache / 'cross.safetensors'
    save_file(tensors, str(shard), metadata=_metadata(identity))
    (cache / 'index.json').write_text(json.dumps(dict(schema_version='visual_rl.h3-text-cache.v1', identity=identity,
        shards={shard.name: _sha(shard.read_bytes())}, entries={key: dict(shard=shard.name, tensor=key, tokens=2)
                                                            for key in tensors})))
    _cross_files(root, raw, ('\n'.join(_CROSS_TRAIN) + '\n').encode(), '\r\n'.join(_CROSS_EVAL).encode())
    profile = compile_h3_profile(root / 'profile.yaml')
    monkeypatch.setattr(h3_loading, '_resolve_h3_class', lambda: _tiny_loading_class())
    adapter = H3Adapter.from_config(profile.model.config, runtime_context=dict(
        artifact_paths={'h3-model': profile.launch.model, 'h3-text': profile.launch.text_cache},
        expected_cache_identity=profile.expected['cache_identity']))
    manager = ComponentManager(adapter, execution_device='cpu')
    manager.load()  # Real meta/PEFT, only heavy class resolution replaced.
    return raw, adapter, manager


def _cross_files(root, raw, train, evaluation):
    for name, content in (('train', train), ('eval', evaluation)):
        (root / f'{name}.txt').write_bytes(content)
        raw['launch']['expected'][f'{name}_sha256'] = _sha(content)
    return _save(root, raw)


def _source_from_profile(profile, adapter, *, rank=1):
    from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
    from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
    from visual_rl.data.h3_source import H3PromptSource

    k, n = profile.training.group_size, profile.training.num_steps
    return H3PromptSource(train_path=profile.launch.train, eval_path=profile.launch.eval,
        train_sha256=profile.expected['train_sha256'], eval_sha256=profile.expected['eval_sha256'],
        adapter=adapter, dynamics=H3CPSDynamics(adapter.geometry.joint_layout, H3ScheduleSnapshot.build(n)),
        rank=rank, world_size=8, group_size=k, base_seed=profile.launch.seed, device='cpu',
        max_samples=k, max_cells=k*n, max_trajectory_bytes=profile.launch.budgets.trajectory_bytes)


def test_h3_profile_source_exact_text_versions_and_real_reserve_commit(tmp_path, monkeypatch):
    from visual_rl.data import h3_source

    raw, adapter, manager = _profile_source_fixture(tmp_path, monkeypatch)
    try:
        # These spies run AFTER the one explicit tiny adapter load. Neither
        # preflight nor the real source constructor/reserve may load again.
        with monkeypatch.context() as scoped:
            _fail_heavy(scoped)
            for separator, final in (('\n', ''), ('\r\n', '\r\n'), ('\n', '\n')):
                train = (separator.join(_CROSS_TRAIN) + final).encode('utf-8')
                evaluation = (separator.join(_CROSS_EVAL) + final).encode('utf-8')
                profile = compile_h3_profile(_cross_files(tmp_path, raw, train, evaluation))
                facts = preflight_h3_profile(profile)
                source = _source_from_profile(profile, adapter)
                assert source._train == _CROSS_TRAIN and source._eval == _CROSS_EVAL
                state = source.state_dict()
                assert facts.semantic_payload['data']['parse'] == state['binding']['parse'] == h3_source.PARSE_VERSION
                assert facts.semantic_payload['math']['sampling'] == state['binding']['sampling'] == h3_source.SAMPLING_VERSION
                assert facts.semantic_payload['math']['random_streams'] == state['binding']['stream'] == h3_source.STREAM_VERSION
                assert facts.semantic_payload['math']['initial_state_version'] == state['binding']['geometry']['initial_state_version'] == adapter.initial_state_version
                assert facts.train_sha256 == _sha(train) and facts.eval_sha256 == _sha(evaluation)
                request = source.reserve(0)
                assert list(request.prompts.values()) == ['猫'] * profile.training.group_size
                assert all(condition.shape == (1, 2, 5120) for condition in request.conditions.values())
                assert source.read_behavior_id() == request.behavior_id
                source.commit(request)
                assert source.state_dict()['next_optimizer_step'] == 1
                assert source.state_dict()['global_cursor'] == 8
                assert source.read_behavior_id() == request.next_behavior_id
    finally:
        adapter.close(); manager.close()


def test_h3_profile_source_reject_same_bom_cr_empty_and_encoding_files(tmp_path, monkeypatch):
    raw, adapter, manager = _profile_source_fixture(tmp_path, monkeypatch)
    train = ('\n'.join(_CROSS_TRAIN) + '\n').encode()
    evaluation = '\n'.join(_CROSS_EVAL).encode()
    try:
        with monkeypatch.context() as scoped:
            _fail_heavy(scoped)
            for bad_train, bad_eval, message in (
                (b'\xef\xbb\xbf' + train, evaluation, 'BOM'),
                (train, b'\xef\xbb\xbf' + evaluation, 'BOM'),
                (train.replace(b'\n', b'\r', 1), evaluation, 'lone CR'),
                (train, evaluation + b'\r', 'lone CR'),
                (train + b'\n', evaluation, 'empty prompt'),
                (train, b'', 'empty prompt'),
            ):
                profile = compile_h3_profile(_cross_files(tmp_path, raw, bad_train, bad_eval))
                with pytest.raises(ValueError, match=message):
                    preflight_h3_profile(profile)
                with pytest.raises(ValueError, match=message):
                    _source_from_profile(profile, adapter)
            profile = compile_h3_profile(_cross_files(tmp_path, raw, train, b'\xff\n'))
            with pytest.raises(UnicodeDecodeError):
                preflight_h3_profile(profile)
            with pytest.raises(UnicodeDecodeError):
                _source_from_profile(profile, adapter)
    finally:
        adapter.close(); manager.close()


def test_h3_profile_source_hard_file_cap_and_smaller_explicit_budget(tmp_path, monkeypatch):
    from visual_rl.data import h3_source

    raw, adapter, manager = _profile_source_fixture(tmp_path, monkeypatch)
    train = (tmp_path / 'train.txt').read_bytes()
    evaluation = (tmp_path / 'eval.txt').read_bytes()
    assert h3_source._FILE_LIMIT == 16 * 1024 * 1024
    try:
        with monkeypatch.context() as scoped:
            _fail_heavy(scoped)
            raw['launch']['budgets']['prompt_file_bytes'] = 2 * h3_source._FILE_LIMIT
            # Real >16MiB input and true SHA, no patched global cap or hash.
            oversized = b'cat\n' * (h3_source._FILE_LIMIT // 4 + 1)
            for big_train, big_eval in ((oversized, evaluation), (train, oversized)):
                profile = compile_h3_profile(_cross_files(tmp_path, raw, big_train, big_eval))
                with pytest.raises(ValueError, match='byte budget'):
                    preflight_h3_profile(profile)
                with pytest.raises(ValueError, match='byte limit'):
                    _source_from_profile(profile, adapter)
            profile = compile_h3_profile(_cross_files(tmp_path, raw, train, evaluation))
            large_budget = preflight_h3_profile(profile)
            raw['launch']['budgets']['prompt_file_bytes'] = max(len(train), len(evaluation))
            profile = compile_h3_profile(_save(tmp_path, raw))
            small_budget = preflight_h3_profile(profile)
            assert small_budget.semantic_id == large_budget.semantic_id
            assert small_budget.binding_id != large_budget.binding_id
            raw['launch']['budgets']['prompt_file_bytes'] = min(len(train), len(evaluation)) - 1
            profile = compile_h3_profile(_save(tmp_path, raw))
            with pytest.raises(ValueError, match='byte budget'):
                preflight_h3_profile(profile)
            # An explicitly smaller profile budget is an intentional additional
            # restriction, not a relaxation of the source's fixed hard limit.
            source = _source_from_profile(profile, adapter)
            source.commit(source.reserve(0))
            assert source.state_dict()['next_optimizer_step'] == 1
    finally:
        adapter.close(); manager.close()
