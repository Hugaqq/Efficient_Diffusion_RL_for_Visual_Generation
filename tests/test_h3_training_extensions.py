"""CPU extension contracts; heavy model/NCCL acceptance remains a separate gate."""
from dataclasses import asdict, replace
import subprocess
import sys
import types

import pytest

from visual_rl.composition.config.h3_profile import H3EMAConfig, H3Training, compile_h3_profile, preflight_h3_profile
from tests.test_h3_mvp_cpu_profile import _fixture, _save


def legacy_module(path):
    """Read the immutable baseline, never the dirty checkout or another worktree."""
    text = subprocess.check_output(['git', 'show', '7c85751:' + path], text=True)
    module = types.ModuleType('_h3_baseline_' + path.replace('/', '_').replace('.', '_'))
    sys.modules[module.__name__] = module
    exec(compile(text, '7c85751:' + path, 'exec'), module.__dict__)
    return module


def test_default_profile_exact_legacy_identity(tmp_path):
    old = legacy_module('visual_rl/composition/config/h3_profile.py')
    path, raw = _fixture(tmp_path)
    for algorithm in ('cps', 'dual_flow', 'dual_flash'):
        raw['overrides']['training'] = dict(algorithm=algorithm)
        path = _save(tmp_path, raw)
        before = old.preflight_h3_profile(old.compile_h3_profile(path))
        now = preflight_h3_profile(compile_h3_profile(path))
        assert asdict(now) == asdict(before)
        explicit = dict(kl_beta=0., ema=asdict(H3EMAConfig()), advantage_std_domain='batch')
        raw['overrides']['training'].update(explicit)
        raw['launch']['hps']['origins'] = [raw['launch']['hps']['origin']]
        assert asdict(preflight_h3_profile(compile_h3_profile(_save(tmp_path, raw)))) == asdict(before)
        raw['launch']['hps'].pop('origins')
    for batch in (1, 4):
        options = dict(algorithm='dual_flash', group_layout='paired_4x8', rollout_batch=batch)
        assert H3Training(**options).payload() == old.H3Training(**options).payload()


def test_multi_origin_semantics_use_scoring_not_placement_identity(tmp_path):
    from visual_rl.core.protocols.flash_hpsv3 import decode_json, canonical_json, digest
    from visual_rl.core.immutable import FrozenMapping
    path, _ = _fixture(tmp_path)
    profile = compile_h3_profile(path)
    profile = replace(profile, launch=replace(profile.launch,
        hps_origins=(profile.launch.hps_origin, 'http://127.0.0.1:18889')))
    health = decode_json(profile.launch.hps_manifest.read_bytes())
    ids = []
    for uuid in ('GPU-a', 'GPU-b'):
        health['identity_facts']['gpu_residency_policy'] = dict(expected_gpu_uuid=uuid, enabled=True)
        health['service_identity'] = digest(dict(identity_facts=health['identity_facts'], video_contract=health['video_contract']))
        profile.launch.hps_manifest.write_bytes(canonical_json(health))
        profile = replace(profile, expected=FrozenMapping(dict(profile.expected, hps_service_identity=health['service_identity'])))
        facts = preflight_h3_profile(profile)
        assert 'hps_service_identity' not in facts.semantic_payload
        assert 'hps_scoring_identity' in facts.semantic_payload
        ids.append((facts.semantic_id, facts.binding_id))
    assert ids[0] == ids[1]


def test_nondefault_profile_identity_and_validation(tmp_path):
    path, raw = _fixture(tmp_path)
    raw['overrides']['training'] = dict(algorithm='dual_flash')
    profile = compile_h3_profile(_save(tmp_path, raw))
    baseline = preflight_h3_profile(profile).semantic_id
    changes = (dict(kl_beta=.01), dict(ema=H3EMAConfig(enabled=True)),
               dict(ema=H3EMAConfig(decay=.8)), dict(ema=H3EMAConfig(update_interval=2)),
               dict(advantage_std_domain='group'))
    identities = [preflight_h3_profile(replace(profile, training=replace(profile.training, **change))).semantic_id
                  for change in changes]
    assert baseline not in identities and len(set(identities)) == len(changes)
    origins = (profile.launch.hps_origin, 'http://127.0.0.1:18889')
    other = replace(profile, launch=replace(profile.launch, hps_origins=origins))
    assert preflight_h3_profile(other).semantic_id != baseline
    assert preflight_h3_profile(other).binding_id != profile.launch.binding_id
    for change in (dict(kl_beta=-1), dict(kl_beta=float('nan')), dict(kl_beta=True),
                   dict(advantage_std_domain='world')):
        with pytest.raises(ValueError):
            replace(profile.training, **change)
    with pytest.raises(ValueError, match='Gaussian'):
        H3Training(kl_beta=.1)
    for change in (dict(enabled=1), dict(decay=1), dict(decay=float('inf')),
                   dict(update_interval=0), dict(update_interval=True), dict(counting='microbatch')):
        with pytest.raises((ValueError, TypeError)):
            H3EMAConfig(**change)
    for origins in ((profile.launch.hps_origin,) * 2, ('https://localhost:1',), ('http://example.org:1',)):
        with pytest.raises(ValueError):
            replace(profile.launch, hps_origins=origins)
