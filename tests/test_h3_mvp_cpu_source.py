"""Real prompt files/cache/adapter/source and existing six-stage CPU consumer."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import weakref

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tests.test_h3_mvp_cpu_adapter import _adapter
from tests.test_h3_mvp_cpu_decode import _artifacts, _leaf_setup
from tests.test_h3_mvp_cpu_text_cache import _identity, _metadata
from tests.test_h3_mvp_cpu_stages import _world_leaf, _CoordinationLeaf
import visual_rl
from visual_rl.data import h3_source
from visual_rl.data.h3_source import H3PromptSource
from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.trainer.h3_stages import H3IterationStages
from visual_rl.algorithms.trainer.interface import PrepareRunContext
from visual_rl.models.h3_interface import H3ModelInput, H3PromptInput, H3_GUIDANCE_ID
from visual_rl.models.h3_joint import H3JointLayout
from visual_rl.models.implementations import h3_loading
from visual_rl.models.lifecycle.components import ComponentManager
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel
from visual_rl.models.lifecycle.prepared import PreparedModelBundle


TRAIN = ('cat', ' cat', 'cat ', 'fish', '猫', ' dog', 'cat', 'bird')
EVAL = ('heldout', '  kept  ')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path, monkeypatch):
    adapter, context, _ = _adapter(tmp_path, monkeypatch)
    cache = context['artifact_paths']['h3-text']
    tensors, entries = {}, {}
    for i, prompt in enumerate(dict.fromkeys(TRAIN + EVAL)):
        key = _sha(prompt.encode())
        tensors[key] = torch.full((2, 5120), .001 * (i+1))
        entries[key] = dict(shard='data.safetensors', tensor=key, tokens=2)
    save_file(tensors, str(cache / 'data.safetensors'), metadata=_metadata(_identity()))
    (cache / 'index.json').write_text(json.dumps(dict(
        schema_version='visual_rl.h3-text-cache.v1', identity=_identity(),
        shards={'data.safetensors': _sha((cache / 'data.safetensors').read_bytes())}, entries=entries)))
    manager = ComponentManager(adapter, execution_device='cpu')
    components = manager.load()
    train, heldout = tmp_path / 'train.txt', tmp_path / 'eval.txt'
    train.write_bytes(('\r\n'.join(TRAIN) + '\r\n').encode('utf-8'))
    heldout.write_bytes('\n'.join(EVAL).encode('utf-8'))  # final line without newline
    dynamics = H3CPSDynamics(adapter.geometry.joint_layout, H3ScheduleSnapshot.build(3))
    width = dynamics.layout.packed_width
    args = dict(train_path=train, eval_path=heldout, train_sha256=_sha(train.read_bytes()),
                eval_sha256=_sha(heldout.read_bytes()), adapter=adapter, dynamics=dynamics,
                rank=0, world_size=8, group_size=2, base_seed=42, device='cpu',
                max_samples=2, max_cells=6, max_trajectory_bytes=6*(4*width+4)+4*width)
    return adapter, manager, components, args


def _rng():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_rng(before):
    after = _rng()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def _same_request(a, b):
    assert a.identity == b.identity
    for name in ('collection_id', 'behavior_id', 'next_behavior_id', 'samples', 'prompts', 'geometry_id'):
        assert getattr(a, name) == getattr(b, name)
    for sid in a.prompts:
        assert torch.equal(a.conditions[sid], b.conditions[sid])
        assert torch.equal(a.initial_states[sid], b.initial_states[sid])
        assert all(torch.equal(x.get_state(), y.get_state()) for x, y in zip(a.generators[sid], b.generators[sid]))


def test_source_real_eight_rank_occurrences_exact_prompts_and_rng(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == root / 'visual_rl'
    assert Path(h3_source.__file__).resolve() == root / 'visual_rl/data/h3_source.py'
    adapter, manager, _, args = _fixture(tmp_path, monkeypatch)
    coverage = []
    real_coverage = adapter.verify_coverage
    def observe(train, heldout):
        coverage.append((tuple(train), tuple(heldout)))
        return real_coverage(train, heldout)
    monkeypatch.setattr(adapter, 'verify_coverage', observe)
    before = _rng()
    sources = [H3PromptSource(**(args | dict(rank=r))) for r in range(8)]
    assert coverage == [(TRAIN, EVAL)] * 8
    assert len({s.source_id for s in sources}) == len({s.data_identity for s in sources}) == 1
    assert len({s.geometry_id for s in sources}) == len({s.next_global_batch_id for s in sources}) == 1
    assert len({s.read_behavior_id() for s in sources}) == 1
    requests = [s.reserve(0) for s in sources]
    assert len({row[0] for request in requests for row in request.samples}) == 16
    assert len({row[1] for request in requests for row in request.samples}) == 8
    assert len({r.collection_id for r in requests}) == 1
    for rank, (source, request) in enumerate(zip(sources, requests)):
        assert list(request.prompts.values()) == [TRAIN[rank]] * 2
        assert [row[2] for row in request.samples] == [0, 1]
        assert source.read_behavior_id() == request.behavior_id  # reserve is not commit
        for sid, _, member in request.samples:
            typed = source.conditioning_for(sid)
            assert typed.prompt == TRAIN[rank] and torch.equal(typed.embeddings, request.conditions[sid])
            assert typed.condition_identity[0].endswith(_sha(TRAIN[rank].encode()))
            assert request.conditions[sid].shape == (1, 2, 5120)
            assert not request.conditions[sid].requires_grad and request.initial_states[sid].dtype == torch.float16
            # Independent reference for the specified canonical SHA / 63-bit streams.
            seeds = []
            for purpose in ('initial', 'rollout'):
                for modality in ('video', 'audio'):
                    payload = (h3_source.STREAM_VERSION, 42, source.source_id, rank, member, purpose, modality)
                    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
                    seeds.append(int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & (2**63-1))
            assert len(set(seeds)) == 4
            assert [g.initial_seed() for g in request.generators[sid]] == seeds[2:]
            layout = args['dynamics'].layout
            video = torch.randn(1, layout.video_rows, layout.video_channels, generator=torch.Generator().manual_seed(seeds[0])).half()
            audio = torch.randn(1, layout.audio_rows, layout.audio_channels, generator=torch.Generator().manual_seed(seeds[1])).half()
            assert torch.equal(request.initial_states[sid], layout.pack(video, audio))
    assert requests[0].prompts != requests[6].prompts  # keys differ despite identical text
    assert set(requests[0].prompts.values()) == set(requests[6].prompts.values()) == {'cat'}
    for source, request in zip(sources, requests):
        source.commit(request)
    assert len({s.next_global_batch_id for s in sources}) == 1
    second = sources[0].reserve(1)
    assert list(second.prompts.values()) == ['cat', 'cat']  # occurrence 8 cycles original order
    assert second.samples[0][1] != requests[0].samples[0][1]
    sources[0].commit(second)
    assert sources[0].state_dict()['global_cursor'] == 16
    _assert_rng(before)
    manager.close()


def test_source_reservation_exact_commit_abort_and_reference_release(tmp_path, monkeypatch):
    _, manager, _, args = _fixture(tmp_path, monkeypatch)
    source = H3PromptSource(**args)
    original = source.state_dict()
    request = source.reserve(0)
    refs = [weakref.ref(request), weakref.ref(source.conditioning_for(request.samples[0][0]))]
    with pytest.raises(RuntimeError, match='active'):
        source.reserve(0)
    with pytest.raises(RuntimeError, match='boundary'):
        source.state_dict()
    with pytest.raises(RuntimeError, match='exact active'):
        source.commit(replace(request))
    assert source.read_behavior_id() == original['behavior_id']
    request.conditions[request.samples[0][0]].fill_(99)
    assert not bool((source.conditioning_for(request.samples[0][0]).embeddings == 99).all())
    source.commit(request)
    with pytest.raises(RuntimeError, match='exact active'):
        source.commit(request)
    with pytest.raises(RuntimeError, match='reservation'):
        source.conditioning_for(request.samples[0][0])
    del request
    assert all(ref() is None for ref in refs)
    state = source.state_dict()
    state['binding']['base_seed'] = 9
    assert source.state_dict()['binding']['base_seed'] == 42
    request = source.reserve(1)
    behavior = source.read_behavior_id()
    source.abort(request)
    assert source.read_behavior_id() == behavior
    with pytest.raises(RuntimeError, match='poisoned'):
        source.reserve(1)
    with pytest.raises(RuntimeError, match='boundary'):
        source.state_dict()
    with pytest.raises(RuntimeError, match='fresh'):
        source.load_state_dict(original)
    manager.close()


def test_source_boundary_fresh_restore_next_request_exact_and_path_independent(tmp_path, monkeypatch):
    _, manager, _, args = _fixture(tmp_path, monkeypatch)
    before = _rng()
    continuous = H3PromptSource(**args)
    for step in range(2):
        request = continuous.reserve(step)
        continuous.commit(request)
    state = json.loads(json.dumps(continuous.state_dict(), allow_nan=False))
    copied = tmp_path / 'same-content.txt'
    copied.write_bytes(args['train_path'].read_bytes())
    restored = H3PromptSource(**(args | dict(train_path=copied)))
    restored.load_state_dict(state)
    assert restored.data_identity == continuous.data_identity
    assert restored.next_global_batch_id == continuous.next_global_batch_id
    assert restored.state_dict() == state
    a, b = continuous.reserve(2), restored.reserve(2)
    _same_request(a, b)
    for sid in a.prompts:
        assert continuous.conditioning_for(sid).condition_identity == restored.conditioning_for(sid).condition_identity
    _assert_rng(before)
    with pytest.raises(RuntimeError, match='fresh'):
        restored.load_state_dict(state)
    continuous.commit(a); restored.commit(b)
    assert restored.state_dict() == continuous.state_dict()
    manager.close()


def test_source_rejects_restore_identity_topology_progress_and_hot_state(tmp_path, monkeypatch):
    _, manager, _, args = _fixture(tmp_path, monkeypatch)
    source = H3PromptSource(**args)
    r = source.reserve(0); source.commit(r)
    state = source.state_dict()
    mutations = [lambda s: s.update(schema_version='old'), lambda s: s.update(rank=1),
                 lambda s: s.update(global_cursor=0), lambda s: s.update(behavior_id='foreign'),
                 lambda s: s.update(next_global_batch_id='foreign'), lambda s: s.update(next_optimizer_step=True),
                 lambda s: s.update(extra='unknown'), lambda s: s['binding'].update(stream='other'),
                 lambda s: s['binding'].update(world_size=4), lambda s: s['binding'].update(group_size=3),
                 lambda s: s['binding']['geometry'].update(cache_identity='0'*64),
                 lambda s: s['binding']['video_sigmas'].__setitem__(1, .5)]
    for change in mutations:
        broken = json.loads(json.dumps(state)); change(broken)
        target = H3PromptSource(**args)
        with pytest.raises(ValueError):
            target.load_state_dict(broken)
        with pytest.raises(RuntimeError, match='poisoned'):
            target.reserve(0)
    for change in (dict(rank=1), dict(base_seed=43), dict(max_cells=7),
                   dict(dynamics=H3CPSDynamics(args['dynamics'].layout, args['dynamics'].schedule, eta=.6))):
        with pytest.raises(ValueError, match='mismatch'):
            H3PromptSource(**(args | change)).load_state_dict(state)
    for step in (1, True, -1):
        target = H3PromptSource(**args)
        with pytest.raises(ValueError, match='next optimizer step'):
            target.reserve(step)
    manager.close()


def test_source_prompt_validation_coverage_drift_shape_and_budgets(tmp_path, monkeypatch):
    adapter, manager, _, args = _fixture(tmp_path, monkeypatch)
    original = args['train_path'].read_bytes()
    for raw in (b'', b'cat\n'*7, b'cat\n\n' + b'cat\n'*7, b'\xff\n'*8,
                b'\xef\xbb\xbfcat\n'*8, b'cat\rcat\n'*8):
        args['train_path'].write_bytes(raw)
        with pytest.raises((ValueError, UnicodeError)):
            H3PromptSource(**(args | dict(train_sha256=_sha(raw))))
    args['train_path'].write_bytes(original)
    for change in (dict(train_sha256='0'*64), dict(train_sha256='bad'), dict(world_size=4), dict(rank=True),
                   dict(base_seed=-1), dict(base_seed=True), dict(group_size=1), dict(max_samples=1),
                   dict(max_cells=5), dict(max_trajectory_bytes=args['max_trajectory_bytes']-1),
                   dict(device='meta'), dict(device='cuda:1'), dict(device='cpu:0'),
                   dict(train_path=tmp_path / 'missing.txt'),
                   dict(dynamics=H3CPSDynamics(H3JointLayout(1, 2, 1, 3), args['dynamics'].schedule))):
        with pytest.raises(ValueError):
            H3PromptSource(**(args | change))
    alias = tmp_path / 'alias.txt'
    alias.symlink_to(args['train_path'])
    with pytest.raises(ValueError, match='non-symlink'):
        H3PromptSource(**(args | dict(train_path=alias)))
    args['train_path'].write_bytes(b'not-in-cache\n'*8)
    with pytest.raises(ValueError, match='missing train'):
        H3PromptSource(**(args | dict(train_sha256=_sha(args['train_path'].read_bytes()))))
    args['train_path'].write_bytes(original)
    heldout = args['eval_path'].read_bytes()
    args['eval_path'].write_bytes(b'')
    with pytest.raises(ValueError, match='empty prompt'):
        H3PromptSource(**(args | dict(eval_sha256=_sha(b''))))
    args['eval_path'].write_bytes(b'not-in-cache')
    with pytest.raises(ValueError, match='missing eval'):
        H3PromptSource(**(args | dict(eval_sha256=_sha(b'not-in-cache'))))
    args['eval_path'].write_bytes(heldout)
    source = H3PromptSource(**args)
    request = source.reserve(0)
    args['train_path'].write_bytes(original + b'cat\n')
    with pytest.raises(ValueError, match='drift'):
        source.commit(request)
    with pytest.raises(RuntimeError, match='boundary'):
        source.state_dict()
    args['train_path'].write_bytes(original)
    source = H3PromptSource(**args)
    # Actual cache drift is detected by the real adapter/cache on reservation.
    cache = adapter.encode(H3PromptInput('cat')).cache_identity
    assert len(cache) == 64
    index = tmp_path / 'cache/index.json'
    index.write_bytes(index.read_bytes() + b' ')
    with pytest.raises(ValueError, match='index changed'):
        source.reserve(0)
    manager.close()


def test_real_source_drives_existing_trainer_tail_and_restorable_boundary(tmp_path, monkeypatch):
    adapter, manager, components, args = _fixture(tmp_path, monkeypatch)
    # Real tiny loader/PEFT state materialization and guarded prepared owner.
    meta = adapter.meta_model
    generator = torch.Generator().manual_seed(91)
    base = {key: torch.randn(meta.state_specs[name].shape, generator=generator)*.02
            for key, name in meta.base_to_model.items()}
    save_file(base, str(meta.transformer_dir / 'diffusion_pytorch_model.safetensors'))
    state = h3_loading.load_h3_cpu_state(meta, is_init_owner=True, host_weight_limit_bytes=10000000, init_seed=42)
    meta.model.load_state_dict(state.state, strict=True, assign=True)
    bundle = PreparedModelBundle(components)
    optimizer = torch.optim.AdamW(bundle.trainable_parameters(), lr=1e-4)
    prepared = H3PreparedModel(source_bundle=bundle, prepared_root=bundle, optimizer=optimizer,
                               scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.), accelerator=object())
    adapter._bind_prepared_components(prepared)
    source = H3PromptSource(**args)
    events, pools = [], []
    group = _world_leaf(monkeypatch, events)  # CPU collective leaf; real world math.
    _artifacts(meta.transformer_dir.parent)
    _leaf_setup(monkeypatch, adapter.geometry)
    def predict(bound, *, sample_id, x, index, condition, dynamics):
        assert bound is prepared
        typed = source.conditioning_for(sample_id)
        assert torch.equal(condition, typed.embeddings)
        spec = adapter.latent_spec_for_batch(H3PromptInput(typed.prompt), device=x.device, dtype=x.dtype)
        vs, _, aus, _ = dynamics.schedule.transition(index)
        return adapter.predict(H3ModelInput(x, vs, typed, None, spec, typed.condition_identity,
                                            (H3_GUIDANCE_ID,), audio_sigma=aus)).value
    spec = adapter.latent_spec_for_batch(H3PromptInput('cat'), device='cpu', dtype=torch.float16)
    stages = H3IterationStages(data=source, prepared=prepared, predict=predict,
        decode=lambda terminal: adapter.decode(terminal, spec),
        score_one=lambda **kw: float(kw['media'].video.mean()),
        coordination_factory=lambda plan: _CoordinationLeaf(plan, events),
        process_group=group, rank=0, device='cpu', read_behavior_id=source.read_behavior_id)
    trainer = stages.trainer()
    real_rollout = trainer._stages['rollout']
    def observe(value):
        pools.append(value.payload.pool)
        return real_rollout(value)
    trainer._stages['rollout'] = observe
    trainer.prepare_run(PrepareRunContext('source-test', 'h3', 0))
    previous = source.read_behavior_id()
    result = trainer.run_iteration(0)
    assert result.value.payload.local_cells == 6 and trainer.next_optimizer_step == 1
    assert pools[0].state == 'released' and source.read_behavior_id() != previous
    assert all(entry['step'].item() == 1 for entry in optimizer.state.values())
    saved = source.state_dict()
    assert saved['global_cursor'] == 8 and saved['next_optimizer_step'] == 1
    fresh = H3PromptSource(**args)
    fresh.load_state_dict(json.loads(json.dumps(saved)))
    _same_request(source.reserve(1), fresh.reserve(1))
    trainer.close(); adapter.close(); manager.close()
