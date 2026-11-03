"""S1 diagnostics on CPU; no claim of batched bf16/GPU numerical acceptance."""
import argparse
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization import h3_dual_anchor as anchor_module
from visual_rl.algorithms.optimization import h3_world
from visual_rl.algorithms.trainer.h3_dual_rollout import H3DualRecord
from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
from visual_rl.algorithms.trainer.interface import IterationIdentity, StageValue
from visual_rl.models.h3_joint import H3JointLayout
from visual_rl.runtime import h3_lifecycle as lifecycle


WARN = anchor_module.ROLLOUT_REPLAY_WARN_THRESHOLD


def _anchor_case(monkeypatch, deltas, *, tolerance=1e-3, old=True, reuse=False):
    dynamics = H3DualFlowSDE(H3JointLayout(1, 2, 1, 3), H3ScheduleSnapshot.build(10))
    x = torch.full((1, 5), .25, dtype=torch.float16)
    records = tuple(H3DualRecord(f's{i}', i, x, x, torch.zeros(1) if old else None)
                    for i in range(len(deltas)))
    # FP64 synthetic joint values make the exact Python threshold boundaries
    # testable without FP32 rounding. Real FP32 dynamics are exercised below.
    def evaluate(self, x, action, prediction, index):
        return torch.tensor([deltas[index]], dtype=torch.float64), (), self.coefficients(index, device='cpu')
    monkeypatch.setattr(H3DualFlowSDE, 'evaluate', evaluate)
    events, gates = [], []
    def emit(line, *, flush):
        assert flush is True
        events.append(json.loads(line))
    monkeypatch.setattr(anchor_module, 'print', emit, raising=False)
    def gate(event, **kw):
        gates.append((event, kw.get('local_error', False), len(events)))
    anchor = anchor_module.H3DualReplayAnchor(dynamics=dynamics, behavior_id='b',
        expected_keys=tuple((r.sample_id, r.index) for r in records), max_records=len(records),
        batched_rollout_tolerance=tolerance, reuse_rollout=reuse)
    def capture():
        return anchor.capture(records, device='cpu', predict=lambda sid, x, index: x.float(),
                              read_behavior_id=lambda: 'b', gate=gate)
    return anchor, capture, events, gates


def test_warning_boundaries_flushed_before_result_gate_and_capture_summary(monkeypatch):
    deltas = [0., WARN, 2.19e-4, 1e-3]
    anchor, capture, events, gates = _anchor_case(monkeypatch, deltas)
    assert capture() == dict(n_checks=4, n_warn=2, max_abs_delta=1e-3,
                            warn_threshold=1e-4, declared_tolerance=1e-3)
    checks = [e for e in events if e['event'] == 'h3_rollout_replay_score_check']
    warnings = [e for e in events if e['event'] == 'h3_rollout_replay_score_warning']
    assert len(checks) == 4 and all(e['within_tolerance'] for e in checks)
    assert warnings == [dict(event='h3_rollout_replay_score_warning', sample_id=f's{i}', index=i,
        max_abs_delta=deltas[i], warn_threshold=1e-4, declared_tolerance=1e-3) for i in (2, 3)]
    assert [(e, error, count) for e, error, count in gates if e.endswith('/result')] == [
        ('anchor/0/result', False, 1), ('anchor/1/result', False, 2),
        ('anchor/2/result', False, 4), ('anchor/3/result', False, 6)]
    assert all(anchor.get(f's{i}', i)[0].item() == delta for i, delta in enumerate(deltas))
    # Summary is emitted only by rank 0 after the existing lifecycle exchange.
    assert not any(e['event'] == 'h3_rollout_replay_score_summary' for e in events)
    with pytest.raises(RuntimeError, match='already attempted'):
        capture()


@pytest.mark.parametrize('tolerance,delta', [(1e-3, .0010001), (1e-4, 2.19e-4), (5e-5, 6e-5)])
def test_above_declared_limit_still_aborts_and_has_no_warning_or_summary(monkeypatch, tolerance, delta):
    anchor, capture, events, gates = _anchor_case(monkeypatch, [delta], tolerance=tolerance)
    with pytest.raises(ValueError, match='beyond the declared batched-rollout tolerance'):
        capture()
    assert len(events) == 1 and events[0]['within_tolerance'] is False
    assert gates[-1][:2] == ('anchor/0/result', True)
    with pytest.raises(RuntimeError, match='incomplete'):
        anchor.get('s0', 0)


def test_warning_output_failure_reaches_row_result_gate(monkeypatch):
    anchor, capture, events, gates = _anchor_case(monkeypatch, [2.19e-4])
    def fail_warning(line, *, flush):
        event = json.loads(line)
        if event['event'] == 'h3_rollout_replay_score_warning':
            raise OSError('warning output failed')
        events.append(event)
    monkeypatch.setattr(anchor_module, 'print', fail_warning, raising=False)
    with pytest.raises(OSError, match='warning output failed'):
        capture()
    assert gates[-1][:2] == ('anchor/0/result', True)
    with pytest.raises(RuntimeError, match='incomplete'):
        anchor.get('s0', 0)


def test_zero_checks_does_not_invent_deltas(monkeypatch):
    _, capture, events, _ = _anchor_case(monkeypatch, [2.19e-4], old=False)
    assert capture() == dict(n_checks=0, n_warn=0, max_abs_delta=0.,
                            warn_threshold=WARN, declared_tolerance=1e-3)
    assert events == []


@pytest.mark.parametrize('delta', [0., WARN, 2.19e-4])
def test_b1_events_return_and_strict_check_unchanged(monkeypatch, delta):
    _, capture, events, gates = _anchor_case(monkeypatch, [delta], tolerance=None)
    if delta > WARN:
        with pytest.raises(ValueError, match='sampled/replayed old score mismatch'):
            capture()
        assert events == [] and gates[-1][:2] == ('anchor/0/result', True)
    else:
        assert capture() is None
        assert events == [dict(event='h3_rollout_replay_score_check', sample_id='s0', index=0, max_abs_delta=delta)]


def test_o1_remains_b1_only_with_no_new_diagnostics(monkeypatch):
    _, capture, events, _ = _anchor_case(monkeypatch, [0.], tolerance=None, reuse=True)
    assert capture() is None
    assert events == [dict(event='h3_rollout_anchor_reused', sample_id='s0', index=0)]
    with pytest.raises(ValueError, match='reuse is rejected'):
        _anchor_case(monkeypatch, [0.], reuse=True)


@pytest.mark.parametrize('batch', [1, 4])
def test_real_rollout_and_optimize_keep_b1_anchor_and_reset_local_counters(monkeypatch, capsys, batch):
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda value, **kw: value.mul_(8))
    dynamics = H3DualFlowSDE(H3JointLayout(1, 2, 1, 3), H3ScheduleSnapshot.build(10))
    ids = tuple(f's{i}' for i in range(4))
    samples = tuple((sid, 'g', i) for i, sid in enumerate(ids))
    lora = torch.nn.Parameter(torch.zeros(5))
    optimizer = torch.optim.AdamW([lora], lr=1e-4)
    def prediction(x, index):
        return x.float() * .1 + index * .02 + lora
    extra = {} if batch == 1 else dict(rollout_batch=4, rollout_batch_replay_tolerance=1e-3,
        predict_batch=lambda prepared, sample_ids, x, index, **kw: tuple(prediction(x, index).split(1)))
    gates = []
    stage = H3DualIterationStages(mode='dual_flash', seed=42,
        data=SimpleNamespace(reserve=lambda s: None, commit=lambda r: None, abort=lambda r: None),
        prepared=SimpleNamespace(optimizer=optimizer, prepared_root=torch.nn.Identity(), reducer_divisor=8, scheduler=None),
        predict=lambda prepared, sample_id, x, index, **kw: prediction(x, index),
        decode=lambda t: None, score_one=lambda **kw: 1., coordination_factory=lambda plan: None,
        process_group=None, rank=3, device='cpu', read_behavior_id=lambda: 'b', **extra)
    for step in (3, 4):
        identity = IterationIdentity(step, 'source', 'train', ids, ('g',) * 4, (0, 1, 2, 3))
        request = SimpleNamespace(identity=identity, samples=samples,
            initial_states={sid: torch.full((1, 5), i * .2, dtype=torch.float16) for i, sid in enumerate(ids)},
            generators={sid: (torch.Generator().manual_seed(i), torch.Generator().manual_seed(i + 10))
                        for i, sid in enumerate(ids)}, behavior_id='b', conditions={sid: None for sid in ids},
            dynamics=dynamics, max_cells=4, collection_id='c', next_behavior_id='n')
        ctx = SimpleNamespace(request=request, dual=dynamics, plan=SimpleNamespace(samples=samples),
            advantages={sid: torch.tensor(v) for sid, v in zip(ids, (.5, -.5, .25, -.25))},
            coordinator=SimpleNamespace(gate=lambda event, **kw: gates.append((event, kw)),
                clip_grad_norm=lambda prepared, max_norm: float(lora.grad.norm())))
        stage._active = ctx
        value = StageValue(identity, ctx)
        stage.rollout(value)
        if batch == 4 and step == 3:
            ctx.dual_records = (replace(ctx.dual_records[0], old_joint=ctx.dual_records[0].old_joint + 2.19e-4),
                                *ctx.dual_records[1:])
        result = stage.optimize(value)
        assert result.payload.local_cells == 4
        if batch == 4:
            summary = stage.rollout_replay_score_summary
            assert summary['optimizer_step'] == step and summary['n_checks'] == 4
            assert summary['n_warn'] == (1 if step == 3 else 0)
            assert summary['max_abs_delta'] == pytest.approx(2.19e-4 if step == 3 else 0., abs=1e-7)
        else:
            assert not hasattr(stage, 'rollout_replay_score_summary')
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        ratios = [e for e in events if e['event'] == 'h3_initial_policy_ratio']
        assert len(ratios) == 4 and all(e['max_abs_logprob_delta'] == 0. for e in ratios)
        assert sum(e['event'] == 'h3_rollout_replay_score_warning' for e in events) == (batch == 4 and step == 3)
        assert not any(e['event'] == 'h3_rollout_replay_score_summary' for e in events)
    assert sum(event == 'dual/anchor/complete' for event, _ in gates) == 2


def _summary(step, rank):
    return dict(optimizer_step=step, n_checks=4, n_warn=rank % 3,
                max_abs_delta=(rank + 1) * 1e-4, warn_threshold=WARN, declared_tolerance=1e-3)


def _lifecycle_case(monkeypatch, tmp_path, *, batch, rank, corrupt=None):
    """Real bounded receipt decoder + lifecycle; fake trainer/checkpoint leaves."""
    events, exchanges, saves = [], [], []
    stages = SimpleNamespace()
    trainer = SimpleNamespace(next_optimizer_step=0)
    def iteration(step):
        trainer.next_optimizer_step = step + 1
        if batch == 4:
            stages.rollout_replay_score_summary = _summary(step, rank)
        return SimpleNamespace(value=SimpleNamespace(payload=SimpleNamespace(
            scaled_loss=rank + .5, local_cells=4, grad_norm=.25)))
    trainer.run_iteration = iteration
    def gather(outputs, raw, *, group):
        receipt = json.loads(raw)
        kind, local = receipt[1], receipt[4]
        exchanges.append((kind, receipt[5]))
        peers = []
        for peer in range(8):
            row = json.loads(raw)
            row[2] = peer
            if kind == 'lifecycle/loss':
                step = trainer.next_optimizer_step - 1
                if batch == 1:
                    assert local == rank + .5
                    row[4] = peer + .5
                else:
                    assert local == dict(loss=rank + .5, replay=_summary(step, rank))
                    summary = _summary(step, peer)
                    if corrupt is not None and peer == 7:
                        summary.update(corrupt)
                    row[4] = dict(loss=peer + .5, replay=summary)
            elif kind == 'lifecycle/pause-request':
                row[4] = False if peer == 0 else None
            peers.append(json.dumps(row).encode())
        outputs[:] = peers
    monkeypatch.setattr(h3_world.dist, 'all_gather_object', gather)
    def emit(line, *, flush):
        assert flush is True
        events.append(json.loads(line))
    monkeypatch.setattr(lifecycle, 'print', emit, raising=False)
    binding = SimpleNamespace(contract='contract', state_identity=object(),
        checkpoint_snapshot=lambda *a: ((object(), object(), {}, object()), object()))
    def save(*args, **kwargs):
        step = trainer.next_optimizer_step
        committed = SimpleNamespace(step=step, path=tmp_path / f'checkpoints/step-{step}',
            checkpoint_contract_id='contract', progress_id='progress', state_tree_id='tree')
        committed.path.mkdir(parents=True)
        saves.append(committed)
        return committed
    graph = SimpleNamespace(session=SimpleNamespace(rank=rank, process_group=object()), run_id='run',
        facts=SimpleNamespace(semantic_payload={}), profile=SimpleNamespace(
            training=SimpleNamespace(global_samples=32, group_layout='paired_4x8', algorithm='dual_flash', num_steps=10),
            launch=SimpleNamespace(output_dir=tmp_path, max_optimizer_steps=2, checkpoint_every_optimizer_steps=1,
                                   gpu_uuids=['gpu'] * 8)))
    bound = SimpleNamespace(graph=graph, stages=stages, trainer=trainer, prepared=object(), binding=binding,
        transaction=SimpleNamespace(checkpoint=save, finalize_files=lambda work: work(),
            manager=SimpleNamespace(inspect_complete=lambda *a, **kw: None)))
    backend = lifecycle.H3LifecycleBackend(**({} if batch == 1 else dict(
        rollout_batch=4, rollout_batch_replay_tolerance=1e-3)))
    return backend, bound, events, exchanges, saves


@pytest.mark.parametrize('batch,rank', [(1, 0), (1, 7), (4, 0), (4, 7)])
def test_world_summary_uses_existing_exchange_and_preserves_metrics_schema(monkeypatch, tmp_path, batch, rank):
    backend, bound, events, exchanges, saves = _lifecycle_case(monkeypatch, tmp_path, batch=batch, rank=rank)
    result = backend.run(bound)
    assert len(saves) == 2
    assert exchanges == [(label, True) for _ in range(2) for label in (
        'lifecycle/loss', 'lifecycle/metrics', 'lifecycle/pause-request', 'lifecycle/checkpoint-boundary')]
    if batch == 4 and rank == 0:
        assert events == [dict(event='h3_rollout_replay_score_summary', rank=0, scope='world', world_size=8,
            optimizer_step=step, n_checks=32, n_warn=7, max_abs_delta=8e-4,
            warn_threshold=1e-4, declared_tolerance=1e-3) for step in range(2)]
    else:
        assert events == []
    metrics = dict(step=1, sample_count=32, active_transition_count=32, loss=4., gradient_norm=.25)
    assert result[1] == metrics
    if rank == 0:
        backend.checkpoint(bound, result)  # exercise the unchanged exact terminal schema
        assert json.loads((tmp_path / 'h3-last-metrics.json').read_text()) == metrics
        assert json.loads((tmp_path / 'metrics.jsonl').read_text()) == dict(schema_version=1, **metrics)
    else:
        assert not (tmp_path / 'h3-last-metrics.json').exists()
        assert not (tmp_path / 'metrics.jsonl').exists()


@pytest.mark.parametrize('corrupt', [dict(optimizer_step=99), dict(warn_threshold=1e-3),
    dict(declared_tolerance=1e-4), dict(n_warn=5), dict(n_checks=-1), dict(max_abs_delta=-1.)])
def test_bad_peer_summary_fails_inside_existing_metrics_gate(monkeypatch, tmp_path, corrupt):
    backend, bound, events, exchanges, saves = _lifecycle_case(
        monkeypatch, tmp_path, batch=4, rank=0, corrupt=corrupt)
    with pytest.raises(ValueError, match='replay summary'):
        backend.run(bound)
    assert exchanges == [('lifecycle/loss', True), ('lifecycle/metrics', False)]
    assert events == saves == []
    assert not (tmp_path / 'h3-last-metrics.json').exists()


def test_summary_output_failure_reaches_existing_metrics_gate(monkeypatch, tmp_path):
    backend, bound, _, exchanges, saves = _lifecycle_case(monkeypatch, tmp_path, batch=4, rank=0)
    def fail_summary(*args, **kwargs):
        raise OSError('summary output failed')
    monkeypatch.setattr(lifecycle, 'print', fail_summary)
    with pytest.raises(OSError, match='summary output failed'):
        backend.run(bound)
    assert exchanges == [('lifecycle/loss', True), ('lifecycle/metrics', False)]
    assert saves == []


def test_defaults_help_and_identity_payloads_stay_unchanged():
    from scripts import launch_h3_formal as formal
    from scripts.launch_h3_resolution_memory import add_common_arguments
    from visual_rl.composition.config.h3_profile import H3Training
    required = ['config.yaml', '--expected-sha', 'a' * 40, '--ffmpeg', 'ffmpeg', '--ffprobe', 'ffprobe', '--monitor-dir', 'm']
    for parser in (add_common_arguments(argparse.ArgumentParser()), formal.build_parser()):
        default = parser.parse_args(required)
        explicit = parser.parse_args(required + ['--rollout-batch', '4', '--rollout-batch-replay-tolerance', '1e-3'])
        assert (default.rollout_batch, default.rollout_batch_replay_tolerance) == (1, 1e-4)
        assert (explicit.rollout_batch, explicit.rollout_batch_replay_tolerance) == (4, 1e-3)
        help_text = ' '.join(parser.format_help().split())
        assert '--rollout-batch-replay-tolerance 1e-3' in help_text
        assert 'S1 warns above 1e-4' in help_text
        flags = formal.extension_flags(explicit)
        assert flags == dict(rollout_batch=4, rollout_batch_replay_tolerance=1e-3, parallel_owners=False)
    for owner in (H3DualIterationStages, lifecycle.H3LifecycleBackend):
        assert inspect.signature(owner).parameters['rollout_batch_replay_tolerance'].default == 1e-4
    paired = H3Training(group_size=8, algorithm='dual_flash', group_layout='paired_4x8')
    batched = replace(paired, rollout_batch=4, rollout_batch_replay_tolerance=1e-3)
    assert paired.rollout_batch_replay_tolerance == 1e-4
    assert not any('warn' in key or 'replay' in key for key in paired.payload() if key != 'replay_microbatch')
    assert {key: value for key, value in batched.payload().items()
            if key not in ('rollout_microbatch', 'rollout_batch', 'rollout_batch_replay_tolerance')} == {
            key: value for key, value in paired.payload().items() if key != 'rollout_microbatch'}
    assert not any('warn' in key for key in batched.payload())
