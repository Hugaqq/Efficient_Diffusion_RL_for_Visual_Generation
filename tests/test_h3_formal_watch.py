"""Offline watch contract tests. No GPU/network/SSH/LLM/notifier execution."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ops.h3_formal_watch import process
from ops.h3_formal_watch.common import Store, read_json
from ops.h3_formal_watch.drill import Sandbox, fake_checkpoint, run_drill
from ops.h3_formal_watch.watchdog import (Watchdog, checkpoint, disk_check, hps_check,
                                        latest_checkpoint, monitor_check, progress_evidence)


@pytest.fixture
def box(tmp_path):
    value = Sandbox(tmp_path / 'fake')
    try:
        yield value
    finally:
        value.cleanup()


def test_progress_and_checkpoint_publication(box):
    out = box.root / 'output'
    assert progress_evidence(out)['step'] == 1
    fake_checkpoint(out, 1)
    Store(out / 'checkpoints').save('latest.json', dict(path='step-99', step=99))
    (out / 'checkpoints' / 'step-200.tmp-ignored').mkdir()
    assert latest_checkpoint(out, True)['step'] == 1
    Store(out).save('h3-last-metrics.json', dict(step=3))
    assert progress_evidence(out)['step'] == 4  # zero-based metric index, NOT step 3
    box.tick()
    first = box.watch.s['last_progress_at']
    box.advance(30)
    box.monitor()
    box.tick()
    assert box.watch.s['last_progress_at'] == first
    Store(out).save('h3-last-metrics.json', dict(step=4))
    box.tick()
    assert box.watch.s['last_progress_at'] == box.now
    Store(out / 'checkpoints' / 'step-1').text('fake-state', 'corrupt')
    with pytest.raises(ValueError):
        latest_checkpoint(out, True)


@pytest.mark.parametrize('field,value', [('healthy', False), ('monotonic_ns', 0), ('monotonic_ns', 10**30)])
def test_monitor_failure(box, field, value):
    box.tick()
    box.monitor(**{field: value})
    assert box.tick()['state'] == 'WATCHING'
    assert box.tick()['state'] == 'WATCHING'
    expected = 'NEEDS_HUMAN' if value == 10**30 else 'STOPPING'
    assert box.tick()['state'] == expected
    if expected == 'STOPPING':
        assert box.watch.s['incidents'][0]['cause'] == 'monitor_stop'


def test_monitor_degraded_is_nonfatal_and_logged(box):
    box.monitor(degraded=True, consecutive_failures=2, max_seconds=1000000, max_bytes=12345)
    assert box.tick()['state'] == 'WATCHING'
    assert 'monitor_degraded' in (box.root / 'state' / 'events.jsonl').read_text()
    assert box.watch.s['last_checks']['monitor']['health']['max_bytes'] == 12345
    box.advance(30)
    assert not monitor_check(box.root / 'monitor', lambda: box.ns, 30)['ok']


def test_hps_protocol_and_bounds():
    class Response:
        code = 200
        def __init__(self, row): self.row = row
        def read(self, n): return json.dumps(self.row).encode()[:n]
        def __enter__(self): return self
        def __exit__(self, *args): pass
    identity = 'a' * 64
    for row, ok, changed in [(dict(status='ready', service_identity=identity), True, False),
                             (dict(status='unready', service_identity=identity), False, False),
                             (dict(status='ready', service_identity='b' * 64), False, True),
                             (dict(status='ready'), False, False),
                             (dict(status='ready', service_identity=identity, identity_facts={}), False, True)]:
        with patch('urllib.request.OpenerDirector.open', return_value=Response(row)) as fetch:
            result = hps_check('http://unused', identity)
            assert result['ok'] is ok and result['identity_changed'] is changed
            assert fetch.call_args.kwargs['timeout'] == 8
    with patch('urllib.request.OpenerDirector.open', side_effect=TimeoutError):
        with pytest.raises(TimeoutError):
            hps_check('http://unused', identity)


def test_disk_bytes_and_inodes(box):
    with patch('os.statvfs', return_value=SimpleNamespace(f_bavail=100, f_frsize=10, f_favail=5, f_files=20)):
        assert disk_check(box.root, 1000, 5)['ok']
        assert not disk_check(box.root, 1001, 5)['ok']
        assert not disk_check(box.root, 1000, 6)['ok']
    with patch('os.statvfs', return_value=SimpleNamespace(f_bavail=100, f_frsize=10, f_favail=5, f_files=0)):
        assert disk_check(box.root, 0, 1)['ok'] is None
        assert disk_check(box.root, 0, 0)['ok']
    box.tick()
    with patch('ops.h3_formal_watch.watchdog.disk_check', return_value=dict(ok=False)):
        assert box.tick()['state'] == 'WATCHING'
        assert box.tick()['state'] == 'WATCHING'
        assert box.tick()['state'] == 'STOPPING'
        assert box.watch.s['incidents'][-1]['cause'] == 'disk_low'


def test_identity_failure_is_permanent(box):
    fake_checkpoint(box.root / 'output')
    box.tick()
    box.health = dict(ok=False, identity_changed=True)
    assert box.tick()['state'] == 'STOPPING'
    box.child.wait(timeout=5)
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    box.health = dict(ok=True)
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    assert not box.watch.s['resumes']


def test_progress_timeout_and_restart_clock(box):
    box.tick()
    box.advance(10800)
    box.monitor()
    assert box.tick()['state'] == 'STOPPING'
    assert box.watch.s['incidents'][-1]['cause'] == 'no_progress'
    recovered = Watchdog(box.config, clock=lambda: box.now, monotonic_ns=lambda: box.ns)
    assert recovered.s['last_progress_at'] == box.watch.s['last_progress_at']
    assert recovered.s['state'] == 'STOPPING'


def test_pause_then_term_then_kill_only_recorded_group(box):
    watch = box.watch
    box.tick()
    box.monitor(healthy=False)
    box.tick(); box.tick()
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'STOPPING'
        assert (box.root / 'output' / 'PAUSE_REQUESTED').exists()
        send.assert_not_called()
        box.advance(12600)
        box.tick()
        assert send.call_args.args == (watch.s['process'], signal.SIGTERM)
        box.advance(.2)
        box.tick()
        assert send.call_args.args == (watch.s['process'], signal.SIGKILL)
        box.advance(.2)
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        assert send.call_count == 2
        box.tick()
        assert send.call_count == 2


def test_dry_run_no_progress_prints_without_signalling(box, capsys):
    box.watch.dry_run = True
    box.tick()
    box.advance(10800); box.monitor()
    with patch.object(process, 'signal_group') as send:
        box.tick()
        send.assert_not_called()
    assert 'SIGTERM' in capsys.readouterr().out
    assert not (box.root / 'output' / 'PAUSE_REQUESTED').exists()


def test_process_identity_and_scope(box):
    rec = box.config['process']
    assert process.inspect(rec)['alive']
    p = rec['members'][0]
    with pytest.raises(ValueError):
        process.inspect(rec, {p['pid']: dict(p, token='reused')})
    with pytest.raises(ValueError):
        process.inspect(dict(rec, boot_id='reboot'))
    with pytest.raises(ValueError):
        process.inspect(dict(rec, pgid=os.getpgrp()))
    stranger = dict(p, pid=p['pid'] + 1000000)
    with pytest.raises(ValueError):
        process.inspect(rec, {stranger['pid']: stranger})
    assert not process.inspect(rec, {p['pid']: dict(p, zombie=True)})['alive']
    with patch.object(process, 'inspect', side_effect=ValueError('reused')), patch.object(os, 'killpg') as send:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        send.assert_not_called()


def test_resume_policy_limits_and_original_recipe_unchanged(box):
    watch = box.watch
    fake_checkpoint(box.root / 'output')
    ckpt = dict(path=str(box.root / 'output' / 'checkpoints' / 'step-1'), step=1)
    assert watch.resume_allowed(None) == 'no complete checkpoint'
    watch.s.update(retry_step=1, retry_highwater=5)
    watch.s['resumes'] = [dict(at=box.now, mono=watch.mono(), boot_id=watch.boot, step=1, highwater_step=5)]
    watch.s['highwater_step'] = 5
    assert 'same checkpoint' in watch.resume_allowed(ckpt)
    watch.s['highwater_step'] = 6
    assert watch.resume_allowed(ckpt) is None
    watch.s['resumes'].append(dict(at=box.now, mono=watch.mono(), boot_id=watch.boot, step=5, highwater_step=6))
    assert '24-hour' in watch.resume_allowed(ckpt)
    box.advance(86401)
    assert watch.resume_allowed(ckpt) is None
    watch.s['highwater_step'] = 5
    assert 'same checkpoint' in watch.resume_allowed(ckpt)  # step guard survives 24h
    before = (box.root / 'recipe.json').read_bytes()
    spec = watch.make_resume_spec(ckpt, 'fake')
    assert (box.root / 'recipe.json').read_bytes() == before
    generated = read_json(spec['argv'][-1])
    original = json.loads(before)
    original['launch']['resume_from'] = ckpt['path']
    assert generated == original
    assert spec['env'] == watch.config['launcher']['env']
    assert spec['cwd'] == watch.config['launcher']['cwd']


def test_resume_unhealthy_prerequisites_and_unknown_launch(box):
    box.tick()
    box.log('NCCL watchdog timed out')
    box.die()
    assert box.tick()['state'] == 'RECOVERING'
    box.monitor(healthy=False)
    box.advance(1801)
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    watch = box.watch
    watch.s['state'] = 'LAUNCHING'
    watch.s['pending_launch'] = dict(launch_id='missing', mono=watch.mono() - 61)
    with patch.object(process, 'inspect', return_value=dict(alive=False, members=[])), patch('subprocess.Popen') as spawn:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        spawn.assert_not_called()


def test_corrupt_checkpoint_never_resumes(box):
    box.tick()
    box.log('NCCL watchdog timed out')
    fake_checkpoint(box.root / 'output')
    Store(box.root / 'output' / 'checkpoints' / 'step-1').text('fake-state', 'corrupt')
    box.die()
    box.tick()
    with patch.object(process, 'inspect', return_value=dict(alive=False, members=[])), patch('subprocess.Popen') as spawn:
        assert box.until('NEEDS_HUMAN')['state'] == 'NEEDS_HUMAN'
        spawn.assert_not_called()


def test_no_checkpoint_and_nonzero_without_success(box):
    box.die()
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    assert box.watch.s['incidents'][-1]['cause'] == 'unknown_exit'
    assert box.tick()['state'] == 'NEEDS_HUMAN'


def test_success_requires_matching_exit_receipt_and_exited_group(box):
    out = box.root / 'output'
    marker = fake_checkpoint(out)
    success = dict(kind='visual_rl_h3_run_success', run_id='fake-run', committed_steps=1,
                   checkpoint_relative_path='checkpoints/step-1',
                   **{k: marker[k] for k in ('checkpoint_contract_id', 'progress_id', 'state_tree_id')})
    Store(out).save('SUCCESS', success)
    assert box.tick()['state'] == 'WATCHING'  # not DONE while processes alive
    box.die()
    assert box.tick()['state'] != 'DONE'
    record = box.config['process']
    box.watch.store.save('exit-' + record['launch_id'] + '.json', dict(launch_id=record['launch_id'],
                         pgid=record['pgid'], exit_code=1))
    assert box.tick()['state'] == 'NEEDS_HUMAN'


def test_watchdog_write_confinement_and_symlink_rejection(box):
    def snapshot():
        return {str(p): p.read_bytes() for p in box.root.rglob('*') if p.is_file()
                and not p.is_relative_to(box.root / 'state')}
    before = snapshot()
    box.watch.dry_run = True
    box.tick()
    box.advance(10801)
    box.tick()
    assert snapshot() == before
    store = box.watch.store
    with pytest.raises(ValueError): store.text('../outside', 'no')
    store.path('evil').symlink_to(box.root / 'recipe.json')
    with pytest.raises(ValueError): store.text('evil', 'no')
    with pytest.raises(ValueError): store.append('evil', {})
    assert snapshot() == before


def test_replay_metrics_after_resume_do_not_mask_or_fake_progress(box):
    watch = box.watch
    Store(box.root / 'output').save('h3-last-metrics.json', dict(step=8))
    watch.s.update(highwater_step=9, attempt_step=5,
                   ignored_metrics=progress_evidence(box.root / 'output')['fingerprint'], last_progress_at=box.now)
    box.tick()
    assert watch.s['optimizer_step'] == 5  # old metric from failed attempt ignored
    Store(box.root / 'output').save('h3-last-metrics.json', dict(step=5))
    box.advance(60)
    box.monitor()
    box.tick()
    assert watch.s['last_progress_at'] == box.now  # replay update 6 counts as liveness
    assert watch.s['highwater_step'] == 9  # but cannot authorize same-checkpoint retry


def test_error_classification(box):
    log = box.watch.store.path('training-' + box.config['process']['launch_id'] + '.log')
    for text, expected in [('NCCL watchdog timed out', 'nccl_watchdog_timeout'),
                           ('card monitor stale', 'card_monitor_stop'),
                           ('CUDA out of memory', 'correctness_or_permanent_error'),
                           ('unknown crash', 'unknown_exit'),
                           ('NCCL watchdog timed out; Adam counter differs from actual step', 'correctness_or_permanent_error'),
                           ('gradient not finite nonzero', 'correctness_or_permanent_error'),
                           ('single-cell graph/count/release audit failed', 'correctness_or_permanent_error')]:
        box.watch.store.text(log.name, text)
        assert box.watch.death_cause() == expected


def test_offline_fire_drill(tmp_path):
    root, results = run_drill(tmp_path / 'drills')
    assert len(results) == 9
    assert 'same checkpoint' in results['transient_then_repeated']
    assert (root / 'drill-results.json').exists()


def test_hps_grace_consecutive_ticks_and_wall_clock_jumps(box):
    box.tick()
    box.health = dict(ok=False, identity_changed=False, health=dict(status='unready', code=503))
    assert box.tick()['state'] == 'WATCHING'
    box.now += 864000  # wall time never consumes grace
    box.advance(599); box.monitor()
    assert box.tick()['state'] == 'WATCHING'
    box.health = dict(ok=True)
    box.tick()
    box.health = dict(ok=None, error='timeout')
    box.tick()
    box.advance(599); box.monitor()
    assert box.tick()['state'] == 'WATCHING'
    box.advance(1); box.monitor()
    assert box.tick()['state'] == 'STOPPING'
    assert box.watch.s['stop']['phase'] == 'pause'
    assert box.watch.s['stop']['cause'] == 'hps_failure'
    assert box.child.poll() is None


def test_hps_503_and_no_environment_proxy():
    import io
    import urllib.error
    import urllib.request
    expected = 'a' * 64
    error = urllib.error.HTTPError('http://unused', 503, 'unready', {},
        io.BytesIO(json.dumps(dict(status='unready', service_identity=expected)).encode()))
    with patch('urllib.request.build_opener', wraps=urllib.request.build_opener) as opener, \
            patch('urllib.request.OpenerDirector.open', side_effect=error):
        assert hps_check('http://unused', expected)['ok'] is False
        assert opener.call_args.args[0].proxies == {}


@pytest.mark.parametrize('name', ['monitor', 'disk', 'progress'])
def test_unknown_reads_debounce_never_imply_stall_or_disk_low(box, name):
    box.tick()
    functions = dict(monitor='monitor_check', disk='disk_check', progress='progress_evidence')
    with patch('ops.h3_formal_watch.watchdog.' + functions[name], side_effect=ValueError('bad JSON/read')):
        if name == 'progress':
            box.advance(20000)
        box.monitor()
        assert box.tick()['state'] == 'WATCHING'
        assert box.tick()['state'] == 'WATCHING'
        assert box.tick()['state'] == 'NEEDS_HUMAN'
    assert not box.watch.s['incidents'] and box.child.poll() is None


@pytest.mark.parametrize('failure', ['hps', 'metrics', 'monitor', 'relative_output', 'wrong_output', 'artifact'])
def test_arming_failure_never_signals(box, failure):
    if failure == 'hps': box.health = dict(ok=False)
    elif failure == 'metrics': (box.root / 'output' / 'h3-last-metrics.json').write_text('{broken')
    elif failure == 'monitor': (box.root / 'monitor' / 'health.json').unlink()
    elif failure == 'relative_output': box.watch.output = Path('output')
    else:
        recipe = read_json(box.root / 'recipe.json')
        if failure == 'wrong_output': recipe['launch']['output_dir'] = str(box.root)
        else: recipe['launch']['artifacts']['train'] = 'relative/input'
        Store(box.root).save('recipe.json', recipe)
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        send.assert_not_called()
    assert box.child.poll() is None


def test_restart_rearms_without_deadline_reset_and_boot_mismatch(box):
    box.tick()
    previous = box.watch.s['last_progress_mono']
    box.advance(10); box.monitor()
    box.watch = Watchdog(box.config, clock=lambda: box.now, monotonic_ns=lambda: box.ns)
    box.health = dict(ok=False)
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    assert box.watch.s['last_progress_mono'] == previous
    with patch.object(process, 'boot_id', return_value='different-boot'):
        watch = Watchdog(box.config, clock=lambda: box.now, monotonic_ns=lambda: box.ns)
        assert watch.s['state'] == 'NEEDS_HUMAN'


def test_monitor_sample_time_is_after_atomic_read(box):
    called = []
    def read(*args):
        called.append('read')
        return dict(healthy=True, monotonic_ns=100)
    def now():
        assert called == ['read']
        return 101
    with patch('ops.h3_formal_watch.watchdog.read_json', side_effect=read):
        assert monitor_check(box.root, now, 30)['ok']


def test_pause_deadline_extends_on_metrics_then_waits_for_checkpoint(box):
    box.tick()
    box.monitor(healthy=False)
    box.tick(); box.tick(); box.tick()
    watch = box.watch
    first = watch.s['stop']['deadline']
    box.advance(5000)
    Store(box.root / 'output').save('h3-last-metrics.json', dict(step=1))
    box.tick()
    assert watch.s['stop']['deadline'] == first + 5000
    assert watch.s['stop']['target_step'] == 2
    fake_checkpoint(box.root / 'output', 2)
    box.tick()
    assert watch.s['stop']['checkpoint_saved'] == 2
    assert watch.s['stop']['deadline'] == watch.mono() + .1
    box.log('H3SafeBoundaryPause')
    box.die()
    assert box.tick()['state'] == 'RECOVERING'
    assert watch.s['incidents'][-1]['cause'] == 'watchdog_checkpoint_stop'


def test_no_progress_term_immediate_and_not_transient(box):
    box.tick()
    box.advance(10800); box.monitor()
    assert box.tick()['state'] == 'STOPPING'
    assert box.watch.s['stop']['phase'] == 'term'
    assert not (box.root / 'state' / 'pause-request').exists()
    box.child.wait(timeout=5)
    assert box.tick()['state'] == 'NEEDS_HUMAN'


def test_operator_pause_is_never_removed_or_resumed(box):
    box.tick()
    box.watch.store.text('pause-request', 'operator')  # same symlink but NO STOPPING ownership
    box.log('NCCL watchdog timed out')
    assert box.tick()['state'] == 'WATCHING'
    box.die()
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    assert box.watch.s['incidents'][-1]['cause'] == 'operator_pause'
    assert (box.root / 'output' / 'PAUSE_REQUESTED').read_text() == 'operator'
    assert not box.watch.s['resumes']


def test_human_state_keeps_checks_and_acknowledgement_rearms_only(box):
    box.tick()
    box.watch.human('review needed')
    box.health = dict(ok=False, identity_changed=True)
    box.advance(60); box.monitor()
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        assert box.watch.s['last_checks']['hps']['identity_changed']
        assert box.watch.s['heartbeat_at'] == box.now
        send.assert_not_called()
    box.watch.store.save('acknowledge.json', dict(token=box.watch.s['human_token'], reason='reviewed'))
    box.health = dict(ok=True)
    assert box.tick()['state'] == 'ARMING'
    assert box.tick()['state'] == 'WATCHING'


def test_failed_unit_adoption_does_not_suppress_human_observations(box):
    box.watch.s['process'] = None
    box.config['training_unit'] = 'h3-training-unavailable.service'
    with patch.object(process, 'capture_unit', side_effect=RuntimeError('manager unavailable')), \
            patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        box.advance(60); box.monitor(degraded=True)
        box.health = dict(ok=False, identity_changed=True)
        state = box.tick()
        assert state['state'] == 'NEEDS_HUMAN' and state['heartbeat_at'] == box.now
        assert state['last_checks']['process']['ok'] is None
        assert state['last_checks']['monitor']['degraded']
        assert state['last_checks']['hps']['identity_changed']
        assert state['last_checks']['progress']['ok']
        send.assert_not_called()


def test_checkpoint_worker_does_not_block_heartbeat(box):
    import threading
    entered, release = threading.Event(), threading.Event()
    box.tick()
    fake_checkpoint(box.root / 'output')
    box.log('NCCL watchdog timed out'); box.die()
    assert box.tick()['state'] == 'RECOVERING'
    real = latest_checkpoint
    def slow(output, verify_files=False, report=lambda **kw: None):
        if verify_files:
            report(files_done=0, files_total=1, bytes_done=128)
            entered.set()
            assert release.wait(5)
        return real(output, verify_files, report)
    try:
        with patch('ops.h3_formal_watch.watchdog.latest_checkpoint', side_effect=slow):
            assert box.tick()['state'] == 'RECOVERING'
            assert entered.wait(2)
            box.advance(60); box.monitor()
            state = box.tick()
            assert state['heartbeat_at'] == box.now and state['recovery_phase'] == 'verifying'
            assert state['verification_progress']['bytes_done'] == 128
            assert not state['resumes']
    finally:
        release.set()
        box.watch.worker.join(5)


def test_long_history_is_archived_but_status_and_reload_are_bounded(box):
    for i in range(1100):
        box.watch.incident('unknown_exit', dict(index=i, details='x' * 2048))
    box.watch.persist()
    status_path = box.root / 'state' / 'status.json'
    assert status_path.stat().st_size < 128 * 1024
    assert len(read_json(status_path)['incidents']) == 32
    archives = list((box.root / 'state').glob('incidents.jsonl*'))
    assert len(archives) >= 3
    assert sum(len(p.read_text().splitlines()) for p in archives) == 1100
    assert all(p.stat().st_size <= 1024 * 1024 for p in archives)
    Watchdog(box.config, clock=lambda: box.now, monotonic_ns=lambda: box.ns)


@pytest.mark.parametrize('files,available', [(0, 0), (10, -1), (-1, 4), (10, 2**64 - 1)])
def test_unknown_inode_sentinels_never_become_disk_low(box, files, available):
    stat = SimpleNamespace(f_bavail=100, f_frsize=4096, f_favail=available, f_files=files)
    with patch('os.statvfs', return_value=stat):
        assert disk_check(box.root, 1, 10)['ok'] is None
        assert disk_check(box.root, 999999, 10)['ok'] is False  # independently known byte shortage


@pytest.mark.parametrize('argv', [['--monitor-dir', 'relative-monitor'], ['--config=relative-no-extension'], ['--monitor-dir']])
def test_arming_checks_all_config_arguments_without_signalling(box, argv):
    box.config['launcher']['argv'].extend(argv)
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        send.assert_not_called()


def test_boundary_fault_without_pause_provision_needs_human(box):
    box.tick()
    (box.root / 'output' / 'PAUSE_REQUESTED').unlink()
    box.monitor(healthy=False)
    box.tick(); box.tick()
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        send.assert_not_called()


def test_successful_checkpoint_and_own_term_receipt_allowlist(box):
    box.tick()
    box.monitor(healthy=False)
    box.tick(); box.tick(); box.tick()
    fake_checkpoint(box.root / 'output')
    box.tick()
    box.advance(.2)
    box.tick()
    box.child.wait(5)
    record = box.watch.s['process']
    box.watch.store.save('exit-' + record['launch_id'] + '.json',
        dict(launch_id=record['launch_id'], pgid=record['pgid'], exit_code=-15))
    assert box.tick()['state'] == 'RECOVERING'
    assert box.watch.s['incidents'][-1]['cause'] == 'watchdog_checkpoint_stop'
    box.log('NCCL watchdog timed out\nRuntimeError: Adam state audit error')
    assert box.watch.death_cause() == 'correctness_or_permanent_error'


def test_resume_artifacts_rechecked_and_original_paths_unchanged(box):
    box.tick()
    fake_checkpoint(box.root / 'output')
    (box.root / 'FAKE-RUN-ONLY').unlink()
    with pytest.raises(ValueError, match='existing absolute'):
        box.watch.make_resume_spec(dict(path=str(box.root / 'output' / 'checkpoints' / 'step-1')), 'missing')


# Observer/notifier tests are entirely offline; the only subprocesses below are
# local Python helpers used to verify real argv/timeout/output-cap behavior.
from ops.h3_formal_watch.commands import run_command
from ops.h3_formal_watch.notifier import FeishuNotifier
from ops.h3_formal_watch.observer import (Observer, daily_key, deadman, diagnosis,
                                        fetch, pi_argv, ssh_argv)
from ops.h3_formal_watch.snapshot import snapshot as remote_snapshot


def observer_config(root):
    return dict(state_dir=str(root / 'observer'), ssh_host='ubtroot@gpu-alias',
                remote_status='/fake/state/status.json', remote_events='/fake/state/events.jsonl',
                remote_logs=['/fake/train.log'], timezone='Asia/Shanghai', daily_time='09:00',
                report_dir=str(root / 'reports'), progress_log_path=str(root / 'progress.md'),
                notifier=dict(alert_user_open_id='ou_fake', identity='bot'))


def remote(now=0, state='WATCHING', **fields):
    return dict(status=dict(dict(heartbeat_at=now, state=state, optimizer_step=7,
                                 incidents=[], resumes=[]), **fields), events=[], logs={}, remote_time=now)


class FakeNotifier:
    def __init__(self): self.calls = []
    def send(self, text, key): self.calls.append((text, key))


def pi_result(text='训练尚无新进展；建议人工核对日志。'):
    return dict(stdout=json.dumps(dict(type='message_end', message=dict(role='assistant',
                content=[dict(type='text', text=text)], stopReason='stop')), ensure_ascii=False) + '\n', stderr='')


def test_notifier_chinese_and_special_characters_real_argv():
    text = '中文告警：暂停“训练”！\n引号"\'，反斜杠\\，$HOME; $(touch 不可执行) `whoami` 😀\t尾'
    calls = []
    def runner(argv, **kw):
        calls.append(argv)
        # Actually cross a subprocess argv boundary, without lark or a shell.
        reply = run_command([sys.executable, '-c', 'import json,sys; print(json.dumps(sys.argv[1:],ensure_ascii=False))',
                             *argv[1:]], **kw)
        received = json.loads(reply['stdout'])
        assert received[received.index('--text') + 1] == text
        return dict(stdout='{"ok":true}', stderr='')
    notifier = FeishuNotifier(dict(alert_user_open_id='ou_fake', identity='bot'), runner=runner)
    notifier.send(text, 'safe-id')
    assert calls[0][calls[0].index('--as') + 1] == 'bot'
    assert calls[0][calls[0].index('--user-id') + 1] == 'ou_fake'
    assert '--chat-id' not in calls[0]
    assert calls[0][calls[0].index('--text') + 1] == text
    with pytest.raises(ValueError): FeishuNotifier(dict(chat_id='oc_fake', identity='default'))
    with pytest.raises(RuntimeError):
        FeishuNotifier(dict(alert_user_open_id='ou_fake', identity='bot'), runner=lambda *a, **k: dict(stdout='{"ok":false}')).send(text, 'k')


def test_notifier_dry_run_no_execution(capsys):
    with patch('subprocess.Popen') as spawn:
        notifier = FeishuNotifier(dict(alert_user_open_id='ou_fake', identity='bot'), dry_run=True)
        notifier.send('中文，特殊 $() " \\', 'key')
        spawn.assert_not_called()
    printed = json.loads(capsys.readouterr().out)
    assert '--dry-run' in printed['notifier_argv']
    assert '中文' in printed['notifier_argv'][printed['notifier_argv'].index('--text') + 1]


@pytest.mark.parametrize('status,now,since,expected', [
    (None, 1799, 0, []), (None, 1800, 0, ['unreachable']),
    (dict(state='WATCHING', heartbeat_at=0), 600, None, []),
    (dict(state='WATCHING', heartbeat_at=0), 601, None, ['heartbeat_stale']),
    (dict(state='WATCHING', heartbeat_at=0), 1800, 0, ['heartbeat_stale', 'unreachable']),
    (dict(state='DONE', heartbeat_at=0), 100000, 0, ['unreachable']),
    (dict(state='NEEDS_HUMAN', heartbeat_at=0), 601, None, ['heartbeat_stale']),
    (dict(state='WATCHING', heartbeat_at=1000), 0, None, ['clock_skew'])])
def test_deadman_logic(status, now, since, expected):
    assert deadman(status, now, unreachable_age=None if since is None else now - since) == expected


def test_ssh_read_only_argv_and_retry_bounds(tmp_path):
    import base64
    import shlex
    config = observer_config(tmp_path)
    config['remote_logs'] = ['/fake/中文; $(bad) "log"']
    argv = ssh_argv(config)
    assert argv[0] == 'ssh' and 'StrictHostKeyChecking=yes' in argv
    remote_command = shlex.split(argv[-1])
    request = json.loads(base64.b64decode(remote_command[-1]))
    assert request['logs'] == config['remote_logs']
    assert remote_command[:2] == ['python3', '-c']
    assert "open('rb')" in remote_command[2] and "open('w'" not in remote_command[2]
    calls, sleeps = [], []
    def failing(argv, **kw):
        calls.append((argv, kw))
        raise TimeoutError('fake unreachable')
    with pytest.raises(RuntimeError, match='SSH fetch failed'):
        fetch(config, runner=failing, sleep=sleeps.append)
    assert len(calls) == 3 and sleeps == [10, 20]
    assert all(kw['timeout'] == 30 for _, kw in calls)
    def good(*a, **kw): return dict(stdout=json.dumps(remote()), stderr='')
    assert fetch(config, runner=good)['status']['state'] == 'WATCHING'
    config['ssh_host'] = '-oProxyCommand=bad'
    with pytest.raises(ValueError): ssh_argv(config)


def test_remote_snapshot_bounds_partial_json_and_no_writes(tmp_path):
    state = tmp_path / 'state.json'
    events = tmp_path / 'events.jsonl'
    log = tmp_path / 'log'
    state.write_text(json.dumps(remote()['status']))
    events.write_text('broken partial\n' + json.dumps(dict(id='a', kind='incident')) + '\n{"incomplete":')
    log.write_text('中文' * 20000)
    before = {p: p.read_bytes() for p in (state, events, log)}
    snap = remote_snapshot(dict(status=str(state), events=str(events), logs=[str(log), str(tmp_path / 'missing')]))
    assert snap['events'] == [dict(id='a', kind='incident')]
    assert len(snap['logs'][str(log)].encode()) <= 16390  # UTF-8 tail may begin at a split code point
    assert str(tmp_path / 'missing') in snap['errors']
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(ValueError):
        remote_snapshot(dict(status=str(state), events=str(events), logs=['x'] * 5))


def test_observer_once_per_event_fresh_pi_and_persistence(tmp_path):
    config = observer_config(tmp_path)
    notify, runs = FakeNotifier(), []
    def runner(argv, **kw):
        runs.append((argv, kw))
        return pi_result()
    obs = Observer(config, clock=lambda: 0, runner=runner, notifier=notify)
    snap = remote(incidents=[dict(id='i1', cause='monitor_stop')])
    obs.tick(snap)
    assert len(runs) == 1 and len(notify.calls) == 2
    args, kwargs = runs[0]
    assert '--no-session' in args and '--no-extensions' in args and '--no-tools' in args
    assert '--no-mcp' in args and '--no-context-files' in args
    assert args[args.index('--model') + 1] == 'gpt-6-astra'
    assert args[args.index('--provider') + 1] == 'openai-codex'
    assert args[args.index('--thinking') + 1] == 'high'
    assert args[-2] == '-p' and args[-1].startswith('@')
    brief = Path(args[-1][1:]).read_text()
    assert '禁止运行任何命令' in brief and 'BEGIN_EVIDENCE_JSON' in brief
    assert kwargs['timeout'] == 180 and kwargs['cwd'] == config['state_dir']
    obs.tick(snap)
    restored = Observer(config, clock=lambda: 60, runner=runner, notifier=notify)
    restored.tick(snap)
    assert len(runs) == 1 and len(notify.calls) == 2
    assert (Path(config['state_dir']) / 'remote-status.json').exists()
    assert list(Path(config['state_dir']).glob('analysis-*.jsonl'))
    report = Path(config['report_dir']) / 'INCIDENTS.md'
    assert report.read_text().count('训练尚无新进展') == 1
    assert all(str(report) in text and '训练尚无新进展' not in text for text, _ in notify.calls)


def test_observer_terminal_state_resume_and_daily_analysis(tmp_path):
    import datetime as dt
    config = observer_config(tmp_path)
    now = [dt.datetime(2026, 11, 3, 1, 0, tzinfo=dt.timezone.utc).timestamp()]  # 09:00 Shanghai
    assert daily_key(now[0] - 1, 'Asia/Shanghai', '09:00') is None
    assert daily_key(now[0], 'Asia/Shanghai', '09:00') == 'daily:2026-11-03'
    notify, runs = FakeNotifier(), []
    def runner(*a, **kw): runs.append(a); return pi_result()
    obs = Observer(config, clock=lambda: now[0], runner=runner, notifier=notify)
    snap = remote(now[0], 'NEEDS_HUMAN', resumes=[dict(launch_id='r1', step=5, performed=True)])
    snap['events'] = [dict(kind='auto_resume', launch_id='r1')]
    obs.tick(snap)
    assert len(runs) == 3  # one resume (dedup status/tail), one state, one daily
    obs.tick(snap)
    assert len(runs) == 3
    obs.tick(remote(now[0], 'DONE'))
    assert len(runs) == 4
    now[0] += 86400
    obs.tick(remote(now[0], 'DONE'))
    assert len(runs) == 5
    daily = (Path(config['report_dir']) / 'DAILY.md').read_text()
    assert daily.count('## 2026-11-03\n') == daily.count('## 2026-11-04\n') == 1
    assert daily.count('训练尚无新进展') == 2
    assert len(Path(config['progress_log_path']).read_text().splitlines()) == 2
    assert len(notify.calls) == 6  # alerts + short report pointers for resume/NEEDS_HUMAN/DONE only
    assert all('每日' not in text and '训练尚无新进展' not in text for text, _ in notify.calls)
    assert (Path(config['report_dir']) / 'INCIDENTS.md').read_text().count('只读诊断') == 3


def test_observer_unreachable_and_rate_limit_recovery(tmp_path):
    notify, now = FakeNotifier(), [0]
    config = observer_config(tmp_path)
    config['daily_time'] = '23:59'
    def unreachable(*args): raise RuntimeError('SSH failure (state unknown)')
    obs = Observer(config, clock=lambda: now[0], monotonic=lambda: now[0], fetcher=unreachable, notifier=notify,
                   runner=lambda *a, **kw: pytest.fail('no LLM for dead-man only'))
    obs.tick()
    assert not notify.calls
    now[0] = 1800
    obs.tick()
    assert len(notify.calls) == 1
    obs.tick()
    now[0] = 2100
    obs.tick()
    assert len(notify.calls) == 1
    now[0] = 5400
    obs.tick()
    assert len(notify.calls) == 2
    obs.tick(remote(now[0]))
    assert obs.s['unreachable_since'] is None
    now[0] += 4000
    obs.tick(remote(now[0]))
    assert len(notify.calls) == 2


def test_failed_notifier_rate_limited_and_analysis_never_replayed(tmp_path):
    now, calls, analyses = [0], [], []
    class Broken:
        def send(self, text, key):
            calls.append((text, key))
            raise RuntimeError('fake missing scope')
    def broken_analysis(*a, **kw):
        analyses.append(a)
        raise TimeoutError('fake analysis timeout')
    config = observer_config(tmp_path)
    obs = Observer(config, clock=lambda: now[0], monotonic=lambda: now[0], runner=broken_analysis, notifier=Broken())
    snap = remote(incidents=[dict(id='i1', cause='process_died')])
    with pytest.raises(RuntimeError, match='undelivered'):
        obs.tick(snap)
    assert len(analyses) == 1 and len(calls) == 2
    with pytest.raises(RuntimeError, match='undelivered'):
        obs.tick(snap)
    assert len(calls) == 2
    now[0] += 300
    with pytest.raises(RuntimeError, match='undelivered'):
        obs.tick(remote(now[0], incidents=snap['status']['incidents']))
    assert len(calls) == 4 and len(analyses) == 1
    assert calls[0][1] == calls[2][1]
    assert obs.s['jobs']['incident:i1']['analysis_status'] == 'failed'


def test_crash_claim_never_replays_pi(tmp_path):
    config = observer_config(tmp_path)
    obs = Observer(config, clock=lambda: 0, notifier=FakeNotifier(),
                   runner=lambda *a, **kw: pytest.fail('claimed analysis must not run again'))
    obs.queue('incident:i', '人工检查')
    obs.s['jobs']['incident:i']['analysis_status'] = 'claimed'
    obs.persist()
    obs.tick(remote())
    assert obs.s['jobs']['incident:i']['analysis_status'] == 'unknown'


def test_pi_json_lf_framing_and_error_handling():
    text = '中文\u2028段落\u2029继续'
    assert diagnosis(pi_result(text)['stdout']) == text
    with pytest.raises(ValueError): diagnosis('{"type":"tool_execution_start"}\n')
    with pytest.raises(ValueError): diagnosis('{"type":"agent_settled","aborted":true}\n')
    with pytest.raises(ValueError): diagnosis('{"type":"session"}\n')


def test_observer_dry_run_no_commands_and_analysis_budget(tmp_path, capsys):
    config = observer_config(tmp_path)
    config['max_analyses_per_tick'] = 1
    obs = Observer(config, dry_run=True, clock=lambda: 0,
                   runner=lambda *a, **kw: pytest.fail('dry run must not spawn'))
    obs.tick()
    assert 'ssh_argv' in capsys.readouterr().out
    snap = remote(incidents=[dict(id='a', cause='process_died'), dict(id='b', cause='monitor_stop')])
    obs.tick(snap)
    assert [j['analysis_status'] for j in obs.s['jobs'].values()].count('dry_run') == 1
    assert 'pi_argv' in capsys.readouterr().out
    obs.tick(snap)
    assert all(j['analysis_status'] == 'dry_run' for j in obs.s['jobs'].values())


def test_bounded_helper_timeout_output_limit_and_nonzero():
    with pytest.raises(TimeoutError):
        run_command([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=.05)
    with pytest.raises(ValueError, match='size limit'):
        run_command([sys.executable, '-c', 'print("x" * 10000)'], max_bytes=100)
    with pytest.raises(RuntimeError, match='exited 3'):
        run_command([sys.executable, '-c', 'raise SystemExit(3)'])
    assert run_command([sys.executable, '-c', 'print("中文")'])['stdout'] == '中文\n'


def test_snapshot_current_resumed_log_and_path_safety(tmp_path):
    state = Store(tmp_path)
    status = remote()['status']
    status['process'] = dict(launch_id='resume-123')
    state.save('status.json', status)
    state.text('events.jsonl', '')
    state.text('training-resume-123.log', '恢复后的日志')
    request = dict(status=str(tmp_path / 'status.json'), events=str(tmp_path / 'events.jsonl'), logs=[])
    snap = remote_snapshot(request)
    assert snap['logs'][str(tmp_path / 'training-resume-123.log')] == '恢复后的日志'
    status['process']['launch_id'] = '../../outside'
    state.save('status.json', status)
    assert not remote_snapshot(request)['logs']


def test_exclusive_locks_and_config_identity(tmp_path):
    store = Store(tmp_path / 'state')
    with store.lock('watchdog.lock'):
        with pytest.raises(BlockingIOError):
            store.lock('watchdog.lock')
    config = observer_config(tmp_path)
    obs = Observer(config, clock=lambda: 0, notifier=FakeNotifier())
    obs.persist()
    with pytest.raises(ValueError, match='config/dry-run'):
        Observer(dict(config, daily_time='10:00'), clock=lambda: 0, notifier=FakeNotifier())
    with pytest.raises(ValueError, match='bounded observer'):
        Observer(dict(config, ssh_timeout_seconds=float('inf')), notifier=FakeNotifier())
    with pytest.raises(ValueError, match='max_analyses'):
        Observer(dict(config, max_analyses_per_tick=100), notifier=FakeNotifier())


def test_snapshot_identity_mismatch_counts_as_unreachable(tmp_path):
    config = dict(observer_config(tmp_path), expected_config_id='approved-config')
    obs = Observer(config, clock=lambda: 0, monotonic=lambda: 0, notifier=FakeNotifier())
    obs.tick(remote(config_id='other-run'))
    assert obs.s['unreachable_since'] == 0
    assert 'identity changed' in obs.s['fetch_error']
    assert not (Path(config['state_dir']) / 'remote-status.json').exists()


def test_descendant_sessions_orphans_and_signal_order(box):
    rec = box.config['process']
    leader = dict(rec['members'][0], ppid=1)
    rank = dict(leader, pid=leader['pid'] + 100000, ppid=leader['pid'],
                pgid=leader['pid'] + 100000, sid=leader['pid'] + 100000, token='rank-start')
    tree = {p['pid']: p for p in (leader, rank)}
    rec['members'] = process.inspect(rec, tree)['members']
    orphan = dict(rank, ppid=1)
    assert process.inspect(rec, {rank['pid']: orphan})['alive']
    with patch.object(process, 'processes', return_value=tree), patch.object(os, 'killpg') as kill:
        process.signal_group(rec, signal.SIGKILL)
    assert [c.args[0] for c in kill.call_args_list] == [rank['pgid'], leader['pgid']]
    with pytest.raises(ValueError, match='reused'):
        process.inspect(rec, {rank['pid']: dict(orphan, token='reused')})


def test_cgroup_ownership_all_sessions_and_reuse():
    info = dict(InvocationID='unique', ControlGroup='/user.slice/fake', ActiveState='active',
                KillMode='control-group', Restart='no')
    with patch.object(process, 'unit_info', return_value=info), \
            patch.object(process, 'cgroup_members', return_value=[111, 222]), \
            patch.object(process, 'run_command') as command:
        record = process.capture_unit('h3-training-test.service')
        assert process.inspect(record)['alive']
        process.signal_group(record, signal.SIGTERM)
        assert command.call_args.args[0] == ['systemctl', '--user', 'kill', '--kill-whom=all',
                                             '--signal=SIGTERM', record['unit']]
        process.signal_group(record, signal.SIGKILL)
        assert command.call_args.args[0][-2:] == ['stop', record['unit']]
        info['InvocationID'] = 'reused'
        with pytest.raises(ValueError, match='invocation changed'):
            process.signal_group(record, signal.SIGKILL)
    with pytest.raises(ValueError): process.unit_name('hps.service')


def test_exited_unit_without_cgroup_is_dead_but_surviving_ranks_are_live():
    info = dict(InvocationID='unique', ControlGroup='', ActiveState='active', SubState='exited',
                KillMode='control-group', Restart='no')
    with patch.object(process, 'unit_info', return_value=info):
        record = process.capture_unit('h3-training-complete.service')
        assert not process.inspect(record)['alive']
        info['ControlGroup'] = '/user.slice/remaining-ranks'
        with patch.object(process, 'cgroup_members', return_value=[222]):
            assert process.inspect(record)['alive']
        info.update(ControlGroup='', SubState='running')
        with pytest.raises(ValueError, match='unknown'):
            process.inspect(record)


def test_cgroup_prune_race_requires_exited_same_invocation():
    info = dict(InvocationID='unique', ControlGroup='/user.slice/pruned', ActiveState='active',
                SubState='exited', KillMode='control-group', Restart='no')
    with patch.object(process, 'unit_info', return_value=info), \
            patch.object(process, 'cgroup_members', side_effect=FileNotFoundError):
        record = process.capture_unit('h3-training-pruned.service')
        assert not process.inspect(record)['alive']
        info['SubState'] = 'running'
        with pytest.raises(ValueError, match='disappeared'):
            process.inspect(record)
    with pytest.raises(ValueError, match='unsafe'):
        process.cgroup_members('/')


def test_systemd_launch_claim_and_no_replay(tmp_path):
    store = Store(tmp_path / 'state')
    store.save('spec.json', dict(launch_id='test', unit='h3-training-test.service'))
    with patch.object(process, 'run_command') as command, \
            patch.object(process, 'capture_unit', return_value=dict(unit='test')):
        process.supervise(store.root, 'spec.json')
        argv = command.call_args.args[0]
        assert argv[:2] == ['systemd-run', '--user']
        assert '--property=KillMode=control-group' in argv and '--property=Restart=no' in argv
        assert 'execute' in argv
        with pytest.raises(FileExistsError): process.supervise(store.root, 'spec.json')
        assert command.call_count == 1


def test_observer_fresh_remote_time_never_local_or_cached_age(tmp_path):
    config = dict(observer_config(tmp_path), daily_time='23:59')
    local, mono, notify = [10000000], [0], FakeNotifier()
    def unreachable(*args): raise RuntimeError('network unavailable')
    obs = Observer(config, clock=lambda: local[0], monotonic=lambda: mono[0],
                   fetcher=unreachable, notifier=notify)
    obs.tick(remote(20))  # remote clock far behind VPS: no false stale alert
    assert not notify.calls
    mono[0] = 600
    obs.tick()  # cached heartbeat is NEVER aged
    local[0] += 864000  # wall jump is irrelevant to unreachable window
    mono[0] += 1799
    obs.tick()
    assert not notify.calls
    mono[0] += 1
    obs.tick()
    assert len(notify.calls) == 1 and 'unreachable' in notify.calls[0][0]
    assert 'deadman:heartbeat_stale' not in obs.s['jobs']


def test_observer_remote_monotonic_beats_remote_wall_jump(tmp_path):
    notify = FakeNotifier()
    obs = Observer(observer_config(tmp_path), clock=lambda: 0, notifier=notify)
    snap = remote(0, boot_id='gpu-boot', heartbeat_mono=100)
    snap.update(remote_time=999999, remote_monotonic=101, remote_boot_id='gpu-boot')
    obs.tick(snap)
    assert not notify.calls
    snap['remote_monotonic'] = 701
    obs.tick(snap)
    assert len(notify.calls) == 1 and 'heartbeat_stale' in notify.calls[0][0]


def test_undelivered_resolved_alert_drains_after_restart(tmp_path):
    config = dict(observer_config(tmp_path), daily_time='23:59')
    mono, calls = [0], []
    class OnceBroken:
        def send(self, text, key):
            calls.append((text, key))
            raise RuntimeError('delivery unavailable')
    obs = Observer(config, clock=lambda: 0, monotonic=lambda: mono[0], notifier=OnceBroken())
    stale = remote(0)
    stale['remote_time'] = 601
    with pytest.raises(RuntimeError, match='undelivered'):
        obs.tick(stale)
    assert len(calls) == 1
    mono[0] = 300
    notify = FakeNotifier()
    obs = Observer(config, clock=lambda: 0, monotonic=lambda: mono[0], notifier=notify)
    obs.tick(remote(1000))  # condition resolved but old outbox still must deliver
    assert len(notify.calls) == 1 and notify.calls[0] == calls[0]
    obs.tick(remote(1000))
    assert len(notify.calls) == 1


@pytest.mark.parametrize('reply', ['[]', 'null', 'true', '"unexpected"', '123'])
def test_notifier_non_object_reply_fails_cleanly(reply):
    notifier = FeishuNotifier(dict(alert_user_open_id='ou_fake', identity='bot'), runner=lambda *a, **kw: dict(stdout=reply))
    with pytest.raises(RuntimeError, match='acknowledge'):
        notifier.send('test', 'key')


@pytest.mark.parametrize('text', ['@/private/file', '--option', '  @file', '\n-option'])
def test_notifier_leading_file_or_option_syntax_is_inert(text):
    notifier = FeishuNotifier(dict(alert_user_open_id='ou_fake', identity='bot'))
    argv = notifier.argv(text, 'key')
    assert argv[argv.index('--text') + 1] == '[H3] ' + text


def test_observer_busy_lock_returns_zero_without_work(tmp_path, capsys):
    from ops.h3_formal_watch.observer import main
    config = observer_config(tmp_path)
    Store(tmp_path).save('config.json', config)
    store = Store(config['state_dir'])
    with store.lock('observer.lock'), patch.object(sys, 'argv', ['observer', str(tmp_path / 'config.json')]), \
            patch('ops.h3_formal_watch.observer.Observer') as construct:
        assert main() == 0
        construct.assert_not_called()
    assert 'lock busy' in capsys.readouterr().out


@pytest.mark.parametrize('change', [
    {'chat_id': 'oc_group'}, {'chat_id': None}, {'group_id': 'group'},
    {'user_id': 'ou_other'}, {'targets': ['ou_one', 'ou_two']},
    {'identity': 'user'}, {'identity': None}, {'alert_user_open_id': None},
    {'alert_user_open_id': ['ou_one']}, {'alert_user_open_id': ''},
    {'alert_user_open_id': 'oc_group'}, {'alert_user_open_id': 'ou_one,ou_two'},
    {'alert_user_open_id': 'ou_one ou_two'}, {'alert_user_open_id': 'ou_one\n'},
    {'alert_user_open_id': '@file'}, {'alert_user_open_id': '--option'},
])
def test_notification_policy_rejects_targets_before_any_work(tmp_path, change):
    config = observer_config(tmp_path)
    config['notifier'].update(change)
    notify = FakeNotifier()
    with patch('subprocess.Popen') as spawn:
        with pytest.raises(ValueError, match='alerts require'):
            FeishuNotifier(config['notifier'])
        with pytest.raises(ValueError, match='alerts require'):
            Observer(config, notifier=notify)
        spawn.assert_not_called()
    assert not notify.calls
    assert 'alerts require' in (Path(config['state_dir']) / 'notifier-errors.jsonl').read_text()


def test_observer_rejects_top_level_chat_and_rechecks_before_send(tmp_path):
    config = observer_config(tmp_path)
    with pytest.raises(ValueError, match='chat/group'):
        Observer(dict(config, chat_id='oc_group'), notifier=FakeNotifier())
    notify = FakeNotifier()
    obs = Observer(config, clock=lambda: 0, notifier=notify)
    config['notifier']['chat_id'] = 'oc_later'
    obs.notify('incident:mutated', 'must not send')
    assert not notify.calls
    assert 'alerts require' in obs.s['notices']['incident:mutated']['error']
    assert len((Path(config['state_dir']) / 'notifier-errors.jsonl').read_text().splitlines()) == 2
    notifier = FeishuNotifier(dict(alert_user_open_id='ou_fake', identity='bot'))
    notifier.config['identity'] = 'user'
    with pytest.raises(ValueError):
        notifier.send('must not send', 'key')


def test_daily_only_is_local_even_in_dry_run_and_legacy_outbox(tmp_path):
    import datetime as dt
    now = dt.datetime(2026, 11, 3, 1, tzinfo=dt.timezone.utc).timestamp()
    config = observer_config(tmp_path)
    notify = FakeNotifier()
    obs = Observer(config, dry_run=True, clock=lambda: now, notifier=notify,
                   runner=lambda *a, **kw: pytest.fail('dry run must not execute'))
    obs.s['notices']['analysis:daily:2026-11-02'] = dict(sent=False, text='old daily diagnosis')
    obs.s['notices']['daily:2026-11-02'] = dict(sent=False, text='old daily summary')
    obs.tick(remote(now))
    assert not notify.calls
    assert all(entry['suppressed'] for entry in obs.s['notices'].values())
    daily = Path(config['report_dir']) / 'DAILY.md'
    before = daily.read_text()
    assert '每日训练摘要' in before and 'Dry-run' in before and 'WATCHING' in before
    restored = Observer(config, dry_run=True, clock=lambda: now, notifier=notify)
    restored.tick(remote(now))
    assert daily.read_text() == before and not notify.calls
    assert len(Path(config['progress_log_path']).read_text().splitlines()) == 1
    assert not (Path(config['report_dir']) / 'INCIDENTS.md').exists()


def test_local_report_failure_does_not_block_alert_outbox(tmp_path):
    config = observer_config(tmp_path)
    notify, runs = FakeNotifier(), []
    def runner(*args, **kwargs):
        runs.append(args)
        return pi_result('private incident diagnosis')
    obs = Observer(config, clock=lambda: 0, notifier=notify, runner=runner)
    obs.s['notices']['deadman:resolved'] = dict(sent=False, text='resolved alert retry')
    snap = remote(incidents=[dict(id='i1', cause='process_died')])
    with patch.object(obs.reports, 'record', side_effect=OSError('report disk unavailable')):
        with pytest.raises(RuntimeError, match='local report writes pending'):
            obs.tick(snap)
    assert len(notify.calls) == 2 and len(runs) == 1
    assert obs.s['notices']['deadman:resolved']['sent']
    assert 'analysis:incident:i1' not in obs.s['notices']
    obs.tick(snap)
    assert len(notify.calls) == 3 and len(runs) == 1
    assert 'private incident diagnosis' in obs.reports.path().read_text()
    assert all('private incident diagnosis' not in text for text, _ in notify.calls)


def test_legacy_incident_diagnosis_outbox_is_redacted(tmp_path):
    notify = FakeNotifier()
    obs = Observer(observer_config(tmp_path), clock=lambda: 0, notifier=notify)
    obs.s['notices']['analysis:incident:old'] = dict(sent=False, text='full PRIVATE model response')
    obs.drain_notices()
    assert len(notify.calls) == 1
    assert 'PRIVATE' not in notify.calls[0][0]
    assert str(obs.reports.path()) in notify.calls[0][0]


def test_report_atomic_append_bounded_sections_and_crash_replay(tmp_path):
    from ops.h3_formal_watch.reports import Reports, SECTION_LIMIT
    reports = Reports(tmp_path / 'reports', tmp_path / 'progress.md')
    reports.record('daily:2026-11-03', '2026-11-03', 'first summary', '中文诊断', daily=True)
    daily = reports.path(True)
    before = daily.read_text()
    real_replace = os.replace
    replacements = []
    def replace(source, dest):
        replacements.append((Path(source), Path(dest)))
        assert Path(source).parent == Path(dest).parent and Path(source).name.startswith('.write-')
        real_replace(source, dest)
    with patch('ops.h3_formal_watch.common.os.replace', side_effect=OSError('disk error')):
        with pytest.raises(OSError):
            reports.record('daily:2026-11-04', '2026-11-04', 'next', daily=True)
    assert daily.read_text() == before
    with patch('ops.h3_formal_watch.reports.append_progress', side_effect=OSError('progress write failed')):
        with pytest.raises(OSError):
            reports.record('daily:2026-11-04', '2026-11-04', 'summary', '长' * 20000, daily=True)
    after = daily.read_text()
    assert len(after) - len(before) <= SECTION_LIMIT
    with patch('ops.h3_formal_watch.common.os.replace', side_effect=replace):
        reports.record('daily:2026-11-04', '2026-11-04', 'summary', '长' * 20000, daily=True)
        reports.record('incident:i1', '2026-11-04', 'incident', 'private diagnosis')
        reports.record('incident:i1', '2026-11-04', 'incident', 'private diagnosis')
        reports.record('incident:i2', '2026-11-04', 'second incident', 'second diagnosis')
    assert daily.read_text() == after
    assert len((tmp_path / 'progress.md').read_text().splitlines()) == 2
    assert reports.path().read_text().count('private diagnosis') == 1
    assert reports.path().read_text().count('second diagnosis') == 1
    assert replacements


@pytest.mark.parametrize('failure', [
    'ValueError: sampled/replayed old score mismatch beyond the declared batched-rollout tolerance',
    'RuntimeError: novel application invariant violated', 'AssertionError: audit/graph invalid',
    'RuntimeError: nonfinite result'])
def test_failure_records_veto_transient_retry(box, failure):
    box.log('[rank0]: ' + failure + '\nNCCL watchdog timed out')
    assert box.watch.death_cause() == 'correctness_or_permanent_error'


@pytest.mark.parametrize('error', ['ValueError: invalid application state', 'RuntimeError: missing cache'])
def test_unrecognized_error_cannot_borrow_same_line_nccl_signature(box, error):
    box.log(error + '; NCCL watchdog timed out')
    assert box.watch.death_cause() == 'correctness_or_permanent_error'


def test_launcher_identity_json_does_not_veto_nccl(box):
    box.log('{"identity": "approved", "status": "ready"}\n'
            '[rank1]: RuntimeError: NCCL watchdog timed out')
    assert box.watch.death_cause() == 'nccl_watchdog_timeout'


@pytest.mark.parametrize('listed', [False, True])
def test_exact_tree_nested_metadata_names(box, listed):
    import hashlib
    from ops.h3_formal_watch.common import digest
    out = box.root / 'output'
    fake_checkpoint(out)
    store = Store(out / 'checkpoints' / 'step-1')
    Store(store.root / 'nested').text('progress.json', 'nested data')
    if listed:
        tree = store.load('state_tree.json')['tree']
        tree['files'].append(dict(path='nested/progress.json', size=11,
                                 sha256=hashlib.sha256(b'nested data').hexdigest()))
        store.save('state_tree.json', dict(tree=tree, state_tree_id=digest(tree)))
        marker = store.load('complete.json')
        marker['state_tree_id'] = digest(tree)
        store.save('complete.json', marker)
        assert latest_checkpoint(out, True)['step'] == 1
    else:
        with pytest.raises(ValueError, match='exact files'):
            latest_checkpoint(out, True)


def test_hps_unicode_producer_digest():
    import hashlib
    payload = dict(identity_facts={'path': '/模型/权重'}, video_contract={'frames': 124})
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode('utf-8')).hexdigest()
    class Response:
        code = 200
        def read(self, n): return json.dumps(dict(payload, status='ready', service_identity=identity)).encode()
        def __enter__(self): return self
        def __exit__(self, *args): pass
    with patch('urllib.request.OpenerDirector.open', return_value=Response()):
        assert hps_check('http://unused', identity)['ok']


def test_proc_scan_skips_disappearing_pid():
    paths = [Path('/proc/111/stat'), Path('/proc/222/stat')]
    raw = '222 (fake) S 1 222 222 ' + '0 ' * 15 + '123 0'
    with patch.object(Path, 'exists', return_value=True), patch.object(Path, 'glob', return_value=paths), \
            patch.object(Path, 'read_text', side_effect=[ProcessLookupError('gone'), raw]):
        assert list(process.processes()) == [222]


def test_ownership_unknown_is_debounced(box):
    box.tick()
    with patch.object(process, 'inspect', side_effect=OSError('temporary scan failure')), \
            patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'WATCHING'
        assert box.tick()['state'] == 'WATCHING'
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        send.assert_not_called()


def test_verification_deadline_survives_restart(box):
    box.tick()
    box.watch.s.update(state='RECOVERING', verification_started=box.watch.mono() - 7200)
    box.watch.persist()
    restored = Watchdog(box.config, clock=lambda: box.now, monotonic_ns=lambda: box.ns)
    with patch('threading.Thread'):
        with pytest.raises(ValueError, match='deadline'):
            restored.verify_async()
    assert restored.s['verification_started'] == box.watch.s['verification_started']


def test_verification_done_on_deadline_is_accepted(box):
    watch = box.watch
    watch.worker = SimpleNamespace(is_alive=lambda: False, join=lambda **kw: None)
    watch.worker_result = dict(done=True, checkpoint={'step': 1})
    watch.s['verification_started'] = watch.mono() - watch.thresholds['verification_wait_seconds']
    assert watch.verify_async()['done']
    watch.worker = None


def test_new_launch_adoption_clears_stale_pause_ownership(box):
    box.tick()
    watch = box.watch
    old = watch.s['process']['launch_id']
    watch.s.update(state='LAUNCHING', pending_launch=dict(launch_id='new', mono=watch.mono()),
                   stop=dict(pause_issued=True, launch_id=old, pause_token='old'))
    watch.s['resumes'].append(dict(launch_id='new', step=1))
    watch.store.save('process-new.json', dict(watch.s['process'], launch_id='new'))
    assert box.tick()['state'] == 'WATCHING'
    assert watch.s['stop'] is None
    watch.store.text('pause-request', 'operator pause for new launch')
    assert watch.operator_pause()


def test_pause_ownership_requires_launch_and_token(box):
    box.tick()
    watch = box.watch
    watch.s['stop'] = dict(pause_issued=True, launch_id='old', pause_token='old')
    watch.store.text('pause-request', 'operator')
    assert watch.operator_pause()


def test_cancelled_operator_pause_latches_human(box):
    box.tick()
    box.watch.store.text('pause-request', 'operator')
    box.tick()
    box.watch.store.path('pause-request').unlink()
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        assert 'cancel' in box.watch.s['reason']
        send.assert_not_called()


def test_acknowledge_dead_process_cannot_resume(box):
    box.tick()
    box.log('NCCL watchdog timed out')
    box.watch.human('manual review')
    old = box.watch.s['human_token']
    box.die()
    box.watch.store.save('acknowledge.json', dict(token=old, reason='rearm only'))
    box.tick()
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    assert box.watch.s['human_token'] != old
    assert not box.watch.s['resumes']


def test_unpublished_pause_replayed_before_escalation(box):
    box.tick()
    watch = box.watch
    # Crash after intent persistence, before target publication.
    class Crash(BaseException): pass
    real_text = watch.store.text
    def crash(name, text):
        if name == 'pause-request': raise Crash()
        return real_text(name, text)
    with patch.object(watch.store, 'text', side_effect=crash):
        with pytest.raises(Crash):
            watch.request_stop('disk_low', watch.s['last_checks'])
    box.advance(13000); box.monitor()
    box.watch = Watchdog(box.config, clock=lambda: box.now, monotonic_ns=lambda: box.ns)
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'STOPPING'
        assert box.watch.store.path('pause-request').exists()
        assert box.watch.s['stop']['pause_published']
        assert box.watch.s['stop']['deadline'] > box.watch.mono()
        send.assert_not_called()


def pending_unknown(box):
    box.tick()
    watch = box.watch
    watch.s['pending_launch'] = dict(launch_id='unknown', mono=watch.mono())
    watch.s['resumes'].append(dict(launch_id='unknown', step=1))
    spec = dict(launch_id='unknown', unit='h3-training-unknown.service', argv=['/fake'], cwd='/fake', env={})
    watch.store.save('launch-unknown.json', spec)
    watch.store.text('claimed-unknown.json', '{}')
    watch.human('launch outcome unknown')
    watch.persist()
    return watch, spec


def test_unknown_launch_reconcile_adopt_or_abandon(box):
    from ops.h3_formal_watch.common import digest
    watch, spec = pending_unknown(box)
    info = dict(InvocationID='ours', ControlGroup='/fake', KillMode='control-group', Restart='no')
    watch.store.save('execution-unknown.json', dict(launch_id='unknown', unit=spec['unit'],
        invocation_id='ours', boot_id=watch.boot, spec_id=digest(spec)))
    with patch.object(process, 'unit_info', return_value=info), \
            patch.object(process, 'cgroup_members', return_value=[123]):
        watch.reconcile_launch('adopt', 'verified live unit')
    assert watch.s['pending_launch'] is None and watch.s['stop'] is None
    assert watch.s['process']['invocation_id'] == 'ours'
    assert watch.s['state'] == 'NEEDS_HUMAN'  # separate rearm, never submission


def test_unknown_launch_abandon_prevents_delayed_submission(box):
    watch, spec = pending_unknown(box)
    box.die()
    with patch.object(process, 'unit_info', return_value={'LoadState': 'not-found'}):
        watch.reconcile_launch('abandon', 'verified absent unit')
    assert watch.s['pending_launch'] is None
    assert watch.store.path('abandoned-unknown.json').exists()
    with patch.object(process, 'run_command') as command:
        with pytest.raises(ValueError, match='abandoned'):
            process.supervise(watch.store.root, 'launch-unknown.json')
        command.assert_not_called()


def test_unknown_launch_reconcile_rejects_foreign_unit(box):
    watch, spec = pending_unknown(box)
    with patch.object(process, 'unit_info', return_value=dict(InvocationID='foreign', ControlGroup='/fake',
                      KillMode='control-group', Restart='no')):
        with pytest.raises(ValueError, match='receipt'):
            watch.reconcile_launch('adopt', 'not proven')
    assert watch.s['pending_launch']


@pytest.mark.parametrize('observe_ack', [False, True])
def test_human_alerts_follow_token_not_polled_state(tmp_path, observe_ack):
    notify = FakeNotifier()
    obs = Observer(observer_config(tmp_path), clock=lambda: 0, notifier=notify,
                   runner=lambda *a, **kw: pi_result())
    obs.tick(remote(state='NEEDS_HUMAN', human_token='first'))
    if observe_ack:
        obs.tick(remote(state='WATCHING'))
    obs.tick(remote(state='NEEDS_HUMAN', human_token='second'))
    obs.tick(remote(state='NEEDS_HUMAN', human_token='second'))
    assert len(notify.calls) == 4  # alert + local diagnosis pointer for EACH token
    assert len({key for _, key in notify.calls}) == 4


def test_persisted_deadman_job_gets_outbox_after_recovery(tmp_path):
    config = dict(observer_config(tmp_path), daily_time='23:59')
    obs = Observer(config, clock=lambda: 0, notifier=FakeNotifier())
    obs.queue('deadman:heartbeat_stale', 'detected before crash', analyze=False, repeat=True)
    obs.persist()  # crash before notify/outbox publication
    notify = FakeNotifier()
    restored = Observer(config, clock=lambda: 0, notifier=notify)
    restored.tick(remote())
    assert len(notify.calls) == 1 and 'detected before crash' in notify.calls[0][0]
    restored.tick(remote())
    assert len(notify.calls) == 1


def test_repeat_alert_window_key_stable_across_failed_retry(tmp_path):
    config = dict(observer_config(tmp_path), alert_repeat_seconds=10, notify_retry_seconds=1)
    now, notify = [0], FakeNotifier()
    obs = Observer(config, clock=lambda: now[0], monotonic=lambda: now[0], notifier=notify)
    obs.notify('deadman:stale', 'alert', repeat=True)
    first = notify.calls[-1][1]
    now[0] = 10
    class Broken:
        def send(self, text, key):
            notify.calls.append((text, key))
            raise RuntimeError('ambiguous delivery')
    obs.notifier = Broken()
    obs.notify('deadman:stale', 'alert', repeat=True)
    second = notify.calls[-1][1]
    assert second != first
    now[0] = 100  # retries keep the original repeat attempt's key, even in another window
    obs = Observer(config, clock=lambda: now[0], monotonic=lambda: now[0], notifier=notify)
    obs.drain_notices()
    assert notify.calls[-1][1] == second


def test_observer_history_bounded_pending_and_dedup_preserved(tmp_path):
    config = observer_config(tmp_path)
    obs = Observer(config, clock=lambda: 0, notifier=FakeNotifier())
    for i in range(220):
        key = 'incident:' + str(i)
        obs.s['jobs'][key] = dict(label=key, analyze=True, alert=True, repeat=False,
            created_at=0, analysis_status='complete', diagnosis='中' * 4000,
            report_written=True, finished=True)
        obs.s['notices'][key] = dict(sent=True, text='old alert')
    for i in range(100):
        obs.queue('pending:' + str(i), 'pending', analyze=False)
        obs.s['notices']['pending:' + str(i)] = dict(sent=False, text='must deliver')
    obs.persist()
    # Even before compaction, pending/diagnosis payloads cannot enlarge the JSON header.
    assert obs.store.path('observer-state.json').stat().st_size < 65536
    obs.compact()
    restored = Observer(config, clock=lambda: 0, notifier=FakeNotifier())
    assert len(restored.s['jobs']) <= 132
    assert len(restored.s['notices']) <= 164
    restored.queue('incident:0', 'must not replay analysis')
    assert 'incident:0' not in restored.s['jobs']  # durable indexed dedup survives archive eviction
    assert sum(not n['sent'] for n in restored.s['notices'].values()) == 100
    assert list(restored.store.root.glob('observer-history.jsonl*'))


def test_legacy_observer_migration_cannot_replay_newer_claims(tmp_path):
    from ops.h3_formal_watch.common import digest
    config = observer_config(tmp_path)
    store = Store(config['state_dir'])
    legacy = dict(config_id=digest(config), dry_run=False, boot_id=process.boot_id(),
                  unreachable_since=None, last_state=None, fetch_error=None,
                  jobs={'incident:i': dict(label='incident', analyze=True, alert=True,
                        repeat=False, created_at=0, analysis_status='pending')},
                  notices={'incident:i': dict(sent=False, text='original')})
    store.save('observer-state.json', legacy)
    obs = Observer(config, clock=lambda: 0, notifier=FakeNotifier())
    obs.s['jobs']['incident:i']['analysis_status'] = 'claimed'
    obs.s['notices']['incident:i'].update(sent=True, repeat_window=7)
    obs.ledger.flush()  # simulate crash before legacy JSON header was replaced
    restored = Observer(config, clock=lambda: 0, notifier=FakeNotifier())
    assert restored.s['jobs']['incident:i']['analysis_status'] == 'claimed'
    assert restored.s['notices']['incident:i']['sent']
    assert restored.s['notices']['incident:i']['repeat_window'] == 7


def test_human_observations_only_on_health_change(box):
    box.tick()
    box.watch.human('review')
    box.tick()
    for _ in range(5):
        box.advance(60); box.monitor(); box.tick()
    events = [json.loads(line) for line in box.watch.store.path('events.jsonl').read_text().splitlines()]
    assert sum(e['kind'] == 'human_observation' for e in events) == 1
    box.health = dict(ok=False, identity_changed=True)
    box.tick()
    events = [json.loads(line) for line in box.watch.store.path('events.jsonl').read_text().splitlines()]
    assert sum(e['kind'] == 'human_observation' for e in events) == 2


def test_rotated_segments_have_retention_cap(tmp_path):
    store = Store(tmp_path)
    for i in range(30):
        store.append_rotated('events.jsonl', dict(i=i, payload='x' * 60), limit=100)
    assert len(list(tmp_path.glob('events.jsonl*'))) <= 9  # active plus eight archives
    assert json.loads(store.path('events.jsonl').read_text())['i'] == 29


def test_observer_unit_covers_bounded_command_budget(tmp_path):
    from ops.h3_formal_watch import observer
    text = Path('ops/h3_formal_watch/systemd/h3-formal-observer.service').read_text()
    minutes = int(next(line.split('=')[1][:-3] for line in text.splitlines() if line.startswith('TimeoutStartSec=')))
    # Max configured SSH/analysis timeouts alone already exceed the old 30min.
    assert minutes * 60 > 3 * 125 + 30 + 4 * 605
    assert minutes * 60 > 3 * 125 + 30 + 4 * 605 + observer.MAX_NOTIFICATIONS_PER_TICK * 125
    notify = FakeNotifier()
    obs = Observer(observer_config(tmp_path), clock=lambda: 0, notifier=notify)
    for i in range(100):
        obs.s['notices'][str(i)] = dict(sent=False, text='queued')
    with pytest.raises(RuntimeError, match='undelivered'):
        obs.drain_notices()
    assert len(notify.calls) == observer.MAX_NOTIFICATIONS_PER_TICK


def test_progress_log_uses_append_preserves_concurrent_edits_and_mode(tmp_path):
    from ops.h3_formal_watch.reports import Reports
    progress = tmp_path / 'progress.md'
    progress.write_text('operator history\n')
    progress.chmod(0o640)
    inode = progress.stat().st_ino
    reports = Reports(tmp_path / 'reports', progress)
    real_open, real_write = os.open, os.write
    append_fds, injected = set(), []
    def opening(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == progress and flags & os.O_APPEND:
            append_fds.add(fd)
        return fd
    def writing(fd, raw):
        if fd in append_fds and not injected:
            injected.append(True)
            with progress.open('a') as stream:
                stream.write('concurrent operator edit\n')
        return real_write(fd, raw)
    with patch('os.open', side_effect=opening), patch('os.write', side_effect=writing):
        reports.record('daily:date', 'date', 'summary', daily=True)
    assert append_fds and injected
    assert progress.stat().st_ino == inode and progress.stat().st_mode & 0o777 == 0o640
    assert 'concurrent operator edit' in progress.read_text()
    reports.record('daily:date', 'date', 'summary', daily=True)
    assert progress.read_text().count('H3 daily summary') == 1
    assert not list(tmp_path.glob('.h3-progress*')) and not list(tmp_path.glob('.write-*'))


def test_failed_atomic_report_write_leaves_no_temp(tmp_path):
    store = Store(tmp_path)
    with patch('os.replace', side_effect=OSError('failed replace')):
        with pytest.raises(OSError):
            store.text('report.md', 'new section')
    assert not list(tmp_path.glob('.write-*'))


def test_user_unit_contract():
    units = Path(__file__).resolve().parents[1] / 'ops/h3_formal_watch/systemd'
    watchdog = (units / 'h3-formal-watchdog.service').read_text()
    observer = (units / 'h3-formal-observer.service').read_text()
    timer = (units / 'h3-formal-observer.timer').read_text()
    assert 'Restart=on-failure' in watchdog and 'RestartSec=30' in watchdog
    assert 'KillMode=process' in watchdog and 'After=default.target' not in watchdog
    assert 'Type=oneshot' in observer
    assert 'OnUnitActiveSec=5min' in timer and 'Persistent=true' in timer and 'OnCalendar=' in timer
