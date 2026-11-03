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
                notifier=dict(chat_id='oc_fake', identity='bot'))


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
    notifier = FeishuNotifier(dict(chat_id='oc_fake', identity='user'), runner=runner)
    notifier.send(text, 'safe-id')
    assert calls[0][calls[0].index('--as') + 1] == 'user'
    assert calls[0][calls[0].index('--chat-id') + 1] == 'oc_fake'
    assert calls[0][calls[0].index('--text') + 1] == text
    with pytest.raises(ValueError): FeishuNotifier(dict(chat_id='oc_fake', identity='default'))
    with pytest.raises(RuntimeError):
        FeishuNotifier(dict(chat_id='oc_fake', identity='bot'), runner=lambda *a, **k: dict(stdout='{"ok":false}')).send(text, 'k')


def test_notifier_dry_run_no_execution(capsys):
    with patch('subprocess.Popen') as spawn:
        notifier = FeishuNotifier(dict(chat_id='oc_fake', identity='bot'), dry_run=True)
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
    notifier = FeishuNotifier(dict(chat_id='oc_fake', identity='bot'), runner=lambda *a, **kw: dict(stdout=reply))
    with pytest.raises(RuntimeError, match='acknowledge'):
        notifier.send('test', 'key')


@pytest.mark.parametrize('text', ['@/private/file', '--option', '  @file', '\n-option'])
def test_notifier_leading_file_or_option_syntax_is_inert(text):
    notifier = FeishuNotifier(dict(chat_id='oc_fake', identity='bot'))
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


def test_user_unit_contract():
    units = Path(__file__).resolve().parents[1] / 'ops/h3_formal_watch/systemd'
    watchdog = (units / 'h3-formal-watchdog.service').read_text()
    observer = (units / 'h3-formal-observer.service').read_text()
    timer = (units / 'h3-formal-observer.timer').read_text()
    assert 'Restart=on-failure' in watchdog and 'RestartSec=30' in watchdog
    assert 'KillMode=process' in watchdog and 'After=default.target' not in watchdog
    assert 'Type=oneshot' in observer
    assert 'OnUnitActiveSec=5min' in timer and 'Persistent=true' in timer and 'OnCalendar=' in timer
