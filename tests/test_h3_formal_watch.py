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
    assert progress_evidence(out)['step'] == 0
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
    box.monitor(**{field: value})
    assert not monitor_check(box.root / 'monitor', box.ns, 30)['ok']
    assert box.tick()['state'] == 'STOPPING'
    assert box.watch.s['incidents'][0]['cause'] == 'monitor_stop'


def test_monitor_degraded_is_nonfatal_and_logged(box):
    box.monitor(degraded=True, consecutive_failures=2, max_seconds=1000000, max_bytes=12345)
    assert box.tick()['state'] == 'WATCHING'
    assert 'monitor_degraded' in (box.root / 'state' / 'events.jsonl').read_text()
    assert box.watch.s['last_checks']['monitor']['health']['max_bytes'] == 12345
    box.advance(30)
    assert not monitor_check(box.root / 'monitor', box.ns, 30)['ok']


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
        with patch('urllib.request.urlopen', return_value=Response(row)) as fetch:
            result = hps_check('http://unused', identity)
            assert result['ok'] is ok and result['identity_changed'] is changed
            assert fetch.call_args.kwargs['timeout'] == 8
    with patch('urllib.request.urlopen', side_effect=TimeoutError):
        with pytest.raises(TimeoutError):
            hps_check('http://unused', identity)


def test_disk_bytes_and_inodes(box):
    with patch('os.statvfs', return_value=SimpleNamespace(f_bavail=100, f_frsize=10, f_favail=5, f_files=20)):
        assert disk_check(box.root, 1000, 5)['ok']
        assert not disk_check(box.root, 1001, 5)['ok']
        assert not disk_check(box.root, 1000, 6)['ok']
    with patch('os.statvfs', return_value=SimpleNamespace(f_bavail=100, f_frsize=10, f_favail=5, f_files=0)):
        assert not disk_check(box.root, 0, 0)['ok']  # unknown inode capacity is not healthy
    with patch('ops.h3_formal_watch.watchdog.disk_check', return_value=dict(ok=False)):
        assert box.tick()['state'] == 'STOPPING'
        assert box.watch.s['incidents'][-1]['cause'] == 'disk_low'


def test_identity_failure_is_permanent(box):
    fake_checkpoint(box.root / 'output')
    box.health = dict(ok=False, identity_changed=True)
    assert box.tick()['state'] == 'STOPPING'
    box.child.wait(timeout=5)
    assert box.tick()['state'] == 'RECOVERING'
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
    watch.config['pause_via_symlink'] = True
    (box.root / 'output' / 'PAUSE_REQUESTED').symlink_to(box.root / 'state' / 'pause-request')
    box.monitor(healthy=False)
    with patch.object(process, 'signal_group') as send:
        assert box.tick()['state'] == 'STOPPING'
        assert (box.root / 'output' / 'PAUSE_REQUESTED').exists()
        send.assert_not_called()
        box.advance(.2)
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


def test_no_pause_link_falls_back_and_dry_run_prints(box, capsys):
    box.watch.config['pause_via_symlink'] = True
    box.watch.dry_run = True
    box.monitor(healthy=False)
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
    ckpt = dict(path=str(box.root / 'output' / 'checkpoints' / 'step-1'), step=1)
    assert watch.resume_allowed(None) == 'no complete checkpoint'
    watch.s['resumes'] = [dict(at=box.now, step=1, highwater_step=5)]
    watch.s['highwater_step'] = 5
    assert 'same checkpoint' in watch.resume_allowed(ckpt)
    watch.s['highwater_step'] = 6
    assert watch.resume_allowed(ckpt) is None
    watch.s['resumes'].append(dict(at=box.now, step=5, highwater_step=6))
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
    box.die()
    assert box.tick()['state'] == 'RECOVERING'
    box.monitor(healthy=False)
    box.advance(1801)
    assert box.tick()['state'] == 'NEEDS_HUMAN'
    watch = box.watch
    watch.s['state'] = 'LAUNCHING'
    watch.s['pending_launch'] = dict(launch_id='missing', at=box.now - 61)
    with patch('subprocess.Popen') as spawn:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        spawn.assert_not_called()


def test_corrupt_checkpoint_never_resumes(box):
    fake_checkpoint(box.root / 'output')
    Store(box.root / 'output' / 'checkpoints' / 'step-1').text('fake-state', 'corrupt')
    box.die()
    box.tick()
    with patch.object(process, 'inspect', return_value=dict(alive=False, members=[])), patch('subprocess.Popen') as spawn:
        assert box.tick()['state'] == 'NEEDS_HUMAN'
        spawn.assert_not_called()


def test_no_checkpoint_and_nonzero_without_success(box):
    box.die()
    assert box.tick()['state'] == 'RECOVERING'
    assert box.watch.s['incidents'][-1]['cause'] == 'process_died'
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
    watch.s.update(highwater_step=9, attempt_step=5, progress_mtime_floor=box.now + 100,
                   last_progress_at=box.now)
    box.tick()
    assert watch.s['optimizer_step'] == 5  # old metric from failed attempt ignored
    watch.s['progress_mtime_floor'] = box.now - 100
    Store(box.root / 'output').save('h3-last-metrics.json', dict(step=5))
    box.advance(60)
    box.monitor()
    box.tick()
    assert watch.s['last_progress_at'] == box.now  # replay update 6 counts as liveness
    assert watch.s['highwater_step'] == 9  # but cannot authorize same-checkpoint retry


def test_error_classification(box):
    log = box.watch.store.path('training-' + box.config['process']['launch_id'] + '.log')
    for text, expected in [('NCCL watchdog timed out', 'nccl_timeout'),
                           ('card monitor stale', 'monitor_stop'),
                           ('CUDA out of memory', 'permanent_training_error'),
                           ('unknown crash', 'process_died')]:
        box.watch.store.text(log.name, text)
        assert box.watch.death_cause() == expected


def test_offline_fire_drill(tmp_path):
    root, results = run_drill(tmp_path / 'drills')
    assert len(results) == 5
    assert 'same checkpoint' in results['transient_then_repeated']
    assert (root / 'drill-results.json').exists()


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
                                 incidents=[], resumes=[]), **fields), events=[], logs={})


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
    (dict(state='DONE', heartbeat_at=0), 100000, 0, []),
    (dict(state='NEEDS_HUMAN', heartbeat_at=0), 601, None, ['heartbeat_stale']),
    (dict(state='WATCHING', heartbeat_at=1000), 0, None, ['clock_skew'])])
def test_deadman_logic(status, now, since, expected):
    assert deadman(status, now, since) == expected


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
    obs = Observer(config, clock=lambda: now[0], fetcher=unreachable, notifier=notify,
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
    obs = Observer(config, clock=lambda: now[0], runner=broken_analysis, notifier=Broken())
    snap = remote(incidents=[dict(id='i1', cause='process_died')])
    obs.tick(snap)
    assert len(analyses) == 1 and len(calls) == 2
    obs.tick(snap)
    assert len(calls) == 2
    now[0] += 300
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
    obs = Observer(config, clock=lambda: 0, notifier=FakeNotifier())
    obs.tick(remote(config_id='other-run'))
    assert obs.s['unreachable_since'] == 0
    assert 'identity changed' in obs.s['fetch_error']
    assert not (Path(config['state_dir']) / 'remote-status.json').exists()


def test_user_unit_contract():
    units = Path(__file__).resolve().parents[1] / 'ops/h3_formal_watch/systemd'
    watchdog = (units / 'h3-formal-watchdog.service').read_text()
    observer = (units / 'h3-formal-observer.service').read_text()
    timer = (units / 'h3-formal-observer.timer').read_text()
    assert 'Restart=on-failure' in watchdog and 'RestartSec=30' in watchdog
    assert 'KillMode=process' in watchdog and 'After=default.target' not in watchdog
    assert 'Type=oneshot' in observer
    assert 'OnUnitActiveSec=5min' in timer and 'Persistent=true' in timer and 'OnCalendar=' in timer
