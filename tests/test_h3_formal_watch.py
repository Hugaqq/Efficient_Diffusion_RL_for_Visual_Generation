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
