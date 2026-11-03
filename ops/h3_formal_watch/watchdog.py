"""Deterministic GPU-host watchdog. No LLM, SSH, scorer restart or training edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from .common import Store, digest, read_json, regular, tail
from . import process

DEFAULTS = dict(interval_seconds=60, monitor_max_age=30, max_update_seconds=10800,
                disk_floor_bytes=20 * 1024**3, inode_floor=20000,
                stop_wait_seconds=900, term_wait_seconds=60, kill_wait_seconds=60,
                recovery_wait_seconds=1800, launch_wait_seconds=60, max_auto_resumes=2)
TRANSIENT = {'process_died', 'nccl_timeout', 'monitor_stop'}


def checkpoint(path, verify_files=False):
    """Read publication metadata; full shard hashing only on resume, never each tick.

    Mirrors manager.py's complete/contract/progress/tree envelope hashes without
    importing torch. The formal launcher remains the authoritative semantic gate.
    """
    if path.resolve() != path or not re.fullmatch(r'step-[1-9][0-9]*', path.name):
        raise ValueError('invalid checkpoint path')
    blobs = {}
    for name in ('complete', 'checkpoint_contract', 'progress', 'state_tree'):
        file = path / (name + '.json')
        if not regular(file):
            raise ValueError('non-regular checkpoint metadata')
        blobs[name] = read_json(file, 16 * 1024 * 1024)
    c, contract, progress, tree = (blobs[k] for k in
                                   ('complete', 'checkpoint_contract', 'progress', 'state_tree'))
    step = c['step']
    if (c['kind'] != 'visual_rl_checkpoint_complete' or type(step) is not int
            or path.name != f'step-{step}' or step < 1
            or progress['progress']['global_step'] != step
            or progress['progress']['next_optimizer_step'] != step):
        raise ValueError('checkpoint marker/progress mismatch')
    for envelope, field, payload in ((contract, 'checkpoint_contract_id', 'contract'),
                                     (progress, 'progress_id', 'progress'), (tree, 'state_tree_id', 'tree')):
        if c[field] != envelope[field] or digest(envelope[payload]) != c[field]:
            raise ValueError('checkpoint metadata digest mismatch')
    if verify_files:
        files = tree['tree']['files']
        if tree['tree']['schema_version'] != 1 or not files:
            raise ValueError('empty or unsupported state tree')
        names = []
        for row in files:
            name = row['path']
            if not name or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe shard path')
            file = path / name
            if not regular(file) or file.stat().st_size != row['size']:
                raise ValueError('missing/wrong size shard')
            sha = hashlib.sha256()
            with file.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    sha.update(chunk)
            if sha.hexdigest() != row['sha256']:
                raise ValueError('shard digest mismatch')
            names.append(name)
        actual = []
        for file in path.rglob('*'):
            if file.is_symlink():
                raise ValueError('symlink in checkpoint')
            if file.is_file() and file.name not in {k + '.json' for k in blobs}:
                actual.append(file.relative_to(path).as_posix())
        if names != sorted(set(names)) or sorted(actual) != names:
            raise ValueError('state tree does not cover exact files')
    return dict(path=str(path), step=step, complete=c)


def latest_checkpoint(output, verify_files=False):
    # latest.json is only a hint; rename may have committed before latest.json updated.
    paths = [p for p in (output / 'checkpoints').glob('step-*')
             if re.fullmatch(r'step-[1-9][0-9]*', p.name) and not p.is_symlink()
             and (p / 'complete.json').is_file()]
    if not paths:
        return None
    # A corrupt published newest marker is NEEDS_HUMAN, not silent rollback.
    return checkpoint(max(paths, key=lambda p: int(p.name[5:])), verify_files)


def progress_evidence(output):
    evidence = []
    metrics = output / 'h3-last-metrics.json'
    if metrics.exists():
        row = read_json(metrics)
        if type(row.get('step')) is not int or row['step'] < 0:
            raise ValueError('invalid metrics step')
        evidence.append(dict(step=row['step'] + 1, source=str(metrics), mtime=metrics.stat().st_mtime))
    ckpt = latest_checkpoint(output)
    if ckpt:
        evidence.append(dict(step=ckpt['step'], source=ckpt['path'],
                             mtime=(Path(ckpt['path']) / 'complete.json').stat().st_mtime))
    return max(evidence, key=lambda row: row['step']) if evidence else dict(step=0, source=None, mtime=None)


def monitor_check(directory, now_ns, max_age):
    health = read_json(Path(directory) / 'health.json')
    age = (now_ns - health.get('monotonic_ns', 0)) / 1e9
    return dict(ok=health.get('healthy') is True and 0 <= age < max_age, age_seconds=age,
                degraded=health.get('degraded', False), health=health)


def hps_check(url, expected, timeout=8):
    # Read even 503 bodies: a mismatched identity is always fatal, not merely unready.
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('oversize HPS health')
        health = json.loads(raw)
    identity = health.get('service_identity')
    changed = identity is not None and identity != expected
    facts = health.get('identity_facts')
    if facts is not None and digest(dict(identity_facts=facts, video_contract=health.get('video_contract'))) != identity:
        changed = True
    return dict(ok=response.code == 200 and health.get('status') == 'ready' and identity == expected
                and not changed, identity_changed=changed, health=health)


def disk_check(output, byte_floor, inode_floor):
    stat = os.statvfs(output)
    free, inodes = stat.f_bavail * stat.f_frsize, stat.f_favail if stat.f_files else None
    return dict(ok=free >= byte_floor and inodes is not None and inodes >= inode_floor,
                free_bytes=free, free_inodes=inodes)


class Watchdog:
    def __init__(self, config, *, dry_run=False, clock=time.time, monotonic_ns=time.monotonic_ns):
        self.config = config
        self.store = Store(config['state_dir'])
        self.output = Path(config['output_dir']).absolute()
        self.run_dir = Path(config['run_dir']).absolute()
        if self.store.root == self.output or self.store.root == self.run_dir:
            raise ValueError('state directory must be distinct from run/output directories')
        self.thresholds = {**DEFAULTS, **config.get('thresholds', {})}
        if any(type(v) not in (int, float) or not 0 <= v < float('inf') for v in self.thresholds.values()):
            raise ValueError('thresholds must be finite nonnegative numbers')
        if self.thresholds['interval_seconds'] <= 0 or type(self.thresholds['max_auto_resumes']) is not int:
            raise ValueError('invalid interval/resume count')
        if not re.fullmatch('[0-9a-f]{64}', config['hps_identity']):
            raise ValueError('expected full HPS SHA256')
        self.clock, self.monotonic_ns, self.dry_run = clock, monotonic_ns, dry_run
        identity = digest(config)
        saved = self.store.load('status.json')
        if saved and (saved['config_id'] != identity or saved['dry_run'] != dry_run):
            raise ValueError('configuration/mode changed: use a separately reconciled state directory')
        self.s = saved or dict(schema=1, config_id=identity, dry_run=dry_run, state='WATCHING',
                               heartbeat_at=clock(), last_progress_at=clock(), optimizer_step=0,
                               highwater_step=0, incidents=[], resumes=[], process=config.get('process'),
                               pending_launch=None, stop=None, last_checks={})
        if not self.s['process'] and not self.s['pending_launch']:
            self.s['process'] = self.store.load('initial-process.json')
        if not self.s['process'] and not self.s['pending_launch'] and config.get('training_unit'):
            self.s['process'] = process.capture_unit(config['training_unit'])
        if not self.s['process'] and not self.s['pending_launch']:
            raise ValueError('record initial training unit or process tree first')

    def persist(self):
        self.s['heartbeat_at'] = self.clock()
        self.store.save('status.json', self.s)

    def event(self, kind, **fields):
        self.store.append('events.jsonl', dict(dict(id=uuid.uuid4().hex, at=self.clock(), kind=kind), **fields))

    def state(self, name):
        if self.s['state'] != name:
            old = self.s['state']
            self.s['state'] = name
            self.event('state_change', old=old, state=name)

    def action(self, kind, **fields):
        self.event('would_' + kind if self.dry_run else kind, **fields)
        if self.dry_run:
            print(json.dumps(dict(action=kind, **fields), ensure_ascii=False))

    def human(self, reason):
        self.s['reason'] = reason
        self.state('NEEDS_HUMAN')

    def incident(self, cause, details):
        row = dict(id=uuid.uuid4().hex, at=self.clock(), cause=cause,
                   transient=cause in TRANSIENT, details=details)
        self.s['incidents'].append(row)
        self.event('incident', **row)
        return row

    def collect(self):
        checks = {}
        def check(name, work):
            try:
                checks[name] = work()
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                checks[name] = dict(ok=False, error=repr(exc))
        check('process', lambda: dict(ok=True, **process.inspect(self.s['process'])))
        if checks['process'].get('alive'):
            self.s['process']['members'] = checks['process']['members']
        check('monitor', lambda: monitor_check(self.config['monitor_dir'], self.monotonic_ns(),
                                              self.thresholds['monitor_max_age']))
        check('hps', lambda: hps_check(self.config['hps_healthz_url'], self.config['hps_identity']))
        check('disk', lambda: disk_check(self.output, self.thresholds['disk_floor_bytes'], self.thresholds['inode_floor']))
        check('progress', lambda: dict(ok=True, **progress_evidence(self.output)))
        p = checks['progress']
        if p['ok']:
            self.s['optimizer_step'] = p['step']
            eligible = not self.s.get('progress_mtime_floor') or (p['mtime'] or 0) >= self.s['progress_mtime_floor']
            if eligible and p['step'] > self.s.get('attempt_step', 0):
                self.s['attempt_step'] = p['step']
                self.s['highwater_step'] = max(p['step'], self.s['highwater_step'])
                self.s['last_progress_at'] = self.clock()
                self.event('progress', **p)
            if not eligible:
                self.s['optimizer_step'] = self.s.get('attempt_step', 0)
            p['seconds_without_progress'] = max(0, self.clock() - self.s['last_progress_at'])
            p['ok'] = p['seconds_without_progress'] < self.thresholds['max_update_seconds']
        degraded = checks['monitor'].get('degraded', False)
        if degraded != self.s['last_checks'].get('monitor', {}).get('degraded', False):
            self.event('monitor_degraded', degraded=degraded)
        self.s['last_checks'] = checks
        return checks

    def pause_supported(self):
        # Lifecycle only tests exists(). Operator provisions this dangling symlink.
        link = self.output / 'PAUSE_REQUESTED'
        return link.is_symlink() and link.resolve() == self.store.path('pause-request').absolute()

    def request_stop(self, cause, checks):
        self.incident(cause, checks)
        pause = self.config.get('pause_via_symlink', False) and self.pause_supported()
        self.s['stop'] = dict(phase='pause' if pause else 'term',
                             deadline=self.clock() + self.thresholds['stop_wait_seconds'])
        self.state('STOPPING')
        self.persist()  # intent durable before side effects
        if pause:
            self.action('pause_request')
            if not self.dry_run:
                self.store.text('pause-request', 'watchdog requested safe optimizer boundary\n')
        else:
            self.send(signal.SIGTERM)

    def send(self, sig):
        self.action('signal', owner=self.s['process'].get('unit', self.s['process'].get('pgid')), signal=sig.name)
        if not self.dry_run:
            process.signal_group(self.s['process'], sig)

    def stopped_tick(self, alive):
        if not alive:
            self.s['stop'] = None
            self.s['recovery_since'] = self.clock()
            self.state('RECOVERING')
            return
        stop = self.s['stop']
        if self.clock() < stop['deadline']:
            return
        if stop['phase'] == 'kill':
            self.human('training group survived SIGKILL; no further action')
            return
        sig = signal.SIGTERM if stop['phase'] == 'pause' else signal.SIGKILL
        stop.update(phase='term' if sig == signal.SIGTERM else 'kill', deadline=self.clock() +
                    self.thresholds['term_wait_seconds' if sig == signal.SIGTERM else 'kill_wait_seconds'])
        self.persist()
        self.send(sig)

    def exit_receipt(self):
        record = self.s['process']
        receipt = self.store.load('exit-' + record['launch_id'] + '.json')
        fields = ('unit', 'invocation_id', 'boot_id') if record.get('kind') == 'cgroup' else ('pgid',)
        if receipt and (receipt['launch_id'] != record['launch_id'] or any(receipt.get(k) != record[k] for k in fields)):
            raise ValueError('exit receipt identity mismatch')
        return receipt

    def complete(self):
        marker = read_json(self.output / 'SUCCESS')
        ckpt = latest_checkpoint(self.output)
        owner = read_json(self.output / 'h3-run.json')
        receipt = self.exit_receipt()
        return bool(receipt and type(receipt['exit_code']) is int and receipt['exit_code'] == 0 and ckpt
                    and marker['kind'] == 'visual_rl_h3_run_success' and marker['run_id'] == owner['run_id']
                    and marker['committed_steps'] == ckpt['step']
                    and marker['checkpoint_relative_path'] == 'checkpoints/step-' + str(ckpt['step'])
                    and all(marker[k] == ckpt['complete'][k] for k in
                            ('checkpoint_contract_id', 'progress_id', 'state_tree_id')))

    def death_cause(self):
        owned = self.store.path('training-' + self.s['process']['launch_id'] + '.log')
        log = owned if owned.is_file() else self.config.get('training_log', owned)
        text = tail(log).lower() if Path(log).is_file() else ''
        if any(word in text for word in ('identity mismatch', 'different h3 run', 'out of memory',
                                         'insufficient disk', 'refusing resume', 'digest mismatch')):
            return 'permanent_training_error'
        if 'nccl' in text and ('timeout' in text or 'timed out' in text):
            return 'nccl_timeout'
        if 'card monitor' in text and ('unhealthy' in text or 'stale' in text):
            return 'monitor_stop'
        return 'process_died'

    def resume_allowed(self, ckpt):
        now = self.clock()
        if not ckpt:
            return 'no complete checkpoint'
        if len([r for r in self.s['resumes'] if now - r['at'] < 86400]) >= self.thresholds['max_auto_resumes']:
            return '24-hour auto-resume limit'
        if any(r['step'] == ckpt['step'] and self.s['highwater_step'] <= r['highwater_step']
               for r in self.s['resumes']):
            return 'same checkpoint without new progress'
        if (self.output / 'PAUSE_REQUESTED').exists() and not self.pause_supported():
            return 'operator pause request still active'
        return None

    def make_resume_spec(self, ckpt, launch_id):
        launch = self.config['launcher']
        recipe = read_json(launch['recipe_json'])  # JSON is YAML; no PyYAML in ops
        if Path(recipe['launch']['output_dir']) != self.output:
            raise ValueError('resume recipe output differs from watched output (use absolute paths)')
        if recipe['launch']['expected']['hps_service_identity'] != self.config['hps_identity']:
            raise ValueError('resume recipe scorer identity differs from watchdog')
        # Only this launch binding changes. Original recipe/flags/env/cwd are immutable.
        recipe['launch']['resume_from'] = ckpt['path']
        name = 'resume-recipe-' + launch_id + '.json'
        argv = launch['argv']
        if argv.count('{resume_config}') != 1:
            raise ValueError('launcher argv must contain one standalone {resume_config}')
        self.store.save(name, recipe)
        return dict(launch_id=launch_id, unit='h3-training-' + launch_id + '.service',
                    argv=[str(self.store.path(name)) if a == '{resume_config}' else a for a in argv],
                    env=launch['env'], cwd=launch['cwd'])

    def resume(self, checks):
        incident = self.s['incidents'][-1]
        if not incident['transient']:
            self.human('non-transient incident: ' + incident['cause'])
            return
        if any(not checks[k]['ok'] for k in ('monitor', 'hps', 'disk')):
            if checks['hps'].get('identity_changed'):
                self.human('HPS identity changed during recovery')
            elif self.clock() - self.s['recovery_since'] >= self.thresholds['recovery_wait_seconds']:
                self.human('recovery prerequisites did not become healthy')
            return
        ckpt = latest_checkpoint(self.output, verify_files=True)
        reason = self.resume_allowed(ckpt)
        if reason:
            self.human(reason)
            return
        if self.dry_run:
            self.action('auto_resume', checkpoint=ckpt['path'])
            self.human('dry-run: resume suppressed')
            return
        launch_id = uuid.uuid4().hex
        spec = self.make_resume_spec(ckpt, launch_id)
        spec_name = 'launch-' + launch_id + '.json'
        self.store.save(spec_name, spec)
        # Crash hereafter must NEVER blindly relaunch. Reconcile receipt or require human.
        self.s['resumes'].append(dict(at=self.clock(), step=ckpt['step'],
                                     highwater_step=self.s['highwater_step'], launch_id=launch_id))
        self.s['pending_launch'] = dict(launch_id=launch_id, at=self.clock())
        self.state('LAUNCHING')
        self.persist()
        pause = self.store.path('pause-request')
        if pause.exists() and self.pause_supported():
            pause.rename(self.store.path('pause-archived-' + launch_id))
        self.action('resume_intent', checkpoint=ckpt['path'], launch_id=launch_id)
        self.start_supervisor(spec_name)

    def start_supervisor(self, spec_name):
        # Supervisor submits an independently owned training unit, never retries.
        root = Path(__file__).resolve().parents[2]
        self.supervisor = subprocess.Popen([sys.executable, '-m', 'ops.h3_formal_watch.process',
            'supervise', str(self.store.root), spec_name], cwd=root, env={**os.environ, 'PYTHONPATH': str(root)},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def tick(self):
        try:
            if self.s['state'] in ('DONE', 'NEEDS_HUMAN'):
                return self.s
            if self.s['pending_launch']:
                pending = self.s['pending_launch']
                record = self.store.load('process-' + pending['launch_id'] + '.json')
                if record:
                    self.s['process'] = record
                    self.s['pending_launch'] = None
                    self.s['last_progress_at'] = self.clock()
                    self.s['progress_mtime_floor'] = record['started_at']
                    self.s['attempt_step'] = self.s['resumes'][-1]['step']
                    self.s['resumes'][-1]['performed'] = True
                    self.state('WATCHING')
                    self.event('auto_resume', launch_id=record['launch_id'])
                elif self.clock() - pending['at'] >= self.thresholds['launch_wait_seconds']:
                    self.human('launch outcome unknown; no automatic replay')
                return self.s
            checks = self.collect()
            if not checks['process']['ok']:
                self.human('cannot prove training group identity; no signals/resume')
                return self.s
            alive = checks['process']['alive']
            if not alive and (self.output / 'SUCCESS').exists():
                if not self.exit_receipt():
                    since = self.s.setdefault('success_seen_at', self.clock())
                    if self.clock() - since < self.thresholds['launch_wait_seconds']:
                        return self.s  # allow the external supervisor to reap and publish
                if self.complete():
                    self.state('DONE')
                else:
                    self.human('SUCCESS lacks matching successful exit/checkpoint receipt')
                return self.s
            if self.s['state'] == 'STOPPING':
                # A second fatal condition cannot be laundered into a transient resume.
                if checks['hps'].get('identity_changed'):
                    self.s['incidents'][-1]['transient'] = False
                self.stopped_tick(alive)
                return self.s
            if self.s['state'] == 'RECOVERING':
                if alive:
                    self.human('training reappeared during recovery; no duplicate launch')
                else:
                    self.resume(checks)
                return self.s
            if not alive:
                self.incident(self.death_cause(), dict(exit=self.exit_receipt(), checks=checks))
                self.s['recovery_since'] = self.clock()
                self.state('RECOVERING')
                return self.s
            fatal = next((cause for name, cause in (('hps', 'hps_failure'), ('disk', 'disk_low'),
                         ('progress', 'no_progress'), ('monitor', 'monitor_stop')) if not checks[name]['ok']), None)
            if fatal:
                self.request_stop(fatal, checks)
            return self.s
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            self.event('watch_error', error=repr(exc))
            self.human('watchdog safety gate: ' + repr(exc))
            return self.s
        finally:
            self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    watch = Watchdog(read_json(args.config), dry_run=args.dry_run)
    with watch.store.lock('watchdog.lock'):
        while True:
            state = watch.tick()['state']
            if args.once or state == 'DONE':
                return
            # NEEDS_HUMAN continues heartbeating, but never performs another action.
            time.sleep(watch.thresholds['interval_seconds'])


if __name__ == '__main__':
    main()
