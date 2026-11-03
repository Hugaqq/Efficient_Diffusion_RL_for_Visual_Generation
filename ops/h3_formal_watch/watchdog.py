"""Deterministic, boot-bound training watchdog. No LLM, SSH or scorer control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from .common import Store, digest, read_json, regular, tail
from . import process

DEFAULTS = dict(interval_seconds=60, monitor_max_age=30, max_update_seconds=10800,
                disk_floor_bytes=20 * 1024**3, inode_floor=20000, bad_ticks=3,
                hps_grace_seconds=600, checkpoint_allowance_seconds=1800,
                term_wait_seconds=60, kill_wait_seconds=60, verification_wait_seconds=7200,
                recovery_wait_seconds=1800, launch_wait_seconds=60, max_auto_resumes=2)
TRANSIENT = {'nccl_watchdog_timeout', 'card_monitor_stop', 'watchdog_checkpoint_stop'}
BOUNDARY_CAUSES = {'monitor_stop', 'disk_low', 'hps_failure'}
HISTORY_LIMIT = 32


def checkpoint(path, verify_files=False, report=lambda **kw: None):
    """Metadata on ticks; full shard hashes only in the recovery worker."""
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
        names, verified_bytes = [], 0
        report(files_done=0, files_total=len(files), bytes_done=0)
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
                    verified_bytes += len(chunk)
                    report(files_done=len(names), files_total=len(files), bytes_done=verified_bytes)
            if sha.hexdigest() != row['sha256']:
                raise ValueError('shard digest mismatch')
            names.append(name)
            report(files_done=len(names), files_total=len(files), bytes_done=verified_bytes)
        actual = []
        for file in path.rglob('*'):
            if file.is_symlink():
                raise ValueError('symlink in checkpoint')
            if file.is_file() and file.relative_to(path).as_posix() not in {k + '.json' for k in blobs}:
                actual.append(file.relative_to(path).as_posix())
        if names != sorted(set(names)) or sorted(actual) != names:
            raise ValueError('state tree does not cover exact files')
    return dict(path=str(path), step=step, complete=c)


def latest_checkpoint(output, verify_files=False, report=lambda **kw: None):
    paths = [p for p in (output / 'checkpoints').glob('step-*')
             if re.fullmatch(r'step-[1-9][0-9]*', p.name) and not p.is_symlink()
             and (p / 'complete.json').is_file()]
    # No silent rollback past a corrupt newest published marker.
    return checkpoint(max(paths, key=lambda p: int(p.name[5:])), verify_files, report) if paths else None


def progress_evidence(output):
    metrics = output / 'h3-last-metrics.json'
    row = read_json(metrics)  # absent/unreadable is UNKNOWN, not a stalled update
    if type(row.get('step')) is not int or row['step'] < 0:
        raise ValueError('invalid metrics step')
    ckpt = latest_checkpoint(output)
    return dict(step=row['step'] + 1, fingerprint=digest(row), checkpoint_step=ckpt['step'] if ckpt else 0)


def monitor_check(directory, monotonic_ns, max_age):
    health = read_json(Path(directory) / 'health.json', 65536)
    # Read first: the producer atomically replaces health.json while we sample.
    age = (monotonic_ns() - health['monotonic_ns']) / 1e9
    if type(health.get('healthy')) is not bool or age < 0:
        raise ValueError('unknown monitor clock/schema')
    return dict(ok=health['healthy'] and age < max_age, age_seconds=age,
                degraded=health.get('degraded', False), health=health)


def hps_check(url, expected, timeout=8):
    try:
        response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc  # unready 503 bodies still carry identity
    with response:
        raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('oversize HPS health')
        health = json.loads(raw)
    identity = health.get('service_identity')
    changed = isinstance(identity, str) and bool(re.fullmatch('[0-9a-f]{64}', identity)) and identity != expected
    facts = health.get('identity_facts')
    # Match flash_hpsv3.canonical_json exactly (including default ensure_ascii=True).
    canonical = json.dumps(dict(identity_facts=facts, video_contract=health.get('video_contract')),
                           sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if facts is not None and hashlib.sha256(canonical).hexdigest() != identity:
        changed = True
    return dict(ok=response.code == 200 and health.get('status') == 'ready' and identity == expected
                and not changed, identity_changed=changed, health=health)


def disk_check(output, byte_floor, inode_floor):
    stat = os.statvfs(output)
    if stat.f_bavail < 0 or stat.f_frsize <= 0:
        raise ValueError('unknown disk byte capacity')
    free = stat.f_bavail * stat.f_frsize
    inodes = stat.f_favail if stat.f_files > 0 and stat.f_favail is not None and 0 <= stat.f_favail < 2**63 else None
    ok = False if free < byte_floor or (inodes is not None and inodes < inode_floor) else (
        None if inodes is None and inode_floor else True)
    return dict(ok=ok, free_bytes=free, free_inodes=inodes)


def bounded(value, limit=32768):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value if len(raw.encode()) <= limit else dict(truncated=True, summary=raw[:limit // 4])


def existing_absolute(value):
    path = Path(value)
    if not path.is_absolute() or not path.exists():
        raise ValueError('require existing absolute path: ' + str(path))
    return path


class Watchdog:
    def __init__(self, config, *, dry_run=False, clock=time.time, monotonic_ns=time.monotonic_ns):
        self.config, self.clock, self.monotonic_ns, self.dry_run = config, clock, monotonic_ns, dry_run
        self.store = Store(config['state_dir'])
        self.output, self.run_dir = Path(config['output_dir']), Path(config['run_dir'])
        self.thresholds = {**DEFAULTS, **config.get('thresholds', {})}
        if set(self.thresholds) != set(DEFAULTS):
            raise ValueError('unknown threshold; migrate old configuration explicitly')
        if any(type(v) not in (int, float) or not 0 <= v < float('inf') for v in self.thresholds.values()):
            raise ValueError('thresholds must be finite nonnegative numbers')
        for key, low, high in (('bad_ticks', 1, 100), ('max_auto_resumes', 0, 100)):
            if type(self.thresholds[key]) is not int or not low <= self.thresholds[key] <= high:
                raise ValueError('invalid threshold: ' + key)
        if self.thresholds['interval_seconds'] <= 0:
            raise ValueError('invalid interval')
        self.boot = process.boot_id()
        saved = self.store.load('status.json')
        if saved and (saved['config_id'] != digest(config) or saved['dry_run'] != dry_run):
            raise ValueError('configuration/mode changed: reconcile explicitly, never erase retry history')
        self.s = saved or dict(schema=2, config_id=digest(config), dry_run=dry_run, state='ARMING',
            boot_id=self.boot, last_progress_at=clock(), last_progress_mono=self.mono(),
            optimizer_step=0, attempt_step=0, highwater_step=0, incidents=[], resumes=[],
            process=config.get('process'), pending_launch=None, stop=None, last_checks={}, debounce={})
        self.needs_arm = True
        self.worker = None
        self.worker_lock = threading.Lock()
        self.worker_result = {}
        if self.s.get('schema') != 2 or self.s.get('boot_id') != self.boot:
            self.human('schema/boot changed: reconcile identities and deadlines manually')
        if not self.s.get('process') and not self.s.get('pending_launch'):
            self.s['process'] = self.store.load('initial-process.json')

    def mono(self):
        return self.monotonic_ns() / 1e9

    def persist(self):
        self.s.update(heartbeat_at=self.clock(), now=self.clock(), heartbeat_mono=self.mono())
        self.s['incidents'] = [{k: row[k] for k in ('id', 'at', 'cause', 'transient') if k in row}
                               for row in self.s['incidents'][-HISTORY_LIMIT:]]
        self.s['resumes'] = self.s['resumes'][-128:]
        self.s['last_checks'] = {k: bounded(v) for k, v in self.s['last_checks'].items()}
        self.store.save('status.json', self.s)

    def event(self, kind, **fields):
        self.store.append_rotated('events.jsonl', dict(dict(id=uuid.uuid4().hex, at=self.clock(),
                                  mono=self.mono(), boot_id=self.boot, kind=kind), **fields))

    def state(self, name):
        if self.s['state'] != name:
            old, self.s['state'] = self.s['state'], name
            self.event('state_change', old=old, state=name)

    def action(self, kind, **fields):
        self.event('would_' + kind if self.dry_run else kind, **fields)
        if self.dry_run:
            print(json.dumps(dict(action=kind, **fields), ensure_ascii=False))

    def human(self, reason):
        self.s['reason'] = reason[:2000]
        if self.s['state'] != 'NEEDS_HUMAN':
            self.s['human_token'] = uuid.uuid4().hex
        self.state('NEEDS_HUMAN')

    def incident(self, cause, details):
        row = dict(id=uuid.uuid4().hex, at=self.clock(), cause=cause, transient=cause in TRANSIENT)
        self.store.append_rotated('incidents.jsonl', dict(row, details=details))
        self.s['incidents'].append(row)
        self.s['incidents'] = self.s['incidents'][-HISTORY_LIMIT:]
        self.event('incident', **row)
        return row

    def recipe(self):
        if self.store.root in (self.output, self.run_dir):
            raise ValueError('state directory must be distinct from run/output')
        for key in ('state_dir', 'output_dir', 'run_dir', 'monitor_dir'):
            existing_absolute(self.config[key])
        if self.output.resolve() != self.output:
            raise ValueError('output_dir must match lifecycle resolved absolute path')
        if not re.fullmatch('[0-9a-f]{64}', self.config['hps_identity']):
            raise ValueError('expected full HPS SHA256')
        launch = self.config['launcher']
        for key in ('recipe_json', 'cwd'):
            existing_absolute(launch[key])
        if self.config.get('training_log'):
            existing_absolute(self.config['training_log'])
        recipe = read_json(launch['recipe_json'])
        binding = recipe['launch']
        if existing_absolute(binding['output_dir']).resolve() != self.output:
            raise ValueError('recipe output differs from lifecycle resolved output')
        artifacts = binding['artifacts']
        if set(artifacts) != {'model', 'text_cache', 'train', 'eval', 'hps_manifest'}:
            raise ValueError('all launch.artifacts required')
        for path in artifacts.values():
            existing_absolute(path)
        if binding.get('resume_from') is not None:
            existing_absolute(binding['resume_from'])
        if binding['expected']['hps_service_identity'] != self.config['hps_identity']:
            raise ValueError('resume recipe scorer identity differs')
        argv = launch['argv']
        if not argv or argv.count('{resume_config}') != 1:
            raise ValueError('one standalone {resume_config} required')
        existing_absolute(argv[0])
        for index, arg in enumerate(argv[1:], 1):
            option, _, inline = arg.partition('=')
            if (option.startswith('--') and option.endswith(('-config', '-dir', '-path', '-file'))) or option == '--config':
                value = inline if '=' in arg else argv[index + 1]
                if value != '{resume_config}':
                    existing_absolute(value)
            elif arg != '{resume_config}' and Path(arg).suffix in ('.json', '.yaml', '.yml', '.py'):
                existing_absolute(arg.split('=', 1)[-1])
        if not isinstance(launch['env'], dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                       for k, v in launch['env'].items()):
            raise ValueError('complete string launch environment required')
        identity = digest(recipe)
        if self.s.get('recipe_id', identity) != identity:
            raise ValueError('original recipe changed after arming')
        self.s['recipe_id'] = identity
        return recipe

    def collect(self):
        checks = {}
        def check(name, work):
            try:
                checks[name] = work()
            except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, subprocess.SubprocessError) as exc:
                checks[name] = dict(ok=None, error=repr(exc)[:2000])
        def ownership():
            if not self.s.get('process') and self.config.get('training_unit'):
                self.s['process'] = process.capture_unit(self.config['training_unit'])
            return dict(ok=True, **process.inspect(self.s['process']))
        check('process', ownership)
        if checks['process'].get('alive') and self.s['process'].get('kind') != 'cgroup':
            self.s['process']['members'] = checks['process']['members']
        check('monitor', lambda: monitor_check(self.config['monitor_dir'], self.monotonic_ns,
                                              self.thresholds['monitor_max_age']))
        check('hps', lambda: hps_check(self.config['hps_healthz_url'], self.config['hps_identity']))
        check('disk', lambda: disk_check(self.output, self.thresholds['disk_floor_bytes'], self.thresholds['inode_floor']))
        check('progress', lambda: dict(ok=True, **progress_evidence(self.output)))
        p = checks['progress']
        if p['ok'] is True:
            # Content/step, never wall-clock mtime, distinguishes old failed-attempt metrics.
            eligible = p['fingerprint'] != self.s.get('ignored_metrics')
            if eligible and p['step'] > self.s.get('attempt_step', 0):
                self.s.update(attempt_step=p['step'], last_progress_at=self.clock(), last_progress_mono=self.mono())
                self.s['highwater_step'] = max(p['step'], self.s['highwater_step'])
                self.event('progress', **p)
            self.s['optimizer_step'] = self.s['attempt_step']
            p['seconds_without_progress'] = self.mono() - self.s['last_progress_mono']
        if checks['monitor'].get('degraded', False) != self.s['last_checks'].get('monitor', {}).get('degraded', False):
            self.event('monitor_degraded', degraded=checks['monitor'].get('degraded', False))
        for name in ('process', 'monitor', 'disk', 'progress', 'hps'):
            if checks[name]['ok'] is True:
                self.s['debounce'].pop(name, None)
            else:
                entry = self.s['debounce'].setdefault(name, dict(count=0, since=self.mono()))
                entry['count'] += 1
        self.s['last_checks'] = checks
        return checks

    def arm(self, checks):
        self.recipe()
        if any(checks[k]['ok'] is not True for k in ('process', 'monitor', 'hps', 'progress')):
            raise ValueError('arming requires identity-matched ready HPS, monitor health, readable metrics and process ownership')
        self.needs_arm = False
        self.s['armed_at'] = self.clock()
        if self.s['state'] == 'ARMING':
            self.state('WATCHING')

    def pause_supported(self):
        link = self.output / 'PAUSE_REQUESTED'
        return link.is_symlink() and link.resolve() == self.store.path('pause-request')

    def owns_pause_intent(self):
        stop = self.s.get('stop') or {}
        return bool(stop.get('pause_issued') and stop.get('pause_token') and
                    stop.get('launch_id') == (self.s.get('process') or {}).get('launch_id'))

    def pause_payload(self):
        stop = self.s['stop']
        return dict(launch_id=stop['launch_id'], pause_token=stop['pause_token'])

    def owns_pause(self):
        if not self.owns_pause_intent() or not self.pause_supported():
            return False
        try:
            return self.store.load('pause-request') == self.pause_payload()
        except (OSError, ValueError):
            return False

    def operator_pause(self):
        return (self.output / 'PAUSE_REQUESTED').exists() and not self.owns_pause()

    def publish_pause(self, *, replay=False):
        stop = self.s['stop']
        if not self.owns_pause_intent() or not self.pause_supported() or self.operator_pause():
            raise ValueError('cannot replay pause: ownership/link changed')
        self.action('pause_request')
        if not self.dry_run and not self.owns_pause():
            self.store.save('pause-request', self.pause_payload())
        # A restart replay gives training a full boundary opportunity, even if the
        # old intent's deadline elapsed before the request reached training.
        stop['pause_published'] = True
        if replay:
            stop['deadline'] = max(stop['deadline'], self.mono() + self.thresholds['max_update_seconds'] +
                                   self.thresholds['checkpoint_allowance_seconds'])
        self.persist()

    def request_stop(self, cause, checks):
        self.incident(cause, checks)
        pause = cause in BOUNDARY_CAUSES
        if pause and (not self.config.get('pause_via_symlink') or not self.pause_supported()):
            self.human('boundary stop requires provisioned pause symlink; no signal issued')
            return
        if self.operator_pause():
            self.human('operator pause active; no watchdog intervention')
            return
        self.s['stop'] = dict(cause=cause, phase='pause' if pause else 'term', pause_issued=pause,
            pause_published=False, launch_id=self.s['process']['launch_id'], pause_token=uuid.uuid4().hex,
            target_step=self.s['optimizer_step'], checkpoint_before=checks['progress'].get('checkpoint_step', 0),
            deadline=(self.s['last_progress_mono'] + self.thresholds['max_update_seconds'] +
                      self.thresholds['checkpoint_allowance_seconds']) if pause else
                     self.mono() + self.thresholds['term_wait_seconds'])
        self.state('STOPPING')
        self.persist()  # ownership/intent durable BEFORE writing request or signalling
        if pause:
            self.publish_pause()
        else:
            self.send(signal.SIGTERM)

    def send(self, sig):
        self.action('signal', owner=self.s['process'].get('unit', self.s['process'].get('pgid')), signal=sig.name)
        if not self.dry_run:
            process.signal_group(self.s['process'], sig)
        self.s['stop']['last_signal'] = sig.name

    def stopped_tick(self, checks):
        stop = self.s['stop']
        if stop.get('launch_id') != self.s['process']['launch_id']:
            self.human('stop ownership does not match current launch')
            return
        if stop['phase'] == 'pause' and not stop.get('pause_published') and checks['process']['alive']:
            self.publish_pause(replay=True)
            return  # never publish and escalate on the same tick
        if checks['hps'].get('identity_changed'):
            stop['cause'] = 'hps_identity_change'
            if checks['process']['alive'] and stop['phase'] == 'pause':
                stop.update(phase='term', deadline=self.mono() + self.thresholds['term_wait_seconds'])
                self.persist()
                self.send(signal.SIGTERM)
        p = checks['progress']
        if p['ok'] is True and stop['phase'] == 'pause':
            if self.s['optimizer_step'] > stop['target_step']:
                stop.update(target_step=self.s['optimizer_step'], deadline=self.s['last_progress_mono'] +
                    self.thresholds['max_update_seconds'] + self.thresholds['checkpoint_allowance_seconds'])
                stop.pop('checkpoint_saved', None)
            if p['checkpoint_step'] >= stop['target_step'] and p['checkpoint_step'] > stop['checkpoint_before']:
                if not stop.get('checkpoint_saved'):
                    stop.update(checkpoint_saved=p['checkpoint_step'], deadline=self.mono() + self.thresholds['term_wait_seconds'])
        if not checks['process']['alive']:
            self.handle_death(checks)
            return
        if self.mono() < stop['deadline']:
            return
        if stop['phase'] == 'kill':
            self.human('training ownership still populated after SIGKILL')
            return
        sig = signal.SIGTERM if stop['phase'] == 'pause' else signal.SIGKILL
        stop.update(phase='term' if sig == signal.SIGTERM else 'kill', deadline=self.mono() +
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
        ckpt, owner, receipt = latest_checkpoint(self.output), read_json(self.output / 'h3-run.json'), self.exit_receipt()
        return bool(receipt and type(receipt['exit_code']) is int and receipt['exit_code'] == 0 and ckpt
            and marker['kind'] == 'visual_rl_h3_run_success' and marker['run_id'] == owner['run_id']
            and marker['committed_steps'] == ckpt['step']
            and marker['checkpoint_relative_path'] == 'checkpoints/step-' + str(ckpt['step'])
            and all(marker[k] == ckpt['complete'][k] for k in ('checkpoint_contract_id', 'progress_id', 'state_tree_id')))

    def death_cause(self):
        owned = self.store.path('training-' + self.s['process']['launch_id'] + '.log')
        log = owned if owned.is_file() else Path(self.config.get('training_log', owned))
        text = tail(log, 256 * 1024).lower() if log.is_file() else ''
        # Correctness evidence takes precedence over EVERY transient pattern.
        permanent = ('audit failed', 'graph/count/release', 'adam counter', 'gradient not finite',
            'non-finite', 'nonfinite', 'frozen base changed', 'lora shard unchanged',
            'different h3 run', 'out of memory', 'insufficient disk', 'refusing resume', 'digest mismatch')
        nccl = r'nccl[^\n]*watchdog[^\n]*(?:timed out|timeout)|watchdog[^\n]*(?:timed out|timeout)[^\n]*nccl'
        monitor = r'card(?:/process)? monitor[^\n]*(?:unhealthy|stale)'
        for line in text.splitlines():
            # Normal structured launcher metadata is not a failure record.
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            if isinstance(row, dict):
                if not (row.get('error') or row.get('exception') or
                        row.get('level') in ('error', 'critical', 'fatal')):
                    continue
            if any(word in line for word in permanent):
                return 'correctness_or_permanent_error'
            failure = re.search(r'\b[\w.]*(?:error|exception)\s*:|\b(?:error|fatal|critical)\b|'
                                r'\b(?:audit|graph|gradient|identity)[^\n]*(?:failed|mismatch)', line)
            # Elastic's aggregate wrapper is not an additional application failure.
            wrapper = re.search(r'\b(?:torch\.distributed\.elastic\.multiprocessing\.errors\.)?childfailederror:\s*$', line)
            if failure and not wrapper and not re.search(nccl + '|' + monitor + '|h3safeboundarypause', line):
                return 'correctness_or_permanent_error'
        stop = self.s.get('stop') or {}
        if stop.get('cause') == 'hps_identity_change':
            return 'hps_identity_change'
        if self.s.get('operator_pause_seen') or self.operator_pause():
            return 'operator_pause'
        if stop.get('cause') in ('hps_failure', 'no_progress'):
            return stop['cause']
        receipt = self.exit_receipt()
        if (stop.get('cause') in ('monitor_stop', 'disk_low') and stop.get('checkpoint_saved')
                and (('h3safeboundarypause' in text) or (receipt and receipt['exit_code'] in (-15, 143)
                     and stop.get('last_signal') == 'SIGTERM'))):
            return 'watchdog_checkpoint_stop'
        if re.search(r'nccl.*watchdog.*(?:timed out|timeout)|watchdog.*(?:timed out|timeout).*nccl', text):
            return 'nccl_watchdog_timeout'
        if re.search(r'card(?:/process)? monitor.*(?:unhealthy|stale)', text):
            return 'card_monitor_stop'
        return 'unknown_exit'

    def handle_death(self, checks):
        cause = self.death_cause()
        self.incident(cause, dict(exit=self.exit_receipt(), checks=checks, stop=self.s.get('stop')))
        if cause not in TRANSIENT:
            self.human('non-allowlisted exit: ' + cause)
        else:
            if self.worker is not None and self.worker.is_alive():
                self.human('previous read-only verifier still running; reconcile before recovery')
                return
            self.worker, self.worker_result = None, {}
            self.s['recovery_since'] = self.mono()
            self.s.pop('verification_started', None)  # new incident, not a verifier restart
            self.state('RECOVERING')

    def resume_allowed(self, ckpt):
        if not ckpt:
            return 'no complete checkpoint'
        if len([r for r in self.s['resumes'] if r['boot_id'] == self.boot and self.mono() - r['mono'] < 86400]) >= self.thresholds['max_auto_resumes']:
            return '24-hour auto-resume limit'
        if (ckpt['step'] <= self.s.get('retry_step', 0)
                and self.s['highwater_step'] <= self.s.get('retry_highwater', 0)):
            return 'same checkpoint without new progress'
        if self.s.get('operator_pause_seen') or self.operator_pause():
            return 'operator pause request active/observed'
        return None

    def make_resume_spec(self, ckpt, launch_id):
        recipe = self.recipe()  # revalidate every artifact just before submission
        recipe['launch']['resume_from'] = str(existing_absolute(ckpt['path']))
        name = 'resume-recipe-' + launch_id + '.json'
        self.store.save(name, recipe)
        launch = self.config['launcher']
        return dict(launch_id=launch_id, unit='h3-training-' + launch_id + '.service',
                    argv=[str(self.store.path(name)) if a == '{resume_config}' else a for a in launch['argv']],
                    env=launch['env'], cwd=launch['cwd'])

    def verify_async(self):
        if self.worker is None:
            self.s['recovery_phase'] = 'verifying'
            self.s.setdefault('verification_started', self.mono())
            self.persist()  # deadline durable before the read-only worker starts
            if self.mono() - self.s['verification_started'] >= self.thresholds['verification_wait_seconds']:
                raise ValueError('checkpoint verification deadline exceeded; worker has no action authority')
            def report(**kw):
                with self.worker_lock:
                    self.worker_result['progress'] = kw
            def work():
                try:
                    result = dict(checkpoint=latest_checkpoint(self.output, True, report))
                except Exception as exc:
                    result = dict(error=repr(exc))
                with self.worker_lock:
                    self.worker_result.update(result, done=True)
            self.worker = threading.Thread(target=work, name='checkpoint-verifier', daemon=True)
            self.worker.start()
        with self.worker_lock:
            result = dict(self.worker_result)
        self.s['verification_progress'] = result.get('progress', {})
        if not result.get('done') and self.mono() - self.s['verification_started'] >= self.thresholds['verification_wait_seconds']:
            raise ValueError('checkpoint verification deadline exceeded; worker has no action authority')
        if result.get('error'):
            raise ValueError('checkpoint verification failed: ' + result['error'])
        return result

    def resume(self, checks):
        if self.s['incidents'][-1]['cause'] not in TRANSIENT:
            self.human('non-transient incident')
            return
        if self.operator_pause() or self.s.get('operator_pause_seen'):
            self.human('operator pause; never remove request')
            return
        if checks['hps'].get('identity_changed'):
            self.human('HPS identity changed during recovery')
            return
        if any(checks[k]['ok'] is not True for k in ('monitor', 'hps', 'disk', 'progress')):
            if self.mono() - self.s['recovery_since'] >= self.thresholds['recovery_wait_seconds']:
                self.human('recovery prerequisites did not become healthy')
            return
        result = self.verify_async()
        if not result.get('done'):
            return
        ckpt = result['checkpoint']
        reason = self.resume_allowed(ckpt)
        if reason:
            self.human(reason)
            return
        if ckpt != latest_checkpoint(self.output):
            raise ValueError('checkpoint changed during verification')
        if self.dry_run:
            self.action('auto_resume', checkpoint=ckpt['path'])
            self.human('dry-run: resume suppressed')
            return
        launch_id = uuid.uuid4().hex
        spec_name = 'launch-' + launch_id + '.json'
        self.store.save(spec_name, self.make_resume_spec(ckpt, launch_id))
        self.s['resumes'].append(dict(at=self.clock(), mono=self.mono(), boot_id=self.boot, step=ckpt['step'],
                                     highwater_step=self.s['highwater_step'], launch_id=launch_id))
        self.s.update(retry_step=ckpt['step'], retry_highwater=self.s['highwater_step'],
                      pending_launch=dict(launch_id=launch_id, mono=self.mono()),
                      ignored_metrics=checks['progress']['fingerprint'])
        self.state('LAUNCHING')
        self.persist()  # ambiguous submission is NEVER replayed
        if self.owns_pause_intent():
            if not self.pause_supported() or self.operator_pause():
                raise ValueError('pause ownership/link changed before resume')
            if self.store.path('pause-request').exists():
                self.store.path('pause-request').rename(self.store.path('pause-archived-' + launch_id))
        self.s['stop'] = None
        self.action('resume_intent', checkpoint=ckpt['path'], launch_id=launch_id)
        self.start_supervisor(spec_name)

    def start_supervisor(self, spec_name):
        root = Path(__file__).resolve().parents[2]
        self.supervisor = subprocess.Popen([sys.executable, '-m', 'ops.h3_formal_watch.process',
            'supervise', str(self.store.root), spec_name], cwd=root, env={**os.environ, 'PYTHONPATH': str(root)},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def adopt_pending(self, record):
        pending = self.s['pending_launch']
        if record['launch_id'] != pending['launch_id']:
            raise ValueError('pending launch receipt mismatch')
        process.inspect(record)  # ownership proof, not merely a JSON file
        resume = next(r for r in self.s['resumes'] if r['launch_id'] == record['launch_id'])
        self.s.update(process=record, pending_launch=None, stop=None, operator_pause_seen=False,
                      last_progress_at=self.clock(), last_progress_mono=self.mono(), attempt_step=resume['step'])
        resume['performed'] = True
        self.worker, self.worker_result = None, {}
        self.s['debounce'] = {}
        self.event('auto_resume', launch_id=record['launch_id'])

    def reconcile_launch(self, action, reason):
        """Explicit human reconciliation only; never submits or signals training."""
        if self.s['state'] != 'NEEDS_HUMAN' or not self.s.get('pending_launch') or not reason.strip():
            raise ValueError('reconciliation requires a pending launch, NEEDS_HUMAN and reason')
        if self.s['boot_id'] != self.boot:
            raise ValueError('boot changed: manual identity reconciliation required')
        launch_id = self.s['pending_launch']['launch_id']
        spec = self.store.load('launch-' + launch_id + '.json')
        if not spec or spec['launch_id'] != launch_id or spec['unit'] != 'h3-training-' + launch_id + '.service':
            raise ValueError('pending launch spec mismatch')
        with self.store.lock('supervisor.lock'):
            if action == 'adopt':
                record = process.reconcile_unit(self.store, spec)
                self.store.save('process-' + launch_id + '.json', record)
                self.adopt_pending(record)
            elif action == 'abandon':
                if process.inspect(self.s['process'])['alive']:
                    raise ValueError('old training still alive; cannot abandon')
                info = process.unit_info(spec['unit'])
                if info.get('LoadState') != 'not-found':
                    record = process.reconcile_unit(self.store, spec)
                    if process.inspect(record)['alive']:
                        raise ValueError('pending training still alive; cannot abandon')
                self.store.save('abandoned-' + launch_id + '.json', dict(reason=reason, at=self.clock()))
                self.s.update(pending_launch=None, stop=None)
            else:
                raise ValueError('reconcile action must be adopt or abandon')
            self.event('launch_reconciled', launch_id=launch_id, action=action, reason=reason)
            self.s['reason'] = 'launch reconciled; acknowledge to rearm only'
            self.persist()

    def acknowledge(self):
        ack = self.store.load('acknowledge.json')
        if not ack or ack.get('token') != self.s.get('human_token'):
            return
        if self.s.get('pending_launch'):
            self.s['reason'] = 'launch outcome unknown; use --reconcile-launch adopt|abandon before acknowledgement'
            return
        if self.s.get('boot_id') != self.boot or (self.output / 'PAUSE_REQUESTED').exists():
            return  # acknowledgement does not authorize bypassing identity/pause gates
        self.event('human_acknowledged', acknowledgement=ack)
        self.s.pop('human_token', None)
        self.s.pop('reason', None)
        self.s['operator_pause_seen'] = False
        self.s['rearm_only'] = True
        self.s['stop'] = None
        self.s['debounce'] = {}
        self.needs_arm = True
        self.state('ARMING')  # no launch; a dead process remains a human decision

    def observe_human(self, checks):
        # Health transitions, not ticking ages/free-byte counters/producer timestamps.
        fields = ('ok', 'alive', 'error', 'degraded', 'identity_changed', 'step',
                  'checkpoint_step', 'fingerprint')
        summary = {name: {k: row[k] for k in fields if k in row} for name, row in checks.items()}
        identity = digest(dict(token=self.s.get('human_token'), checks=summary))
        if identity != self.s.get('human_observation_id'):
            self.event('human_observation', checks=bounded(checks))
            self.s['human_observation_id'] = identity

    def tick(self):
        try:
            if self.s['state'] == 'DONE':
                return self.s
            checks = self.collect()  # even NEEDS_HUMAN keeps read-only evidence current
            if self.s['state'] == 'NEEDS_HUMAN':
                self.observe_human(checks)
                self.acknowledge()
                return self.s
            if self.needs_arm:
                self.arm(checks)  # any restart failure -> human, without signalling
            if self.s.get('pending_launch'):
                pending = self.s['pending_launch']
                record = self.store.load('process-' + pending['launch_id'] + '.json')
                if record:
                    self.adopt_pending(record)
                    self.state('WATCHING')
                elif self.mono() - pending['mono'] >= self.thresholds['launch_wait_seconds']:
                    self.human('launch outcome unknown; no automatic replay')
                return self.s
            if checks['process']['ok'] is not True:
                if self.s['debounce'].get('process', {}).get('count', 0) >= self.thresholds['bad_ticks']:
                    self.human('cannot prove training ownership for consecutive ticks; no signal/resume')
                return self.s
            alive = checks['process']['alive']
            if self.s.pop('rearm_only', False) and not alive:
                self.human('acknowledged but training is dead; manual launch decision required')
                return self.s
            if self.operator_pause():
                self.s['operator_pause_seen'] = True
            elif self.s.get('operator_pause_seen') and alive:
                self.event('operator_pause_cancelled')
                self.human('operator pause cancelled before boundary; acknowledge to rearm checks')
                return self.s
            if not alive and (self.output / 'SUCCESS').exists() and not self.s.get('operator_pause_seen'):
                if not self.exit_receipt():
                    since = self.s.setdefault('success_seen_mono', self.mono())
                    if self.mono() - since < self.thresholds['launch_wait_seconds']:
                        return self.s
                self.state('DONE') if self.complete() else self.human('SUCCESS lacks matching exit/checkpoint receipt')
                return self.s
            if self.s['state'] == 'STOPPING':
                self.stopped_tick(checks)
                return self.s
            if self.s['state'] == 'RECOVERING':
                self.human('training reappeared during recovery') if alive else self.resume(checks)
                return self.s
            if not alive:
                self.handle_death(checks)
                return self.s
            if checks['hps'].get('identity_changed'):
                self.request_stop('hps_identity_change', checks)
                return self.s
            if self.s.get('operator_pause_seen'):
                return self.s  # operator owns the pending safe-boundary pause
            if checks['progress']['ok'] is True and checks['progress']['seconds_without_progress'] >= self.thresholds['max_update_seconds']:
                self.request_stop('no_progress', checks)
                return self.s
            for name in ('monitor', 'disk', 'progress'):
                if self.s['debounce'].get(name, {}).get('count', 0) >= self.thresholds['bad_ticks']:
                    if checks[name]['ok'] is None:
                        self.human(name + ' unknown for consecutive ticks; no signal')
                    else:
                        self.request_stop({'monitor': 'monitor_stop', 'disk': 'disk_low'}[name], checks)
                    return self.s
            if 'hps' in self.s['debounce'] and self.mono() - self.s['debounce']['hps']['since'] >= self.thresholds['hps_grace_seconds']:
                self.request_stop('hps_failure', checks)
            return self.s
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, RuntimeError, subprocess.SubprocessError) as exc:
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
    parser.add_argument('--acknowledge', metavar='REASON', help='acknowledge current NEEDS_HUMAN; rearm only, never force resume')
    parser.add_argument('--reconcile-launch', choices=('adopt', 'abandon'), help='resolve pending launch; never submit')
    parser.add_argument('--reason', help='required explanation for launch reconciliation')
    args = parser.parse_args()
    config = read_json(existing_absolute(args.config))
    if args.reconcile_launch:
        if args.acknowledge or not args.reason or not args.reason.strip():
            parser.error('reconciliation requires --reason and a separate acknowledgement')
        store = Store(config['state_dir'])
        with store.lock('watchdog.lock'):
            Watchdog(config, dry_run=args.dry_run).reconcile_launch(args.reconcile_launch, args.reason)
        return
    if args.acknowledge:
        store = Store(config['state_dir'])
        status = store.load('status.json')
        if status['state'] != 'NEEDS_HUMAN' or not args.acknowledge.strip():
            parser.error('acknowledgement requires NEEDS_HUMAN and a reason')
        store.save('acknowledge.json', dict(token=status['human_token'], reason=args.acknowledge, at=time.time()))
        return
    store = Store(config['state_dir'])
    with store.lock('watchdog.lock'):
        watch = Watchdog(config, dry_run=args.dry_run)
        while True:
            state = watch.tick()['state']
            if args.once or state == 'DONE':
                return
            time.sleep(watch.thresholds['interval_seconds'])


if __name__ == '__main__':
    main()
