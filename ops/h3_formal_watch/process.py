"""Training-only session identity and durable exit receipts; no process-name matching.

Linux is production. The ps fallback is solely for local POSIX fire drills.
A detached supervisor owns waitpid, so a watchdog restart cannot lose exit status.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .common import Store, read_json


def boot_id():
    path = Path('/proc/sys/kernel/random/boot_id')
    return path.read_text().strip() if path.exists() else 'posix-local-drill'


def processes():
    rows = {}
    if Path('/proc/self/stat').exists():
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                raw = path.read_text()
                fields = raw[raw.rfind(')') + 2:].split()
                pid = int(path.parent.name)
                rows[pid] = dict(pid=pid, pgid=int(fields[2]), sid=int(fields[3]),
                                 token=fields[19], zombie=fields[0] in ('Z', 'X'))
            except FileNotFoundError:
                pass  # process exited during snapshot; permission failures are NOT death
    else:
        result = subprocess.run(['ps', '-axo', 'pid=,pgid=,lstart=,stat='],
                                check=True, capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 8:
                pid, pgid = map(int, parts[:2])
                rows[pid] = dict(pid=pid, pgid=pgid, sid=os.getsid(pid) if pid == os.getpid() else pgid,
                                 token=' '.join(parts[2:7]), zombie=parts[7].startswith('Z'))
    return rows


def capture(pid, launch_id=None):
    rows = processes()
    leader = rows[pid]
    if leader['pgid'] != pid or leader['sid'] != pid or pid <= 1 or pid == os.getpgrp():
        raise ValueError('training must have its own session and PGID, never the watchdog/HPS group')
    members = [p for p in rows.values() if p['pgid'] == pid and not p['zombie']]
    return dict(launch_id=launch_id or uuid.uuid4().hex, boot_id=boot_id(), pgid=pid,
                members=members, started_at=time.time())


def inspect(record, rows=None):
    """Fail closed on PID/PGID reuse, reboot or unidentified surviving members."""
    if record['pgid'] <= 1 or record['pgid'] == os.getpgrp():
        raise ValueError('unsafe training group')
    if record['boot_id'] != boot_id():
        raise ValueError('host reboot: process identity no longer usable')
    rows = processes() if rows is None else rows
    live = [p for p in rows.values() if p['pgid'] == record['pgid'] and not p['zombie']]
    known = {p['pid']: p['token'] for p in record['members']}
    if live and (any(p['pid'] in known and p['token'] != known[p['pid']] for p in live)
                 or not any(known.get(p['pid']) == p['token'] for p in live)
                 or any(p['sid'] != record['pgid'] for p in live)):
        raise ValueError('training group identity changed or cannot be proven')
    return dict(alive=bool(live), members=live)


def signal_group(record, sig):
    if inspect(record)['alive']:
        os.killpg(record['pgid'], sig)


def supervise(state_dir, spec_name):
    """One-shot spawn. A consumed spec is never replayed, even across crashes."""
    store = Store(state_dir)
    spec = store.load(spec_name)
    launch_id = spec['launch_id']
    with store.lock('supervisor.lock'):
        claim = store.path('claimed-' + launch_id + '.json')
        with claim.open('x', encoding='utf-8') as stream:
            stream.write('{}\n')
            stream.flush()
            os.fsync(stream.fileno())
        with store.path('training-' + launch_id + '.log').open('ab') as log:
            child = subprocess.Popen(spec['argv'], cwd=spec['cwd'],
                                     env=spec['env'], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # capture immediately, including a very fast exit (zombie still has /proc identity)
            record = capture(child.pid, launch_id)
            store.save('process-' + launch_id + '.json', record)
            code = child.wait()
            store.save('exit-' + launch_id + '.json', dict(launch_id=launch_id,
                       pgid=record['pgid'], exit_code=code, finished_at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    record = sub.add_parser('record', help='adopt an existing isolated torchrun group (exit unknown)')
    record.add_argument('pid', type=int)
    record.add_argument('state_dir')
    run = sub.add_parser('supervise', help='wait for an operator-approved launch spec in state_dir')
    run.add_argument('state_dir')
    run.add_argument('spec_name')
    args = parser.parse_args()
    if args.mode == 'record':
        store = Store(args.state_dir)
        if store.path('initial-process.json').exists():
            raise ValueError('initial identity already recorded; do not overwrite')
        store.save('initial-process.json', capture(args.pid))
    else:
        supervise(args.state_dir, args.spec_name)


if __name__ == '__main__':
    main()
