"""Own training by a unique systemd cgroup, or conservatively adopt a PPID tree.

No name matching. Linux is production; ps is only for the offline POSIX drill.
Never reuse a unit name, launch claim, or process receipt for another attempt.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from .common import Store
from .commands import run_command


def boot_id():
    path = Path('/proc/sys/kernel/random/boot_id')
    if path.exists():
        return path.read_text().strip()
    return subprocess.check_output(['sysctl', '-n', 'kern.boottime'], text=True).strip()


def processes():
    rows = {}
    if Path('/proc/self/stat').exists():
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                raw = path.read_text()
                fields = raw[raw.rfind(')') + 2:].split()
                pid = int(path.parent.name)
                rows[pid] = dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
                                 token=fields[19], zombie=fields[0] in ('Z', 'X'))
            except FileNotFoundError:
                pass
    else:
        result = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,lstart=,stat='],
                                check=True, capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 9:
                pid, ppid, pgid = map(int, parts[:3])
                rows[pid] = dict(pid=pid, ppid=ppid, pgid=pgid, sid=pgid,
                                 token=' '.join(parts[3:8]), zombie=parts[8].startswith('Z'))
    return rows


def unit_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'h3-training-[A-Za-z0-9_-]+\.service', name):
        raise ValueError('exclusive unit must be h3-training-UNIQUE.service')
    return name


def unit_info(name):
    result = run_command(['systemctl', '--user', 'show', unit_name(name),
                          '--property=InvocationID,ControlGroup,ActiveState,KillMode,Restart'], timeout=10)
    return dict(line.split('=', 1) for line in result['stdout'].splitlines() if '=' in line)


def capture_unit(name, launch_id=None):
    info = unit_info(name)
    if (not info.get('InvocationID') or not info.get('ControlGroup')
            or info.get('KillMode') != 'control-group' or info.get('Restart') != 'no'):
        raise ValueError('unit must exist, own a cgroup, and have KillMode=control-group, Restart=no')
    return dict(kind='cgroup', unit=unit_name(name), invocation_id=info['InvocationID'],
                launch_id=launch_id or uuid.uuid4().hex, boot_id=boot_id(), started_at=time.time())


def cgroup_members(group):
    root = Path('/sys/fs/cgroup')
    path = root / group.lstrip('/')
    if not group.startswith('/') or path.resolve() != path or not path.is_relative_to(root):
        raise ValueError('unsafe cgroup path')
    if not path.is_dir():
        raise ValueError('active cgroup missing')
    return sorted({int(pid) for file in [path / 'cgroup.procs', *path.rglob('*/cgroup.procs')]
                   for pid in file.read_text().split()})


def capture(pid, launch_id=None):
    rows = processes()
    leader = rows[pid]
    if leader['pgid'] != pid or leader['sid'] != pid or pid <= 1 or pid == os.getpgrp():
        raise ValueError('training agent must have its own session, never watchdog/HPS group')
    record = dict(kind='descendants', launch_id=launch_id or uuid.uuid4().hex, boot_id=boot_id(), pgid=pid,
                  members=[leader], started_at=time.time())
    record['members'] = inspect(record, rows)['members']
    return record


def inspect(record, rows=None):
    if record['boot_id'] != boot_id():
        raise ValueError('host reboot: process identity no longer usable')
    if record.get('kind') == 'cgroup':
        info = unit_info(record['unit'])
        if info.get('InvocationID') != record['invocation_id']:
            raise ValueError('unit invocation changed or disappeared; reconcile manually')
        if info.get('KillMode') != 'control-group' or info.get('Restart') != 'no':
            raise ValueError('unit ownership policy changed')
        members = cgroup_members(info['ControlGroup']) if info.get('ControlGroup') else []
        if not info.get('ControlGroup') and info.get('ActiveState') not in ('inactive', 'failed'):
            raise ValueError('unit cgroup unknown')
        return dict(alive=bool(members), members=members)
    if record['pgid'] <= 1 or record['pgid'] == os.getpgrp():
        raise ValueError('unsafe training group')
    rows = processes() if rows is None else rows
    known = {p['pid']: p['token'] for p in record['members']}
    if any(pid in rows and rows[pid]['token'] != token for pid, token in known.items()):
        raise ValueError('recorded PID reused; no signal')
    live = {pid: rows[pid] for pid in known if pid in rows and not rows[pid]['zombie']}
    while True:
        children = {pid: p for pid, p in rows.items() if p.get('ppid') in live and not p['zombie']}
        if children.keys() <= live.keys():
            break
        live.update(children)
    # A group signal must not include unowned peers. Known orphan ranks remain owned
    # by their start token even after their parent agent has vanished.
    groups = {record['pgid'], *(p['pgid'] for p in live.values())}
    if any(not p['zombie'] and p['pgid'] in groups and p['pid'] not in live for p in rows.values()):
        raise ValueError('unowned member in training group')
    if any(p['pgid'] <= 1 or p['pgid'] == os.getpgrp() for p in live.values()):
        raise ValueError('descendant shares controller group')
    return dict(alive=bool(live), members=list(live.values()))


def signal_group(record, sig):
    current = inspect(record)
    if not current['alive']:
        return
    if record.get('kind') == 'cgroup':
        run_command(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=' + sig.name, record['unit']], timeout=10)
        if sig == signal.SIGKILL:
            run_command(['systemctl', '--user', '--no-block', 'stop', record['unit']], timeout=10)
        return
    record['members'] = current['members']
    groups = {p['pgid'] for p in current['members']}
    # Elastic ranks have independent sessions. Signal ranks BEFORE the agent;
    # killing only torchrun's PGID would orphan every one of these groups.
    for pgid in sorted(groups, key=lambda g: (g == record['pgid'], g)):
        checked = inspect(record)
        if any(p['pgid'] == pgid for p in checked['members']):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass


def execute(state_dir, spec_name):
    """Inside the training unit: capture the exact training exit, never restart."""
    store = Store(state_dir)
    spec = store.load(spec_name)
    with store.path('training-' + spec['launch_id'] + '.log').open('ab') as log:
        child = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=spec['env'],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        # The unit receives TERM too. Keep this waiter alive long enough to reap.
        signal.signal(signal.SIGTERM, lambda *_: None)
        code = child.wait()
        store.save('exit-' + spec['launch_id'] + '.json', dict(launch_id=spec['launch_id'],
                   unit=spec['unit'], invocation_id=os.environ['INVOCATION_ID'], boot_id=boot_id(),
                   exit_code=code, finished_at=time.time()))


def supervise(state_dir, spec_name):
    """One-shot systemd submission. A consumed spec is NEVER automatically replayed."""
    store = Store(state_dir)
    spec = store.load(spec_name)
    launch_id = spec['launch_id']
    unit = unit_name(spec['unit'])
    root = Path(__file__).resolve().parents[2]
    with store.lock('supervisor.lock'):
        with store.path('claimed-' + launch_id + '.json').open('x', encoding='utf-8') as stream:
            stream.write('{}\n')
            stream.flush()
            os.fsync(stream.fileno())
        run_command(['systemd-run', '--user', '--unit=' + unit, '--service-type=exec',
                     '--property=KillMode=control-group', '--property=Restart=no',
                     '--property=RemainAfterExit=yes', '--working-directory=' + str(root),
                     '--setenv=PYTHONPATH=' + str(root), sys.executable,
                     '-m', 'ops.h3_formal_watch.process', 'execute', str(store.root), spec_name], timeout=30)
        store.save('process-' + launch_id + '.json', capture_unit(unit, launch_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    record = sub.add_parser('record', help='fallback: adopt an isolated agent and current descendants')
    record.add_argument('pid', type=int)
    record.add_argument('state_dir')
    adopt = sub.add_parser('adopt-unit', help='adopt an exclusive existing systemd training unit')
    adopt.add_argument('unit')
    adopt.add_argument('state_dir')
    for mode in ('supervise', 'execute'):
        run = sub.add_parser(mode)
        run.add_argument('state_dir')
        run.add_argument('spec_name')
    args = parser.parse_args()
    if args.mode in ('record', 'adopt-unit'):
        store = Store(args.state_dir)
        if store.path('initial-process.json').exists():
            raise ValueError('initial identity already recorded; do not overwrite')
        store.save('initial-process.json', capture(args.pid) if args.mode == 'record' else capture_unit(args.unit))
    else:
        globals()[args.mode](args.state_dir, args.spec_name)


if __name__ == '__main__':
    main()
