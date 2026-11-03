"""Offline drill: temporary FAKE data, sleep processes, mocked HTTP/systemd only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

if __package__ in (None, ''):  # also support python ops/h3_formal_watch/drill.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'ops.h3_formal_watch'

from .common import Store, digest
from . import process
from .watchdog import Watchdog

IDENTITY = 'a' * 64


def fake_checkpoint(output, step=1):
    path = output / 'checkpoints' / f'step-{step}'
    store = Store(path)
    store.text('fake-state', 'NOT A TRAINING CHECKPOINT\n')
    import hashlib
    tree = dict(schema_version=1, files=[dict(path='fake-state', size=len(b'NOT A TRAINING CHECKPOINT\n'),
                sha256=hashlib.sha256(b'NOT A TRAINING CHECKPOINT\n').hexdigest())])
    contract = dict(fake=True)
    progress = dict(global_step=step, next_optimizer_step=step)
    marker = dict(kind='visual_rl_checkpoint_complete', step=step,
                  checkpoint_contract_id=digest(contract), progress_id=digest(progress), state_tree_id=digest(tree))
    store.save('checkpoint_contract.json', dict(checkpoint_contract_id=digest(contract), contract=contract))
    store.save('progress.json', dict(progress_id=digest(progress), progress=progress))
    store.save('state_tree.json', dict(state_tree_id=digest(tree), tree=tree))
    store.save('complete.json', marker)
    return marker


class Sandbox:
    def __init__(self, root):
        self.root = root.resolve()
        root = self.root
        self.now, self.ns = time.time(), time.monotonic_ns()
        for name in ('state', 'output', 'monitor'):
            (root / name).mkdir(parents=True)
        Store(root).text('FAKE-RUN-ONLY', 'Created by h3_formal_watch drill\n')
        self.children = []
        self.child = subprocess.Popen(['/bin/sleep', '600'], start_new_session=True)
        self.children.append(self.child)
        record = process.capture(self.child.pid)
        self.config = dict(run_dir=str(root), output_dir=str(root / 'output'), state_dir=str(root / 'state'),
            process=record, monitor_dir=str(root / 'monitor'), hps_identity=IDENTITY,
            hps_healthz_url='http://127.0.0.1:1/never-contacted', pause_via_symlink=True,
            thresholds=dict(disk_floor_bytes=0, inode_floor=0, term_wait_seconds=.1, kill_wait_seconds=.1),
            launcher=dict(argv=[sys.executable, '-c', 'import time; time.sleep(600)', '{resume_config}'],
                          env={}, cwd=str(root), recipe_json=str(root / 'recipe.json')))
        Store(root).save('recipe.json', dict(launch=dict(output_dir=str(root / 'output'), resume_from=None,
            artifacts={k: str(root / 'FAKE-RUN-ONLY') for k in ('model', 'text_cache', 'train', 'eval', 'hps_manifest')},
            expected=dict(hps_service_identity=IDENTITY))))
        Store(root / 'output').save('h3-run.json', dict(run_id='fake-run'))
        Store(root / 'output').save('h3-last-metrics.json', dict(step=0))
        (root / 'output' / 'PAUSE_REQUESTED').symlink_to(root / 'state' / 'pause-request')
        self.health = dict(ok=True, identity_changed=False, health=dict(status='ready', service_identity=IDENTITY))
        self.monitor()
        self.watch = Watchdog(self.config, clock=lambda: self.now, monotonic_ns=lambda: self.ns)

    def monitor(self, **kw):
        Store(self.root / 'monitor').save('health.json', dict(dict(healthy=True, degraded=False, monotonic_ns=self.ns), **kw))

    def tick(self):
        def fake_supervisor(spec_name):
            spec = self.watch.store.load(spec_name)
            child = subprocess.Popen(['/bin/sleep', '600'], start_new_session=True)
            self.children.append(child)
            self.watch.store.save('process-' + spec['launch_id'] + '.json', process.capture(child.pid, spec['launch_id']))
            self.watch.supervisor = child
        with patch('ops.h3_formal_watch.watchdog.hps_check', return_value=self.health), \
                patch.object(self.watch, 'start_supervisor', side_effect=fake_supervisor):
            return self.watch.tick()

    def until(self, state):
        for _ in range(200):
            result = self.tick()
            if result['state'] == state:
                return result
            if result['state'] == 'NEEDS_HUMAN':
                raise AssertionError(result['reason'])
            time.sleep(.01)
        raise AssertionError('fake state deadline: ' + state)

    def log(self, text):
        self.watch.store.text('training-' + self.watch.s['process']['launch_id'] + '.log', text)

    def advance(self, seconds):
        self.now += seconds
        self.ns += int(seconds * 1e9)

    def die(self):
        self.child.terminate()
        self.child.wait(timeout=5)

    def cleanup(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        if self.watch.worker:
            self.watch.worker.join(timeout=5)


def run_drill(parent):
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='h3-FAKE-drill-', dir=parent)).resolve()
    results = {}
    for scenario in ('monitor_stale', 'no_progress', 'hps_identity_change', 'hps_503_burst',
                     'audit_error', 'operator_pause', 'orphan_rank', 'transient_then_repeated', 'success'):
        box = Sandbox(root / scenario)
        try:
            assert box.tick()['state'] == 'WATCHING'
            if scenario in ('monitor_stale', 'no_progress', 'hps_identity_change'):
                if scenario == 'monitor_stale':
                    box.advance(31)
                    box.tick(); box.tick()
                elif scenario == 'no_progress':
                    box.advance(10801)
                    box.monitor()
                else:
                    box.health = dict(ok=False, identity_changed=True)
                assert box.tick()['state'] == 'STOPPING'
                if scenario == 'monitor_stale':
                    assert box.child.poll() is None  # wait a real update boundary, NOT 15 minutes
                    box.advance(12600)
                    box.tick()
                box.child.wait(timeout=5)
                assert not process.inspect(box.config['process'])['alive']
                results[scenario] = 'training tree stopped; HPS untouched'
            elif scenario == 'hps_503_burst':
                box.health = dict(ok=False, identity_changed=False, health=dict(status='unready', code=503))
                box.tick()
                box.advance(599); box.monitor()
                assert box.tick()['state'] == 'WATCHING'
                box.health = dict(ok=True)
                assert box.tick()['state'] == 'WATCHING' and box.child.poll() is None
                results[scenario] = '503 burst recovered; no stop'
            elif scenario in ('audit_error', 'operator_pause'):
                if scenario == 'audit_error':
                    box.log('NCCL watchdog timed out\nsingle-cell graph/count/release audit failed')
                else:
                    box.watch.store.text('pause-request', 'operator requested, NOT watchdog\n')
                    box.tick()
                    box.log('H3SafeBoundaryPause')
                box.die()
                assert box.tick()['state'] == 'NEEDS_HUMAN'
                assert not box.watch.s['resumes']
                results[scenario] = 'NEEDS_HUMAN; no resume'
            elif scenario == 'orphan_rank':
                # Fake tree models elastic's separate rank session, then agent death.
                record = box.config['process']
                leader = record['members'][0]
                rank = dict(leader, pid=leader['pid'] + 1000000, ppid=leader['pid'],
                            pgid=leader['pid'] + 1000000, sid=leader['pid'] + 1000000, token='rank')
                record['members'] = process.inspect(record, {p['pid']: p for p in (leader, rank)})['members']
                tree = {rank['pid']: dict(rank, ppid=1)}
                assert process.inspect(record, tree)['alive']
                with patch.object(process, 'processes', return_value=tree), patch.object(os, 'killpg') as kill:
                    process.signal_group(record, signal.SIGKILL)
                    assert kill.call_args.args == (rank['pgid'], signal.SIGKILL)
                results[scenario] = 'orphan rank remains owned and targeted (fake PPID tree)'
            elif scenario == 'transient_then_repeated':
                fake_checkpoint(box.root / 'output')
                box.log('NCCL watchdog timed out')
                box.die()
                assert box.tick()['state'] == 'RECOVERING'
                box.until('LAUNCHING'); box.until('WATCHING')
                box.log('NCCL watchdog timed out')
                process.signal_group(box.watch.s['process'], signal.SIGTERM)
                box.watch.supervisor.wait(timeout=5)
                assert box.tick()['state'] == 'RECOVERING'
                box.until('NEEDS_HUMAN')
                assert len(box.watch.s['resumes']) == 1
                results[scenario] = box.watch.s['reason']
            else:
                marker = fake_checkpoint(box.root / 'output')
                box.die()
                Store(box.root / 'output').save('SUCCESS', dict(kind='visual_rl_h3_run_success', run_id='fake-run',
                    committed_steps=1, checkpoint_relative_path='checkpoints/step-1',
                    **{k: marker[k] for k in ('checkpoint_contract_id', 'progress_id', 'state_tree_id')}))
                record = box.config['process']
                box.watch.store.save('exit-' + record['launch_id'] + '.json', dict(launch_id=record['launch_id'],
                                     pgid=record['pgid'], exit_code=0))
                assert box.tick()['state'] == 'DONE'
                results[scenario] = 'DONE with matching SUCCESS and exit receipt'
        finally:
            box.cleanup()
    Store(root).save('drill-results.json', results)
    return root, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    root, results = run_drill(args.parent)
    print(json.dumps(dict(artifacts=str(root), scenarios=results), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
