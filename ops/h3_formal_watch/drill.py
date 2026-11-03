"""Offline fire drill. Creates NEW fake data and isolated sleep groups only.

No run/config argument is accepted. Artifacts are retained under --parent.
Never use a real run directory as --parent. No HTTP, SSH, GPU, pi or lark calls.
"""
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
        self.root = root
        self.now = time.time()
        self.ns = time.monotonic_ns()
        for name in ('state', 'output', 'monitor'):
            (root / name).mkdir(parents=True)
        Store(root).text('FAKE-RUN-ONLY', 'Created by h3_formal_watch drill\n')
        self.children = []
        self.child = subprocess.Popen(['/bin/sleep', '600'], start_new_session=True)
        self.children.append(self.child)
        record = process.capture(self.child.pid)
        self.config = dict(run_dir=str(root), output_dir=str(root / 'output'), state_dir=str(root / 'state'),
                           process=record, monitor_dir=str(root / 'monitor'), hps_identity=IDENTITY,
                           hps_healthz_url='http://127.0.0.1:1/never-contacted',
                           thresholds=dict(disk_floor_bytes=0, inode_floor=0, stop_wait_seconds=0.1,
                                           term_wait_seconds=0.1, kill_wait_seconds=0.1),
                           launcher=dict(argv=[sys.executable, '-c', 'import time; time.sleep(600)', '{resume_config}'],
                                         env={}, cwd=str(root), recipe_json=str(root / 'recipe.json')))
        Store(root).save('recipe.json', dict(launch=dict(output_dir=str(root / 'output'), resume_from=None,
                        expected=dict(hps_service_identity=IDENTITY))))
        Store(root / 'output').save('h3-run.json', dict(run_id='fake-run'))
        self.health = dict(ok=True, identity_changed=False, health=dict(status='ready', service_identity=IDENTITY))
        self.monitor()
        self.watch = Watchdog(self.config, clock=lambda: self.now, monotonic_ns=lambda: self.ns)

    def monitor(self, **kw):
        Store(self.root / 'monitor').save('health.json', dict(dict(healthy=True, degraded=False, monotonic_ns=self.ns), **kw))

    def tick(self):
        real_popen = subprocess.Popen
        def fake_supervisor(spec_name):
            # ONLY the offline drill replaces systemd with a new local sleep.
            spec = self.watch.store.load(spec_name)
            child = real_popen(['/bin/sleep', '600'], start_new_session=True)
            self.children.append(child)
            self.watch.store.save('process-' + spec['launch_id'] + '.json', process.capture(child.pid, spec['launch_id']))
            self.watch.supervisor = child
        with patch('ops.h3_formal_watch.watchdog.hps_check', return_value=self.health), \
                patch.object(self.watch, 'start_supervisor', side_effect=fake_supervisor):
            return self.watch.tick()

    def advance(self, seconds):
        self.now += seconds
        self.ns += int(seconds * 1e9)

    def die(self):
        self.child.terminate()
        self.child.wait(timeout=5)

    def cleanup(self):
        record = self.watch.s.get('process')
        if record and record['pgid'] not in {child.pid for child in self.children}:
            process.signal_group(record, signal.SIGKILL)
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        if hasattr(self.watch, 'supervisor'):
            self.watch.supervisor.wait(timeout=5)


def run_drill(parent):
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='h3-FAKE-drill-', dir=parent)).resolve()
    results = {}
    for scenario in ('monitor_stale', 'no_progress', 'hps_identity_change', 'transient_then_repeated', 'success'):
        box = Sandbox(root / scenario)
        try:
            if scenario in ('monitor_stale', 'no_progress', 'hps_identity_change'):
                box.tick()
                if scenario == 'monitor_stale':
                    box.advance(31)
                elif scenario == 'no_progress':
                    box.advance(10801)
                    box.monitor()
                else:
                    box.health = dict(ok=False, identity_changed=True)
                assert box.tick()['state'] == 'STOPPING'
                box.child.wait(timeout=5)
                assert not process.inspect(box.config['process'])['alive']
                results[scenario] = 'training group stopped; HPS untouched'
            elif scenario == 'transient_then_repeated':
                fake_checkpoint(box.root / 'output')
                box.die()
                assert box.tick()['state'] == 'RECOVERING'
                assert box.tick()['state'] == 'LAUNCHING'
                for _ in range(100):
                    if box.tick()['state'] == 'WATCHING':
                        break
                    time.sleep(.05)
                else:
                    raise AssertionError('fake resume did not publish process identity')
                resumed = box.watch.s['process']
                process.signal_group(resumed, signal.SIGTERM)
                box.watch.supervisor.wait(timeout=5)
                assert box.tick()['state'] == 'RECOVERING'
                assert box.tick()['state'] == 'NEEDS_HUMAN'
                assert len(box.watch.s['resumes']) == 1
                results[scenario] = 'one resume; same checkpoint failure needs human'
            else:
                marker = fake_checkpoint(box.root / 'output')
                box.die()
                Store(box.root / 'output').save('SUCCESS', dict(kind='visual_rl_h3_run_success', run_id='fake-run',
                    committed_steps=1, checkpoint_relative_path='checkpoints/step-1',
                    **{k: marker[k] for k in ('checkpoint_contract_id', 'progress_id', 'state_tree_id')}))
                record = box.config['process']
                # Synthetic receipt ONLY for this fake run, never inferred for adopted production processes.
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
    parser.add_argument('--parent', type=Path, default=Path('tmp/h3-watch-drills'))
    args = parser.parse_args()
    root, results = run_drill(args.parent)
    print(json.dumps(dict(artifacts=str(root), scenarios=results), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
