"""Least-inflight HPS leases, shared by threads and optionally all local ranks."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import threading
from uuid import uuid4


def _fresh(origins):
    return dict(schema='hps.least-inflight.v1', origins=list(origins), generation=uuid4().hex,
                counts=[0]*len(origins), next=0, failed=False)


class HpsDispatcher:
    """One process by default; a run-owned file coordinates P2's eight processes.

    Failure poisons the entire pool. No lost lease reclamation or server removal:
    an unknown/crashed request must stop this attempt, not be retried elsewhere.
    """
    def __init__(self, origins, *, path=None):
        self.origins = tuple(origins)
        self.path = None if path is None else Path(path)
        self.lock = threading.Lock()
        self.state = _fresh(origins) if path is None else None
        with self._state() as state:
            self.generation = state['generation']

    @staticmethod
    def initialize(path, origins):
        # Called once by rank0 under the lifecycle's output lock, before acquire.
        from visual_rl.runtime.h3_lifecycle import _write_json
        path = Path(path)
        if path.is_symlink():
            raise ValueError('HPS dispatch state cannot be a symlink')
        _write_json(path, _fresh(origins))

    def _validate(self, state):
        if (type(state) is not dict or set(state) != {'schema','origins','generation','counts','next','failed'}
                or state['schema'] != 'hps.least-inflight.v1' or state['origins'] != list(self.origins)
                or type(state['generation']) is not str or len(state['generation']) != 32
                or (hasattr(self, 'generation') and state['generation'] != self.generation)
                or type(state['counts']) is not list or len(state['counts']) != len(self.origins)
                or any(type(v) is not int or not 0 <= v <= 4096 for v in state['counts'])
                or type(state['next']) is not int or not 0 <= state['next'] < len(self.origins)
                or type(state['failed']) is not bool):
            raise ValueError('HPS dispatcher schema/origins/generation/counts mismatch')

    @contextmanager
    def _state(self):
        """文件态仅在 with 正常退出并校验后回写/flush（不承诺 fsync 持久性）。

        with 体抛异常不回写；内存态直接借出对象，不提供回滚。崩溃只按
        fail-closed 处理，不做 lease 回收、计数修复或实例剔除；遗留在途
        计数不能解释为可安全重试，必须由生命周期终止该 attempt。
        """
        with self.lock:
            if self.path is None:
                self._validate(self.state)
                yield self.state
                return
            fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
            with os.fdopen(fd, 'r+b') as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                raw = stream.read(16385)
                if len(raw) > 16384:
                    raise ValueError('oversized HPS dispatch state')
                state = json.loads(raw)
                self._validate(state)
                yield state
                self._validate(state)
                stream.seek(0)
                stream.write(json.dumps(state, sort_keys=True, allow_nan=False).encode())
                stream.truncate()
                stream.flush()

    @contextmanager
    def lease(self):
        with self._state() as state:
            if state['failed']:
                raise RuntimeError('HPS instance failed; entire multi-origin pool is poisoned')
            order = [(state['next']+i) % len(self.origins) for i in range(len(self.origins))]
            chosen = min(order, key=lambda i: state['counts'][i])
            if state['counts'][chosen] == 4096:
                raise RuntimeError('HPS dispatch in-flight bound exceeded')
            state['counts'][chosen] += 1
            state['next'] = (chosen+1) % len(self.origins)
        success = False
        try:
            yield chosen
            success = True
        finally:
            with self._state() as state:
                if state['counts'][chosen] <= 0:
                    raise RuntimeError('HPS dispatch lease underflow')
                state['counts'][chosen] -= 1
                if not success:
                    state['failed'] = True
                failed = state['failed']
            if success and failed:
                raise RuntimeError('another HPS instance failed; refusing partial-pool results')
