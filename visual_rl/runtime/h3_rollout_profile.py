"""Opt-in, one fixed rollout cell per rank; no runner or training ownership.

The trace covers forward/result gates and transition, NOT row-initial H2D or
whole rollout. CPU enqueue and CUDA durations overlap; NCCL includes gates,
not just FSDP. Publication limits do not cap Kineto RAM or temporary disk.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time

import torch

from visual_rl.runtime.h3_file_phase import file_phase

VERSION = 'h3.rollout-cell-profile.v1'
TARGET = dict(optimizer_step=1, row=1, index=0)
TRACE_LIMIT = 128 * 1024**2
RECEIPT_LIMIT = 64 * 1024
SETTINGS = dict(record_shapes=False, profile_memory=False, with_stack=False, with_flops=False)


def configuration(enabled):
    if type(enabled) is not bool:
        raise TypeError('profile_rollout must be bool')
    return dict(version=VERSION, enabled=enabled, target=dict(TARGET), activities=['CPU', 'CUDA'],
                settings=dict(SETTINGS), trace_limit=TRACE_LIMIT, receipt_limit=RECEIPT_LIMIT)


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('CUDA rollout profiler must start/stop on the main training thread')


def _check_support():
    _main_thread()
    api = torch.profiler
    if (not {api.ProfilerActivity.CPU, api.ProfilerActivity.CUDA} <= api.supported_activities()
            or not all(callable(getattr(api.profile, name, None))
                       for name in ('start', 'stop', 'events', 'export_chrome_trace'))):
        raise RuntimeError('public CPU+CUDA profiler activities/APIs required; no CPU fallback')


def _new_profiler():
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA], **SETTINGS)


def _device_counts(profiler):
    counts = {'CPU': 0, 'CUDA': 0}
    for event in profiler.events():
        for name in counts:
            if event.device_type == getattr(torch.autograd.DeviceType, name):
                counts[name] += 1
    if not all(counts.values()):
        raise RuntimeError('profile did not record both CPU and CUDA events; no CPU-only receipt')
    return counts


def _identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('profile file must be a regular non-symlink file')
    return info.st_dev, info.st_ino


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


class RolloutCellProfile:
    def __init__(self, *, output_dir, rank, gpu_uuid, launcher_sha, process_group):
        _check_support()
        import re
        if (type(rank) is not int or not 0 <= rank < 8
                or re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', gpu_uuid) is None
                or re.fullmatch(r'[0-9a-f]{40}', launcher_sha) is None):
            raise ValueError('admitted rank/UUID and clean launcher SHA required')
        output = Path(output_dir).absolute()
        if not output.is_dir() or any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError('profile output must be the new run directory without symlinks')
        root = output / 'rollout-profile'
        if root.is_symlink():
            raise ValueError('profile root symlink rejected')
        root.mkdir(exist_ok=True)
        self.directory = root / f'rank-{rank}'
        self.directory.mkdir()  # fail closed on ANY preexisting rank target
        info = self.directory.stat()
        self.directory_identity = (info.st_dev, info.st_ino)
        self.trace = self.directory / 'update-2-row-1-index-0.trace.json'
        self.receipt = self.directory / 'update-2-row-1-index-0.receipt.json'
        self.rank, self.gpu_uuid, self.launcher_sha = rank, gpu_uuid, launcher_sha
        self.process_group = process_group
        self.hits = 0
        self.receipt_sha = None

    @staticmethod
    def _gate(gate, label, work):
        result = error = None
        try: result = work()
        except Exception as exc: error = exc
        try: gate('rollout-profile/' + label, local_error=error is not None)
        except BaseException as peer:
            if error is not None: raise error from peer
            raise
        if error is not None: raise error
        return result

    def _check_directory(self):
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            raise ValueError('profile directory symlink rejected')
        info = self.directory.stat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self.directory_identity:
            raise ValueError('profile directory ownership changed')

    def _temp(self, owned):
        fd, name = tempfile.mkstemp(prefix='.profile-', dir=self.directory)
        path = Path(name)
        try: owned[path] = _identity(path)
        finally: os.close(fd)
        return path

    @staticmethod
    def _hash_owned(path, identity, limit):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (info.st_dev, info.st_ino) != identity or not stat.S_ISREG(info.st_mode):
                raise ValueError('profile temporary ownership changed')
            if info.st_size > limit:
                raise ValueError('profile trace publication byte budget exceeded')
            digest, size = hashlib.sha256(), 0
            while chunk := stream.read(1024**2):
                size += len(chunk)
                if size > limit:
                    raise ValueError('profile trace publication byte budget exceeded')
                digest.update(chunk)
            if not size: raise ValueError('empty profile trace')
            os.fsync(stream.fileno())
        return size, digest.hexdigest()

    def _publish(self, profiler, metadata, wall_seconds, counts):
        # stop/finalize has already returned on main. Only CPU serialization,
        # hashing and file IO run in the existing heartbeat file worker.
        owned = {}
        primary = None
        try:
            self._check_directory()
            if self.trace.exists() or self.trace.is_symlink() or self.receipt.exists() or self.receipt.is_symlink():
                raise FileExistsError('profile target already exists; never overwrite')
            temporary = self._temp(owned)
            profiler.export_chrome_trace(str(temporary))
            size, digest = self._hash_owned(temporary, owned[temporary], TRACE_LIMIT)
            receipt = dict(schema=VERSION, launcher_sha=self.launcher_sha, rank=self.rank,
                admitted_gpu_uuid=self.gpu_uuid, target=dict(TARGET), actual=metadata,
                activities=['CPU', 'CUDA'], observed_device_events=counts, settings=dict(SETTINGS),
                instrumented_wall_seconds=wall_seconds, trace_bytes=size, trace_sha256=digest,
                coverage=dict(row_initial_h2d=False, forward_gate=True, prediction=True,
                              transition=True, result_gate=True, whole_rollout=False),
                limitations=['instrumented one-cell window, not end-to-end timing',
                             'CPU and CUDA overlap; do not sum into wall time',
                             'NCCL includes gates, not solely FSDP',
                             'publication cap is not a Kineto RAM/temporary-disk cap'])
            raw = (json.dumps(receipt, sort_keys=True, allow_nan=False) + '\n').encode()
            if len(raw) > RECEIPT_LIMIT: raise ValueError('profile receipt byte budget exceeded')
            pending = self._temp(owned)
            with pending.open('wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            # Hard-link publication is atomic and NOREPLACE, unlike replace().
            os.link(temporary, self.trace, follow_symlinks=False)
            owned[self.trace] = owned[temporary]
            _sync_directory(self.directory)
            os.link(pending, self.receipt, follow_symlinks=False)
            owned[self.receipt] = owned[pending]
            _sync_directory(self.directory)
            receipt_sha = hashlib.sha256(raw).hexdigest()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup_error = None
            def remove(path, identity):
                if _identity(path) != identity:
                    raise RuntimeError('refusing to remove profile file with unknown ownership')
                path.unlink()
            for path, identity in owned.items():
                if primary is None and path in (self.trace, self.receipt): continue
                try: remove(path, identity)
                except BaseException as cleanup:
                    if primary is not None: primary.add_note('profile cleanup failed: ' + repr(cleanup))
                    elif cleanup_error is None: cleanup_error = cleanup
                    else: cleanup_error.add_note('profile cleanup failed: ' + repr(cleanup))
            if primary is None and cleanup_error is None:
                try: _sync_directory(self.directory)
                except BaseException as cleanup: cleanup_error = cleanup
            if cleanup_error is not None:
                # Even temp cleanup failure makes publication fail closed.
                for path in (self.receipt, self.trace):
                    if path in owned:
                        try: remove(path, owned[path])
                        except BaseException as cleanup: cleanup_error.add_note('profile rollback failed: ' + repr(cleanup))
                raise cleanup_error
        return receipt_sha

    @contextmanager
    def cell(self, *, optimizer_step, row, index, sample_id, selected_index, gate):
        if (optimizer_step, row, index) != (TARGET['optimizer_step'], TARGET['row'], TARGET['index']):
            yield
            return
        self.receipt_sha = None  # a failed/repeated target cannot retain a success status
        profiler = None
        started = False
        start = time.monotonic()
        def begin():
            nonlocal profiler, started
            _main_thread()
            if self.hits: raise RuntimeError('rollout profile target already attempted; no replay')
            self.hits += 1
            if (not isinstance(sample_id, str) or not sample_id or len(sample_id.encode()) > 256
                    or type(selected_index) is not int or not 0 <= selected_index < 5):
                raise ValueError('bounded sample identity and selected first-window index required')
            profiler = _new_profiler()
            profiler.start()
            started = True
        try:
            self._gate(gate, 'start', begin)
        except BaseException as primary:
            if profiler is not None:
                try: profiler.stop()
                except BaseException as cleanup: primary.add_note('profile start cleanup failed: ' + repr(cleanup))
            raise
        try:
            yield
        except BaseException as primary:
            # A failed model/collective follows whole-attempt teardown. Do not
            # add a rescue collective, publish a success receipt or replay body.
            if started:
                try:
                    _main_thread(); profiler.stop()
                except BaseException as cleanup: primary.add_note('profile stop failed: ' + repr(cleanup))
            raise
        else:
            def stop():
                _main_thread(); profiler.stop()
                return _device_counts(profiler)
            counts = self._gate(gate, 'stop', stop)
            wall = time.monotonic() - start
            metadata = dict(optimizer_step=optimizer_step, row=row, index=index,
                            sample_id=sample_id, selected_index=selected_index)
            self.receipt_sha = self._gate(gate, 'publish', lambda: file_phase(
                lambda: self._publish(profiler, metadata, wall, counts), rank=self.rank,
                group=self.process_group, label='rollout-profile-export'))

    def assert_complete(self):
        """Local-only final gate; no CUDA/PG work, no fallback for a missed hit."""
        if self.hits != 1 or self.receipt_sha is None:
            raise RuntimeError('enabled rollout profile did not publish exactly one target window')
        self._check_directory()
        _identity(self.trace); _identity(self.receipt)
        with self.receipt.open('rb') as stream: raw = stream.read(RECEIPT_LIMIT + 1)
        if len(raw) > RECEIPT_LIMIT or hashlib.sha256(raw).hexdigest() != self.receipt_sha:
            raise ValueError('profile receipt integrity mismatch')
        receipt = json.loads(raw)
        if self.trace.stat().st_size != receipt['trace_bytes']:
            raise ValueError('profile trace size changed')
