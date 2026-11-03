#!/usr/bin/env python3
"""Private persistent HPSv3 JSON worker; never changes an existing service.

CPU weights persist between requests. One video is scored serially (all frames,
frame microbatch1); default GPU use offloads per request. Opt-in dedicated-UUID
residency retains one model only with original headroom and a 60s idle lease.
Opt-in --max-queue N admits at most N requests (one scoring, N-1 waiting) and
still scores strictly one at a time in admission order; a full queue answers
503 + Retry-After before any scoring. Default 0 keeps the single-threaded server.
The artifact manifest hashes actual tensors/processor/code and declared policy.
"""
from __future__ import annotations
import argparse
import base64
from dataclasses import asdict, is_dataclass
import hashlib
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from io import BytesIO
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback

from visual_rl.core.protocols.flash_hpsv3 import (
    PROTOCOL, VERSION, MAX_REQUEST_BYTES, VideoContract, aggregate_frame_scores,
    canonical_json, decode_json, digest, scalar_fp32,
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def loaded_tensor_sha(model) -> str:
    """Streaming hash of actual CPU state, without a model-sized bytes copy."""
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.device.type != 'cpu':
            raise ValueError('identity must be measured with actual weights on CPU')
        metadata = canonical_json({'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)})
        h.update(len(metadata).to_bytes(8, 'big')); h.update(metadata)
        data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        view = memoryview(data).cast('B')
        for start in range(0, len(view), 16 * 1024 * 1024):
            h.update(view[start:start + 16 * 1024 * 1024])
    return h.hexdigest()


RESIDENCY_TTL = 60.


def residency_policy(enabled=False, expected_uuid=None):
    """Declared policy, not evidence of device availability or exclusivity."""
    if type(enabled) is not bool:
        raise TypeError('gpu_resident must be bool')
    if not enabled:
        if expected_uuid is not None:
            raise ValueError('resident GPU UUID requires --gpu-resident')
    else:
        from visual_rl.runtime.h3_session import _uuid
        if not isinstance(expected_uuid, str) or _uuid(expected_uuid) != expected_uuid:
            raise ValueError('resident mode requires a complete canonical GPU UUID')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid:
            raise ValueError('resident mode requires exactly its single UUID in CUDA_VISIBLE_DEVICES')
    return dict(version='hps.dedicated-idle-residency.v1', enabled=enabled,
                expected_gpu_uuid=expected_uuid, idle_ttl_seconds=RESIDENCY_TTL)


QUEUE_FULL_ERROR = 'hpsv3_queue_full'
QUEUE_RETRY_AFTER_S = 1
MAX_QUEUE = 64


def queue_policy(max_queue=0):
    """Declared admission policy; None (absent from identity) for the default server."""
    if type(max_queue) is not int or not 0 <= max_queue <= MAX_QUEUE:
        raise ValueError(f'max_queue must be an int in 0..{MAX_QUEUE}')
    if max_queue == 0:
        return None
    return dict(version='hps.bounded-fifo-queue.v1', max_queue=max_queue, scoring_concurrency=1,
                order='admission', full='503-retry-after-before-scoring')


class QueueFull(Exception):
    """The request was rejected BEFORE admission; nothing was scored."""


class QueueClosed(Exception):
    """The server is shutting down; this request was never scored."""


class ScoreQueue:
    """Bounded FIFO tickets: at most `capacity` admitted requests, one holds the turn.

    Tickets are served strictly in admission order. Every admitted ticket is
    released in a finally block, so a failed request never blocks later ones.
    The idle-housekeeping ticket is counted separately and never consumes
    request capacity. close() wakes all waiters (they abandon unscored).
    """
    def __init__(self, capacity):
        if queue_policy(capacity) is None:
            raise ValueError('a score queue needs capacity >= 1')
        self.capacity = capacity
        self._condition = threading.Condition()
        self._next = self._serving = 0
        self._idle_held = self._closed = self._turn_held = False
        self.admitted_order = deque(maxlen=256)  # recent admission labels; observability only

    @property
    def outstanding(self):
        """Admitted REQUESTS not yet released (excludes a held idle ticket)."""
        with self._condition:
            return self._next - self._serving - int(self._idle_held)

    def _admit(self, label):
        if self._closed:
            raise QueueClosed('HPS request queue closed')
        if self._next - self._serving - int(self._idle_held) >= self.capacity:
            raise QueueFull('HPS request queue full')
        ticket = self._next
        self._next += 1
        self.admitted_order.append(label)
        return ticket

    def _release(self, ticket, *, idle=False):
        with self._condition:
            if ticket != self._serving:
                raise RuntimeError('HPS queue released out of order')
            if idle:
                self._idle_held = False
            self._turn_held = False
            self._serving += 1
            self._condition.notify_all()

    @contextmanager
    def turn(self, label=None):
        with self._condition:
            ticket = self._admit(label)
            self._condition.wait_for(lambda: self._serving == ticket or self._closed)
            if self._closed or self._serving != ticket:
                raise QueueClosed('HPS server closing; queued request abandoned unscored')
            self._turn_held = True
        try:
            yield ticket
        finally:
            self._release(ticket)

    def try_idle_turn(self):
        """Ticket only when nothing is admitted; housekeeping then excludes scoring."""
        with self._condition:
            if self._closed or self._next != self._serving:
                return None
            ticket = self._next
            self._next += 1
            self._idle_held = self._turn_held = True
            return ticket

    def release_idle(self, ticket):
        self._release(ticket, idle=True)

    def close(self, timeout_s):
        """Refuse/abandon waiters, then wait (bounded) for the active turn only."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            return self._condition.wait_for(lambda: not self._turn_held, timeout=timeout_s)


def _cleanup(primary, action):
    try: action()
    except BaseException as error:
        if primary is None: return error
        primary.add_note('HPS cleanup failed: ' + repr(error))
    return primary


class HpsBackend:
    def __init__(self, checkpoint: Path, config: Path, contract: VideoContract, min_free_gib: float,
                 *, gpu_resident=False, resident_gpu_uuid=None, max_queue=0):
        self._initialize_residency(gpu_resident, resident_gpu_uuid)
        request_queue = queue_policy(max_queue)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        import hpsv3
        from hpsv3 import HPSv3RewardInferencer
        torch.set_num_threads(2)
        if checkpoint.suffix != '.safetensors':
            raise ValueError('only explicit safetensors reward checkpoints are accepted')
        self.torch, self.contract, self.min_free_gib = torch, contract, min_free_gib
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if available < 80 * 1024**3:
            raise BlockingIOError('need80GiB available host RAM for safe cold HPSv3 loading')
        self.inferencer = HPSv3RewardInferencer(device='cpu', checkpoint_path=str(checkpoint), config_path=str(config))
        self.inferencer.model.eval().requires_grad_(False)
        processor = self.inferencer.processor
        processor_state = {
            'tokenizer_backend': processor.tokenizer.backend_tokenizer.to_str(),
            'special_tokens': processor.tokenizer.special_tokens_map,
            'chat_template': processor.chat_template,
            'image_processor': processor.image_processor.to_dict(),
        }
        package_root = Path(inspect.getfile(hpsv3)).parent
        source_files = {str(path.relative_to(package_root)): file_sha(path) for path in sorted(package_root.rglob('*.py'))}
        facts = {
            'loaded_model_state_sha256': loaded_tensor_sha(self.inferencer.model),
            'loaded_processor_sha256': digest(processor_state),
            'checkpoint_file_sha256': file_sha(checkpoint),
            'config_file_sha256': file_sha(config),
            'hps_package_source_sha256': digest(source_files),
            'service_source_sha256': file_sha(Path(__file__)),
            'wire_source_sha256': file_sha(Path(inspect.getfile(VideoContract))),
            'torch_version': torch.__version__,
            'transformers_version': importlib.metadata.version('transformers'),
            'use_special_tokens': self.inferencer.use_special_tokens,
            'data_config': asdict(self.inferencer.data_config) if is_dataclass(self.inferencer.data_config) else vars(self.inferencer.data_config),
            'gpu_residency_policy': dict(self._policy),
        }
        if request_queue is not None:  # absent by default: unchanged identity fact set
            facts['request_queue_policy'] = request_queue
        self._health = {
            'status': 'ready', 'reward': 'hpsv3', 'protocol': PROTOCOL, 'protocol_version': VERSION,
            'identity_facts': facts, 'video_contract': contract.payload(),
            'service_identity': digest({'identity_facts': facts, 'video_contract': contract.payload()}),
            'result_device': 'cpu', 'result_dtype': 'fp32',
        }

    def score(self, sample_id: str, prompt: str, images: list) -> list[float]:
        from visual_rl.runtime import h3_memory as memory
        try:
            with memory.span('hps_score',sample_id=sample_id,input_height=self.contract.height,input_width=self.contract.width):
                return self._score(sample_id,prompt,images)
        except BaseException as exc:
            if not (isinstance(exc, BlockingIOError) and self._location == 'cpu' and not self._poisoned):
                self._poisoned = True
            if not self._active and self._location != 'cpu' and not self._release_failed:
                _cleanup(exc, lambda: self._offload('score-scope-error', sample_id=sample_id))
            if memory.enabled(): _cleanup(exc, lambda: memory.journal().exception(exc))
            raise

    def _initialize_residency(self, enabled=False, expected_uuid=None):
        self._policy = residency_policy(enabled, expected_uuid)
        self._location = 'cpu'
        self._active = self._poisoned = self._closed = self._release_failed = False
        self._last_used = None
        self._onloads = self._offloads = self._hits = 0
        self._onload_attempts = self._offload_attempts = 0

    @property
    def ready(self):
        return not (self._closed or self._poisoned) and self._location in ('cpu', 'cuda')

    @property
    def health(self):
        return self._health if self.ready else dict(self._health, status='unready')

    def _event(self, kind, **fields):
        from visual_rl.runtime import h3_memory as memory
        payload = dict(state=self._location, enabled=self._policy['enabled'],
            declared_uuid=self._policy['expected_gpu_uuid'], onloads=self._onloads,
            offloads=self._offloads, hits=self._hits, onload_attempts=self._onload_attempts,
            offload_attempts=self._offload_attempts, **fields)
        memory.event('hps_residency_' + kind, **payload)
        print(json.dumps(dict(event='hps_residency_' + kind, **payload)), flush=True)

    def _guard_gpu(self):
        if not self._policy['enabled']: return None
        try:
            expected = self._policy['expected_gpu_uuid']
            if os.environ.get('CUDA_VISIBLE_DEVICES') != expected or self.torch.cuda.current_device() != 0:
                raise ValueError('resident CUDA_VISIBLE_DEVICES/current logical device changed')
            from visual_rl.runtime.h3_session import _uuid
            value = getattr(self.torch.cuda.get_device_properties(0), 'uuid', None)
            uuid_type = getattr(self.torch._C, '_CUuuid', ())
            observed = _uuid(str(value) if isinstance(value, uuid_type) else value)
            if observed != expected:
                raise ValueError('observed resident GPU UUID differs from declaration')
            return observed
        except BaseException:
            self._poisoned = True
            if self._location != 'cpu': self._location = 'unknown'
            raise

    def _check_location(self, expected):
        params = tuple(self.inferencer.model.parameters())
        tensors = (*params, *self.inferencer.model.buffers())
        if (self.inferencer.model.training or any(p.requires_grad for p in params)
                or not tensors or self.torch.device(self.inferencer.device).type != expected
                or any(t.device.type != expected or
                       (expected == 'cuda' and self._policy['enabled'] and t.device.index != 0)
                       for t in tensors)):
            self._location = 'unknown'
            raise RuntimeError('HPS model/inferencer actual position is inconsistent')

    def _free(self):
        free, total = self.torch.cuda.mem_get_info()
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (free, total))
                or not 0 <= free <= total or total <= 0):
            raise RuntimeError('invalid GPU free-memory telemetry')
        return free / 1024**3

    def _transfer(self, device, phase, sample_id):
        from visual_rl.runtime import h3_memory as memory
        attempted = False
        def move():
            nonlocal attempted
            attempted = True
            self._location = 'unknown'
            if device == 'cuda': self._onload_attempts += 1
            else: self._offload_attempts += 1
            self.inferencer.model.to(device)  # original dtype, no CPU scoring fallback
            self.inferencer.device = device
            self._check_location(device)
            self._location = device
            if device == 'cuda': self._onloads += 1
            else: self._offloads += 1
        try:
            with memory.span(phase, sample_id=sample_id): move()
        except BaseException as primary:
            # A broken journal must fail closed, but must not prevent the ONE
            # release attempt. Never replay a .to() that already ran/failed.
            if device == 'cpu' and not attempted: _cleanup(primary, move)
            raise

    def _unused_cache(self, reason):
        from visual_rl.runtime import h3_memory as memory
        self._guard_gpu()
        error = _cleanup(None, lambda: memory.event('empty_cache_before', reason=reason))
        error = _cleanup(error, self.torch.cuda.empty_cache)
        error = _cleanup(error, lambda: memory.event('empty_cache_after', reason=reason))
        if error is not None: raise error

    def _offload(self, reason, *, sample_id=None, phase='hps_offload'):
        if self._location == 'cpu': return
        if self._release_failed:
            raise RuntimeError('previous HPS release failed; GPU state is unknown')
        try:
            observed = self._guard_gpu()
            error = _cleanup(None, lambda: self._transfer('cpu', phase, sample_id))
            error = _cleanup(error, lambda: self._unused_cache('existing HPS return-to-CPU cleanup: ' + reason))
            if error is not None: raise error
            self._last_used = None
            self._event('offload', reason=reason, observed_uuid=observed)
        except BaseException:
            self._poisoned = True
            self._release_failed = self._location != 'cpu'
            raise

    def _admit(self, sample_id):
        observed = self._guard_gpu()
        self._check_location(self._location)
        free = self._free()
        if self._location == 'cuda':
            expired = self._last_used is None or time.monotonic() - self._last_used >= RESIDENCY_TTL
            if expired or free < self.min_free_gib:
                reason = 'ttl' if expired else 'headroom'
                self._event('fallback', reason=reason, observed_uuid=observed, free_gib=free)
                self._offload(reason, sample_id=sample_id)
                free = self._free()  # REAL free after unloading, not free+weight
            else:
                self._hits += 1
                self._event('hit', sample_id=sample_id, observed_uuid=observed, free_gib=free)
        if free < self.min_free_gib:
            raise BlockingIOError('insufficient GPU admission headroom')
        return free

    def idle_housekeeping(self):
        if (not self._policy['enabled'] or self._active or not self.ready or self._location != 'cuda'):
            return
        try:
            observed = self._guard_gpu()
            self._check_location('cuda')
            free = self._free()
            if time.monotonic() - self._last_used >= RESIDENCY_TTL or free < self.min_free_gib:
                reason = 'idle-ttl' if time.monotonic() - self._last_used >= RESIDENCY_TTL else 'idle-headroom'
                self._event('idle', reason=reason, observed_uuid=observed, free_gib=free)
                self._offload(reason, phase='hps_idle_offload')
        except BaseException as error:
            self._poisoned = True
            if self._location != 'cpu' and not self._release_failed:
                _cleanup(error, lambda: self._offload('idle-error', phase='hps_idle_offload'))
            raise

    def close(self):
        if self._closed: return
        if self._active: raise RuntimeError('cannot close an active HPS request concurrently')
        self._closed = True
        if self._location != 'cpu': self._offload('shutdown', phase='hps_close_offload')

    def _score(self, sample_id: str, prompt: str, images: list) -> list[float]:
        if not self.ready or self._active:
            raise RuntimeError('HPS backend is unready/closed or already scoring')
        torch = self.torch
        from visual_rl.runtime import h3_memory as memory
        started = time.monotonic()
        stop = threading.Event()
        thread = None
        monitor_started = False
        monitor_errors, free_samples = [], []
        primary = None
        self._active = True
        def check_monitor():
            if monitor_errors: raise RuntimeError('HPS GPU memory monitor failed') from monitor_errors[0]
            if free_samples and min(free_samples) < 16:
                raise BlockingIOError('GPU safety margin fell below16GiB; do not commit scores')
        def monitor():
            try:
                while not stop.wait(.1): free_samples.append(self._free())
            except BaseException as error:
                monitor_errors.append(error); stop.set()
        def join_monitor():
            thread.join(timeout=3)
            if thread.is_alive(): raise RuntimeError('HPS monitor did not terminate; state unsafe')
        try:
            free_samples.append(self._admit(sample_id))
            thread = threading.Thread(target=monitor, name='hps-request-memory', daemon=True)
            thread.start(); monitor_started = True
            if memory.enabled(): memory.event('legacy_reset_suppressed', reason='unified HPS score segment owns reset')
            else: torch.cuda.reset_peak_memory_stats()
            if self._location == 'cpu':
                observed = self._guard_gpu()
                self._event('cold', sample_id=sample_id, observed_uuid=observed, free_gib=free_samples[0])
                self._transfer('cuda', 'hps_onload', sample_id)
            scores = []
            with torch.inference_mode():
                for image in images:
                    check_monitor()
                    value = self.inferencer.reward(prompts=[prompt], image_paths=[image])
                    scores.append(scalar_fp32(float(value[0][0].item())))
            self._guard_gpu()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            free_samples.append(self._free())
            check_monitor()
            observed = self._guard_gpu()
            self._check_location('cuda')
            if self._policy['enabled'] and self._free() >= self.min_free_gib:
                self._unused_cache('existing HPS unused-cache cleanup; weights retained')
                free = self._free()
                if free >= self.min_free_gib:
                    self._last_used = time.monotonic()
                    self._event('retain', sample_id=sample_id, observed_uuid=observed, free_gib=free)
                else: self._offload('post-score-headroom', sample_id=sample_id)
            else: self._offload('per-request' if not self._policy['enabled'] else 'post-score-headroom', sample_id=sample_id)
        except BaseException as error:
            primary = error
        finally:
            # Keep monitoring through the original offload cleanup as well.
            if primary is not None and self._location != 'cpu' and not self._release_failed:
                primary = _cleanup(primary, lambda: self._offload('request-error', sample_id=sample_id))
            stop.set()
            if monitor_started: primary = _cleanup(primary, join_monitor)
            primary = _cleanup(primary, check_monitor)
            # A late monitor/margin error must not leave a reusable warm model.
            if primary is not None and self._location != 'cpu' and not self._release_failed:
                primary = _cleanup(primary, lambda: self._offload('late-monitor-error', sample_id=sample_id))
            if primary is not None:
                clean_admission_rejection = (thread is None and isinstance(primary, BlockingIOError)
                                             and self._location == 'cpu' and not self._poisoned)
                if not clean_admission_rejection: self._poisoned = True
            self._active = False
        if primary is not None: raise primary
        try:
            print(json.dumps({'event': 'real_hpsv3_video_scored', 'sample_id': sample_id,
                              'frames': len(scores), 'seconds_including_transfers': time.monotonic()-started,
                              'own_peak_allocated_gib': peak, 'min_card_free_gib': min(free_samples),
                              'peak_scope':('last unified segment; use event ancestry for full score peak' if memory.enabled() else 'score-call')}), flush=True)
        except BaseException as error:
            self._poisoned = True
            if self._location != 'cpu' and not self._release_failed:
                _cleanup(error, lambda: self._offload('score-log-error', sample_id=sample_id))
            raise
        return scores


def make_handler(backend, queue=None):
    if queue is not None and not isinstance(queue, ScoreQueue):
        raise TypeError('queue must be a ScoreQueue')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def reply(self, status: int, value: dict, headers=()):
            body = canonical_json(value)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            for name, header in headers: self.send_header(name, header)
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path != '/healthz':
                return self.reply(404, {'error': 'not_found'})
            self.reply(200 if backend.ready else 503, backend.health)
        def do_POST(self):
            if self.path != '/score':
                return self.reply(404, {'error': 'not_found'})
            # Queued mode: `ready` is transiently False while the turn holder
            # moves weights (_location 'unknown'); only terminal state rejects
            # here, and readiness is rechecked inside the exclusive turn.
            if (backend._closed or backend._poisoned) if queue is not None else not backend.ready:
                return self.reply(503, {'error': 'hpsv3_unready'})
            try:
                self.connection.settimeout(30)
                if self.headers.get('Content-Type', '').split(';', 1)[0] != 'application/json':
                    raise ValueError('JSON required')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError('invalid request byte count')
                data = decode_json(self.rfile.read(length))
                if set(data) != {'protocol', 'protocol_version', 'service_identity', 'sample_id', 'prompt', 'jpeg_frames'}:
                    raise ValueError('invalid request fields')
                if (data['protocol'], data['protocol_version'], data['service_identity']) != (PROTOCOL, VERSION, backend.health['service_identity']):
                    raise ValueError('wrong service/protocol identity')
                if not isinstance(data['sample_id'], str) or not 0 < len(data['sample_id']) <= 2048:
                    raise ValueError('invalid sample identity')
                if not isinstance(data['prompt'], str) or len(data['prompt'].encode('utf-8')) > 32768:
                    raise ValueError('invalid prompt')
                frames = data['jpeg_frames']
                if not isinstance(frames, list) or len(frames) != backend.contract.frames:
                    raise ValueError('all frames are required')
                from PIL import Image
                images = []
                for encoded in frames:
                    raw = base64.b64decode(encoded, validate=True)
                    if len(raw) > 2 * 1024 * 1024:
                        raise ValueError('oversized frame')
                    with Image.open(BytesIO(raw), formats=['JPEG']) as image:
                        if image.size != (backend.contract.width, backend.contract.height):
                            raise ValueError('frame shape mismatch')
                        images.append(image.convert('RGB'))
            except Exception:
                return self.reply(400, {'error': 'invalid_video_request'})
            try:
                if queue is None:
                    scores = backend.score(data['sample_id'], data['prompt'], images)
                else:
                    # Admission happens only after full validation; the turn is
                    # released before the reply so a slow reader never blocks.
                    with queue.turn(data['sample_id']):
                        if not backend.ready:  # an earlier queued request poisoned it
                            return self.reply(503, {'error': 'hpsv3_unready'})
                        scores = backend.score(data['sample_id'], data['prompt'], images)
                score, _indices = aggregate_frame_scores(scores, backend.contract)
                self.reply(200, {'protocol': PROTOCOL, 'protocol_version': VERSION,
                    'service_identity': backend.health['service_identity'], 'sample_id': data['sample_id'],
                    'frame_scores': scores, 'video_score': score})
            except QueueFull:
                self.reply(503, {'error': QUEUE_FULL_ERROR}, (('Retry-After', str(QUEUE_RETRY_AFTER_S)),))
            except QueueClosed:
                self.reply(503, {'error': 'hpsv3_unready'})
            except BlockingIOError:
                self.reply(503, {'error': 'gpu_admission_or_margin'})
            except Exception:
                traceback.print_exc()
                self.reply(500, {'error': 'hpsv3_scoring_failed'})
    return Handler


class HpsHTTPServer(HTTPServer):
    def service_actions(self):
        # Called by serve_forever on the SAME serial HTTP thread, never a new
        # idle worker. TTL has poll/scheduling delay, not hard real-time bounds.
        self.backend.idle_housekeeping()


class HpsQueuedHTTPServer(ThreadingHTTPServer):
    """Opt-in: per-connection threads only parse/validate; ScoreQueue serializes GPU work."""
    daemon_threads = True  # shutdown is bounded by ScoreQueue.close, never an unbounded join
    shutdown_timeout_s = 120.

    def __init__(self, address, handler, score_queue):
        if not isinstance(score_queue, ScoreQueue):
            raise TypeError('queued server requires a ScoreQueue')
        self.score_queue = score_queue
        self.request_queue_size = max(5, 2 * score_queue.capacity)  # listen backlog
        super().__init__(address, handler)

    def service_actions(self):
        # Idle housekeeping takes a queue ticket, so it never overlaps a score.
        ticket = self.score_queue.try_idle_turn()
        if ticket is None:
            return
        try:
            self.backend.idle_housekeeping()
        finally:
            self.score_queue.release_idle(ticket)

    def server_close(self):
        # Waiters abandon unscored (503); the active score gets a bounded
        # drain so backend.close never offloads underneath it. If it does not
        # finish, backend.close refuses (active request) and the process exits.
        drained = self.score_queue.close(self.shutdown_timeout_s)
        super().server_close()
        if not drained:
            raise RuntimeError('active HPS score did not finish within the shutdown bound')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--artifact-output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18820)
    p.add_argument('--frames', type=int, default=81)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--min-free-gib', type=float, default=40)
    p.add_argument('--gpu-resident', action='store_true', help='opt-in dedicated UUID idle residency; requires new deployment approval')
    p.add_argument('--resident-gpu-uuid')
    p.add_argument('--max-queue', type=int, default=0,
                   help='opt-in bounded FIFO admission of N requests (0 = single-threaded default); '
                        'changes the service identity; requires new deployment approval')
    a = p.parse_args()
    try: residency_policy(a.gpu_resident, a.resident_gpu_uuid)
    except (ValueError, TypeError, RuntimeError) as error: p.error(str(error))
    try: queue_policy(a.max_queue)
    except ValueError as error: p.error(str(error))
    if not 1 <= a.port <= 65535 or not 40 <= a.min_free_gib <= 80:
        p.error('invalid private port or safety admission budget')
    from visual_rl.runtime import h3_memory as memory
    import atexit
    if memory.enabled():memory.journal();atexit.register(memory.close)
    backend = server = None
    primary = None
    try:
        with memory.span('hps_initialization'):
            options = dict(gpu_resident=a.gpu_resident, resident_gpu_uuid=a.resident_gpu_uuid)
            if a.max_queue: options['max_queue'] = a.max_queue
            backend = HpsBackend(a.checkpoint.resolve(strict=True), a.config.resolve(strict=True),
                VideoContract(a.frames, a.height, a.width), a.min_free_gib, **options)
        if a.max_queue:
            score_queue = ScoreQueue(a.max_queue)
            server = HpsQueuedHTTPServer(('127.0.0.1', a.port), make_handler(backend, score_queue), score_queue)
        else:
            server = HpsHTTPServer(('127.0.0.1', a.port), make_handler(backend))
        server.backend = backend
        artifact = canonical_json(backend.health)
        a.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        if a.artifact_output.exists():
            if a.artifact_output.read_bytes() != artifact:
                raise ValueError('refusing to overwrite a different immutable HPSv3 artifact')
        else:
            with a.artifact_output.open('xb') as handle: handle.write(artifact)
        print('HPSV3_VIDEO_READY ' + backend.health['service_identity'], flush=True)
        server.serve_forever(poll_interval=.5)
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup = primary
        if server is not None: cleanup = _cleanup(cleanup, server.server_close)
        if backend is not None: cleanup = _cleanup(cleanup, backend.close)
        if primary is None and cleanup is not None: raise cleanup


if __name__ == '__main__':
    main()
