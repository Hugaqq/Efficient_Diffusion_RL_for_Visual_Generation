"""Conservative, single-owner video VAE borrowing; never an audio/media cache.

Only the fixed stateless vendor video class is eligible. Metadata/version
checks are invalidation sentinels, not a proof for arbitrary Python plugins or
an atomic filesystem lock. File SHA authority comes from binding's full hash.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import random
import stat
import threading

import numpy as np
import torch

from visual_rl.composition.preflight.artifacts import FileHashObservation, _file_metadata
from visual_rl.runtime import h3_memory as memory

VIDEO_CAP = 16 * 1024**3
REUSE_VERSION = 'h3.owner-video-reuse.v1'


@dataclass(frozen=True)
class _VideoFiles:
    root: Path
    model_sha: str
    records: tuple[FileHashObservation, ...]


def _video_files(root, model_sha, records):
    values = tuple(r for r in records if r.relative.startswith('vae/'))
    if not values or any(not isinstance(r, FileHashObservation) for r in values):
        raise ValueError('video reuse requires records from the successful full model hash')
    return _VideoFiles(Path(root).resolve(), model_sha, values)


def _check_files(evidence, root):
    if not isinstance(evidence, _VideoFiles) or evidence.root != root:
        raise ValueError('video reuse requires binding-owned model file evidence')
    expected = {r.relative: r for r in evidence.records}
    video = root / 'vae'
    if any(p.is_symlink() for p in (root, video, *root.parents)):
        raise ValueError('video artifact path symlink changed')
    actual = {}
    directories = {'vae'}
    expected_directories = {str(p) for name in expected for p in Path(name).parents if str(p) != '.'}
    for path in video.rglob('*'):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError('video artifact node changed')
        if stat.S_ISDIR(info.st_mode): directories.add(path.relative_to(root).as_posix())
        if stat.S_ISREG(info.st_mode): actual[path.relative_to(root).as_posix()] = _file_metadata(info)
    if directories != expected_directories or actual != {name: r.metadata for name, r in expected.items()}:
        raise ValueError('video artifact list/stat/ctime changed since full verification')


def _rng(device):
    if device.type == 'cuda':
        if not torch.cuda.is_initialized() or torch.cuda.current_device() != device.index:
            return None  # no new CUDA device/RNG initialization for qualification
        try: cuda = torch.cuda.get_rng_state(device).numpy().tobytes()
        except Exception: return None
    else: cuda = None
    n = np.random.get_state()
    return (random.getstate(), n[0], n[1].tobytes(), n[2:], torch.get_rng_state().numpy().tobytes(), cuda)


def _supported_class(model):
    return (type(model).__module__ == 'diffusers.models.autoencoders.autoencoder_kl_minimax_h3'
            and type(model).__name__ == 'AutoencoderKLMiniMaxH3')


def _plain(value):
    if isinstance(value, torch.Tensor): raise ValueError('unregistered tensor state')
    if value is None or type(value) in (str, bool, int, float): return value
    if isinstance(value, (torch.dtype, torch.device)): return str(value)
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)): return sorted(_plain(v) for v in value)
    raise ValueError('uncontrolled Python module state')


def _state(model, device):
    tensors, modules = [], []
    for name, module in model.named_modules():
        if module.training: raise ValueError('video VAE left eval mode')
        attrs = {}
        for key, value in vars(module).items():
            if key in ('_parameters', '_buffers', '_modules'): continue
            if 'hook' in key and value: raise ValueError('uncontrolled video hooks')
            attrs[key] = _plain(value)
        modules.append((name, id(module), type(module), json.dumps(attrs, sort_keys=True)))
    size = 0
    for name, t in (*model.named_parameters(), *model.named_buffers()):
        if t.device != device or (t.is_floating_point() and t.dtype != torch.float32) or t.requires_grad or t.grad is not None:
            raise ValueError('video parameter/buffer device/dtype/grad drift')
        size += t.numel() * t.element_size()
        tensors.append((name, id(t), t.data_ptr(), tuple(t.shape), t.dtype, t.device, t._version))
    return (type(model), id(type(model).decode), tuple(modules), tuple(tensors)), size


class VideoReuse:
    def __init__(self, decoder, evidence):
        _check_files(evidence, decoder._root)
        self.decoder, self.evidence = decoder, evidence
        self.thread = threading.get_ident()
        self.device = getattr(decoder, '_device', torch.device('cpu'))
        self.model = self.signature = self.geometry = None
        self.bytes = 0
        self.allowed = True
        self.load_baseline = 0

    def check(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError('video reuse belongs to the owner producer thread')
        if getattr(self.decoder, '_device', torch.device('cpu')) != self.device:
            raise ValueError('video reuse device changed')
        _check_files(self.evidence, self.decoder._root)

    def event(self, kind, **fields):
        memory.event('video_reuse_' + kind, cached_bytes=self.bytes, **fields)

    def drop(self, reason, *, disable=False):
        # Clear owned references BEFORE any fallible logging/cache cleanup.
        self.model = self.signature = self.geometry = None
        self.bytes = 0
        if disable: self.allowed = False
        error = None
        try: self.event('drop', reason=reason)
        except BaseException as exc: error = exc
        try: self.decoder._release_device_cache()
        except BaseException as cleanup:
            if error is None: error = cleanup
            else: error.add_note('video drop cleanup failed: ' + repr(cleanup))
        if error is not None: raise error

    def admission(self, *, audio=False):
        self.check()
        if self.device.type != 'cuda': return
        free = self.decoder._card_free()
        if free < 40 and self.model is not None:
            self.drop('audio-headroom' if audio else 'warm-headroom', disable=True)
            free = self.decoder._card_free()
        if free < 40: raise BlockingIOError('video reuse needs actual40GiB free; no virtual free')

    def borrow(self, geometry, config, loader):
        self.check()
        key = (geometry, json.dumps(config, sort_keys=True))
        if self.model is not None:
            signature, size = _state(self.model, self.device)
            if self.geometry != key or signature != self.signature:
                raise ValueError('cached video VAE registered state/config/geometry changed')
            self.event('hit')
            return self.model
        before = _rng(self.device) if self.allowed else None
        self.load_baseline = self.decoder._allocated() if self.device.type == 'cuda' else 0
        model = loader()
        after = _rng(self.device) if self.allowed else None
        if not self.allowed: return model
        if before is None or before != after:
            self.allowed = False
            self.event('ineligible', reason='loader-global-RNG-or-unobservable')
            return model
        try:
            if not _supported_class(model): raise ValueError('only fixed stateless video class supported')
            signature, size = _state(model, self.device)
            actual = max(size, self.decoder._allocated() - self.load_baseline) if self.device.type == 'cuda' else size
            if actual > VIDEO_CAP: raise ValueError('video persistent byte cap exceeded')
        except (ValueError, RuntimeError) as error:
            self.allowed = False
            self.event('ineligible', reason=str(error))
            return model
        self.model, self.signature, self.geometry = model, signature, key
        self.bytes = actual
        self.event('miss')
        return model

    def after_decode(self, model):
        self.check()
        if self.model is None: return
        signature, size = _state(model, self.device)
        if model is not self.model or signature != self.signature:
            raise ValueError('video decode changed registered state')
        actual = max(size, self.decoder._allocated() - self.load_baseline) if self.device.type == 'cuda' else size
        if actual > VIDEO_CAP:
            self.allowed = False  # drop after the borrower's local ref expires
        self.bytes = actual


@contextmanager
def video_scope(decoder, evidence):
    decoder._require_open()
    if decoder._reuse is not None or decoder._lock.locked():
        raise RuntimeError('video reuse scope cannot overlap a scope/decode lease')
    state = VideoReuse(decoder, evidence)
    decoder._reuse = state
    primary = None
    try:
        state.event('scope_begin', version=REUSE_VERSION)
        yield
        state.check()
    except BaseException as error:
        primary = error
        decoder._poisoned = True
        raise
    finally:
        if threading.get_ident() != state.thread or decoder._lock.locked():
            decoder._poisoned = True
            raise RuntimeError('video scope release requires its producer and no active lease')
        try:
            state.drop('owner-scope-release')
        except BaseException as cleanup:
            decoder._poisoned = True
            if primary is None: raise
            primary.add_note('video scope cleanup failed: ' + repr(cleanup))
        finally: decoder._reuse = None
