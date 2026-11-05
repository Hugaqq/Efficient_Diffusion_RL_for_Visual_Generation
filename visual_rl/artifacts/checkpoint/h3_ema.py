"""Rank-local EMA attachments, covered by the existing atomic checkpoint tree."""
import os
from pathlib import Path

import torch

from visual_rl.algorithms.trainer.h3_ema import EMA_STATE_SCHEMA
from visual_rl.artifacts.checkpoint.manager import _fsync_directory

DIRECTORY = 'h3-ema'


def check_ema_contract(ema, contract):
    expected = 'none' if ema is None else EMA_STATE_SCHEMA
    if contract.ema_state_schema != expected:
        raise ValueError('EMA owner differs from checkpoint ema_state_schema')


def save_ema(ema, path, *, step):
    ema.validate(step=step)
    root = Path(path) / DIRECTORY
    root.mkdir(exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('EMA attachment directory must be regular')
    with (root / f'rank-{ema.rank}.pt').open('xb') as stream:
        torch.save(ema.state_dict(), stream)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(root)
    return dict(schema=EMA_STATE_SCHEMA, step=step, updates=ema.updates)


def load_ema(ema, path, *, step):
    """Only after outer manager verifies the COMPLETE tree; no standalone trust bypass."""
    root = Path(path) / DIRECTORY
    if root.is_symlink() or not root.is_dir() or sorted(p.name for p in root.iterdir()) != [f'rank-{r}.pt' for r in range(8)]:
        raise ValueError('EMA checkpoint must cover exactly ranks 0..7')
    file = root / f'rank-{ema.rank}.pt'
    # Tensor bytes plus bounded per-parameter pickle/zip metadata. No unbounded file read.
    limit = sum(t.numel()*4 for t in ema.shadow.values()) + len(ema.shadow)*8192 + 65536
    if file.is_symlink() or not file.is_file() or not 0 < file.stat().st_size <= limit:
        raise ValueError('missing/oversized EMA rank attachment')
    state = torch.load(file, map_location='cpu', weights_only=True)
    ema.load_state_dict(state, step=step)
    return dict(schema=EMA_STATE_SCHEMA, step=step, updates=ema.updates)
