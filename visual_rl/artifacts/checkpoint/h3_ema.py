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


def export_ema_lora(checkpoint):
    """CPU-only reconstruction of complete WORLD8 Shard(0) EMA LoRA tensors."""
    import math
    from dataclasses import asdict
    from torch.distributed.checkpoint import FileSystemReader
    from visual_rl.artifacts.checkpoint.manager import AtomicCheckpointManager
    from visual_rl.composition.config.h3_profile import H3EMAConfig
    path = Path(checkpoint).resolve()
    inspection = AtomicCheckpointManager(path.parent).inspect_complete(path)
    if inspection.contract.ema_state_schema != EMA_STATE_SCHEMA:
        raise ValueError('checkpoint does not contain the H3 EMA schema')
    metadata = FileSystemReader(path / 'h3-state' / 'dcp').read_metadata()
    specs = {name[len('model.'):]: value for name,value in metadata.state_dict_metadata.items()
             if name.startswith('model.') and ('.lora_A.' in name or '.lora_B.' in name)}
    if not specs or any(value.properties.dtype != torch.float32 or len(value.size) != 2 for value in specs.values()):
        raise ValueError('EMA export requires nonempty FP32 2D LoRA metadata')
    pieces = {name: [] for name in specs}
    config = None
    root = path / DIRECTORY
    if root.is_symlink() or not root.is_dir() or sorted(p.name for p in root.iterdir()) != [f'rank-{r}.pt' for r in range(8)]:
        raise ValueError('EMA export requires all eight rank files')
    for rank in range(8):
        expected = {}
        for name,item in specs.items():
            shape = list(item.size)
            chunk = math.ceil(shape[0]/8)
            offset = min(rank*chunk, shape[0])
            expected[name] = dict(shape=shape, local_shape=[min(chunk, shape[0]-offset), *shape[1:]],
                                  offset=offset, dtype='torch.float32')
        file = root / f'rank-{rank}.pt'
        limit = sum(math.prod(spec['local_shape'])*4 for spec in expected.values()) + len(specs)*8192 + 65536
        if file.is_symlink() or not file.is_file() or not 0 < file.stat().st_size <= limit:
            raise ValueError('missing/oversized EMA export attachment')
        state = torch.load(file, map_location='cpu', weights_only=True)
        if type(state) is not dict or set(state) != {'schema','rank','world_size','config','committed_steps','updates','specs','shadow'}:
            raise ValueError('EMA export state schema mismatch')
        current = H3EMAConfig(**state['config'])
        config = current if config is None else config
        if (state['schema'] != EMA_STATE_SCHEMA or type(state['rank']) is not int or state['rank'] != rank
                or type(state['world_size']) is not int or state['world_size'] != 8
                or not current.enabled or current != config or state['config'] != asdict(config)
                or type(state['committed_steps']) is not int or state['committed_steps'] != inspection.progress.global_step
                or type(state['updates']) is not int or state['updates'] != state['committed_steps']//config.update_interval
                or state['specs'] != expected or set(state['shadow']) != set(specs)):
            raise ValueError('EMA export config/counter/shard metadata mismatch')
        for name,tensor in state['shadow'].items():
            if (type(tensor) is not torch.Tensor or tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                    or list(tensor.shape) != expected[name]['local_shape'] or tensor.requires_grad
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError('invalid EMA export tensor')
            pieces[name].append(tensor)
    return {name: torch.cat(parts, dim=0) for name,parts in pieces.items()}, str(path / 'h3-state' / 'dcp')


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
