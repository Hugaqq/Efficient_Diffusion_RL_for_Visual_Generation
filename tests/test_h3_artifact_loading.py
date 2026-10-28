"""Focused direct-IO regression cases; not evidence of real H3/GPU loading."""
import json

import pytest
import torch
from safetensors.torch import save_file

from visual_rl.models.implementations.h3_loading import _read_owned_tensor


def test_h3_owned_io_matches_safetensors_and_outlives_file(tmp_path):
    source = {
        'scalar': torch.tensor(1.25),
        'bf16': torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        'fp16': torch.linspace(-2, 2, 12, dtype=torch.float16).reshape(3, 4),
        'fp32': torch.linspace(-3, 3, 18).reshape(2, 9),
    }
    path = tmp_path / 'weights.safetensors'
    save_file(source, str(path))
    loaded = {}
    with path.open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        header = json.loads(stream.read(size))
        for name in source:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                tensor = _read_owned_tensor(stream, header[name], size + 8, dtype)
                assert torch.equal(tensor, source[name].to(dtype))
                assert tensor.device.type == 'cpu' and tensor.dtype == dtype
                loaded[name, dtype] = tensor
    # No input mmap alias can survive: the bytes can change without changing
    # the returned state, including the no-dtype-conversion path.
    path.write_bytes(b'overwritten')
    for (name, dtype), tensor in loaded.items():
        assert torch.equal(tensor, source[name].to(dtype))


def test_h3_owned_io_rejects_truncation_nonfinite_and_overflow(tmp_path):
    path = tmp_path / 'bad.safetensors'
    for value in (float('nan'), float('inf'), 1e10):
        save_file({'x': torch.tensor([value], dtype=torch.float32)}, str(path))
        with path.open('rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            header = json.loads(stream.read(size))
            with pytest.raises(ValueError, match='nonfinite'):
                _read_owned_tensor(stream, header['x'], size + 8, torch.float16)
    save_file({'x': torch.ones(8)}, str(path))
    with path.open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        header = json.loads(stream.read(size))
    path.write_bytes(path.read_bytes()[:-1])
    with path.open('rb') as stream:
        with pytest.raises(ValueError, match='truncated'):
            _read_owned_tensor(stream, header['x'], size + 8, torch.float32)
