"""Allocator tests use tiny tensors, not a claim of full192-video acceptance."""
import gc
import os
import shutil

import pytest
import torch

from visual_rl.data.media_storage import allocate_cpu_media, FileBackedMedia


@pytest.fixture
def portable_small_file(monkeypatch):
    # macOS test host lacks posix_fallocate. Only these tiny unit files use a
    # truncate shim; production always requires actual POSIX block reservation.
    if not hasattr(os, 'posix_fallocate'):
        monkeypatch.setattr(os, 'posix_fallocate', lambda fd, offset, count: os.ftruncate(fd, offset + count), raising=False)
    usage = shutil._ntuple_diskusage(128*1024**3, 0, 128*1024**3)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: usage)


def test_default_in_memory_unchanged(monkeypatch):
    monkeypatch.delenv('VISUAL_RL_MEDIA_MMAP_DIR', raising=False)
    tensor, backing = allocate_cpu_media((2, 3, 2, 2), torch.float32)
    assert tensor.shape == (2, 3, 2, 2) and tensor.device.type == 'cpu'
    assert backing is None


def test_exact_pixels_flush_and_mapping_lifetime(tmp_path, monkeypatch, portable_small_file):
    monkeypatch.setenv('VISUAL_RL_MEDIA_MMAP_DIR', str(tmp_path))
    tensor, backing = allocate_cpu_media((3, 2, 3), torch.float32)
    expected = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 19
    for start in range(3):
        tensor[start:start+1].copy_(expected[start:start+1])
        backing.flush_rows(start, start+1)
    assert torch.equal(tensor, expected)
    del backing
    gc.collect()
    assert torch.equal(tensor, expected)
    # Tensor storage, not a filename in the directory, owns the unlinked file.
    assert list(tmp_path.iterdir()) == []


def test_disk_headroom_before_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(1, 1, 0))
    with pytest.raises(BlockingIOError):
        FileBackedMedia((2, 3), torch.float32, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_no_implicit_precision_change(tmp_path, portable_small_file):
    with pytest.raises(ValueError):
        FileBackedMedia((2, 3), torch.float16, tmp_path)
