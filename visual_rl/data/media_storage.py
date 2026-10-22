"""Optional file-backed CPU media allocation, preserving every float32 pixel.

VISUAL_RL_MEDIA_MMAP_DIR is a launch-only storage location, analogous to TMPDIR;
it changes neither model dtype, media values, row geometry nor algorithm math.
Use a disk filesystem, NOT tmpfs. Anonymous unlinked files are owned by their
mappings and reclaimed automatically after the last tensor reference disappears.
"""
from __future__ import annotations

import math
import mmap
import os
from pathlib import Path
import shutil
import tempfile


class FileBackedMedia:
    def __init__(self, shape: tuple[int, ...], dtype, directory: Path):
        import torch
        if dtype != torch.float32 or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('file-backed decoded media requires a positive float32 shape')
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError('media mmap location must be an existing disk directory')
        self.shape = shape
        self.row_bytes = math.prod(shape[1:]) * 4
        self.nbytes = math.prod(shape) * 4
        if shutil.disk_usage(directory).free < self.nbytes + 32 * 1024**3:
            raise BlockingIOError('insufficient media disk space including32GiB headroom')
        if not hasattr(os, 'posix_fallocate'):
            raise RuntimeError('safe media allocation requires POSIX preallocation')
        with tempfile.TemporaryFile(dir=directory) as file:
            # Reserve blocks up front: a sparse mapping could otherwise SIGBUS
            # halfway through decoding when another job fills the filesystem.
            os.posix_fallocate(file.fileno(), 0, self.nbytes)
            self.mapping = mmap.mmap(file.fileno(), self.nbytes, access=mmap.ACCESS_WRITE)
        # torch keeps the buffer alive, even after this allocation owner goes
        # out of scope. Closing the original fd does not invalidate the map.
        self.tensor = torch.frombuffer(self.mapping, dtype=dtype).reshape(shape)

    def flush_rows(self, start: int, stop: int) -> None:
        if not 0 <= start < stop <= self.shape[0]:
            raise ValueError('invalid media flush rows')
        first = start * self.row_bytes
        end = stop * self.row_bytes
        aligned = first // mmap.PAGESIZE * mmap.PAGESIZE
        self.mapping.flush(aligned, end - aligned)
        if hasattr(self.mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            self.mapping.madvise(mmap.MADV_DONTNEED, aligned, end - aligned)


def allocate_cpu_media(shape: tuple[int, ...], dtype):
    """Return (real CPU tensor, optional mapping owner); defaults unchanged."""
    import torch
    location = os.environ.get('VISUAL_RL_MEDIA_MMAP_DIR')
    if location is None:
        return torch.empty(shape, dtype=dtype, device='cpu'), None
    if not location:
        raise ValueError('VISUAL_RL_MEDIA_MMAP_DIR must not be empty')
    owner = FileBackedMedia(shape, dtype, Path(location))
    return owner.tensor, owner
