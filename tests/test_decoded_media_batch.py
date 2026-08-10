"""Typed, import-safe decoded-media boundary contracts."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from visual_rl.data.media import DecodedMediaBatch


class _OpaqueTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


@pytest.mark.parametrize(
    ("layout", "shape"),
    (
        ("BCHW", (2, 3, 16, 16)),
        ("BFCHW", (2, 4, 3, 16, 16)),
        ("BFHWC", (2, 4, 16, 16, 3)),
    ),
)
def test_decoded_media_batch_validates_canonical_layout_geometry(
    layout: str,
    shape: tuple[int, ...],
) -> None:
    value = DecodedMediaBatch(
        tensor=_OpaqueTensor(shape),
        layout=layout,  # type: ignore[arg-type]
    )

    assert value.shape == shape
    assert value.batch_size == shape[0]
    assert value.rank == len(shape)
    with pytest.raises(FrozenInstanceError):
        value.layout = "BCHW"  # type: ignore[misc]


def test_decoded_media_batch_rejects_rank_batch_and_shape_drift() -> None:
    with pytest.raises(ValueError, match="rank 5"):
        DecodedMediaBatch(tensor=_OpaqueTensor((2, 3, 8, 8)), layout="BFCHW")
    with pytest.raises(ValueError, match="positive"):
        DecodedMediaBatch(tensor=_OpaqueTensor((0, 3, 8, 8)), layout="BCHW")

    tensor = _OpaqueTensor((2, 3, 8, 8))
    value = DecodedMediaBatch(tensor=tensor, layout="BCHW")
    tensor.shape = (1, 3, 8, 8)
    with pytest.raises(ValueError, match="shape changed"):
        value.assert_integrity()


def test_decoded_media_contract_import_does_not_import_torch() -> None:
    script = r"""
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('torch imported by decoded-media contract')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from visual_rl.data.media import DecodedMediaBatch
class Tensor:
    shape = (1, 3, 8, 8)
value = DecodedMediaBatch(Tensor(), 'BCHW')
assert value.batch_size == 1
assert 'torch' not in sys.modules
print('import-safe')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "import-safe"
