"""H3 contract CPU tests; no model weights, GPU, or distributed claims."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import visual_rl
from visual_rl.algorithms.dynamics import h3_schedule
from visual_rl.models import h3_joint
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.models.h3_joint import H3JointLayout


def test_h3_imports_are_from_this_worktree():
    checkout = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == checkout / "visual_rl"
    assert Path(h3_joint.__file__).resolve() == checkout / "visual_rl/models/h3_joint.py"
    assert Path(h3_schedule.__file__).resolve() == checkout / "visual_rl/algorithms/dynamics/h3_schedule.py"


def test_joint_roundtrip_batch_order_dtype_and_gradients():
    layout = H3JointLayout(video_rows=2, video_channels=3, audio_rows=4, audio_channels=5)
    assert (layout.video_numel, layout.audio_numel, layout.packed_width) == (6, 20, 26)
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        # Noncontiguous inputs exercise row ordering without requiring contiguous().
        video = torch.arange(12, dtype=dtype).reshape(2, 3, 2).transpose(1, 2).requires_grad_()
        audio = (100 + torch.arange(40, dtype=dtype)).reshape(2, 5, 4).transpose(1, 2).requires_grad_()
        packed = layout.pack(video, audio)
        expected = torch.tensor([
            [0, 2, 4, 1, 3, 5, 100, 104, 108, 112, 116, 101, 105, 109, 113, 117,
             102, 106, 110, 114, 118, 103, 107, 111, 115, 119],
            [6, 8, 10, 7, 9, 11, 120, 124, 128, 132, 136, 121, 125, 129, 133, 137,
             122, 126, 130, 134, 138, 123, 127, 131, 135, 139],
        ], dtype=dtype)
        assert torch.equal(packed, expected)
        v, a = layout.unpack(packed)
        assert torch.equal(v, video) and torch.equal(a, audio)
        assert packed.dtype == v.dtype == a.dtype == dtype
        assert packed.device == v.device == a.device == video.device == audio.device
        (v.sum() * 2 + a.sum() * 3).backward()
        assert torch.equal(video.grad, torch.full_like(video, 2))
        assert torch.equal(audio.grad, torch.full_like(audio, 3))
        # pack owns new storage; unpack intentionally aliases only that storage.
        with torch.no_grad():
            packed[0, 0] = -7
        assert video[0, 0, 0].item() == 0
        assert v[0, 0, 0].item() == -7
        assert torch.equal(a, audio)
    with pytest.raises(FrozenInstanceError):
        layout.audio_rows = 0


def test_joint_rejects_invalid_geometry_shapes_and_implicit_conversion():
    for field in ("video_rows", "video_channels", "audio_rows", "audio_channels"):
        for bad in (0, -1, True, 2.0, "2"):
            values = dict(video_rows=2, video_channels=3, audio_rows=4, audio_channels=5)
            values[field] = bad
            with pytest.raises(ValueError, match="positive integer"):
                H3JointLayout(**values)
    layout = H3JointLayout(2, 3, 4, 5)
    video, audio = torch.zeros(2, 2, 3), torch.zeros(2, 4, 5)
    for bad in (torch.zeros(2, 6), torch.zeros(0, 2, 3), torch.zeros(2, 3, 2)):
        with pytest.raises(ValueError, match="video"):
            layout.pack(bad, audio)
    for bad in (torch.zeros(1, 4, 5), torch.zeros(2, 0, 5), torch.zeros(2, 5, 4)):
        with pytest.raises(ValueError, match="audio"):
            layout.pack(video, bad)
    for bad in (audio.double(), torch.empty(2, 4, 5, device="meta")):
        with pytest.raises(ValueError, match="dtype and device"):
            layout.pack(video, bad)
    for bad in (torch.zeros(2, 25), torch.zeros(0, 26), torch.zeros(2, 2, 13)):
        with pytest.raises(ValueError, match="packed"):
            layout.unpack(bad)
    with pytest.raises(TypeError, match="tensors"):
        layout.pack(video, None)
    with pytest.raises(TypeError, match="tensor"):
        layout.unpack([0] * 26)


def test_dual_schedule_exact_default_grids_and_indexed_terminal():
    schedule = H3ScheduleSnapshot.build(10)
    base = torch.linspace(1, 0, 11, dtype=torch.float32, device="cpu")
    expected_video = 12 * base / (1 + 11 * base)
    expected_audio = 3 * base / (1 + 2 * base)
    assert schedule.num_steps == 10
    assert torch.equal(schedule.video_sigmas, expected_video)
    assert torch.equal(schedule.audio_sigmas, expected_audio)
    assert schedule.video_sigmas.dtype == schedule.audio_sigmas.dtype == torch.float32
    assert schedule.video_sigmas.device.type == schedule.audio_sigmas.device.type == "cpu"
    assert schedule.video_sigmas[5].item() == torch.tensor(12 / 13, dtype=torch.float32).item()
    assert schedule.audio_sigmas[5].item() == 0.75
    assert bool((schedule.video_sigmas[1:-1] > schedule.audio_sigmas[1:-1]).all())
    # Non-sequential/repeated reads cannot advance a hidden scheduler cursor.
    for i in (9, 0, 5, 0, 9):
        actual = schedule.transition(i)
        expected = (expected_video[i], expected_video[i + 1], expected_audio[i], expected_audio[i + 1])
        assert all(x.shape == () and x.dtype == torch.float32 for x in actual)
        assert all(torch.equal(x, y) for x, y in zip(actual, expected))
    vs, vu, aus, au = schedule.transition(9)
    assert vs > 0 and aus > 0 and vu == 0 and au == 0
    one = H3ScheduleSnapshot.build(1)
    assert tuple(x.item() for x in one.transition(0)) == (1, 0, 1, 0)
    unshifted = H3ScheduleSnapshot.build(10, video_shift=1, audio_shift=1)
    assert torch.equal(unshifted.video_sigmas, base)
    assert torch.equal(unshifted.audio_sigmas, base)


def test_schedule_owns_exact_values_and_returns_independent_tensors():
    video = torch.tensor([1, 0.9876543, 0.1234567, 0], dtype=torch.float32, requires_grad=True)
    audio = torch.tensor([1, 0.8, 0.2, 0], dtype=torch.float32)
    expected_v, expected_a = video.detach().clone(), audio.clone()
    schedule = H3ScheduleSnapshot(video, audio)
    with torch.no_grad():
        video.fill_(0)
        audio.fill_(0)
    schedule.video_sigmas.fill_(0)
    schedule.audio_sigmas.fill_(0)
    for scalar in schedule.transition(1):
        scalar.fill_(0)
    assert torch.equal(schedule.video_sigmas.view(torch.int32), expected_v.view(torch.int32))
    assert torch.equal(schedule.audio_sigmas.view(torch.int32), expected_a.view(torch.int32))
    assert not schedule.video_sigmas.requires_grad
    restored = H3ScheduleSnapshot(schedule.video_sigmas, schedule.audio_sigmas)
    assert restored == schedule
    with pytest.raises(FrozenInstanceError):
        schedule._video = (1, 0)


def test_schedule_rejects_invalid_grids_counts_shifts_and_indices():
    good = torch.tensor([1, 0.5, 0], dtype=torch.float32)
    invalid = (
        [], [1], [1, float("nan"), 0], [1, float("inf"), 0],
        [1, -0.1, 0], [1, 1.1, 0], [1, 1, 0], [1, 0, 0.5, 0],
        [0.9, 0.5, 0], [1, 0.5, 0.1], [1, 0],
    )
    for values in invalid:
        bad = torch.tensor(values, dtype=torch.float32)
        for grids in ((bad, good), (good, bad)):
            with pytest.raises(ValueError):
                H3ScheduleSnapshot(*grids)
    for bad in (good[None], good.double(), good.half(), good.to(torch.int64), good.to("meta")):
        with pytest.raises(ValueError):
            H3ScheduleSnapshot(bad, good)
    with pytest.raises(TypeError):
        H3ScheduleSnapshot([1, 0.5, 0], good)
    for bad in (0, -1, True, 10.0, "10"):
        with pytest.raises(ValueError, match="num_steps"):
            H3ScheduleSnapshot.build(bad)
    for bad in (0, -1, float("nan"), float("inf"), True, "3"):
        for key in ("video_shift", "audio_shift"):
            with pytest.raises(ValueError, match="shifts"):
                H3ScheduleSnapshot.build(10, **{key: bad})
    # Large finite shift collapses distinct base values in FP32: never shorten N.
    with pytest.raises(ValueError, match="strictly decreasing"):
        H3ScheduleSnapshot.build(10, video_shift=1e10)
    schedule = H3ScheduleSnapshot.build(10)
    for bad in (-1, 10, 11):
        with pytest.raises(IndexError):
            schedule.transition(bad)
    for bad in (True, 0.0, 0.5, "0", torch.tensor(0)):
        with pytest.raises(TypeError, match="index"):
            schedule.transition(bad)
