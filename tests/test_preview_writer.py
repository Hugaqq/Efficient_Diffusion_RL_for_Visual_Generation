"""Bounded, deterministic preview encoding contracts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from visual_rl.artifacts.preview import PreviewWriter
from visual_rl.core.types import RolloutBatch, StepContext


def _batch(media: Any, layout: str, *, step: int = 7) -> RolloutBatch:
    batch_size = int(media.shape[0])
    return RolloutBatch(
        prompts=tuple(f"prompt {index}" for index in range(batch_size)),
        metadata=tuple({} for _ in range(batch_size)),
        media=media,
        latents=torch.zeros(batch_size, 2, 1),
        next_latents=torch.ones(batch_size, 2, 1),
        timesteps=torch.tensor([[9, 4]] * batch_size, dtype=torch.int64),
        old_log_probs=torch.zeros(batch_size, 2),
        transition_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        prompt_id=tuple(f"prompt-{index}" for index in range(batch_size)),
        group_id=tuple(f"group-{index}" for index in range(batch_size)),
        branch_id=None,
        media_layout=layout,
        camera_trajectory=None,
        context=StepContext(step=step, seed=123),
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={"features": torch.ones(batch_size, 2)},
        artifact_metadata={},
    )


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_equal(
    left: tuple[Any, tuple[Any, ...], torch.Tensor],
    right: tuple[Any, tuple[Any, ...], torch.Tensor],
) -> None:
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_image_writer_is_bounded_canonical_and_preserves_input_and_rng(
    tmp_path: Path,
) -> None:
    media = torch.linspace(
        0,
        1,
        steps=3 * 4 * 6,
        dtype=torch.bfloat16,
    ).reshape(1, 3, 4, 6)
    original = media.clone()
    writer = PreviewWriter(tmp_path.resolve())
    before = _rng_snapshot()

    result = writer.write_batch(_batch(media, "BCHW"), max_samples=1)

    after = _rng_snapshot()
    _assert_rng_equal(before, after)
    assert torch.equal(media, original)
    assert result.warnings == ()
    assert result.media_paths == ("previews/step_000007/rank_0/sample_000000.jpg",)
    output = tmp_path / result.media_paths[0]
    assert output.is_file() and output.stat().st_size > 0
    from PIL import Image

    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (6, 4)


@pytest.mark.parametrize("layout", ["BFCHW", "BFHWC"])
def test_video_writer_streams_frames_in_order_with_fixed_codec_contract(
    tmp_path: Path,
    layout: str,
) -> None:
    values = torch.stack(
        (
            torch.zeros(3, 4, 6),
            torch.full((3, 4, 6), 0.5),
            torch.ones(3, 4, 6),
        )
    )
    media = values.unsqueeze(0)
    if layout == "BFHWC":
        media = media.permute(0, 1, 3, 4, 2)
    captured: dict[str, Any] = {"frames": []}

    def factory(filename: str, **kwargs: Any):
        captured["filename"] = filename
        captured["kwargs"] = kwargs
        Path(filename).write_bytes(b"fake-mp4")

        def sink():
            while True:
                frame = yield
                captured["frames"].append(frame.copy())

        return sink()

    result = PreviewWriter(
        tmp_path.resolve(),
        ffmpeg_factory=factory,
    ).write_batch(_batch(media, layout), max_samples=1)

    assert result.warnings == ()
    assert result.media_paths == ("previews/step_000007/rank_0/sample_000000.mp4",)
    assert captured["kwargs"] == {
        "size": (6, 4),
        "pix_fmt_in": "rgb24",
        "pix_fmt_out": "yuv420p",
        "fps": 8,
        "codec": "libx264",
        "bitrate": "4M",
        "macro_block_size": 1,
        "ffmpeg_log_level": "error",
    }
    frames = captured["frames"]
    assert len(frames) == 3
    assert all(frame.shape == (4, 6, 3) for frame in frames)
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 128, 255]
    assert (tmp_path / result.media_paths[0]).read_bytes() == b"fake-mp4"


def test_real_ffmpeg_video_round_trip_smoke(tmp_path: Path) -> None:
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    media = torch.stack(
        (
            torch.zeros(3, 4, 6),
            torch.ones(3, 4, 6),
        )
    ).unsqueeze(0)

    result = PreviewWriter(tmp_path.resolve()).write_batch(
        _batch(media, "BFCHW"),
        max_samples=1,
    )

    assert result.warnings == ()
    output = tmp_path / result.media_paths[0]
    assert output.is_file() and output.stat().st_size > 0
    reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        frames = [next(reader), next(reader)]
    finally:
        reader.close()
    assert metadata["size"] == (6, 4)
    assert all(len(frame) == 6 * 4 * 3 for frame in frames)


def test_preview_failure_is_per_sample_best_effort_and_restores_rng(
    tmp_path: Path,
) -> None:
    media = torch.zeros(2, 3, 4, 6)
    media[0, 0, 0, 0] = 1.5
    original = media.clone()
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    before = _rng_snapshot()

    result = PreviewWriter(tmp_path.resolve()).write_batch(
        _batch(media, "BCHW"),
        max_samples=2,
    )

    _assert_rng_equal(before, _rng_snapshot())
    assert torch.equal(media, original)
    assert result.media_paths == (
        None,
        "previews/step_000007/rank_0/sample_000001.jpg",
    )
    assert len(result.warnings) == 1
    assert "floating preview media must be in [0, 1]" in result.warnings[0]
    assert not tuple(tmp_path.rglob("*.part"))
    assert not tuple(tmp_path.rglob("*.part.mp4"))


@pytest.mark.parametrize(
    ("max_samples", "invalid_first_sample", "saved_count"),
    (
        (0, False, 0),
        (2, False, 2),
        (2, True, 1),
    ),
)
def test_preview_preserves_current_rank_cuda_rng(
    tmp_path: Path,
    max_samples: int,
    invalid_first_sample: bool,
    saved_count: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the current-rank RNG contract")
    device = torch.device("cuda", torch.cuda.current_device())
    media = torch.zeros(2, 3, 4, 6, device=device)
    if invalid_first_sample:
        media[0, 0, 0, 0] = 1.5
    torch.cuda.manual_seed(119)
    before = torch.cuda.get_rng_state(device).clone()

    result = PreviewWriter(tmp_path.resolve()).write_batch(
        _batch(media, "BCHW"),
        max_samples=max_samples,
    )

    assert torch.equal(before, torch.cuda.get_rng_state(device))
    assert sum(path is not None for path in result.media_paths) == saved_count


@pytest.mark.parametrize(
    ("media", "layout", "message"),
    [
        (
            torch.zeros(1, 2, 4, 6),
            "BCHW",
            "exactly three RGB channels",
        ),
        (
            torch.zeros(1, 2, 3, 3, 6),
            "BFCHW",
            "dimensions must be even",
        ),
        (
            torch.full((1, 3, 4, 6), float("nan")),
            "BCHW",
            "must be finite",
        ),
        (
            torch.zeros(1, 3, 4, 6, dtype=torch.int16),
            "BCHW",
            "uint8 or floating point",
        ),
    ],
)
def test_invalid_preview_media_is_skipped_without_partial_file(
    tmp_path: Path,
    media: torch.Tensor,
    layout: str,
    message: str,
) -> None:
    result = PreviewWriter(tmp_path.resolve()).write_batch(
        _batch(media, layout),
        max_samples=1,
    )

    assert result.media_paths == (None,)
    assert len(result.warnings) == 1
    assert message in result.warnings[0]
    assert not tuple(tmp_path.rglob("*.jpg"))
    assert not tuple(tmp_path.rglob("*.mp4"))


@pytest.mark.parametrize("max_samples", [True, -1, 3])
def test_preview_count_rejects_noncanonical_values(
    tmp_path: Path,
    max_samples: Any,
) -> None:
    writer = PreviewWriter(tmp_path.resolve())
    with pytest.raises(ValueError, match="between 0 and 2"):
        writer.write_batch(
            _batch(torch.zeros(1, 3, 4, 6), "BCHW"),
            max_samples=max_samples,
        )
