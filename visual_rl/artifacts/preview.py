"""Bounded JPEG/MP4 previews for committed training artifacts."""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visual_rl.core.types import RolloutBatch

__all__ = ["PreviewWriteResult", "PreviewWriter"]

_JPEG_QUALITY = 90
_VIDEO_FPS = 8
_VIDEO_BITRATE = "4M"


@dataclass(frozen=True)
class PreviewWriteResult:
    """Tensor-free result of one best-effort preview staging attempt."""

    media_paths: tuple[str | None, ...]
    warnings: tuple[str, ...]


class PreviewWriter:
    """Write selected media below one existing transaction staging directory."""

    def __init__(
        self,
        staging_root: Path,
        *,
        ffmpeg_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(staging_root, Path) or not staging_root.is_absolute():
            raise TypeError("staging_root must be an absolute pathlib.Path")
        if not staging_root.is_dir():
            raise ValueError("staging_root must be an existing directory")
        self.staging_root = staging_root
        self._ffmpeg_factory = ffmpeg_factory

    def write_batch(
        self,
        batch: RolloutBatch,
        *,
        max_samples: int,
    ) -> PreviewWriteResult:
        """Stage at most ``max_samples`` media files without advancing RNG."""

        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if type(max_samples) is not int or not 0 <= max_samples <= 2:
            raise ValueError("max_samples must be an integer between 0 and 2")
        paths: list[str | None] = [None] * batch.batch_size
        if max_samples == 0:
            return PreviewWriteResult(tuple(paths), ())

        rng_state = _capture_rng_state()
        warning_messages: list[str] = []
        try:
            for index in range(min(max_samples, batch.batch_size)):
                relative = _relative_path(batch, index)
                destination = self.staging_root / relative
                try:
                    _prepare_parent(destination, root=self.staging_root)
                    if batch.media_layout == "BCHW":
                        self._write_image(batch.media[index], destination)
                    else:
                        self._write_video(
                            batch.media[index],
                            layout=batch.media_layout,
                            destination=destination,
                        )
                    paths[index] = relative
                except Exception as exc:  # noqa: BLE001 - preview is best-effort
                    _cleanup_owned_outputs(destination)
                    warning_messages.append(
                        f"preview sample {index} failed: {type(exc).__name__}: {exc}"
                    )
        finally:
            _restore_rng_state(rng_state)
        return PreviewWriteResult(tuple(paths), tuple(warning_messages))

    @staticmethod
    def _write_image(value: Any, destination: Path) -> None:
        from PIL import Image

        frame = _rgb_uint8(value, channels_first=True)
        temporary = destination.with_name(f".{destination.name}.part")
        try:
            Image.fromarray(frame, mode="RGB").save(
                temporary,
                format="JPEG",
                quality=_JPEG_QUALITY,
                optimize=False,
                progressive=False,
            )
            _publish_staged_file(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_video(
        self,
        value: Any,
        *,
        layout: str,
        destination: Path,
    ) -> None:
        shape = _shape(value)
        if len(shape) != 4:
            raise ValueError("video sample must have four dimensions")
        channels_first = layout == "BFCHW"
        if channels_first:
            frames, channels, height, width = shape
        elif layout == "BFHWC":
            frames, height, width, channels = shape
        else:
            raise ValueError("video layout must be BFCHW or BFHWC")
        if frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("video frame count and dimensions must be positive")
        if channels != 3:
            raise ValueError("preview media must have exactly three RGB channels")
        if height % 2 or width % 2:
            raise ValueError("H.264 yuv420p preview dimensions must be even")

        factory = self._ffmpeg_factory
        if factory is None:
            from imageio_ffmpeg import write_frames

            factory = write_frames
        temporary = destination.with_name(f".{destination.stem}.part.mp4")
        writer = None
        try:
            writer = factory(
                str(temporary),
                size=(width, height),
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                fps=_VIDEO_FPS,
                codec="libx264",
                bitrate=_VIDEO_BITRATE,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
            writer.send(None)
            for frame_index in range(frames):
                frame = _rgb_uint8(
                    value[frame_index],
                    channels_first=channels_first,
                )
                writer.send(frame)
            writer.close()
            writer = None
            _publish_staged_file(temporary, destination)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001,S110 - preserve primary error
                    pass
            temporary.unlink(missing_ok=True)


def _relative_path(batch: RolloutBatch, index: int) -> str:
    extension = "jpg" if batch.media_layout == "BCHW" else "mp4"
    return (
        f"previews/step_{batch.context.step:06d}/"
        f"rank_{batch.context.rank}/sample_{index:06d}.{extension}"
    )


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("preview media must expose a shape")
    return tuple(int(item) for item in shape)


def _rgb_uint8(value: Any, *, channels_first: bool):
    import numpy as np

    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError("preview media tensors must be detached")
            value = value.detach().to(device="cpu")
            if value.is_floating_point():
                # NumPy has no portable bfloat16 representation. Normalizing
                # all floating tensors to float32 also keeps the encoder path
                # independent from the model's mixed-precision dtype.
                value = value.to(dtype=torch.float32)
            if channels_first:
                value = value.permute(1, 2, 0)
            array = value.contiguous().numpy()
        else:
            array = np.asarray(value)
            if channels_first:
                array = np.moveaxis(array, 0, -1)
    except ModuleNotFoundError:
        array = np.asarray(value)
        if channels_first:
            array = np.moveaxis(array, 0, -1)

    if array.ndim != 3:
        raise ValueError("preview frame must have three dimensions")
    if array.shape[-1] != 3:
        raise ValueError("preview media must have exactly three RGB channels")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("preview frame dimensions must be positive")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("preview media must be uint8 or floating point")
    if not np.isfinite(array).all():
        raise ValueError("preview media must be finite")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError("floating preview media must be in [0, 1]")
    return np.ascontiguousarray(np.rint(array * 255.0).astype(np.uint8))


def _prepare_parent(path: Path, *, root: Path) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise ValueError("preview destination escapes transaction staging") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preview destination already exists: {path}")


def _publish_staged_file(temporary: Path, destination: Path) -> None:
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError("preview encoder did not produce a non-empty file")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _cleanup_owned_outputs(destination: Path) -> None:
    destination.unlink(missing_ok=True)
    for temporary in (
        destination.with_name(f".{destination.name}.part"),
        destination.with_name(f".{destination.stem}.part.mp4"),
    ):
        temporary.unlink(missing_ok=True)


def _capture_rng_state() -> tuple[Any, Any, Any, tuple[Any, ...] | None]:
    import numpy as np
    import torch

    cuda_states = None
    if torch.cuda.is_initialized():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
        cuda_states,
    )


def _restore_rng_state(
    state: tuple[Any, Any, Any, tuple[Any, ...] | None],
) -> None:
    import numpy as np
    import torch

    python_state, numpy_state, torch_state, cuda_states = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(list(cuda_states))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
