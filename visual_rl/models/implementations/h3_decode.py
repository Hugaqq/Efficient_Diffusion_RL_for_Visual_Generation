"""One-sample sequential H3 AV decoding, with explicit CPU and CUDA backends.

Independent math/API implementation from UniRL
8cfbac17765004afb68fef08a4e011a4c8daf555 minimax_h3/vae.py and vendor packing /
autoencoders (Apache-2.0). Public module paths are recorded in its
VENDOR_COMMIT.txt (Diffusers abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc).
No vendor imports, fallback media, scalar scaling_factor or audio clipping.

Budgets bound the single returned media payload and a conservative tensor
working-set estimate. They EXCLUDE VAE weights, loader copies and internal VAE
activations/workspace; real host resource admission still requires measurement.
The caller must serialize aux across ranks and consume media inside its lease.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import json
import time
from pathlib import Path
from threading import Lock, Event, Thread

import torch

from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry

_CLASSES = {"vae": ("autoencoder_kl_minimax_h3", "AutoencoderKLMiniMaxH3"),
            "audio_vae": ("autoencoder_kl_minimax_h3_audio", "AutoencoderKLMiniMaxH3Audio")}


def _resolve_vae_class(component: str):
    module, symbol = _CLASSES[component]
    path = f"diffusers.models.autoencoders.{module}"
    try:
        cls = getattr(importlib.import_module(path), symbol)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"H3 decode requires optional {path}.{symbol}; no reference/fake fallback") from exc
    if not callable(getattr(cls, "from_pretrained", None)):
        raise ImportError(f"H3 VAE {symbol} lacks from_pretrained API")
    return cls


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate VAE config key: {key}")
        result[key] = value
    return result


def _stats(config: Mapping, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(config, Mapping):
        raise ValueError("VAE config must expose explicit per-channel latent statistics")
    values = []
    for key in ("latents_mean", "latents_std"):
        raw = config.get(key)
        if (not isinstance(raw, (list, tuple)) or len(raw) != channels
                or any(type(v) not in (int, float) for v in raw)):
            raise ValueError(f"{key} must contain exactly {channels} numeric values")
        tensor = torch.tensor(raw, dtype=torch.float32, device="cpu")
        if not bool(torch.isfinite(tensor).all()) or (key == "latents_std" and not bool((tensor > 0).all())):
            raise ValueError("VAE latent statistics must be finite FP32 with std > 0")
        values.append(tensor)
    return tuple(values)


@dataclass(frozen=True)
class H3AVMetadata:
    fps: int
    sample_rate: int
    frame_count: int
    sample_count: int

    @property
    def video_duration_seconds(self) -> float:
        return self.frame_count / self.fps

    @property
    def audio_duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate


class H3AVMedia:
    """Lease-owned media; tensor access is invalid after lease exit.

    Consumers must not retain tensor aliases beyond the lease. Python cannot
    revoke references a consumer deliberately keeps; no RSS reclamation claim.
    """

    def __init__(self, video: torch.Tensor, audio: torch.Tensor, fps: int, sample_rate: int):
        self._video, self._audio = video, audio
        self.metadata = H3AVMetadata(fps, sample_rate, video.shape[0], audio.shape[0])

    @property
    def video(self) -> torch.Tensor:
        if self._video is None:
            raise RuntimeError("AV media lease has expired")
        return self._video

    @property
    def audio(self) -> torch.Tensor:
        if self._audio is None:
            raise RuntimeError("AV media lease has expired")
        return self._audio

    def _release(self) -> None:
        self._video = self._audio = None


class _DecodeLease:
    def __init__(self, owner, terminal, geometry):
        self._owner, self._terminal, self._geometry = owner, terminal, geometry
        self._used, self._media = False, None

    def __enter__(self) -> H3AVMedia:
        if self._used:
            raise RuntimeError("decode lease cannot be reentered")
        self._used = True
        terminal, self._terminal = self._terminal, None
        if not self._owner._lock.acquire(blocking=False):
            raise RuntimeError("only one media lease may be in flight")
        try:
            self._media = self._owner._decode(terminal, self._geometry)
            return self._media
        except BaseException:
            self._owner._lock.release()
            raise

    def __exit__(self, *_exc) -> None:
        if self._media is None:
            raise RuntimeError("decode lease is not active")
        self._media._release()
        self._media = None
        self._owner._lock.release()


class H3CPUDecoder:
    """Load/use/drop video VAE, then audio VAE, with at most one media lease."""

    def __init__(self, local_model_dir: str | Path, *, max_media_bytes: int, max_decode_bytes: int):
        self._root = Path(local_model_dir).expanduser().resolve()
        if not self._root.is_dir():
            raise ValueError("H3 decode requires an existing local artifact directory")
        for name, value in (("max_media_bytes", max_media_bytes), ("max_decode_bytes", max_decode_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an explicit positive integer")
        self._max_media_bytes, self._max_decode_bytes = max_media_bytes, max_decode_bytes
        self._lock = Lock()

    def decode_one(self, terminal: torch.Tensor, geometry: H3T2VAGeometry) -> _DecodeLease:
        """Use as `with decoder.decode_one(terminal, geometry) as media:`."""
        return _DecodeLease(self, terminal, geometry)

    @staticmethod
    def estimate_bytes(geometry: H3T2VAGeometry) -> tuple[int, int]:
        """(AV payload, tensor working set), NOT weights/internal VAE workspace.

        Working set: FP16 input + three FP32 latent copies + three AV output
        copies, covering unpack/denormalization/output/layout conversions.
        """
        if not isinstance(geometry, H3T2VAGeometry):
            raise TypeError("geometry must be H3T2VAGeometry")
        media = 4 * (3 * geometry.num_frames * geometry.height * geometry.width
                     + 2 * geometry.audio_latents * 800)
        working = 14 * geometry.joint_layout.packed_width + 3 * media
        return media, working

    def _configs(self) -> dict[str, dict]:
        configs = {}
        for component, (_, symbol) in _CLASSES.items():
            directory = self._root / component
            path = directory / "config.json"
            if directory.is_symlink() or not directory.is_dir() or not path.is_file():
                raise ValueError(f"missing VAE config or path escape: {component}")
            # Includes local shard-index symlink escapes; no traversal outside
            # the declared component is permitted before from_pretrained.
            if any(p.is_symlink() for p in directory.rglob("*")):
                raise ValueError("VAE artifact symlink/path escape is unsupported")
            for index_path in directory.glob("*.index.json"):
                with index_path.open("rb") as stream:
                    raw_index = stream.read(1024 * 1024 + 1)
                if len(raw_index) > 1024 * 1024:
                    raise ValueError("VAE shard index exceeds 1 MiB")
                index = json.loads(raw_index, object_pairs_hook=_unique_json)
                mapping = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError("VAE shard index requires a weight_map")
                for filename in mapping.values():
                    if (not isinstance(filename, str) or Path(filename).is_absolute()
                            or ".." in Path(filename).parts
                            or not (directory / filename).resolve().is_relative_to(directory.resolve())
                            or not (directory / filename).is_file()):
                        raise ValueError("VAE shard missing or path escape")
            with path.open("rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("VAE config exceeds 1 MiB")
            config = json.loads(raw, object_pairs_hook=_unique_json)
            if not isinstance(config, dict) or config.get("_class_name") != symbol:
                raise ValueError(f"invalid VAE class config: {component}")
            _stats(config, 24 if component == "vae" else 32)
            if component == "audio_vae" and (type(config.get("sampling_rate")) is not int or config["sampling_rate"] != 32000):
                raise ValueError("audio VAE config requires sampling_rate=32000")
            configs[component] = config
        return configs

    def _load(self, component: str):
        cls = _resolve_vae_class(component)
        with torch.device("cpu"):
            model = cls.from_pretrained(str(self._root), subfolder=component,
                                        torch_dtype=torch.float32, local_files_only=True)
        if not isinstance(model, torch.nn.Module) or not callable(getattr(model, "decode", None)):
            raise TypeError("H3 VAE must be a torch module with decode API")
        model.eval().requires_grad_(False)
        for tensor in (*model.parameters(), *model.buffers()):
            if tensor.device.type != "cpu" or (tensor.is_floating_point() and tensor.dtype != torch.float32):
                raise ValueError("VAE parameters/buffers must already be CPU FP32")
        return model

    @staticmethod
    def _output(output, shape) -> torch.Tensor:
        if not isinstance(output, tuple) or len(output) != 1 or not isinstance(output[0], torch.Tensor):
            raise ValueError("VAE decode(return_dict=False) must return one tensor tuple")
        tensor = output[0]
        if tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("VAE output shape/device/FP32 mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("VAE output must be finite")
        return tensor.detach()

    def _run_decode(self, model, latents, shape):
        return self._output(model.decode(latents, return_dict=False), shape)

    def _release_device_cache(self):
        pass  # CPU reference owns no CUDA allocation.

    def _decode_stream(self, component, rows, geometry, config):
        started = time.monotonic()
        print(json.dumps(dict(event='h3_vae_load_begin', component=component)), flush=True)
        model = self._load(component)
        print(json.dumps(dict(event='h3_vae_load_end', component=component,
                              seconds=time.monotonic()-started)), flush=True)
        started = time.monotonic()
        print(json.dumps(dict(event='h3_vae_decode_begin', component=component)), flush=True)
        try:
            channels = 24 if component == "vae" else 32
            mean, std = _stats(model.config, channels)
            expected_mean, expected_std = _stats(config, channels)
            if not torch.equal(mean, expected_mean) or not torch.equal(std, expected_std):
                raise ValueError("loaded VAE latent statistics differ from local config")
            if component == "vae":
                _, frames, height, width = geometry.latent_shape
                # Rows: (time,patch_y,patch_x); features: (channel,dy,dx).
                latents = rows.float().reshape(1, frames, height//2, width//2, 24, 2, 2)
                latents = latents.permute(0, 4, 1, 2, 5, 3, 6).reshape(1, 24, frames, height, width)
                latents = latents * std.view(1, 24, 1, 1, 1) + mean.view(1, 24, 1, 1, 1)
                if not bool(torch.isfinite(latents).all()):
                    raise ValueError("video denormalization must be finite")
                decoded = self._run_decode(model, latents,
                                           (1, 3, geometry.num_frames, geometry.height, geometry.width))
                pixel_mean = torch.tensor((.485, .456, .406), dtype=torch.float32, device="cpu").view(1, 3, 1, 1, 1)
                pixel_std = torch.tensor((.229, .224, .225), dtype=torch.float32, device="cpu").view(1, 3, 1, 1, 1)
                video = (decoded * pixel_std + pixel_mean).clamp_(0, 1)
                return video[0].permute(1, 0, 2, 3).contiguous()
            rate = model.config.get("sampling_rate")
            if type(rate) is not int or rate != 32000 or type(getattr(model, "hop_length", None)) is not int or model.hop_length != 800:
                raise ValueError("audio VAE requires sampling_rate=32000 and hop_length=800")
            latents = rows.float().reshape(2, geometry.audio_latents, 32).transpose(1, 2).contiguous()
            latents = latents * std.view(1, 32, 1) + mean.view(1, 32, 1)
            if not bool(torch.isfinite(latents).all()):
                raise ValueError("audio denormalization must be finite")
            decoded = self._run_decode(model, latents, (2, 1, geometry.audio_latents * 800))
            return decoded[:, 0].transpose(0, 1).contiguous(), rate
        finally:
            print(json.dumps(dict(event='h3_vae_decode_exit', component=component,
                                  seconds=time.monotonic()-started)), flush=True)
            del model  # No owner-held leaf survives success or failure.
            self._release_device_cache()

    def _decode(self, terminal, geometry) -> H3AVMedia:
        media_bytes, decode_bytes = self.estimate_bytes(geometry)
        if media_bytes > self._max_media_bytes or decode_bytes > self._max_decode_bytes:
            raise ValueError("decode media/tensor workspace budget exceeded before model load")
        video, audio = geometry.joint_layout.unpack(terminal)
        if (terminal.shape[0] != 1 or terminal.device.type != "cpu" or terminal.dtype != torch.float16
                or terminal.requires_grad or terminal.grad_fn is not None):
            raise ValueError("terminal must be detached CPU FP16 B1 joint state")
        if not bool(torch.isfinite(terminal).all()):
            raise ValueError("terminal must be finite")
        configs = self._configs()
        # No grad/autocast scope leaks into the consumer's with-body.
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            frames = self._decode_stream("vae", video, geometry, configs["vae"])
            try:
                waveform, rate = self._decode_stream("audio_vae", audio, geometry, configs["audio_vae"])
                return H3AVMedia(frames, waveform, geometry.fps, rate)
            finally:
                del frames  # No half-AV result if audio fails.


class H3GPUDecoder(H3CPUDecoder):
    """Rank-local CUDA FP32 VAE; return the same bounded CPU FP32 AV lease.

    Aux work is serialized across WORLD8 by the caller. Do not overlap decode
    with policy forward or HPS scoring on this card. Never fall back to CPU.
    Official VAE tiling defaults and all layout/statistics checks are preserved.
    """
    def __init__(self, local_model_dir, *, max_media_bytes, max_decode_bytes, device):
        super().__init__(local_model_dir, max_media_bytes=max_media_bytes,
                         max_decode_bytes=max_decode_bytes)
        self._device = torch.device(device)
        if self._device.type != 'cuda' or self._device.index is None:
            raise ValueError('GPU decoder requires an explicit rank-local CUDA device')

    def _load(self, component):
        model = super()._load(component)
        try:
            model.to(device=self._device, dtype=torch.float32)
            if any(t.device != self._device for t in (*model.parameters(), *model.buffers())):
                raise ValueError('VAE did not materialize completely on the selected GPU')
            return model
        except BaseException:
            del model
            self._release_device_cache()
            raise

    def _run_decode(self, model, latents, shape):
        with torch.autocast('cuda', enabled=False):
            inputs = latents.to(device=self._device, dtype=torch.float32)
            outputs = model.decode(inputs, return_dict=False)
            if (not isinstance(outputs, tuple) or len(outputs) != 1
                    or not isinstance(outputs[0], torch.Tensor)
                    or outputs[0].device != self._device or outputs[0].dtype != torch.float32
                    or tuple(outputs[0].shape) != shape):
                raise ValueError('GPU VAE output shape/device/FP32 mismatch')
            # Blocking CPU copy completes this stream before GPU temporaries die.
            result = self._output((outputs[0].detach().to('cpu'),), shape)
            del outputs, inputs
            return result

    def _release_device_cache(self):
        with torch.cuda.device(self._device):
            torch.cuda.current_stream(self._device).synchronize()
            torch.cuda.empty_cache()

    def _decode(self, terminal, geometry):
        with torch.cuda.device(self._device):
            self._release_device_cache()  # release only unused allocator cache
            free = torch.cuda.mem_get_info(self._device)[0] / 1024**3
            if free < 40:
                raise BlockingIOError('GPU VAE needs40GiB free; no CPU fallback')
            baseline = torch.cuda.memory_allocated(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)
            minimum = [free]
            monitor_errors = []
            stop = Event()
            def monitor():
                try:
                    while not stop.wait(.1):
                        minimum[0] = min(minimum[0], torch.cuda.mem_get_info(self._device)[0] / 1024**3)
                except Exception as exc:
                    monitor_errors.append(exc)
            watcher = Thread(target=monitor, name='h3-gpu-decode-memory', daemon=True)
            watcher.start()
            started = time.monotonic()
            media = None
            try:
                media = super()._decode(terminal, geometry)
                if monitor_errors:
                    raise RuntimeError('GPU VAE memory monitor failed') from monitor_errors[0]
                if minimum[0] < 16:
                    raise BlockingIOError('GPU VAE safety margin fell below16GiB')
                print(json.dumps(dict(event='h3_gpu_decode_complete', device=str(self._device),
                                      seconds=time.monotonic()-started, dtype='float32',
                                      incremental_peak_gib=(torch.cuda.max_memory_allocated(self._device)-baseline)/1024**3,
                                      min_card_free_gib=minimum[0], frames=geometry.num_frames)), flush=True)
                return media
            except BaseException:
                if media is not None:
                    media._release()
                raise
            finally:
                stop.set()
                watcher.join(timeout=3)
                self._release_device_cache()
