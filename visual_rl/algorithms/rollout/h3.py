"""Serial H3 inference rollout and real AV export, not an RL training loop.

Model calculation stays in H3InferenceAdapter. This algorithm explicitly owns
both sigma schedules, the joint Euler loop, artifact isolation and receipts.
"""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from visual_rl.models.h3 import DIFFUSERS_SHA, H3Result, sha256_file


def sigma_schedule(points, shift):
    """Pinned H3 baseline: grid includes terminal0, t=1-sigma, no extra step."""
    import torch
    if type(points) is not int or points < 2 or shift not in (12.0, 3.0):
        raise ValueError("H3 requires >=2 grid points and native video12/audio3 shift")
    base = torch.linspace(1.0, 0.0, points, dtype=torch.float32)
    return torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))


def joint_euler_step(state, velocity, sigmas, next_sigmas):
    """Data-ward +v, independent clocks, FP32 state; no CFG or re-noising.

    Matches MiniMaxH3Scheduler.step, including the float32 sigma -> t -> sigma
    round trip for x0, while the blend ratio uses the original sigma grid.
    This is deterministic inference, NOT a stochastic RL/logprob transition.
    """
    import torch
    if any(set(values) != {"video", "audio"} for values in (state, velocity, sigmas, next_sigmas)):
        raise ValueError("joint H3 step requires both modalities")
    result = {}
    for name in ("video", "audio"):
        x, v = state[name], velocity[name]
        sigma, following = sigmas[name], next_sigmas[name]
        if (x.dtype != torch.float32 or x.ndim != 3 or x.shape[0] != 1 or v.shape != x.shape
                or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(v).all())
                or not 0 <= float(following) < float(sigma) <= 1):
            raise ValueError(f"invalid joint H3 {name} state/velocity/sigma")
        timestep = (1 - sigma).to(x.device, x.dtype)
        denoised = x + (1 - timestep) * v.float()
        ratio = (following / sigma).to(x.device, x.dtype)
        result[name] = ratio * x + (1 - ratio) * denoised
        if not bool(torch.isfinite(result[name]).all()):
            raise ValueError(f"nonfinite H3 {name} step")
    return result


def sample_h3(adapter, request):
    """VisualRL-owned joint sampling. No third-party pipeline call is used."""
    import torch
    import logging
    started = time.monotonic()
    with torch.inference_mode():
        prompt = adapter.encode(request)
        state, layout, geometry = adapter.prepare_latents(request, prompt)
        video_sigmas = sigma_schedule(request.steps, 12.0)
        audio_sigmas = sigma_schedule(request.steps, 3.0)
        if len(video_sigmas) != len(audio_sigmas):
            raise ValueError("H3 joint clocks have different transition counts")
        with adapter.denoising():
            for index in range(len(video_sigmas) - 1):
                sigmas = {"video": video_sigmas[index], "audio": audio_sigmas[index]}
                following = {"video": video_sigmas[index + 1], "audio": audio_sigmas[index + 1]}
                velocity = adapter.predict(state, prompt, layout, 1 - sigmas["video"], 1 - sigmas["audio"])
                state = joint_euler_step(state, velocity, sigmas, following)
                logging.info(json.dumps({"event": "h3_joint_step", "sample_id": request.sample_id,
                    "step": index + 1, "transitions": len(video_sigmas) - 1,
                    "video_sigma": float(sigmas["video"]), "audio_sigma": float(sigmas["audio"])}))
                del velocity
        # No denoiser weights survive into either decoder phase.
        result = adapter.decode(request, state, geometry)
    return replace(result, metrics={**result.metrics, "generate_seconds": time.monotonic() - started,
        "model_evaluations": len(video_sigmas) - 1, "sigma_grid_points": request.steps})


def ffmpeg_executable():
    path = shutil.which("ffmpeg")
    if path:
        return path
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _write_media(command, chunks):
    """Feed bounded chunks; preserve diagnostic stderr without pipe deadlock."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for chunk in chunks:
                process.stdin.write(chunk)
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.stdin.close()
            process.wait()
            raise
        if code:
            errors.seek(0)
            raise RuntimeError(f"ffmpeg encode failed ({code}): {errors.read().decode(errors='replace')[-4000:]}")


def decode_media(path, *, width, height):
    """Fully decode both tracks; count actual frames/samples, check finite audio.

    No ffprobe metadata-only shortcut and no reliance on an old preview video.
    """
    import numpy as np
    ffmpeg = ffmpeg_executable()
    counts = {}
    for name, args, stride in (
        ("video_frames", ["-map", "0:v:0", "-vf", "showinfo", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24"], width * height * 3),
        ("audio_samples", ["-map", "0:a:0", "-af", "ashowinfo", "-f", "f32le", "-acodec", "pcm_f32le"], 8),
    ):
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([ffmpeg, "-v", "info", "-xerror", "-nostdin", "-i", str(path), *args, "pipe:1"],
                                       stdout=subprocess.PIPE, stderr=errors)
            total, nonzero = 0, False
            while block := process.stdout.read(65536):
                total += len(block)
                if name == "audio_samples":
                    samples = np.frombuffer(block, dtype="<f4")
                    if not np.isfinite(samples).all():
                        process.stdout.close()
                        process.wait()
                        raise ValueError("decoded H3 audio is nonfinite")
                    nonzero |= bool(np.any(samples != 0))
            process.stdout.close()
            code = process.wait()
            errors.seek(0)
            details = errors.read().decode(errors="replace")
            if code or total == 0 or total % stride:
                raise ValueError(f"full {name} decode failed: {details[-4000:]}")
            if name == "video_frames":
                sizes = set(re.findall(r"\bs:(\d+)x(\d+)", details))
                rates = set(re.findall(r"config in .*?frame_rate:\s*(\d+)/(\d+)", details))
                if sizes != {(str(width), str(height))} or rates != {("24", "1")}:
                    raise ValueError(f"decoded video dimensions/rate mismatch: {sizes}, {rates}")
                counts.update(width=width, height=height, fps=24)
            else:
                formats = set(re.findall(r"channels:(\d+).*?rate:(\d+)", details))
                if formats != {("2", "32000")} or not nonzero:
                    raise ValueError("exported audio is silent or has wrong native channels/rate")
                counts.update(channels=2, sample_rate=32000)
            counts[name] = total // stride
    return counts


def export_result(result: H3Result, directory):
    """Write lossless float WAV plus H264/AAC MP4; no padding/fake silence."""
    if not isinstance(result, H3Result):
        raise TypeError("export requires a validated H3Result")
    directory = Path(directory)
    ffmpeg = ffmpeg_executable()
    video, audio = directory / "video.mp4", directory / "audio.wav"
    r = result.request
    # Preserve the native float waveform exactly in the sidecar. MP4's AAC is
    # lossy and can carry encoder padding; record both observed sample counts.
    waveform = result.audio.numpy().T.astype("<f4", copy=False)
    _write_media([ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "f32le", "-ar", str(result.sample_rate),
                  "-ac", "2", "-i", "pipe:0", "-c:a", "pcm_f32le", str(audio)],
                 (waveform[start:start+8192].tobytes() for start in range(0, len(waveform), 8192)))
    _write_media([ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                  "-s", f"{r.width}x{r.height}", "-r", str(result.fps), "-i", "pipe:0", "-i", str(audio),
                  "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(video)],
                 (frame.numpy().tobytes() for frame in result.video))
    counts = decode_media(video, width=r.width, height=r.height)
    if counts["video_frames"] != r.frames or abs(counts["audio_samples"] - waveform.shape[0]) > 2048:
        raise ValueError("encoded H3 AV geometry/length changed beyond AAC padding")
    # WAV format is float32 stereo32k, written from these exact native samples.
    # Verify decoded bytes too; a mux/codec failure must not produce a receipt.
    decoded = subprocess.check_output([ffmpeg, "-v", "error", "-xerror", "-i", str(audio),
                                      "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])
    if decoded != waveform.tobytes():
        raise ValueError("lossless H3 WAV differs from native waveform")
    version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]
    return {"ffmpeg": {"executable": str(ffmpeg), "version": version},
            "video": {"path": "video.mp4", "sha256": sha256_file(video), "bytes": video.stat().st_size},
            "audio": {"path": "audio.wav", "sha256": sha256_file(audio), "bytes": audio.stat().st_size},
            "decoded_mp4": counts, "native_audio_samples": waveform.shape[0],
            "video_seconds": r.frames / result.fps, "audio_seconds": waveform.shape[0] / result.sample_rate,
            "fps": result.fps, "sample_rate": result.sample_rate, "channels": 2,
            "video_layout": "FHWC uint8", "audio_layout": "CS float32", "full_decode_verified": True}


def run_h3_rollout(adapter, requests, output_dir, *, diagnostic=False):
    """One or three serial original-prompt requests. Never re-use an output."""
    requests = tuple(requests)
    if len(requests) not in {1, 3} or len({r.sample_id for r in requests}) != len(requests):
        raise ValueError("H3 validation rollout requires one or three unique samples")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    try:
        for request in requests:
            started = time.monotonic()
            staging = output / f".{request.sample_id}.partial"
            staging.mkdir()
            condition_images = [{"path": str(p), "sha256": sha256_file(p)} for p in request.images]
            result = sample_h3(adapter, request)
            media = export_result(result, staging)
            if condition_images != [{"path": str(p), "sha256": sha256_file(p)} for p in request.images]:
                raise ValueError("H3 conditioning image changed during generation")
            receipt = {"schema_version": 1, "sample_id": request.sample_id,
                       "request": {**asdict(request), "images": [str(p) for p in request.images]},
                       "conditioning_images": condition_images,
                       "implementation": "visualrl-pytorch-joint-loop", "diffusers_commit": DIFFUSERS_SHA,
                       "model_revision_declared": adapter.assets["declared_revision"],
                       "sampler": "euler_eta0", "sigma_grid_points": request.steps,
                       "model_evaluations": request.steps - 1, "video_shift": 12.0, "audio_shift": 3.0,
                       "quality": "lossless", "turbo_lora": False, "rl_training_supported": False,
                       "diagnostic": diagnostic, "media": media, "metrics": result.metrics,
                       "generate_export_seconds": time.monotonic() - started}
            (staging / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
            os.replace(staging, output / request.sample_id)
            receipts.append(receipt)
            del result
        (output / "ROLLOUT_COMPLETE.json").write_text(json.dumps({"status": "native_av_rollout_complete",
            "samples": [r.sample_id for r in requests], "diagnostic": diagnostic,
            "rl5_gpu_passed": False}, sort_keys=True) + "\n")
    except BaseException as error:
        (output / "FAILED.json").write_text(json.dumps({"status": "blocked_or_failed", "error": str(error),
            "error_type": type(error).__name__, "completed_samples": [r["sample_id"] for r in receipts]}) + "\n")
        raise
    return receipts
