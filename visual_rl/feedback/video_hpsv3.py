"""Flash-GRPO's all-frame HPSv3 reward over localhost JSON."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import numpy as np

from visual_rl.artifacts.hashing import file_sha256, mapping_sha256, tree_sha256
from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.feedback.clients import (
    RewardProtocolError,
    RewardTransportError,
    requests_session,
)

VIDEO_HPSV3_PROTOCOL = "visual_rl.video_hpsv3.json.v1"
VIDEO_HPSV3_PATH = "/v1/video-hpsv3/score"
VIDEO_HPSV3_SCORER_BATCH_SIZE = 4
_TOP_FRACTION = 0.3


def _sha256(value: Any, field: str) -> str:
    digest = value.strip().lower() if isinstance(value, str) else ""
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Video HPSv3 {field} must be a 64-digit SHA-256.")
    return digest


def video_hpsv3_runtime_manifest(
    *,
    checkpoint_path: str | Path,
    hps_source_root: str | Path,
    config_path: str | Path,
    base_model_root: str | Path,
) -> dict[str, str]:
    """Hash every local asset that determines HPSv3 inference."""

    package_root = Path(hps_source_root).expanduser().resolve(strict=True) / "hpsv3"
    return {
        "base_model_tree_sha256": tree_sha256(base_model_root),
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "config_file_sha256": file_sha256(config_path),
        "hpsv3_package_tree_sha256": tree_sha256(package_root),
    }


def video_hpsv3_runtime_manifest_sha256(**paths: Any) -> str:
    return mapping_sha256(video_hpsv3_runtime_manifest(**paths))


def video_hpsv3_identity(
    *,
    scorer_revision: str,
    checkpoint_sha256: str,
    runtime_manifest_sha256: str,
) -> dict[str, str]:
    if not isinstance(scorer_revision, str) or not scorer_revision.strip():
        raise ValueError("Video HPSv3 scorer_revision must be non-empty.")
    return {
        "scorer_revision": scorer_revision.strip(),
        "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint_sha256"),
        "runtime_manifest_sha256": _sha256(
            runtime_manifest_sha256, "runtime_manifest_sha256"
        ),
    }


def aggregate_flash_grpo_hpsv3(
    frame_scores: Sequence[float],
) -> tuple[float, list[int]]:
    try:
        scores = [_finite(value, "frame score") for value in frame_scores]
    except TypeError:
        raise TypeError("Video HPSv3 frame_scores must be a sequence.") from None
    count = int(len(scores) * _TOP_FRACTION)
    if count == 0:
        raise ValueError("Video HPSv3 requires at least 4 frames.")
    # Python's sort is stable, so equal scores retain temporal order.
    selected = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:count]
    return math.fsum(scores[index] for index in selected) / count, selected


def _encode_videos(media: Any, expected_batch: int) -> list[list[str]]:
    import torch

    if not isinstance(media, torch.Tensor):
        raise TypeError("Video HPSv3 media must be a BFCHW torch.Tensor.")
    if media.ndim != 5 or media.shape[0] != expected_batch or media.shape[2] != 3:
        raise ValueError("Video HPSv3 media must be batch-matched RGB BFCHW.")
    if media.shape[1] < 4 or media.shape[3] < 1 or media.shape[4] < 1:
        raise ValueError("Video HPSv3 requires at least 4 non-empty frames.")
    if not torch.is_floating_point(media):
        raise ValueError("Video HPSv3 media must be floating point.")
    media = media.detach()
    if not torch.isfinite(media).all().item():
        raise ValueError("Video HPSv3 media contains NaN or infinity.")
    if media.min().item() < 0 or media.max().item() > 1:
        raise ValueError("Video HPSv3 canonical media must be in [0, 1].")
    pixels = (
        (media * 255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .numpy()
    )
    from PIL import Image

    encoded: list[list[str]] = []
    for video in pixels:
        frames = []
        for frame in video:  # every frame, in temporal order
            buffer = BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG")
            frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        encoded.append(frames)
    return encoded


class VideoHPSv3RewardClient:
    name = "video_hpsv3"

    def __init__(
        self,
        url: str,
        *,
        scorer_revision: str,
        checkpoint_sha256: str,
        runtime_manifest_sha256: str,
        timeout: float = 300.0,
        transport: Any = None,
    ) -> None:
        self.url = _loopback_url(url)
        self.identity = video_hpsv3_identity(
            scorer_revision=scorer_revision,
            checkpoint_sha256=checkpoint_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
        )
        self.timeout = _finite(timeout, "timeout")
        if self.timeout <= 0:
            raise ValueError("Video HPSv3 timeout must be positive.")
        self.transport = requests_session() if transport is None else transport
        if not callable(getattr(self.transport, "post", None)):
            raise TypeError("Video HPSv3 transport must provide post(...).")

    def score(
        self,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        sample_id: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prompts, sample_ids = _batch_identity(prompts, metadata, sample_id)
        videos = _encode_videos(media, len(prompts))
        payload = {
            "protocol": VIDEO_HPSV3_PROTOCOL,
            "scorer_identity": self.identity,
            "samples": [
                {"sample_id": identity, "prompt": prompt, "jpeg_frames": frames}
                for identity, prompt, frames in zip(
                    sample_ids, prompts, videos, strict=True
                )
            ],
        }
        response = self._post(payload)
        if response.get("protocol") != VIDEO_HPSV3_PROTOCOL:
            raise RewardProtocolError("Video HPSv3 protocol mismatch.")
        if _identity(response.get("scorer_identity")) != self.identity:
            raise RewardProtocolError("Video HPSv3 scorer identity mismatch.")
        samples = _mapping_list(response.get("samples"), "response samples")
        if len(samples) != len(sample_ids):
            raise RewardProtocolError("Video HPSv3 response batch mismatch.")

        values, evidence = [], []
        for expected_id, frames, sample in zip(
            sample_ids, videos, samples, strict=True
        ):
            if sample.get("sample_id") != expected_id:
                raise RewardProtocolError("Video HPSv3 sample_id order mismatch.")
            scores = _numbers(sample.get("raw_scores"), "raw_scores")
            if len(scores) != len(frames):
                raise RewardProtocolError("Video HPSv3 frame score count mismatch.")
            aggregate, selected = aggregate_flash_grpo_hpsv3(scores)
            values.append(aggregate)
            evidence.append(
                {
                    "sample_id": expected_id,
                    "raw_scores": scores,
                    "selected_indices": selected,
                    "aggregate": aggregate,
                }
            )
        details = {
            "protocol": VIDEO_HPSV3_PROTOCOL,
            "scorer_identity": dict(self.identity),
            "sample_evidence": evidence,
            "valid_mask": [True] * len(values),
        }
        return np.asarray(values, dtype=np.float32), details

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self.transport.post(
            self.url,
            data=json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(),
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
            allow_redirects=False,
            stream=True,
        )
        try:
            status = getattr(response, "status_code", None)
            if not isinstance(status, int) or not 200 <= status < 300:
                raise RewardTransportError(
                    f"Video HPSv3 server returned HTTP {status}."
                )
            try:
                decoded = json.loads(response.content)
            except (TypeError, ValueError):
                raise RewardProtocolError(
                    "Video HPSv3 response is not valid JSON."
                ) from None
            if not isinstance(decoded, Mapping):
                raise RewardProtocolError("Video HPSv3 response must be a mapping.")
            return decoded
        finally:
            response.close()


class VideoHPSv3JSONApplication:
    def __init__(self, *, scorer_identity: Mapping[str, Any], scorer: Any) -> None:
        self.identity = _identity(scorer_identity)
        self.scorer = scorer
        if not callable(getattr(scorer, "reward", None)):
            raise TypeError("Video HPSv3 scorer must provide reward(...).")

    def handle(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("protocol") != VIDEO_HPSV3_PROTOCOL:
            raise RewardProtocolError("Video HPSv3 request protocol mismatch.")
        if _identity(payload.get("scorer_identity")) != self.identity:
            raise RewardProtocolError("Video HPSv3 requested the wrong scorer.")
        requested = _mapping_list(payload.get("samples"), "request samples")
        if not requested:
            raise RewardProtocolError("Video HPSv3 request batch is empty.")
        decoded, seen = [], set()
        for sample in requested:
            identity, prompt = sample.get("sample_id"), sample.get("prompt")
            frames = sample.get("jpeg_frames")
            if (
                not isinstance(identity, str)
                or not identity.strip()
                or identity in seen
                or not isinstance(prompt, str)
                or not prompt.strip()
                or isinstance(frames, (str, bytes))
                or not isinstance(frames, Sequence)
                or len(frames) < 4
            ):
                raise RewardProtocolError("Video HPSv3 request sample is invalid.")
            seen.add(identity)
            try:
                jpegs = [base64.b64decode(frame, validate=True) for frame in frames]
            except (TypeError, ValueError):
                raise RewardProtocolError(
                    "Video HPSv3 JPEG base64 is invalid."
                ) from None
            if any(not jpeg for jpeg in jpegs):
                raise RewardProtocolError("Video HPSv3 JPEG is empty.")
            decoded.append((identity, prompt, jpegs))

        prompts, paths, frame_counts = [], [], []
        with TemporaryDirectory(prefix="visual-rl-hpsv3-") as directory:
            for sample_id, prompt, jpegs in decoded:
                frame_counts.append(len(jpegs))
                for frame_index, jpeg in enumerate(jpegs):
                    path = Path(directory) / f"{len(paths):08d}-{frame_index:04d}.jpg"
                    path.write_bytes(jpeg)
                    prompts.append(prompt)
                    paths.append(str(path))
            scores = self._score_microbatches(prompts, paths)

        samples, offset = [], 0
        for (sample_id, _, _), count in zip(decoded, frame_counts, strict=True):
            samples.append(
                {"sample_id": sample_id, "raw_scores": scores[offset : offset + count]}
            )
            offset += count
        return {
            "protocol": VIDEO_HPSV3_PROTOCOL,
            "scorer_identity": dict(self.identity),
            "samples": samples,
        }

    def _score_microbatches(self, prompts: list[str], paths: list[str]) -> list[float]:
        scores = []
        for start in range(0, len(paths), VIDEO_HPSV3_SCORER_BATCH_SIZE):
            batch_prompts = prompts[start : start + VIDEO_HPSV3_SCORER_BATCH_SIZE]
            batch_paths = paths[start : start + VIDEO_HPSV3_SCORER_BATCH_SIZE]
            real_count = len(batch_paths)
            if real_count < VIDEO_HPSV3_SCORER_BATCH_SIZE:
                padding = VIDEO_HPSV3_SCORER_BATCH_SIZE - real_count
                batch_prompts += [batch_prompts[-1]] * padding
                batch_paths += [batch_paths[-1]] * padding
            rewards = self.scorer.reward(prompts=batch_prompts, image_paths=batch_paths)
            scores.extend(
                _hps_scores(rewards, VIDEO_HPSV3_SCORER_BATCH_SIZE)[:real_count]
            )
        return scores


def _identity(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "scorer_revision",
        "checkpoint_sha256",
        "runtime_manifest_sha256",
    }:
        raise RewardProtocolError("Video HPSv3 scorer identity is invalid.")
    try:
        return video_hpsv3_identity(
            scorer_revision=value["scorer_revision"],
            checkpoint_sha256=value["checkpoint_sha256"],
            runtime_manifest_sha256=value["runtime_manifest_sha256"],
        )
    except (KeyError, ValueError):
        raise RewardProtocolError("Video HPSv3 scorer identity is invalid.") from None


def _batch_identity(
    prompts: Any, metadata: Any, sample_id: Any
) -> tuple[list[str], list[str]]:
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("Video HPSv3 prompts must be a sequence.")
    prompts = list(prompts)
    if not prompts or any(
        not isinstance(value, str) or not value.strip() for value in prompts
    ):
        raise ValueError("Video HPSv3 prompts must be non-empty strings.")
    if (
        isinstance(metadata, (str, bytes))
        or not isinstance(metadata, Sequence)
        or len(metadata) != len(prompts)
    ):
        raise ValueError("Video HPSv3 metadata must match prompts.")
    if isinstance(sample_id, (str, bytes)) or not isinstance(sample_id, Sequence):
        raise RewardProtocolError("Video HPSv3 requires explicit sample_id.")
    identities = list(sample_id)
    if (
        len(identities) != len(prompts)
        or len(set(identities)) != len(identities)
        or any(not isinstance(value, str) or not value.strip() for value in identities)
    ):
        raise RewardProtocolError("Video HPSv3 sample_id batch is invalid.")
    return prompts, identities


def _mapping_list(value: Any, field: str) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RewardProtocolError(f"Video HPSv3 {field} must be a sequence.")
    result = list(value)
    if any(not isinstance(item, Mapping) for item in result):
        raise RewardProtocolError(f"Video HPSv3 {field} must contain mappings.")
    return result


def _numbers(value: Any, field: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RewardProtocolError(f"Video HPSv3 {field} is invalid.")
    try:
        return [_finite(item, field) for item in value]
    except (TypeError, ValueError):
        raise RewardProtocolError(f"Video HPSv3 {field} is invalid.") from None


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Video HPSv3 {field} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"Video HPSv3 {field} must be numeric.") from None
    if not math.isfinite(result):
        raise ValueError(f"Video HPSv3 {field} must be finite.")
    return result


def _hps_scores(value: Any, expected: int) -> list[float]:
    try:
        rewards = list(value)
    except TypeError:
        raise RewardProtocolError("HPSv3 scorer result is invalid.") from None
    if len(rewards) != expected:
        raise RewardProtocolError("HPSv3 scorer result count mismatch.")
    result = []
    for value in rewards:
        try:
            value = value[0]
        except (IndexError, KeyError, TypeError):
            pass
        if callable(getattr(value, "item", None)):
            value = value.item()
        try:
            result.append(_finite(value, "score"))
        except (TypeError, ValueError):
            raise RewardProtocolError("HPSv3 scorer result is invalid.") from None
    return result


def _loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.path != VIDEO_HPSV3_PATH
    ):
        raise ValueError(
            f"Video HPSv3 URL must be localhost HTTP at {VIDEO_HPSV3_PATH}."
        )
    return value


REWARD_CLIENTS.register("video_hpsv3", VideoHPSv3RewardClient)
