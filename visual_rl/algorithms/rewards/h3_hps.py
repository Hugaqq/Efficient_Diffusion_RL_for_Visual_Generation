"""H3's explicit video-only HPSv3 reward (weight 1), NOT AV alignment quality.

All 124 frames use the existing flash_hpsv3_json.v1 protocol (top37 mean).
Joint audio remains mandatory decoded media, validated but never sent to HPS.
No HTTP/JPEG/top-k reimplementation, retries, model imports or fallback reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from visual_rl.algorithms.rewards.clients.hpsv3 import (
    Hpsv3VideoClient, Hpsv3VideoResourceFactory, jpeg_frames,
)
from visual_rl.core.protocols.flash_hpsv3 import (
    MAX_HEALTH_BYTES, PROTOCOL, VERSION, VideoContract, decode_json, sha256_id, validate_health,
)
from visual_rl.core.serialization import to_plain_dict
from visual_rl.runtime.reward_resources import RewardResourceAcquireRequest


@dataclass(frozen=True)
class H3HPSIdentity:
    profile: str
    version: int
    weight: float
    protocol: str
    protocol_version: str
    service_identity: str
    video_contract: VideoContract
    manifest_sha256: str
    endpoint_identity: str


class H3HPSReward:
    """Own exactly one client acquired by the existing HPS resource factory.

    Constructor receives the real materialized resource request: local manifest
    location, explicit expected service/contract semantics and remote CPU/fp32
    binding with loopback origin, trusted host, timeout and response byte limit.
    Constructor only reads/validates the bounded local artifact; no networking.
    Explicit acquire() performs the factory's live health/identity admission.
    This owner must not share its client with another resource pool/owner.
    Runtime integration must provide an admitted 124-frame artifact, NOT reuse
    an 81-frame Flash manifest. close() is idempotent; no automatic reacquisition.
    """

    def __init__(self, request: RewardResourceAcquireRequest, *, video_contract: VideoContract):
        if not isinstance(request, RewardResourceAcquireRequest):
            raise TypeError("H3 HPS requires a real RewardResourceAcquireRequest")
        if not isinstance(video_contract, VideoContract) or video_contract.frames != 124:
            raise ValueError("h3_hpsv3_video_only_v1 requires all 124 frames (not legacy 81)")
        descriptor = request.descriptor
        if descriptor.factory_class != "visual_rl.algorithms.rewards.clients.hpsv3:Hpsv3VideoResourceFactory":
            raise ValueError("H3 HPS request must declare the existing HPSv3 resource factory")
        policy = descriptor.allowed_runtime_policy
        if ("cpu" not in policy.allowed_devices or "fp32" not in policy.allowed_dtypes
                or "remote" not in policy.allowed_worker_domains):
            raise ValueError("H3 HPS declared runtime policy must allow remote CPU/fp32")
        if (descriptor.protocol, descriptor.protocol_version) != (PROTOCOL, VERSION):
            raise ValueError("H3 HPS protocol mismatch")
        semantic = to_plain_dict(descriptor.semantic_factory_config)
        if set(semantic) != {"service_identity", "video_contract"} or semantic["video_contract"] != video_contract.payload():
            raise ValueError("H3 HPS declared video contract mismatch")
        self._service_identity = sha256_id(semantic["service_identity"])
        self._contract = video_contract
        self._request = request
        raw = self._manifest_bytes()
        validate_health(decode_json(raw, MAX_HEALTH_BYTES), self._service_identity, self._contract)
        self._manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self._client = None
        self._identity = None
        self._attempted = False
        self._closed = False

    def _manifest_bytes(self) -> bytes:
        location = self._request.artifact_location
        path = location / "hpsv3-service.json" if location.is_dir() else location
        if location.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError("H3 HPS requires an existing local manifest without symlink indirection")
        with path.open("rb") as stream:
            raw = stream.read(MAX_HEALTH_BYTES + 1)
        if len(raw) > MAX_HEALTH_BYTES:
            raise ValueError("H3 HPS artifact manifest too large")
        return raw

    def _check_manifest(self) -> None:
        if hashlib.sha256(self._manifest_bytes()).hexdigest() != self._manifest_sha256:
            raise ValueError("H3 HPS manifest changed since declaration")

    def acquire(self) -> H3HPSReward:
        if self._closed or self._attempted:
            raise RuntimeError("H3 HPS is closed or acquisition already attempted")
        self._attempted = True
        self._check_manifest()
        # This real factory validates binding/trusted loopback, local manifest,
        # live health and exact observed==artifact before returning a client.
        acquired = Hpsv3VideoResourceFactory()(self._request)
        client = acquired.resource
        try:
            self._check_manifest()
            if not isinstance(client, Hpsv3VideoClient):
                raise TypeError("H3 HPS factory must return Hpsv3VideoClient")
            self._identity = H3HPSIdentity(
                "h3_hpsv3_video_only_v1", 1, 1.0, PROTOCOL, VERSION,
                self._service_identity, self._contract, self._manifest_sha256,
                acquired.binding_facts.endpoint_identity,
            )
            self._client = client
        except BaseException:
            client.close()
            raise
        return self

    @property
    def identity(self) -> H3HPSIdentity:
        if self._identity is None:
            raise RuntimeError("H3 HPS identity is not live-admitted yet")
        return self._identity

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            client, self._client = self._client, None
            if client is not None:
                client.close()

    def score_one(self, *, sample_id: str, prompt: str, media) -> float:
        import torch

        if self._closed or self._client is None:
            raise RuntimeError("H3 HPS must be acquired and open before scoring")
        self._check_manifest()
        # Expired H3AVMedia property access deliberately raises here.
        video, audio, info = media.video, media.audio, media.metadata
        expected_meta = {"fps": 24, "sample_rate": 32000, "frame_count": 124, "sample_count": 165600}
        for key, value in expected_meta.items():
            observed = getattr(info, key, None)
            if type(observed) is not int or observed != value:
                raise ValueError(f"H3 AV metadata mismatch: {key}")
        for name, tensor, shape in (
            ("video", video, (124, 3, self._contract.height, self._contract.width)),
            ("audio", audio, (info.sample_count, 2)),
        ):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                    or tensor.requires_grad or tensor.grad_fn is not None):
                raise ValueError(f"H3 {name} must be detached CPU FP32 with exact AV shape")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"H3 {name} must be finite")
        if bool(((video < 0) | (video > 1)).any()):
            raise ValueError("H3 video pixels must be in [0,1]")
        frames = jpeg_frames(video, "BFCHW", self._contract)
        score, _ = self._client.score_video(sample_id=sample_id, prompt=prompt, frames_base64=frames)
        # The client's original validator checks complete frame scores, sample /
        # service identity and the top-k scalar; never trust an unchecked scalar.
        if type(score) is not float or not math.isfinite(score):
            raise ValueError("HPSv3 must produce a finite Python float")
        return score
