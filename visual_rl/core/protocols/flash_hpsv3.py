"""JSON-only HPSv3 video wire; no imports of model/HTTP runtimes.

Scores every frame, then averages the largest floor(F*.3) scores, matching
Flash-GRPO's video_hpsv3_remote. Artifact identity comes from the live server's
loaded weights/processor/code, never from a caller-invented attestation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any

PROTOCOL = 'flash_hpsv3_json'
VERSION = 'v1'
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_HEALTH_BYTES = 256 * 1024
_SHA = re.compile(r'^[0-9a-f]{64}$')


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_id(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError('expected a lowercase SHA256 identity')
    return value


def _pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode_json(raw: bytes, limit: int = MAX_REQUEST_BYTES) -> dict:
    if len(raw) > limit:
        raise ValueError('JSON exceeds byte limit')
    def invalid_constant(value: str) -> None:
        raise ValueError('non-finite JSON constant')
    result = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError('expected a JSON object')
    return result


@dataclass(frozen=True)
class VideoContract:
    frames: int
    height: int
    width: int

    def __post_init__(self) -> None:
        for name, limit in [('frames', 128), ('height', 4096), ('width', 4096)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f'invalid video {name}')

    @property
    def top_k(self) -> int:
        return max(1, int(self.frames * .3))

    def payload(self) -> dict:
        return {'frames': self.frames, 'height': self.height, 'width': self.width, 'top_k': self.top_k}


def scalar_fp32(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('frame scores must be finite real numbers')
    try:
        result = struct.unpack('!f', struct.pack('!f', value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError('score is not representable in fp32') from exc
    if not math.isfinite(result):
        raise ValueError('score is not representable in fp32')
    return result


def aggregate_frame_scores(values: Any, contract: VideoContract) -> tuple[float, tuple[int, ...]]:
    if not isinstance(values, (list, tuple)) or len(values) != contract.frames:
        raise ValueError('must score every frame exactly once')
    scores = [scalar_fp32(value) for value in values]
    indices = tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:contract.top_k])
    # Python mean as in the author code; publish the final training scalar as fp32.
    return scalar_fp32(sum(scores[i] for i in indices) / contract.top_k), indices


def validate_health(payload: dict, expected_identity: str, contract: VideoContract) -> dict:
    expected_keys = {'status', 'reward', 'protocol', 'protocol_version', 'service_identity', 'identity_facts', 'video_contract', 'result_device', 'result_dtype'}
    if set(payload) != expected_keys:
        raise ValueError('invalid HPSv3 health fields')
    if (payload['status'], payload['reward'], payload['protocol'], payload['protocol_version']) != ('ready', 'hpsv3', PROTOCOL, VERSION):
        raise ValueError('wrong or unready HPSv3 service')
    if (payload['result_device'], payload['result_dtype']) != ('cpu', 'fp32'):
        raise ValueError('HPSv3 wire result precision mismatch')
    if payload['video_contract'] != contract.payload():
        raise ValueError('HPSv3 frame/shape/aggregation contract mismatch')
    facts = payload['identity_facts']
    if not isinstance(facts, dict) or not facts:
        raise ValueError('loaded-service identity facts missing')
    identity = sha256_id(payload['service_identity'])
    if digest({'identity_facts': facts, 'video_contract': contract.payload()}) != identity or identity != sha256_id(expected_identity):
        raise ValueError('loaded HPSv3 service identity mismatch')
    return payload


def validate_score(payload: dict, *, sample_id: str, identity: str, contract: VideoContract) -> tuple[float, tuple[int, ...]]:
    if set(payload) != {'protocol', 'protocol_version', 'service_identity', 'sample_id', 'frame_scores', 'video_score'}:
        raise ValueError('invalid HPSv3 score response fields')
    if (payload['protocol'], payload['protocol_version'], payload['service_identity'], payload['sample_id']) != (PROTOCOL, VERSION, identity, sample_id):
        raise ValueError('HPSv3 response identity/order mismatch')
    score, indices = aggregate_frame_scores(payload['frame_scores'], contract)
    if scalar_fp32(payload['video_score']) != score:
        raise ValueError('HPSv3 top-frame aggregate mismatch')
    return score, indices
