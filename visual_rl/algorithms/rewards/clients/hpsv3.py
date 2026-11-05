"""Optional HPSv3 video resource, using the existing video-general plugin port.

No new trainer or bypass of resource acquisition. The factory checks a live
server's loaded-model identity against the materialized artifact before bind.
"""
from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

from visual_rl.core.protocols.flash_hpsv3 import (
    PROTOCOL, VERSION, MAX_HEALTH_BYTES, MAX_REQUEST_BYTES, VideoContract,
    canonical_json, decode_json, digest, sha256_id, validate_health, validate_score, scoring_identity,
)
from visual_rl.runtime.flash140_logging import progress
from visual_rl.algorithms.rewards.clients.input import (
    resolve_pointwise_reward_input, pointwise_reward_output,
)
from visual_rl.algorithms.rewards.clients.mock import (
    close_http_response, read_bounded_http_response, requests_session,
    validate_max_response_bytes,
)


QUEUE_FULL_ERROR = 'hpsv3_queue_full'
QUEUE_RETRY_MAX_DELAY_S = 10.


def loopback_origin(value: str) -> str:
    p = urlsplit(value)
    if (p.scheme != 'http' or p.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or p.username is not None or p.password is not None or p.query or p.fragment
            or p.path not in {'', '/'} or p.port is None):
        raise ValueError('HPSv3 requires an explicit loopback HTTP origin and port')
    host = f'[{p.hostname}]' if p.hostname == '::1' else p.hostname
    return f'http://{host}:{p.port}'


def jpeg_frames(video, layout: str, contract: VideoContract) -> list[str]:
    import base64
    import torch
    from PIL import Image

    if not isinstance(video, torch.Tensor) or video.device.type != 'cpu':
        raise ValueError('HPSv3 input must be CPU video tensors')
    if video.requires_grad or video.grad_fn is not None:
        raise ValueError('HPSv3 video must be detached')
    if layout == 'BFCHW':
        expected = (contract.frames, 3, contract.height, contract.width)
    elif layout == 'BFHWC':
        expected = (contract.frames, contract.height, contract.width, 3)
    else:
        raise ValueError('HPSv3 requires typed BFCHW/BFHWC video')
    if tuple(video.shape) != expected:
        raise ValueError('HPSv3 video shape/frame count mismatch')
    if not video.is_floating_point() and video.dtype != torch.uint8:
        raise ValueError('HPSv3 supports decoded float or quantized uint8 video')
    result = []
    for frame in video:
        if frame.is_floating_point():
            if not bool(torch.isfinite(frame).all()):
                raise ValueError('HPSv3 frame must be finite')
            # Same rounding/clamping and default JPEG settings as author code.
            pixels = (frame * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            pixels = frame
        if layout == 'BFCHW':
            pixels = pixels.permute(1, 2, 0)
        buffer = BytesIO()
        Image.fromarray(pixels.contiguous().numpy()).save(buffer, format='JPEG')
        result.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
    return result


class Hpsv3VideoClient:
    def __init__(self, *, origin: str, service_identity: str, contract: VideoContract,
                 timeout_s: float, max_response_bytes: int, transport=None):
        self.origin = loopback_origin(origin)
        self.service_identity = sha256_id(service_identity)
        if not isinstance(contract, VideoContract):
            raise TypeError('contract must be VideoContract')
        self.contract = contract
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be finite and positive')
        self.timeout_s = timeout_s
        self.max_response_bytes = validate_max_response_bytes(max_response_bytes)
        self.transport = requests_session() if transport is None else transport
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.transport.close()

    def _request(self, method: str, route: str, payload=None):
        if self.closed:
            raise RuntimeError('HPSv3 client is closed')
        raw = None if payload is None else canonical_json(payload)
        if raw is not None and len(raw) > MAX_REQUEST_BYTES:
            raise ValueError('HPSv3 video request exceeds bounded transport size')
        # The whole call, including queue-full backoff, stays inside timeout_s.
        # Only an explicit pre-admission queue-full 503 is retried; any other
        # status, transport error or timeout may follow acceptance and is final.
        # The first attempt keeps the original per-socket timeout_s semantics;
        # a retried attempt is also checked against the wall-clock deadline
        # (its per-socket timeout alone would not cap a slow trickle).
        deadline = time.monotonic() + self.timeout_s
        timeout, attempt = self.timeout_s, 0
        def over_deadline():
            if attempt and time.monotonic() > deadline:
                raise RuntimeError('HPSv3 retried request exceeded the wall-clock timeout budget; not retried')
        while True:
            response = self.transport.request(
                method, self.origin + route, data=raw,
                headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                timeout=timeout, allow_redirects=False, stream=True,
            )
            try:
                over_deadline()
                if response.status_code != 200:
                    delay = self._queue_full_delay(response, method, route, attempt)
                    if delay is None:
                        raise RuntimeError(f'HPSv3 returned HTTP {response.status_code}; no automatic POST retry')
                else:
                    if response.headers.get('Content-Type', '').split(';', 1)[0].strip() != 'application/json':
                        raise ValueError('HPSv3 response must be JSON')
                    body = read_bounded_http_response(response, max_response_bytes=self.max_response_bytes)
                    over_deadline()
                    return decode_json(body, self.max_response_bytes)
            finally:
                close_http_response(response)
            remaining = deadline - time.monotonic()
            if delay >= remaining:
                raise RuntimeError('HPSv3 queue stayed full for the request timeout budget; never admitted')
            progress("hpsv3_queue_full_backoff", attempt=attempt, delay_s=delay)
            time.sleep(delay)
            attempt += 1
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise RuntimeError('HPSv3 queue stayed full for the request timeout budget; never admitted')

    def _queue_full_delay(self, response, method, route, attempt):
        """Backoff seconds ONLY for the service's explicit pre-admission rejection."""
        if (method, route, response.status_code) != ('POST', '/score', 503):
            return None
        retry_after = response.headers.get('Retry-After', '')
        if not isinstance(retry_after, str) or not retry_after.strip().isdigit():
            return None
        if response.headers.get('Content-Type', '').split(';', 1)[0].strip() != 'application/json':
            return None
        try:
            body = decode_json(read_bounded_http_response(response, max_response_bytes=self.max_response_bytes),
                               self.max_response_bytes)
        except Exception:  # noqa: BLE001 - an unreadable 503 is simply not retryable
            return None
        if body != {'error': QUEUE_FULL_ERROR}:
            return None
        base = max(.25, min(float(int(retry_after.strip())), QUEUE_RETRY_MAX_DELAY_S))
        return min(base * 2 ** min(attempt, 6), QUEUE_RETRY_MAX_DELAY_S)

    def healthcheck(self):
        return validate_health(self._request('GET', '/healthz'), self.service_identity, self.contract)

    def score_video(self, *, sample_id: str, prompt: str, frames_base64: list[str]):
        if not isinstance(sample_id, str) or not sample_id or len(sample_id) > 2048:
            raise ValueError('invalid sample identity')
        if not isinstance(prompt, str) or len(prompt.encode('utf-8')) > 32768:
            raise ValueError('invalid or oversized HPSv3 prompt')
        if len(frames_base64) != self.contract.frames:
            raise ValueError('must transmit every video frame')
        progress("hpsv3_video_start", sample_id=sample_id, frames=len(frames_base64))
        payload = self._request('POST', '/score', {
            'protocol': PROTOCOL, 'protocol_version': VERSION,
            'service_identity': self.service_identity, 'sample_id': sample_id,
            'prompt': prompt, 'jpeg_frames': frames_base64,
        })
        score, indices = validate_score(payload, sample_id=sample_id,
                                       identity=self.service_identity, contract=self.contract)
        progress("hpsv3_video_complete", sample_id=sample_id, frames=self.contract.frames, reward=score)
        return score, {
            'scored_frame_indices': list(range(self.contract.frames)),
            'selected_frame_indices': list(indices),
            'frame_scores': payload['frame_scores'],
        }

    def score(self, *, batch):
        resolved = resolve_pointwise_reward_input(batch)
        if len(resolved.media) != resolved.flat_size:
            raise ValueError('HPSv3 media row mapping mismatch')
        scores, metadata = [], []
        for row, (sample_id, prompt) in enumerate(zip(resolved.sample_ids, resolved.prompts)):
            encoded = jpeg_frames(resolved.media[row], resolved.media_layout, self.contract)
            value, record = self.score_video(sample_id=sample_id, prompt=prompt, frames_base64=encoded)
            scores.append(value)
            metadata.append(record)
        return pointwise_reward_output(resolved, scores, shared_metadata={
            'reward': 'hpsv3', 'protocol': PROTOCOL, 'protocol_version': VERSION,
            **({'scoring_identity': self.scoring_identity} if isinstance(self, Hpsv3MultiOriginClient)
               else {'service_identity': self.service_identity}),
            'aggregation': 'all_frames_then_top30percent_mean',
            'video_contract': self.contract.payload(),
        }, sample_metadata=metadata)


class Hpsv3MultiOriginClient(Hpsv3VideoClient):
    """相同协议的多个实例；每次请求独享 Session，跨线程不共享可变 transport。"""
    def __init__(self, *, origins, expected_scoring_identity, dispatcher=None, transport_factory=None, **kwargs):
        from visual_rl.algorithms.rewards.clients.hps_dispatch import HpsDispatcher
        self.origins = tuple(loopback_origin(origin) for origin in origins)
        if not 2 <= len(self.origins) <= 8 or len(set(self.origins)) != len(self.origins):
            raise ValueError('multi-origin HPS requires 2..8 distinct loopback origins')
        self.dispatcher = dispatcher or HpsDispatcher(self.origins)
        if self.dispatcher.origins != self.origins:
            raise ValueError('HPS dispatcher origins differ from client origins')
        self.options, self.transport_factory = kwargs, transport_factory
        self.scoring_identity = sha256_id(expected_scoring_identity)
        self.endpoint_identities = None
        self.clients = []
        self.closed = False
        try:
            for origin in self.origins:
                self.clients.append(self._client(origin))
        except BaseException:
            self.close()
            raise
        self.origin = self.origins[0]
        self.service_identity, self.contract = self.clients[0].service_identity, self.clients[0].contract

    def _client(self, origin):
        options = dict(self.options)
        if self.endpoint_identities is not None:
            options['service_identity'] = self.endpoint_identities[self.origins.index(origin)]
        return Hpsv3VideoClient(origin=origin, **options,
            **({} if self.transport_factory is None else dict(transport=self.transport_factory(origin))))

    def healthcheck(self):
        if self.closed:
            raise RuntimeError('HPSv3 client is closed')
        observed = []
        for index, client in enumerate(self.clients):
            payload = client._request('GET', '/healthz')
            # 首次发现每端点完整 identity；评分身份由已校验的本地 artifact 锚定。
            expected = payload.get('service_identity') if self.endpoint_identities is None else self.endpoint_identities[index]
            validate_health(payload, expected, self.contract)
            if scoring_identity(payload, self.contract) != self.scoring_identity:
                raise ValueError('HPS instances have different scoring identities')
            observed.append(payload)
        self.endpoint_identities = tuple(value['service_identity'] for value in observed)
        for client, identity in zip(self.clients, self.endpoint_identities):
            client.service_identity = identity
        return observed[0]

    def score_video(self, **kwargs):
        if self.closed:
            raise RuntimeError('HPSv3 client is closed')
        if self.endpoint_identities is None:
            raise RuntimeError('HPS pool requires health admission before scoring')
        with self.dispatcher.lease() as index:
            client = self._client(self.origins[index])
            try:
                return client.score_video(**kwargs)
            finally:
                client.close()

    def close(self):
        if not self.closed:
            self.closed = True
            error = None
            for client in self.clients:
                try:
                    client.close()
                except Exception as exc:
                    error = error or exc
            self.clients = []
            if error is not None:
                raise error


class Hpsv3VideoResourceFactory:
    """Explicit module:Class factory supported by DefaultRewardResourceFactory."""
    def __init__(self, *, dispatcher=None):
        self.dispatcher = dispatcher

    def __call__(self, request):
        from visual_rl.core.serialization import to_plain_dict
        from visual_rl.runtime.reward_resources import AcquiredRewardResource, RewardResourceBindingFacts

        descriptor, binding = request.descriptor, request.runtime_binding
        if (descriptor.protocol, descriptor.protocol_version) != (PROTOCOL, VERSION):
            raise ValueError('HPSv3 artifact protocol mismatch')
        if binding is None or (binding.execution_domain, binding.device, binding.dtype) != ('remote', 'cpu', 'fp32'):
            raise ValueError('HPSv3 requires remote CPU/fp32 result binding')
        origin = loopback_origin(binding.endpoint)
        if urlsplit(origin).hostname not in binding.trusted_hosts or binding.ca_bundle is not None:
            raise ValueError('HPSv3 loopback host must be trusted; TLS settings are not accepted for HTTP')
        semantic = to_plain_dict(descriptor.semantic_factory_config)
        if set(semantic) not in ({'service_identity', 'video_contract'}, {'service_identity', 'video_contract', 'origins'}):
            raise ValueError('HPSv3 semantic config fields mismatch')
        identity = sha256_id(semantic['service_identity'])
        video = semantic['video_contract']
        if not isinstance(video, dict) or set(video) != {'frames', 'height', 'width', 'top_k'}:
            raise ValueError('invalid HPSv3 video contract fields')
        contract = VideoContract(video['frames'], video['height'], video['width'])
        if contract.payload() != video:
            raise ValueError('HPSv3 top_k must follow the author30percent rule')
        location = request.artifact_location
        path = location / 'hpsv3-service.json' if location.is_dir() else location
        if path.stat().st_size > MAX_HEALTH_BYTES:
            raise ValueError('HPSv3 artifact manifest too large')
        artifact = decode_json(path.read_bytes(), MAX_HEALTH_BYTES)
        validate_health(artifact, identity, contract)
        options = dict(service_identity=identity, contract=contract,
                       timeout_s=binding.timeout_s, max_response_bytes=binding.max_response_bytes)
        origins = semantic.get('origins', [origin])
        if (type(origins) is not list or not origins or origins[0] != origin
                or any(loopback_origin(item) != item or urlsplit(item).hostname not in binding.trusted_hosts
                       for item in origins)):
            raise ValueError('HPS origins must be canonical trusted loopbacks, first equal to endpoint')
        if len(origins) == 1:
            if self.dispatcher is not None:
                raise ValueError('single-origin HPS must not use a multi-origin dispatcher')
            resource = Hpsv3VideoClient(origin=origin, **options)
        else:
            resource = Hpsv3MultiOriginClient(origins=origins, dispatcher=self.dispatcher,
                expected_scoring_identity=scoring_identity(artifact, contract), **options)
        try:
            observed = resource.healthcheck()
            if len(origins) == 1 and observed != artifact:
                raise ValueError('live HPSv3 identity differs from materialized artifact')
            endpoint_id = digest({'origin': origin, 'observed': observed}) if len(origins) == 1 else digest(
                {'origins': origins, 'scoring_identity': resource.scoring_identity, 'dispatch': 'least-inflight.v1'})
            return AcquiredRewardResource(resource=resource, binding_facts=RewardResourceBindingFacts(
                endpoint_identity=endpoint_id, protocol=PROTOCOL, protocol_version=VERSION,
                device='cpu', dtype='fp32', worker_domain='remote',
            ))
        except BaseException:
            resource.close()
            raise
