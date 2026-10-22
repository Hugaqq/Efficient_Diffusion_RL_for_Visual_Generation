#!/usr/bin/env python3
"""Private persistent HPSv3 JSON worker; never changes an existing service.

CPU weights persist between requests. One video is scored serially (all frames,
frame microbatch1); GPU use is admitted with headroom and offloaded in finally.
The artifact manifest is derived from actual loaded tensors/processor/code.
"""
from __future__ import annotations
import argparse
import base64
from dataclasses import asdict, is_dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import threading
import time
import traceback

from visual_rl.core.protocols.flash_hpsv3 import (
    PROTOCOL, VERSION, MAX_REQUEST_BYTES, VideoContract, aggregate_frame_scores,
    canonical_json, decode_json, digest, scalar_fp32,
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def loaded_tensor_sha(model) -> str:
    """Streaming hash of actual CPU state, without a model-sized bytes copy."""
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.device.type != 'cpu':
            raise ValueError('identity must be measured with actual weights on CPU')
        metadata = canonical_json({'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)})
        h.update(len(metadata).to_bytes(8, 'big')); h.update(metadata)
        data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        view = memoryview(data).cast('B')
        for start in range(0, len(view), 16 * 1024 * 1024):
            h.update(view[start:start + 16 * 1024 * 1024])
    return h.hexdigest()


class HpsBackend:
    def __init__(self, checkpoint: Path, config: Path, contract: VideoContract, min_free_gib: float):
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        import hpsv3
        from hpsv3 import HPSv3RewardInferencer
        torch.set_num_threads(2)
        if checkpoint.suffix != '.safetensors':
            raise ValueError('only explicit safetensors reward checkpoints are accepted')
        self.torch, self.contract, self.min_free_gib = torch, contract, min_free_gib
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if available < 80 * 1024**3:
            raise BlockingIOError('need80GiB available host RAM for safe cold HPSv3 loading')
        self.inferencer = HPSv3RewardInferencer(device='cpu', checkpoint_path=str(checkpoint), config_path=str(config))
        self.inferencer.model.eval().requires_grad_(False)
        processor = self.inferencer.processor
        processor_state = {
            'tokenizer_backend': processor.tokenizer.backend_tokenizer.to_str(),
            'special_tokens': processor.tokenizer.special_tokens_map,
            'chat_template': processor.chat_template,
            'image_processor': processor.image_processor.to_dict(),
        }
        package_root = Path(inspect.getfile(hpsv3)).parent
        source_files = {str(path.relative_to(package_root)): file_sha(path) for path in sorted(package_root.rglob('*.py'))}
        facts = {
            'loaded_model_state_sha256': loaded_tensor_sha(self.inferencer.model),
            'loaded_processor_sha256': digest(processor_state),
            'checkpoint_file_sha256': file_sha(checkpoint),
            'config_file_sha256': file_sha(config),
            'hps_package_source_sha256': digest(source_files),
            'service_source_sha256': file_sha(Path(__file__)),
            'wire_source_sha256': file_sha(Path(inspect.getfile(VideoContract))),
            'torch_version': torch.__version__,
            'transformers_version': importlib.metadata.version('transformers'),
            'use_special_tokens': self.inferencer.use_special_tokens,
            'data_config': asdict(self.inferencer.data_config) if is_dataclass(self.inferencer.data_config) else vars(self.inferencer.data_config),
        }
        self.health = {
            'status': 'ready', 'reward': 'hpsv3', 'protocol': PROTOCOL, 'protocol_version': VERSION,
            'identity_facts': facts, 'video_contract': contract.payload(),
            'service_identity': digest({'identity_facts': facts, 'video_contract': contract.payload()}),
            'result_device': 'cpu', 'result_dtype': 'fp32',
        }

    def score(self, sample_id: str, prompt: str, images: list) -> list[float]:
        torch = self.torch
        free = torch.cuda.mem_get_info()[0] / 1024**3
        if free < self.min_free_gib:
            raise BlockingIOError('insufficient GPU admission headroom')
        started = time.monotonic()
        stop = threading.Event()
        free_samples = [free]
        def monitor():
            while not stop.wait(.1):
                free_samples.append(torch.cuda.mem_get_info()[0] / 1024**3)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        torch.cuda.reset_peak_memory_stats()
        try:
            self.inferencer.model.to('cuda')
            self.inferencer.device = 'cuda'
            scores = []
            with torch.inference_mode():
                for image in images:
                    value = self.inferencer.reward(prompts=[prompt], image_paths=[image])
                    scores.append(scalar_fp32(float(value[0][0].item())))
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            free_samples.append(torch.cuda.mem_get_info()[0] / 1024**3)
        finally:
            self.inferencer.model.to('cpu')
            self.inferencer.device = 'cpu'
            torch.cuda.empty_cache()
            stop.set(); thread.join(timeout=3)
        if min(free_samples) < 16:
            raise BlockingIOError('GPU safety margin fell below16GiB; do not commit scores')
        print(json.dumps({'event': 'real_hpsv3_video_scored', 'sample_id': sample_id,
                          'frames': len(scores), 'seconds_including_transfers': time.monotonic()-started,
                          'own_peak_allocated_gib': peak, 'min_card_free_gib': min(free_samples)}), flush=True)
        return scores


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def reply(self, status: int, value: dict):
            body = canonical_json(value)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path != '/healthz':
                return self.reply(404, {'error': 'not_found'})
            self.reply(200, backend.health)
        def do_POST(self):
            if self.path != '/score':
                return self.reply(404, {'error': 'not_found'})
            try:
                self.connection.settimeout(30)
                if self.headers.get('Content-Type', '').split(';', 1)[0] != 'application/json':
                    raise ValueError('JSON required')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError('invalid request byte count')
                data = decode_json(self.rfile.read(length))
                if set(data) != {'protocol', 'protocol_version', 'service_identity', 'sample_id', 'prompt', 'jpeg_frames'}:
                    raise ValueError('invalid request fields')
                if (data['protocol'], data['protocol_version'], data['service_identity']) != (PROTOCOL, VERSION, backend.health['service_identity']):
                    raise ValueError('wrong service/protocol identity')
                if not isinstance(data['sample_id'], str) or not 0 < len(data['sample_id']) <= 2048:
                    raise ValueError('invalid sample identity')
                if not isinstance(data['prompt'], str) or len(data['prompt'].encode('utf-8')) > 32768:
                    raise ValueError('invalid prompt')
                frames = data['jpeg_frames']
                if not isinstance(frames, list) or len(frames) != backend.contract.frames:
                    raise ValueError('all frames are required')
                from PIL import Image
                images = []
                for encoded in frames:
                    raw = base64.b64decode(encoded, validate=True)
                    if len(raw) > 2 * 1024 * 1024:
                        raise ValueError('oversized frame')
                    with Image.open(BytesIO(raw), formats=['JPEG']) as image:
                        if image.size != (backend.contract.width, backend.contract.height):
                            raise ValueError('frame shape mismatch')
                        images.append(image.convert('RGB'))
            except Exception:
                return self.reply(400, {'error': 'invalid_video_request'})
            try:
                scores = backend.score(data['sample_id'], data['prompt'], images)
                score, _indices = aggregate_frame_scores(scores, backend.contract)
                self.reply(200, {'protocol': PROTOCOL, 'protocol_version': VERSION,
                    'service_identity': backend.health['service_identity'], 'sample_id': data['sample_id'],
                    'frame_scores': scores, 'video_score': score})
            except BlockingIOError:
                self.reply(503, {'error': 'gpu_admission_or_margin'})
            except Exception:
                traceback.print_exc()
                self.reply(500, {'error': 'hpsv3_scoring_failed'})
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--artifact-output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18820)
    p.add_argument('--frames', type=int, default=81)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--min-free-gib', type=float, default=40)
    a = p.parse_args()
    if not 1 <= a.port <= 65535 or not 40 <= a.min_free_gib <= 80:
        p.error('invalid private port or safety admission budget')
    backend = HpsBackend(a.checkpoint.resolve(strict=True), a.config.resolve(strict=True),
                         VideoContract(a.frames, a.height, a.width), a.min_free_gib)
    server = HTTPServer(('127.0.0.1', a.port), make_handler(backend))
    artifact = canonical_json(backend.health)
    a.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    if a.artifact_output.exists():
        if a.artifact_output.read_bytes() != artifact:
            raise ValueError('refusing to overwrite a different immutable HPSv3 artifact')
    else:
        with a.artifact_output.open('xb') as handle:
            handle.write(artifact)
    print('HPSV3_VIDEO_READY ' + backend.health['service_identity'], flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
