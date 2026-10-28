"""Prepare the audited native FL2VA release in a fresh, separate directory.

Invokes (does not vendor) Apache-2.0 Diffusers' pinned streaming converter.
No downloads, checkpoint-local Python, pipeline, optimizer or training runner.
Source: aaf57d03a3edfbd9de457d2872417a92f149a91d; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess

GIB = 1024**3
COMMIT = 'aaf57d03a3edfbd9de457d2872417a92f149a91d'
CONVERTER_HASH = '86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c'
LICENSE_HASH = 'f9e2070c247517b1ddf65f7b11b393484a18da91a958fd18a97bd0f241c3125c'
MODEL_HASHES = {
    'models/transformers/transformer_minimax_h3.py': '92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef',
    'models/autoencoders/autoencoder_kl_minimax_h3.py': '58e0ed47fdee8227d738eec5bb1e7a47acc21c65ee3de01aa3795c97d679d015',
    'models/autoencoders/autoencoder_kl_minimax_h3_audio.py': 'b56c5bb95a7dc99ace03df192544615c89ec01ac3b134a7f1dd7348b03731d57',
}
# Fixed original bytes actually observed in H3-EXP-001-materials-r1. This narrow
# tool refuses other releases/configurations rather than silently using the
# converter's hardcoded 33B / video architecture on an unknown checkpoint.
CONFIG_HASHES = {
    'model_index.json': 'd1113e0f123c69f79cd0de35ca1771606ebc3ec924270d257b771f96f584aa6b',
    'transformer/config.json': 'f619093a231fcfbcc3d035bec26c50ad864e7331a500d5c519f5045dc1e50458',
    'video_vae/config.json': '3edd2cdd1ebc823c868be55ef917e1b3b8a398fde4d3150dae44a3bf05d9f627',
    'video_vae/source/config.json': '66c68f541e6578ce613ce7a0fc985eb59097038829e49f7535e6d08e6d95ab12',
    'audio_vae/config.json': 'd8f3bcc62e23c7e9806970fa63cca6139c06faa3797cf9c94034f60db8512771',
    'audio_vae/config.yaml': '259f675c9f71eedc24ca4f23965cd32a1b3878fc894f56664c40328d99831a5e',
    'audio_vae/metadata.json': '755d0529d43b2b5c83590f6f44ca659bc68e6a21b01d5669c93e8b2965749bff',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def event(name, **values):
    # This preparation entry requires Linux /proc/meminfo below; ru_maxrss is KiB.
    print(json.dumps({'event': name, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      **values}, sort_keys=True), flush=True)


def inventory(paths, root):
    result = {}
    for path in sorted(paths):
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f'file changed while hashing: {path}')
        name = str(path.relative_to(root))
        result[name] = {'bytes': after.st_size, 'sha256': sha, 'resolved': str(path.resolve(strict=True))}
        event('file_hashed', file=name, bytes=after.st_size, sha256=sha)
    return result


def load_converter(source):
    source = source.resolve(strict=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True, timeout=30).strip()
    if git('rev-parse', 'HEAD') != COMMIT or git('status', '--porcelain'):
        raise ValueError('converter requires the fixed clean upstream checkout')
    path = source / 'scripts/convert_minimax_h3_to_diffusers.py'
    if digest(path) != CONVERTER_HASH or digest(source / 'LICENSE') != LICENSE_HASH:
        raise ValueError('converter/license content differs from reviewed source')
    import diffusers
    installed = Path(diffusers.__file__).resolve().parent
    for name, expected in MODEL_HASHES.items():
        if digest(installed / name) != expected:
            raise ValueError(f'installed H3 model source differs: {name}')
    spec = importlib.util.spec_from_file_location('reviewed_h3_converter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, diffusers.__version__


def raw_inputs(root, blob_root):
    paths = [root / name for name in CONFIG_HASHES]
    shards = sorted((root / 'transformer').glob('*.safetensors'))
    if not shards:
        raise ValueError('native transformer shards missing')
    paths += shards + [root / 'video_vae/source/model.safetensors', root / 'audio_vae/model.safetensors']
    paths += sorted((root / 'transformer').glob('*.index.json'))
    for folder in ('transformer', 'video_vae', 'audio_vae'):
        directory = root / folder
        if directory.is_symlink() or any(p.is_dir() and p.is_symlink() for p in directory.rglob('*')):
            raise ValueError('native component directory symlinks are unsupported')
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not (resolved.is_relative_to(root) or resolved.is_relative_to(blob_root)):
            raise ValueError(f'native file escapes explicit snapshot/blob roots: {path}')
    for name, expected in CONFIG_HASHES.items():
        if digest(root / name) != expected:
            raise ValueError(f'unreviewed native configuration: {name}')
    return paths, shards


def headers(paths):
    from safetensors import safe_open
    result = {}
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f'duplicate source tensor: {name}')
                view = handle.get_slice(name)
                result[name] = {'shape': view.get_shape(), 'dtype': view.get_dtype(), 'file': path.name}
    return result


def validate_raw(converter, root, shards):
    import torch
    from safetensors import safe_open
    from visual_rl.models.implementations.h3_loading import _json
    transformer = headers(shards)
    plan = converter.get_transformer_key_plan(converter.MINIMAX_H3_TRANSFORMER_CONFIG)
    if set(transformer) != set(plan):
        raise ValueError('native transformer keys do not exactly cover converter plan')
    indices = list((root / 'transformer').glob('*.index.json'))
    if len(indices) > 1:
        raise ValueError('ambiguous native transformer index')
    if indices:
        mapping = _json(indices[0].read_bytes())['weight_map']
        if mapping != {key: value['file'] for key, value in transformer.items()}:
            raise ValueError('native transformer index/shard coverage mismatch')
    for key, targets in plan.items():
        if not targets:
            continue
        shape = list(targets[0][1])
        if len(targets) > 1:
            shape[0] = sum(target[1][0] for target in targets)
        dtype = 'F32' if key.startswith(converter.MINIMAX_H3_FP32_SOURCE_PREFIXES) else 'BF16'
        if transformer[key]['shape'] != shape or transformer[key]['dtype'] != dtype:
            raise ValueError(f'native transformer shape/dtype mismatch: {key}')
    # The only dropped DiT key must equal the actual constructor's formula.
    with safe_open(str(root / 'transformer' / transformer['rope.inv_freq']['file']), framework='pt', device='cpu') as f:
        rope = f.get_tensor('rope.inv_freq')
        expected = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        if rope.dtype != torch.float32 or not torch.equal(rope, expected):
            raise ValueError('native rotary buffer does not match reconstruction')
    video = headers([root / 'video_vae/source/model.safetensors'])
    video_plan = converter.get_video_vae_key_plan(converter.MINIMAX_H3_VIDEO_VAE_CONFIG)
    if set(video) != set(video_plan) or any(h['dtype'] != 'F32' for h in video.values()):
        raise ValueError('native video VAE keys/precision differ from converter plan')
    with safe_open(str(root / 'video_vae/source/model.safetensors'), framework='pt', device='cpu') as f:
        if bool(torch.count_nonzero(f.get_tensor('decoder.mask_token'))):
            raise ValueError('native unused video mask buffer is not zero')
    return {'transformer': plan, 'vae': video_plan, 'rope_inv_freq': rope.tolist()}


def validate_output(root, rope_values):
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from visual_rl.models.implementations.h3_loading import build_h3_meta, _weight_headers
    from visual_rl.models.implementations.h3_decode import H3CPUDecoder, _resolve_vae_class
    meta = build_h3_meta(root, rank=4, alpha=8)
    if not torch.equal(meta.nonpersistent_buffers['rope.inv_freq'], torch.tensor(rope_values, dtype=torch.float32)):
        raise ValueError('actual public transformer rotary buffer differs from native weights')
    expected = {name: (spec.shape, spec.dtype) for name, dest in meta.base_to_model.items()
                for spec in (meta.state_specs[dest],)}
    report = {}
    H3CPUDecoder(root, max_media_bytes=1, max_decode_bytes=1)._configs()
    for component in ('transformer', 'vae', 'audio_vae'):
        if component != 'transformer':
            cls = _resolve_vae_class(component)
            with init_empty_weights(include_buffers=False):
                model = cls.from_config(cls.load_config(str(root / component), local_files_only=True))
            expected = {name: (tuple(t.shape), t.dtype) for name, t in model.state_dict().items()}
            del model
        locations, info = _weight_headers(root / component)
        if set(locations) != set(expected):
            raise ValueError(f'{component}: output does not exactly cover actual public model state')
        dtypes = {torch.float32: 'F32', torch.bfloat16: 'BF16', torch.float16: 'F16'}
        for name, (shape, dtype) in expected.items():
            if tuple(info[name]['shape']) != shape or info[name]['dtype'] != dtypes[dtype]:
                raise ValueError(f'{component}: output shape/dtype differs from model: {name}')
        # Stream finite checks, never hold a complete materialized model here.
        for path in sorted(set(locations.values())):
            with safe_open(str(path), framework='pt', device='cpu') as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    for chunk in tensor.reshape(-1).split(1024**2):
                        if not bool(torch.isfinite(chunk).all()):
                            raise ValueError(f'nonfinite converted tensor: {component}/{name}')
                    del chunk, tensor
            event('validated_shard', component=component, file=path.name)
        report[component] = {name: {'shape': h['shape'], 'dtype': h['dtype']} for name, h in info.items()}
    return report


def prepare(args):
    if any(os.environ.get(k) != '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
        raise ValueError('explicit offline mode required')
    root, blob = args.checkpoint.resolve(strict=True), args.blob_root.resolve(strict=True)
    output = args.output.absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('output symlinks unsupported')
    output = output.resolve()
    staging = output.with_name(output.name + '.partial')
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (root, blob)):
        raise ValueError('derived output must be separate from snapshot and blobs')
    if output.exists() or staging.exists():
        raise FileExistsError('preserve previous artifact/partial; choose a new output')
    paths, shards = raw_inputs(root, blob)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = [p for p in paths if p.suffix == '.safetensors']
    # Reviewed converter holds shard mappings and a <=2GiB-plus-one-tensor
    # output buffer. Conservative allowance includes two largest input files,
    # output scratch and audio construction; reserve is independent.
    working = 2 * max(p.stat().st_size for p in weights) + 12 * GIB
    available = int(dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())['MemAvailable'].split()[0]) * 1024
    if working > args.host_working_limit_gib * GIB or working + 24 * GIB > available:
        raise ValueError(f'conversion host admission failed: working={working}, available={available}, reserve=24GiB')
    if shutil.disk_usage(output.parent).free < sum(p.stat().st_size for p in weights) + 64 * GIB:
        raise ValueError('insufficient output disk plus 64GiB reserve')
    converter, version = load_converter(args.diffusers_source)
    plan = validate_raw(converter, root, shards)
    staging.mkdir()
    try:
        before = inventory(paths, root)
        event('conversion_start', working_bytes=working, available_bytes=available)
        converter.convert_transformer(str(root), str(staging / 'transformer'), converter.MINIMAX_H3_TRANSFORMER_CONFIG, 2 * GIB)
        converter.write_transformer_config(str(staging / 'transformer'), converter.MINIMAX_H3_TRANSFORMER_CONFIG, version)
        converter.convert_video_vae(str(root), str(staging / 'vae'), converter.MINIMAX_H3_VIDEO_VAE_CONFIG, version, 2 * GIB)
        converter.convert_audio_vae(str(root), str(staging / 'audio_vae'), version)
        converter.write_scheduler_configs(str(root), str(staging), version)
        gc.collect()
        coverage = validate_output(staging, plan['rope_inv_freq'])
        plan['audio_vae'] = {name: [name] for name in coverage['audio_vae']}
        if inventory(paths, root) != before:
            raise ValueError('native input changed during conversion')
        converted = inventory([p for p in staging.rglob('*') if p.is_file()], staging)
        for record in converted.values():
            # The final directory name changes on publication, content does not.
            record.pop('resolved')
        receipt = {'schema': 1, 'status': 'converted_validated_not_gpu_tested', 'converter_commit': COMMIT,
                   'converter_sha256': CONVERTER_HASH, 'license_sha256': LICENSE_HASH,
                   'installed_model_sha256': MODEL_HASHES, 'diffusers_version': version,
                   'native_files': before, 'converted_files': converted, 'key_plan': plan,
                   'actual_model_state': coverage, 'working_budget_bytes': working}
        (staging / 'conversion.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        # Flush component files/directories before publishing the fresh artifact.
        for path in staging.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        for path in sorted([staging, *(p for p in staging.rglob('*') if p.is_dir())], key=lambda p: len(p.parts), reverse=True):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if output.exists():
            raise FileExistsError('output appeared during conversion; refusing replacement')
        staging.rename(output)
        fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        event('artifact_published', output=str(output), gpu_test=False)
    except BaseException as error:
        # Retain all partials for inspection; never retry/overwrite automatically.
        if staging.exists():
            (staging / 'FAILED.json').write_text(json.dumps({'type': type(error).__name__, 'error': str(error)}) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='native FL2VA directory')
    parser.add_argument('--blob-root', type=Path, required=True, help='explicit trusted HF blobs directory')
    parser.add_argument('--diffusers-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host-working-limit-gib', type=int, default=48)
    args = parser.parse_args()
    if args.host_working_limit_gib < 1:
        parser.error('host working limit must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
