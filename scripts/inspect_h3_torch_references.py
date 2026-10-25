"""Fetch inert, pinned Diffusers H3 reference text; never execute it.

Run from the author repository. Downloaded text goes to the task reference
folder; attributed pure numerical test excerpts and LICENSE are emitted into
the candidate. No downloaded Python is executed by this audit fetcher.
"""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

DEST = Path(__file__).resolve().parents[2] / 'h3-pytorch-integration/reference'
BASE = 'https://api.github.com/repos/huggingface/diffusers'


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'VisualRL-H3-source-audit'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    receipt = DEST / 'diffusers-source.json'
    sha = 'aaf57d03a3edfbd9de457d2872417a92f149a91d'
    if receipt.exists() and json.loads(receipt.read_text())['commit'] != sha:
        raise RuntimeError('preserve mismatched reference receipt; do not silently repin')
    tree = json.loads(get(BASE + '/git/trees/' + sha + '?recursive=1'))
    paths = ['LICENSE', 'setup.py', 'src/diffusers/dependency_versions_table.py',
             'src/diffusers/hooks/group_offloading.py', 'src/diffusers/models/modeling_utils.py'] + [item['path'] for item in tree['tree'] if item['type'] == 'blob' and
        ('minimax_h3' in item['path']) and item['path'].endswith('.py')]
    files = {}
    for path in paths:
        raw = get('https://raw.githubusercontent.com/huggingface/diffusers/' + sha + '/' + path)
        target = DEST / 'diffusers' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[path] = hashlib.sha256(raw).hexdigest()
        print(path, len(raw), flush=True)
    receipt.write_text(json.dumps({'repo': 'https://github.com/huggingface/diffusers', 'commit': sha, 'files': files}, indent=2) + '\n')
    print('PIN', sha, flush=True)
    # Inert text extraction for independent CPU parity tests. Never execute the
    # downloaded modules. Only geometry and scheduler kernels are retained.
    candidate = Path(__file__).resolve().parents[1]
    license_path = candidate / 'docs/licenses/h3-diffusers-Apache-2.0.txt'
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_bytes((DEST / 'diffusers/LICENSE').read_bytes())
    before = ast.parse((DEST / 'diffusers/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py').read_text())
    wanted = {'patchify_video_latents', '_spatial_position_grid', '_temporal_position_grid', '_frame_position_grid'}
    nodes = [n for n in before.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    for cls in before.body:
        if isinstance(cls, ast.ClassDef):
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name in {'build_packed_sequence', 'build_row_timesteps'}:
                    method.decorator_list = []
                    nodes.append(method)
    scheduler = ast.parse((DEST / 'diffusers/src/diffusers/schedulers/scheduling_minimax_h3.py').read_text())
    for cls in scheduler.body:
        if isinstance(cls, ast.ClassDef) and cls.name == 'MiniMaxH3Scheduler':
            cls.bases = []
            cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                        {'__init__', 'set_timesteps', 'index_for_timestep', 'step'}]
            for method in cls.body:
                method.decorator_list = []
            nodes.append(cls)
    converter = ast.parse((DEST / 'diffusers/scripts/convert_minimax_h3_to_diffusers.py').read_text())
    for node in converter.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'reorder_interleaved_qkv', 'split_fused_qkv',
                'convert_transformer_key', 'convert_video_vae_key', '_rename_video_vae_key'}:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {'MINIMAX_H3_TRANSFORMER_DROPPED_KEYS', 'MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS'} for t in node.targets):
            nodes.append(node)
    prefix = '# Copyright 2026 The MiniMax and HuggingFace Teams. Apache-2.0.\n# Extracted from Diffusers ' + sha + '; see docs/licenses/h3-diffusers-Apache-2.0.txt\nfrom __future__ import annotations\nimport numpy as np\nimport torch\n_ROPE_FRAME_RESCALE = 5.0 / 3.0\n_ROPE_FRAMES_PER_LATENT = (1,4,4,4,4)\n_ROPE_SPATIAL_SCALE = 32\n\n'
    (candidate / 'tests/h3_pinned_reference.py').write_text(prefix + '\n\n'.join(ast.unparse(n) for n in nodes) + '\n')
    for path in ('src/transformers/models/qwen3_vl/modeling_qwen3_vl.py', 'setup.py'):
        raw = get('https://raw.githubusercontent.com/huggingface/transformers/v5.19.0/' + path)
        target = DEST / 'transformers-5.19.0' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        print('transformers5.19.0', path, len(raw), flush=True)
    revision = '42ed227ee7df40d41602854ae760620d6eb651fe'
    for path in ('model_index.json', 'text_encoder/config.json', 'transformer/config.json',
                 'video_vae/config.json', 'video_vae/source/config.json', 'audio_vae/config.json',
                 'audio_vae/metadata.json'):
        raw = get('https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/' + revision + '/FL2VA/' + path)
        target = DEST / 'official-configs' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        print('official', path, len(raw), flush=True)


if __name__ == '__main__':
    main()
