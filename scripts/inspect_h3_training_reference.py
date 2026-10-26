"""Download ONLY small public API source, never weights; do not execute it."""
from pathlib import Path
import urllib.request
import json

root = Path(__file__).resolve().parents[2] / 'h3-training-integration/reference'
root.mkdir(parents=True, exist_ok=True)
files = {
    'torch-2.10.0-module.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/nn/modules/module.py',
    'torch-2.10.0-future.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/__future__.py',
    'diffusers-aaf57d03-peft.py': 'https://raw.githubusercontent.com/huggingface/diffusers/aaf57d03a3edfbd9de457d2872417a92f149a91d/src/diffusers/loaders/peft.py',
    'peft-0.18.1-config.py': 'https://raw.githubusercontent.com/huggingface/peft/v0.18.1/src/peft/tuners/lora/config.py',
    'peft-0.18.1-layer.py': 'https://raw.githubusercontent.com/huggingface/peft/v0.18.1/src/peft/tuners/lora/layer.py',
    'accelerate-1.14.0-accelerator.py': 'https://raw.githubusercontent.com/huggingface/accelerate/v1.14.0/src/accelerate/accelerator.py',
    'peft-0.18.1-pyproject.toml': 'https://raw.githubusercontent.com/huggingface/peft/v0.18.1/pyproject.toml',
}
for name, url in files.items():
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
        (root / name).write_bytes(data)
        print(json.dumps({'file': str(root / name), 'url': url, 'bytes': len(data)}))
    except Exception as error:
        print(json.dumps({'url': url, 'error': repr(error)}))
