# World-R1 strict_v2 + json_v1 fail-closed companion service

This directory contains the self-contained reward service used by the Flash
and World-R1 recipes. The fail-closed World-R1 managers and the required
Depth Anything 3 inference source are bundled under `native/`; deployment does
not clone, patch, import, or install a separate World-R1 repository. The
service is not part of the `visual_rl` public API, but its implementation,
configuration resources, licenses and frozen requirements file ship in the
same `visual-rl` wheel. It never talks pickle or forwards to legacy HTTP
servers.

The service exposes exactly two reward-kind origins:

- `reward_general` on `127.0.0.1:8090` (HPS general reward);
- `reward_3d` on `127.0.0.1:8089` (3D reward, fixed `scorer_type="qwen"`,
  `use_lpips=True`).

Both origins share `protocol.py` (exact decoder/encoder),
`reference_contract.py` (manager contract, runtime gate, native
fault-injection gate, lifecycle registry) and the single revision-grammar
owner `visual_rl.world_r1_protocol.validate_server_revision()`. Changing any
timeout, scorer or manager protocol requires changing the public
`server_revision`; training YAML cannot override any of it.

## 1. Create the isolated service environment

Create the CUDA 12.8 service environment, then install the same wheel used by
the trainer. No framecode checkout is required:

```bash
conda create -n world-r1-reward \
  -c nvidia -c conda-forge \
  python=3.10 pip cuda-nvcc=12.8
conda activate world-r1-reward
python -m pip install /absolute/path/to/visual_rl-0.7.0-py3-none-any.whl
SERVICE_ROOT="$(python -c 'from importlib.resources import files; print(files("services.world_r1_strict"))')"
python -m pip install -r "$SERVICE_ROOT/requirements-service.txt"
```

## 2. Prepare local model files

All four model paths must be absolute paths owned by the experiment
deployment. Final path components must not be symbolic links. The service
does not download model data at startup:

```bash
export WORLD_R1_HPS_CHECKPOINT=/absolute/path/to/HPS_v2.1_compressed.pt
export WORLD_R1_QWEN_MODEL=/absolute/path/to/Qwen3-VL-4B-Instruct
export WORLD_R1_DA3_MODEL=/absolute/path/to/DA3-GIANT
export WORLD_R1_LPIPS_ALEXNET_CHECKPOINT=/absolute/path/to/alexnet-owt-7be5be79.pth
```

The HPS worker also verifies and installs the bundled OpenAI CLIP tokenizer
vocabulary into the isolated `hpsv2` package before importing it. This repairs
the resource omitted by `hpsv2==1.2.0`; no network request is made.
The 3D manager verifies the AlexNet file is exactly 244,408,911 bytes with
SHA-256
`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
LPIPS is constructed without torchvision pretrained downloads and receives
that verified state dictionary explicitly.

To bound 32 GB service-GPU memory, each input video is uniformly sampled to at
most 16 frames before 3D reconstruction and to at most 8 frames for Qwen
scoring. These bounds are part of the bundled service revision. Per-request
GS video/image/JSON debug files are disabled by default; set
`WORLD_R1_SAVE_DEBUG_ARTIFACTS=1` only for a bounded diagnostic run.

Prove the frozen toolchain before any manager import or model load:

```bash
python -c 'import torch, torchvision; from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; assert torch.__version__ == "2.7.1+cu128"; assert torchvision.__version__ == "0.22.1+cu128"; assert torch.version.cuda == "12.8"'
```

Compile and import the pinned gsplat CUDA extension once, before starting
Gunicorn. The service worker repeats the import as a readiness gate, so a
missing extension produces `INIT_ERROR` and never exposes a healthy origin:

```bash
CUDA_HOME="$CONDA_PREFIX" \
CPATH="$CONDA_PREFIX/targets/x86_64-linux/include" \
CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include" \
LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib" \
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib" \
TORCH_EXTENSIONS_DIR=/absolute/path/to/torch_extensions/cu128 \
TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=4 \
python -c 'from gsplat.cuda._backend import _C; assert _C is not None and hasattr(_C, "CameraModelType")'
```

## 3. Start the two origins (Gunicorn only)

Gunicorn is the only supported WSGI server; there is no VisualRL launcher, no
argparse and no "equivalent WSGI server". Each origin runs exactly one worker
process (`gthread`, four threads). `--preload`, `--reload` and `workers>1` are forbidden,
because a forked worker must never inherit CUDA/queue state.
`build_app()` runs only after the worker fork: it executes
`require_service_runtime()`, the native fault-injection gate on the real
imported manager class, then constructs and initializes the manager exactly
once. The two commands are frozen token-for-token:

```bash
WORLD_R1_HPS_CHECKPOINT=/absolute/path/to/HPS_v2.1_compressed.pt \
WORLD_R1_SERVER_REVISION=world-r1-e156b02bc171 \
python -m gunicorn \
  --config python:services.world_r1_strict.gunicorn_conf \
  --bind 127.0.0.1:8090 \
  --workers 1 --worker-class gthread --threads 4 \
  --timeout 1860 --graceful-timeout 30 \
  --access-logfile - --error-logfile - --capture-output \
  'services.world_r1_strict.reward_general_app:build_app()'

WORLD_R1_QWEN_MODEL=/absolute/path/to/Qwen3-VL-4B-Instruct \
WORLD_R1_DA3_MODEL=/absolute/path/to/DA3-GIANT \
WORLD_R1_LPIPS_ALEXNET_CHECKPOINT=/absolute/path/to/alexnet-owt-7be5be79.pth \
CUDA_HOME=/absolute/path/to/world-r1-reward \
CPATH=/absolute/path/to/world-r1-reward/targets/x86_64-linux/include \
CPLUS_INCLUDE_PATH=/absolute/path/to/world-r1-reward/targets/x86_64-linux/include \
LIBRARY_PATH=/absolute/path/to/world-r1-reward/lib:/absolute/path/to/world-r1-reward/targets/x86_64-linux/lib \
LD_LIBRARY_PATH=/absolute/path/to/world-r1-reward/lib:/absolute/path/to/world-r1-reward/targets/x86_64-linux/lib \
TORCH_EXTENSIONS_DIR=/absolute/path/to/torch_extensions/cu128 \
TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=4 \
WORLD_R1_SERVER_REVISION=world-r1-e156b02bc171 \
python -m gunicorn \
  --config python:services.world_r1_strict.gunicorn_conf \
  --bind 127.0.0.1:8089 \
  --workers 1 --worker-class gthread --threads 4 \
  --timeout 1860 --graceful-timeout 30 \
  --access-logfile - --error-logfile - --capture-output \
  'services.world_r1_strict.reward_3d_app:build_app()'
```

`WORLD_R1_SERVER_REVISION` is required and must equal
`services.world_r1_strict.service_revision.BUNDLED_SERVICE_REVISION`; a stale
or operator-chosen value is rejected before model import. The full
implementation digest for `world-r1-e156b02bc171` is
`e156b02bc171895421de5a4ba9d74d14e2a225da70be7464e851f14edb585c0c`.
It is the SHA-256 of the sorted per-file SHA-256 records for the native Python
and asset files, the top-level service Python files, and
`visual_rl/world_r1_protocol.py`; the generated `service_revision.py` is the
only excluded file, avoiding a self-referential digest.

`GET /healthz` returns 503 until the manager is ready, and afterwards the exact
health body including
`manager_contract: world_r1_fail_closed_v1`.

Timeout invariant: manager deadline 1800 s < client `timeout_s` >= 1830 s <
Gunicorn/proxy 1860 s. Any remote (non-loopback) deployment must put a TLS
reverse proxy in front of each origin with read/send timeouts of at least
1860 seconds; HTTP is only acceptable on loopback.

A deployment smoke run must verify each origin owns exactly one
manager/worker-process tree, then send TERM to both Gunicorn masters and
confirm within a 45-second deadline that master, worker, manager and all
descendants are gone and 8090/8089 no longer listen.
The frozen Gunicorn config restores its Python termination handlers after
native imports, closes managers from the worker lifecycle hooks, makes the
master a Linux child subreaper, and arms each model worker with
`PR_SET_PDEATHSIG=SIGKILL`. This is required because native CUDA/geometry
libraries may otherwise intercept TERM and orphan a GPU-owning child.
