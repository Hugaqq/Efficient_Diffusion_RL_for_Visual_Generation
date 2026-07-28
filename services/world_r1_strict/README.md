# World-R1 strict_v2 + json_v1 fail-closed companion service

This directory is the deployment adapter that serves the patched native
World-R1 reward managers over the frozen `strict_v2 + json_v1` protocol. It is
not part of the `visual_rl` public API, is never packaged into the training
wheel, and never talks pickle or forwards to the legacy HTTP servers.

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

Run from the framecode Git root:

```bash
conda env create -f envs/world-r1-reward-cu128.yml
conda activate world-r1-reward
python -m pip install -r services/world_r1_strict/requirements-service.txt
```

## 2. Prepare the patched World-R1 checkout

The only supported native patch is
`services/world_r1_strict/reference_patches/world_r1_fail_closed_v1.patch`. It
touches exactly `reward_server/general_reward.py`, `reward_server/reward_3d.py`
and `reward_server/reward_3d_backend.py` in your World-R1 checkout and turns
every swallowed `0.5`/`0.0` fallback into a structured fail-closed raise with
spawn workers, one 1800-second deadline and bounded cleanup.

```bash
cd /absolute/path/to/World-R1-main
git apply --check /absolute/path/to/framecode/services/world_r1_strict/reference_patches/world_r1_fail_closed_v1.patch
git apply /absolute/path/to/framecode/services/world_r1_strict/reference_patches/world_r1_fail_closed_v1.patch
cd /absolute/path/to/framecode
python -m pip install --no-deps -e /absolute/path/to/World-R1-main
python -m pip check
```

Then prove the frozen toolchain before any manager import or model load:

```bash
python -c 'import torch, torchvision; from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; assert torch.__version__ == "2.7.1+cu128"; assert torchvision.__version__ == "0.22.1+cu128"; assert torch.version.cuda == "12.8"'
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
WORLD_R1_SERVER_REVISION=world-r1-<patched-commit> \
python -m gunicorn \
  --chdir /absolute/path/to/framecode \
  --config python:services.world_r1_strict.gunicorn_conf \
  --bind 127.0.0.1:8090 \
  --workers 1 --worker-class gthread --threads 4 \
  --timeout 1860 --graceful-timeout 30 \
  --access-logfile - --error-logfile - --capture-output \
  'services.world_r1_strict.reward_general_app:build_app()'

WORLD_R1_SERVER_REVISION=world-r1-<patched-commit> \
python -m gunicorn \
  --chdir /absolute/path/to/framecode \
  --config python:services.world_r1_strict.gunicorn_conf \
  --bind 127.0.0.1:8089 \
  --workers 1 --worker-class gthread --threads 4 \
  --timeout 1860 --graceful-timeout 30 \
  --access-logfile - --error-logfile - --capture-output \
  'services.world_r1_strict.reward_3d_app:build_app()'
```

`WORLD_R1_SERVER_REVISION` is required, must satisfy the
`^world-r1-[0-9a-f]{12,40}$` public grammar, and identifies the patched native
commit you deployed. `GET /healthz` returns 503 until the manager is ready,
and afterwards the exact health body including `manager_contract:
world_r1_fail_closed_v1`.

Timeout invariant: manager deadline 1800 s < client `timeout_s` >= 1830 s <
Gunicorn/proxy 1860 s. Any remote (non-loopback) deployment must put a TLS
reverse proxy in front of each origin with read/send timeouts of at least
1860 seconds; HTTP is only acceptable on loopback.

A deployment smoke run must verify each origin owns exactly one
manager/worker-process tree, then send TERM to both Gunicorn masters and
confirm within a 45-second deadline that master, worker, manager and all
descendants are gone and 8090/8089 no longer listen.
