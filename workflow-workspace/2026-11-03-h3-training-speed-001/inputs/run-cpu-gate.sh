#!/usr/bin/env bash
set -euo pipefail
PATH=/home/user/projects/h3-visual-rl/.scratch/h3-media-tools:/usr/local/bin:/usr/bin:/bin VISUALRL_FLASH_AUTHOR_DATA=/home/user/projects/h3-visual-rl/.scratch/flash-grpo/source/dataset/video PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/user/projects/h3-visual-rl/.scratch/h3-cpu-venv/bin/python -m pytest -q tests/test_flash_collection_runtime.py
