# H3 performance workflow rules

- Current authority: workflow-workspace/2026-11-03-h3-training-speed-001/TASK.md and the saved workflow definition. Read them first. This task supersedes archived Mac-only/SMALL40/resume commands; inputs/source-AGENTS-before.md is historical, not an execution mandate.
- VPS-only code work, workflow-managed worktrees, one independent optimization per cycle. Preserve original 3db2a05 source/run and the paused Euler-B task. No GPU execution, SSH, HPS requests or service changes in roles.
- Preserve model/data/geometry/steps/reward/precision/RNG/gradient and checkpoint semantics. No shortcut that changes the scientific question or removes safety checks.
- Original seven tests only as routine CPU gate: PATH=/home/user/projects/h3-visual-rl/.scratch/h3-media-tools:/usr/local/bin:/usr/bin:/bin VISUALRL_FLASH_AUTHOR_DATA=/home/user/projects/h3-visual-rl/.scratch/flash-grpo/source/dataset/video PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/user/projects/h3-visual-rl/.scratch/h3-cpu-venv/bin/python -m pytest -q tests/test_flash_collection_runtime.py. Keep seven functions; extend coverage within them. Do not skip or replace. CPU pass is not GPU performance proof.
- Coder/Simplifier commit independent logical changes only after tests pass. Include necessary tests in the same commit. No amend/rebase/squash/force push or branch changes. Record intent, SHA, checks and unverified claims in own task subdirectory.
- Do not modify .pi/workflows, .pi/auxiliary, .gitignore, task authorization, dependencies or other role reports. Preserve unknown changes and diagnose before retrying side effects.
- Final integration reruns the real gate. Publish only to the dedicated perf branch; GPU deployment and final two-update benchmark require a separate approved experiment card.
- Planner completion means scoped code review/implementation is finished, not hardware optimality or measured speedup. Budget exhaustion/needs_attention/waiting_external are not completion.
