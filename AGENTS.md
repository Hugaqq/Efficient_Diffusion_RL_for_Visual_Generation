# Development and experiment workflow

## Required workflow (user decision, 2026-10-22)

- The Mac checkout is the source of truth for code edits. Make all future source/config/test fixes locally, not on the server.
- For the current Flash-GRPO task, run only the seven core CPU tests in `tests/test_flash_collection_runtime.py` as the routine gate. Do not silently substitute a full768-test run or a different set of seven tests.
- Commit and push the tested changes to GitHub. The server must fetch/pull the agreed branch/commit from GitHub; do not patch server source, hotfix it in the terminal, or upload edited source as a substitute for Git deployment.
- The server is for GPU validation and experiments only. Runtime outputs, logs, checkpoints and environment setup are distinct from source edits. Use the authorized OpenCLI JumpServer terminal; no SSH. Do not disturb unrelated services/processes.
- Verify the local commit, GitHub commit, server checkout and actual imported source path before claiming matching code versions. Package version0.8.0 alone does not establish source equality.
- Do not overwrite unknown dirty server/local changes. Preserve and reconcile them explicitly before deploying.
- Progress inquiries do not cancel, pause or restart work. Only an explicit stop instruction or a real safety/failure condition warrants stopping.

## Current migration state

- Local branch `main` was reset by switching back to baseline `6b7f6d5`, not by deleting history. The candidate implementation remains on pushed `v0.9`, commit `5367716`.
- One-time reconciliation: six existing server source patches were copied back to this Mac checkout with per-file SHA256 verification. They cover HPSv3 integration, explicit component-param replacement, and stage-managed GPU text encoding. This is not a merge of the fullv0.9 candidate.
- The collection runtime, evaluation/checkpoint integration, candidate config and seven core tests have now been ported locally to main alongside the six reconciled server patches.
- Isolated core testing initially reproduced6 passed/1 RNG identity failure. The first Diffusers lazy import consumed global Python RNG only in the first of several in-process controllers. Tests now prime that import before controller seeding (same approach as the existing continuation test); exact RNG/checkpoint assertions are unchanged. Two fresh-process seven-test runs passed,13.16s and13.77s. This is CPU test evidence, not GPU validation.
- Accepted review simplification implemented in the isolated `outputs/flash140-coding` candidate: retain old_log_prob/normal PPO, remove per-update behavior prepass and persistent behavior weights; both initial gates now use only the normal absolute-step0 forward (not first process call). New code/config requires a fresh run and step0 validation, not a hot change to pending state.
- Collection `begin` computes a runtime-only optimizer-weight fingerprint for disk chunk identity; odd pending recovery does not need it. CPU EMA and its in-place/finally parameter exchange remain. Nested collection checkpoint schema2 rejects old schema1 candidates in owner and decoder; outer data-plane schema1 remains. No migration or dropped pending pools.
- Milestone A core tests in the actual candidate passed7/7 in9.96s, then7/7 in28.97s after evaluation/RNG/config assertions. The gate also verifies imported `visual_rl` and collection paths resolve inside this checkout; rerun it after changes before commit/push. CPU evidence only. Milestone B launch/readiness work and simplified full384/two192 GPU validation remain pending.
- Target remains140 optimizer updates,384-video fixed-policy collections followed by two nominal192-row updates;26,880 training videos. Do not silently substitute one192-video collection per update.
- A real8-video single-update GPU smoke passed. The formal140-update experiment has NOT started; do not describe unit tests or smoke updates as formal progress.

## Progress log

Maintain today's action record at `~/Desktop/ZIP Lab/Diffusion RL Infra/10.22-progress.md`. Action1 is fixing single-step/runtime problems, making only minimal necessary simplifications, and starting the Flash-GRPO140-update experiment.
