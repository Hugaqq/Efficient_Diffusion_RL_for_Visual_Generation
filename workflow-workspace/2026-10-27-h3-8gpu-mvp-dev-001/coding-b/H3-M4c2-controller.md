# H3-M4c2-controller — blocked on prepare owner failure boundary

implementation_ready=false
outcome=blocked
Proposed issue: C-PREPARE-02 (orchestrator to assign to original Coding C/Simplifier C).

## Actual checkout and scope

- Branch: `pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`.
- HEAD: `fcc5dc48700b7f7d900ab7849882aafa92d1c272`; initial status clean. No branch switch or manual Git writes.
- Read current task/plan/assignment B/wave-08 and Simplifier B session approval, current composition/controller/CLI/types, profile, binding, source, prepare/loader and actual related CPU fixtures.
- Only added `tests/test_h3_mvp_cpu_controller.py` and this report. No lifecycle/composition or other production modification; no stubs. Default H3 controller/CLI implementation remains unfinished.

## Reproduced prerequisite failure

W8 requires locally cooperative file/initialization errors to be exchanged before peers enter the next model collective. Existing `visual_rl/models/lifecycle/h3_fsdp.py::prepare_h3_fsdp` currently executes:

1. Common metadata exchange.
2. Only rank0 calls real `load_h3_cpu_state` and validates complete keys.
3. All peers immediately proceed to `_world_mesh`, AC/FSDP wrapping/materialization and collective `_set_full_state`.

There is no status exchange between 2 and 3. A rank0 finite-validation or IO failure returns directly. A lifecycle wrapper around the complete prepare call cannot insert a same-order gate between those internal operations: peers have already crossed the boundary before that wrapper can observe rank0's exception. Adding a catch/finally collective outside prepare would mismatch peers' state collective. Preflight byte hashing/headers do not establish finite payload values; duplicating the full owner load in the lifecycle also cannot handle later IO failure and violates the minimal single-owner loading design.

New test `test_h3_controller_prepare_owner_read_failure_stops_all_ranks_before_mesh`:

- Builds eight real meta/PEFT tiny instances and a real safetensors checkpoint with unchanged keys/shapes/dtypes, containing a nonfinite tensor.
- Executes actual prepare/API admission/metadata digest and actual owner loader/finite check across eight CPU threads.
- Replaces only WORLD/CUDA observations, bounded in-memory metadata transport, and mesh entry. Mesh entry is a sentinel to stop immediately, not a real PG, CUDA operation or timeout experiment.
- Actual outcome: rank0 raises `ValueError: nonfinite checkpoint tensor...`; ranks **1–7 all enter mesh**, triggering `PeerAdvanced` sentinels. The required `entered_mesh == []` assertion fails.
- This proves unsafe ordering in CPU control, not an observed real NCCL deadlock. The production peer path continues from mesh into the collective setter; actual GPU failure behavior remains untested.

The new test is intentionally an ordinary **failing regression**, not a passing assertion of broken behavior, skip or xfail. Do not count its addition as completed controller functionality or approve the red gate.

## Minimum owner repair request

Assignment explicitly keeps prepare/loader and existing tests read-only for B. Please return this dependency to original Coding C/Simplifier C:

- Keep rank0-only actual initialization read and full state coverage validation.
- Capture locally cooperative owner-load errors and perform one bounded allrank status exchange **after the read/coverage check and before mesh/wrapping/materialization/state setter**. Preserve the owner's original cause, reject all peers, release temporary state, no retry/rescue/finally collective after raw distributed failure.
- Add/extend C's own failure/order tests, retaining legal owner/peer paths and real loader checks. Coordinate any necessary test-only exception-chain assertions with B; do not hide this reproducer with skip/xfail or delete its safety assertion.
- Re-run Flash7 and the complete H3 glob; original Simplifier C must review the change. This does not authorize B to change C's production files.

After repair is accepted, return H3-M4c2-controller to B to implement the originally assigned production glue and full default-controller/CLI/fresh-resume tests. No claim that this is the only possible integration issue; this is the concrete reproduced blocker encountered first.

## Actual CPU gates after final test edit

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# exit0: 7 passed in 13.86s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# exit1: 1 failed, 146 passed, 22 warnings in 22.66s
```

The failure is the exact new prerequisite regression above; previous 146 tests passed. Existing DCP warnings remain visible. No full suite or warning suppression. `git diff --check` exit0; authorized historical diff of the entire Flash file against d818b8a is empty. No original tests/assertions were changed.

No GPU/real PG/HTTP/network/server/download/environment upgrade/push or manual commit. No SUCCESS/checkpoint/controller result generated. **MVP unfinished, GPU_PENDING.**
