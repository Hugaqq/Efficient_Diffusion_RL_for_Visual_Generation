# v0.8 modular real-GPU engineering bring-up

This directory records diagnostic runs on `10.130.140.73` before the final
source/wheel freeze. It is not release evidence and must not be counted as the
final A7/A8 gate in `Plan_8_2.md`.

## Evidence boundary

- The canonical schema-v2 recipes and component registries remain the owners
  of algorithm, model, dynamics, reward, and runtime behavior.
- Files in `configs/` only bind server-local artifacts, endpoints, output
  directories, and a bounded optimizer-step count.
- `evidence/reward_general_engineering_bringup.json` describes a source-tree
  deployment and is explicitly `engineering_bringup_not_release`. It is not a
  `world_r1_release_reward_marker`.
- Every one of the six routes must be rerun from the same frozen source/wheel
  and reward deployment identity before A7 can pass.

The post-bring-up local candidate currently hashes the complete `visual_rl`
Python tree as `56507f6ea6b24ed788306988bbecdaaabf3d7eeabf80fb21f46c6bd426aa375b`
(`python-code.v1`, 183 files, 2,648,703 bytes). This is a candidate, not the
final freeze: any subsequent runtime fix must change the recorded value. The
external memory sampler is outside runtime code identity and is separately
hashed as `3a8e2030678b4ca2aa606e635621f850467ad0c6f7e1e0ed657db78b9f19efd8`.
The exact candidate was mirrored without cache/build trees to
`/dev/shm/v-qiaoqifan/visualrl-v08-candidate-56507f6e-source`; the remote
Python-tree identity and sampler digest match the local values byte-for-byte.
This directory will not be modified in place.

The candidate also builds as a clean core wheel with SHA-256
`6f1533ef8e1d3ed471414ddea8ee956db199bd3b154500ed66a44f51b31bca61`.
Installation and the installed-surface smoke passed in an isolated Python 3.11
environment without Torch, Diffusers, PEFT, Accelerate, or Transformers. This
proves the candidate packaging boundary, but does not by itself promote the
diagnostic reward services or runs to final A7 evidence.
All 183 `visual_rl/**/*.py` members in that wheel match the candidate source
byte-for-byte; the sorted per-file SHA-256 record hashes to
`238a9d83bf9fe8f1accc4a6142f4766d7cd2ee53df45b7383c5baf61205335f8`.
The same wheel is installed without dependency changes in the tmpfs reward
service environment. Import-isolated probes resolve both `visual_rl` and
`services.world_r1_strict` from that environment's `site-packages`, not from a
source-tree working directory. Final general/3D Gunicorn origins are launched
from `/tmp` on ports 8093/8092; real health/score receipts and release markers
remain pending until both workers report ready.

The one-time frozen-wheel resource inventory is retained as
`evidence/final_resource_identities_6f1533ef.json` (SHA-256
`d694447aac5bed009bd0518e24ec46d1f9ccabdf71645ac59d80a8b3281de4c8`).
It records all-file identities for the real HPS checkpoint, Qwen3-VL model,
DA3 model, and LPIPS AlexNet checkpoint rather than relying on filenames or
file counts alone.

The frozen-wheel general origin is now release-bound: health is strict-v2 HTTP
200, the real HPS request returned finite score `0.23296520113945007`, and the
marker manifest SHA-256 is
`648cebfbbc4eabdf6003022ee208c3d6653d207e46571e96c5f38600f8cff123`.
The exact manifest is retained as
`evidence/reward_general_release_marker_6f1533ef.json` and at the final remote
artifact path. The frozen-wheel 3D origin likewise returned strict-v2 health
and real finite score `1.9351264214596995`; its release marker SHA-256 is
`fbceb8637bd068e29cbfc111bb38565b987901ea050839e67d2dcac43da60af0`.
Both exact manifests are retained under `evidence/` and at the final remote
artifact paths.

`evidence/a7_freeze_identity_56507f_6f1533ef.json` is the authoritative shared
identity record for all six final runs. It contains no pending fields and binds
the code tree, wheel, six config hashes, two reward markers, resource inventory,
memory sampler, and launcher. Its SHA-256 is
`a6c961fc4b2670f1df947d73015f1572e35840acb8fd9ad6bf734905af5b07f9`,
matching the remote release-candidate copy.

The six bounded server configs live in `configs/a7_candidate_56507f/` and were
mirrored to
`/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/configs/a7-candidate-56507f/`.
Their local and remote per-file hashes match
`evidence/a7_candidate_56507f_config_sha256.txt`. Five configs are semantically
identical to their canonical `configs/v2` recipes; the bounded release route
differs only in `training.max_optimizer_steps` (`150` to `20`).

The final server-only config candidates live in
`configs/a7_final_56507f_6f1533ef/` and are mirrored to the remote
`configs/a7-final-56507f-6f1533ef/` directory. Their six hashes are frozen in
`evidence/a7_final_56507f_6f1533ef_config_sha256.txt`. Relative to the bounded
candidate configs, they change only deployment paths: final reward-marker
locations and tmpfs output directories. The recipe, model, algorithm,
optimizer, rollout geometry, 20-step bound, and reward endpoints are unchanged.
Tmpfs output is required because a real committed checkpoint write to NFS
entered uninterruptible `folio_wait_bit_common`; completed receipts are copied
to NFS only after training has reached SUCCESS.

## Resolved composition evidence

The generated `recipe.resolved.json` files confirm that the runtime graph is
assembled from public axes rather than from a recipe-named trainer branch:

- Flow-GRPO × Wan resolves `algorithm=flow-grpo`, `model=wan-t2v`, and a
  model-bound `wan-flow-sde` projection with `profile=standard` and
  `stochastic_sampling=true`.
- World-R1 core still resolves `algorithm=flow-grpo` and `model=wan-t2v`; the
  integration layer adds `world-r1-camera`, two World-R1 reward components,
  and projects conditioned Wan dynamics with
  `post_hook_base_density_surrogate`, `conditioned_next`, and
  `stochastic_sampling=true`.

These are engineering composition receipts only. The local source has since
added the matching algorithm-side `stochastic` capability requirement, so the
final runs must use a new immutable source snapshot containing that static
fail-closed gate.

## Run acceptance evidence

For a 20-step route, the authoritative commit proof is the conjunction of:

- `SUCCESS` with `committed_steps=20` and the final checkpoint identity;
- `run_manifest.json` with `update_count=20`;
- `checkpoints/step-20/progress.json` with `next_optimizer_step=20`;
- `checkpoints/step-20/complete.json` and matching `latest.json`.

`metrics.jsonl` is intentionally a one-row terminal metric artifact in the
current terminal/inspection contract; it is not an optimizer-step history.
OOM evidence therefore comes from successful terminal convergence plus the
separate time-series GPU memory log. The completed Flow-GRPO × SD3 engineering
run already has all four commit receipts, a final finite-gradient metric row,
and a 15-second GPU sample trace with a 21,039 MiB driver peak.

Final runs use `sample_gpu_memory.sh GPU_INDEX TARGET_PID OUTPUT_CSV` as a
separate background evidence process. The sampler does not launch training or
enter the recipe/runtime identity; it records driver memory every 15 seconds
until the exact training-wrapper PID exits. Its CSV must be retained beside the
terminal receipts for each accepted A7 row.

`launch_a7_route.sh` is the server-only wrapper for those final runs. It refuses
to overwrite an existing route, starts exactly one frozen config with one
physical GPU, binds the sampler to the real trainer PID, and records stdout,
PID, memory CSV, and exit code under the tmpfs evidence root. Its SHA-256 is
`4c1b260babe9135d6ec23017358d210f511cef1675836bce556ea5539a00c5dd`;
like the sampler, it is external evidence tooling rather than runtime code.

For success-dependent queued routes, `capture_a7_launch_receipt.py` reads the
live trainer's procfs command, CWD, environment, parent launcher, PID, physical
GPU binding, config hash, and freeze record before atomically creating its
launch receipt. Its SHA-256 is
`86caf1329bf14d93e8472632e37bfe2d5897ad208b101db3882b8d4a19b3519d`;
a read-only capture of the live Flow-GRPO × SD3 process exactly reproduced its
independently created existing receipt.

After a route terminates, `audit_a7_route.py` performs the read-only acceptance
check. It invokes the canonical terminal `audit_run`, requires a fresh 20/20
manifest and step-20 progress, positive finite final gradient norms, the exact
frozen code/config/reward identities, an exit-zero trainer, a clean failure
signature scan, and one PID/GPU-bound memory series ending in a dead-target
sample. Its SHA-256 is
`13c325bb004024950b31f7794a7a56cb32e5975d9f5a1e0415658142b68577ff`.
Its nested checkpoint-progress parsing was also exercised against a synthetic
20-step terminal run before use on the frozen GPU matrix.
It is executed only after termination and cannot convert a partial/live run into
accepted evidence.

`audit_a7_matrix.py` then accepts only the exact six-route set, all at 20
commits, with one shared code/wheel/freeze identity, exact frozen config
coverage, identical local reward identity across the two SD3 routes, and the
frozen general/3D artifact identities across all Wan routes. Its SHA-256 is
`6387e04285e9ac69fd07146aa0aa9312bd03a45b9372d05c3afaa3b3aea991b1`.

`finalize_a7_acceptance.sh` applies those two auditors idempotently at a key
node: it skips live routes, atomically installs deterministic per-route receipts
for terminal routes, fails closed on a terminal rejection, and emits
`acceptance/matrix.json` only after all six receipts exist. Its SHA-256 is
`363a7813a6ba1225b6ae801b35e90f728a3c0daf0996724fd0ad84d743acd971`.

`watch_a7_terminal.sh` is the one-shot event-driven handoff for the live final
matrix. It waits on the six already-known supervisor PIDs with `tail --pid`,
never reads live step logs, captures both queued World-R1 launch receipts after
their upstream SD3 supervisors exit, invokes the idempotent finalizer only at
terminal events, and invokes archival only after matrix acceptance. A `flock`
prevents duplicate watchers. Its SHA-256 is
`f98073c4a4a89b5017b6521dee777b1c9c14fdc49f4ad0bb2c6d09c33cbb33eb`;
the active one-shot watcher PID is `1704242` and its startup state is
`waiting_for_pid_events`.

The first frozen release-surrogate process later received SIGTERM and exited
143 before producing a checkpoint. The recovery handoff
`recover_a7_world_release_after_core.sh` has SHA-256
`e582f5df5efd7fc4d5f1927e6a6969473fe3a4d3c8e032b50fcbd4889d5f9f57`
and active wrapper PID `2270592`. It waits for the core supervisor and original
watcher by PID event, requires an accepted core receipt, preserves the rejected
release run under checksummed NFS and tmpfs paths, checks the released GPU-3
baseline, and only then starts a fresh same-freeze release route. This is
external recovery/evidence tooling and does not change the frozen `visual_rl`
code or wheel identity.

The hash above identifies the copy deployed for the frozen A7 run. That copy
later exposed an infrastructure defect: it waited for `sync -f` on the NFS
failed-attempt staging directory before launching the retry, leaving the
already-released GPU idle in `folio_wait_bit_common`. The repository copy was
therefore hardened after deployment: it first relocates the rejected attempt
to its checksummed tmpfs namespace, launches the retry, archives that immutable
tmpfs evidence to NFS concurrently, and joins the archive before terminal
matrix acceptance. The hardened repository copy has SHA-256
`ff5d1b79478ad6b7733776743d273df4d139c2ab45ebe0432a63a4d4a010207b`;
it was not substituted into the active frozen A7 process.

Only after that matrix receipt exists, `archive_a7_acceptance.sh` may copy the
tmpfs runs, logs, launch receipts, acceptance receipts, tools, frozen configs,
and freeze records into an NFS staging directory. It verifies the copied trees,
writes `SHA256SUMS`, and atomically renames the staging directory to
`accepted/a7-final-56507f-6f1533ef`; it refuses to overwrite an existing
archive or archive a partial route. Its SHA-256 is
`25b8e9a8f63c5ba59f58bc2cbb3709be072a150832f99a850b948212fcbd7b82`.

## Final frozen A7 matrix

No row below may inherit a diagnostic result. A row becomes accepted only when
the source identity equals the one frozen identity shared by all six rows and
its reward identity, optimizer commits, terminal receipts, and memory trace are
recorded from that exact run.

| Route | Canonical config | Required reward identity | Frozen code identity | 20+ commits | SUCCESS/checkpoint/memory |
| --- | --- | --- | --- | --- | --- |
| Flow-GRPO × SD3.5 | `configs/v2/flow_grpo_sd3.yaml` | local `reward_quality` artifact identity | code `56507f6e…`, wheel `6f1533ef…` | **20/20 accepted** | **SUCCESS; peak 24,892 MiB; acceptance `bc5f5d85…`** |
| Flow-GRPO × Wan2.1 | `configs/v2/flow_grpo_wan.yaml` | general marker `648cebf…` | code `56507f6e…`, wheel `6f1533ef…` | **20/20 accepted** | **SUCCESS; peak 17,568 MiB; acceptance `2d35e86d…`** |
| TempFlow-GRPO × SD3.5 | `configs/v2/tempflow_sd3.yaml` | local `reward_quality` artifact identity | code `56507f6e…`, wheel `6f1533ef…` | **20/20 accepted** | **SUCCESS; peak 25,148 MiB; acceptance `bcba2ecb…`** |
| Flash-GRPO × Wan2.1 | `configs/v2/flash_wan.yaml` | general marker `648cebf…` | code `56507f6e…`, wheel `6f1533ef…` | **20/20 accepted** | **SUCCESS; peak 22,555 MiB; acceptance `35759b33…`** |
| World-R1 core × Wan2.1 | `configs/v2/world_r1_core_wan.yaml` | general `648cebf…` + 3D `fbceb863…` | code `56507f6e…`, wheel `6f1533ef…` | **20/20 accepted** | **SUCCESS; peak 18,664 MiB; acceptance `b11a0c48…`** |
| World-R1 release-surrogate × Wan2.1 | `configs/v2/world_r1_release_surrogate_wan.yaml` | general `648cebf…` + 3D `fbceb863…` | code `56507f6e…`, wheel `6f1533ef…` | retry queued | first final run externally terminated with 143; deployed recovery is blocked in pre-retry NFS `sync -f` while GPU 3 is free |

The two World-R1 rows use one-shot success-dependent queues rather than status
polling. World-R1 core started on GPU 3 only after Flow-GRPO × SD3 exited zero,
published SUCCESS, and passed route acceptance; its live procfs launch receipt
binds trainer PID `1708533` to the frozen source/config/GPU identity.
Release-surrogate likewise started on GPU 6 only after TempFlow-GRPO × SD3
passed acceptance; its live launch receipt binds trainer PID `1756567` to the
same frozen source/wheel and config `6f7feacc…`. A failed upstream route stops
its queue instead of launching more work on an unverified state.

The final Flow-GRPO × SD3.5 run is accepted from physical GPU 3. Its canonical
audit proves 20/20 fresh optimizer commits, exit zero and SUCCESS, one valid
step-20 checkpoint, final pre/post-clip gradient norm
`0.003381970804184675`, and 222 live 15-second memory samples with a 24,892 MiB
peak on a 32,607 MiB device. The acceptance receipt SHA-256 is
`bc5f5d858a68a5b1200fc55adf67b379ed195556f0bf9b03e9fcfb5aebc9bb8f`.

The final TempFlow-GRPO × SD3.5 run is accepted from physical GPU 6. It proves
20/20 fresh commits, exit zero/SUCCESS, final pre/post-clip gradient norm
`0.0005376166081987321`, and 522 live memory samples with a 25,148 MiB peak on
a 32,607 MiB device. Its acceptance SHA-256 is
`bcba2ecb043ca78aa05759d62d07ff20543663dd7379e1b994fe4ce5ce18477d`.
The final Flow-GRPO × Wan2.1 and Flash-GRPO × Wan2.1 runs are accepted. They
completed 20/20 fresh updates with respective driver-memory peaks of 17,568
MiB and 22,555 MiB and positive final pre-clip gradient norms of
`0.00047204949078150094` and `0.0005254881107248366`.

The final World-R1 core route is accepted from physical GPU 3. It proves 20/20
fresh commits, exit zero/SUCCESS, a valid step-20 checkpoint, positive final
pre/post-clip gradient norm `0.003636404639109969`, and 5,429 live memory
samples with an 18,664 MiB peak on a 32,607 MiB device. Its acceptance receipt
SHA-256 is
`b11a0c48a929a17aeac9f96888e0d68b129252fe2f33bdc0bd9767261722b6cf`.

The first final release-surrogate process received SIGTERM (exit 143) before
its first checkpoint; its log contains no OOM or traceback and it is rejected
as final evidence. The deployed event-driven recovery copy accepted the core
preconditions but is currently blocked in its pre-retry NFS `sync -f`, leaving
GPU 3 free. The hardened repository copy described above removes that NFS
operation from the retry launch critical path. Neither copy inspects live step
logs.

The obsolete pre-stochastic Flash bring-up and its old 8091 reward origin were
stopped after final reward markers became available. The final Flash-GRPO ×
Wan2.1 route is now running on GPU 2 from a 511 MiB memory baseline. The old
NFS-based 8091 service left one uninterruptible resource tracker and a reparented
zombie after its port/GPU manager exited; those processes are retained as
diagnostic cleanup evidence and do not own a listening socket or GPU allocation.

## Current diagnostic jobs

| Route/service | GPU | PID | State at launch | Remote evidence |
| --- | ---: | ---: | --- | --- |
| Flow-GRPO × SD3.5 | 6 | completed | 20/20, SUCCESS, no OOM | `runs/flow-grpo-sd3-attempt3-cpu-text` |
| TempFlow-GRPO × SD3.5 | 6 | completed | 20/20, SUCCESS | `runs/tempflow-sd3-attempt1-cpu-text` |
| Flow-GRPO × Wan2.1 attempt 1 | 3 | completed | optimizer recompute OOM at 30.58 GiB before the first commit | `runs/flow-wan-gpu3-attempt1` |
| Flow-GRPO × Wan2.1 attempt 2 | 3 | completed | row microbatch removed OOM; rejected before commit because all gradients were zero | `runs/flow-wan-gpu3-attempt2-row1` |
| Flow-GRPO × Wan2.1 attempt 4 | 3 | completed | no OOM; rejected before commit with 480/480 gradient tensors exactly zero despite four unique rewards and 112/112 nonzero active advantages | `runs/flow-wan-gpu3-attempt4-tmpfs-diagnostics` |
| Flow-GRPO × Wan2.1 attempt 5 | 7 | stopped | obsolete deterministic-Dynamics diagnostic stopped without deleting its partial artifacts after the root cause was established | `runs/flow-wan-gpu7-attempt5-fp32-diagnostics` |
| Flow-GRPO × Wan2.1 attempt 6 | 3 | completed | 1/1, SUCCESS; nonzero gradient norm `0.0006251836894080043`; no OOM/error match | `runs/flow-wan-gpu3-attempt6-stochastic-diagnostics` |
| Flash-GRPO × Wan2.1 attempt 1 | 2 | stopped | obsolete pre-stochastic source diagnostic stopped without deleting partial artifacts so GPU 2 could run the frozen final route | `runs/flash-wan-gpu2-attempt1-tmpfs` |
| World-R1 core × Wan2.1 attempt 3 | 7 | completed | 1/1, SUCCESS; nonzero gradient norm `0.004145184997469187`; current general + 3D rewards; no OOM/error match | `runs/world-r1-core-wan-gpu7-attempt3-stochastic-diagnostics` |
| World-R1 release-surrogate × Wan2.1 | 6 | completed | 1/1, exit 0 and SUCCESS with no OOM/error; NFS checkpoint staging spent about 40 minutes in `folio_wait_bit_common` before recovering, so it remains diagnostic-only | `runs/world-r1-release-surrogate-wan-stochastic-diagnostics` |
| general reward, old diagnostic revision | 7 | 1592326 | healthy, autocast score `0.231689453125` | `reward_receipts/general-2173e1e166a3-diagnostic` |
| general reward, FP32 diagnostic revision | 4 | 1656793 | healthy, FP32 score `0.23295444250106812` | `reward_receipts/general-8e46b1b63498-diagnostic` |
| 3D reward attempt 1/2 | 4 | completed | stopped before bind after the NFS service env entered uninterruptible RPC waits | `logs/reward-3d-2173-gpu4-port8092.log` |
| 3D reward, old diagnostic revision | 4 | completed | real DA3/GS/Qwen chain returned finite score `1.9351264214596995`; stopped cleanly after smoke | `reward_receipts/3d-2173e1e166a3-diagnostic` |
| 3D reward, current diagnostic revision | 4 | 1659059 | healthy on 8092; new-revision real smoke returned `1.9351264214596995` | `reward_receipts/3d-8e46b1b63498-diagnostic` |

Remote paths above are relative to
`/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808`.

## Real issues found so far

1. A model artifact on NFS can block before CUDA use. SD3 and Wan staging
   therefore use node-local SSD or `/dev/shm` copies.
2. Managing frozen text encoders as ordinary CUDA-resident components caused
   a real 30.8 GiB preprocess OOM. SD3 and Wan prompt encoders are now
   CPU-static; conditioning tensors cross to the active model device only at
   the typed model port.
3. `gsplat==1.5.3` needs the Ninja executable for its JIT CUDA extension.
   `ninja==1.11.1.4` is now frozen in the service requirements. Launching an
   isolated environment through an absolute Python path must also prepend its
   `bin` directory to `PATH`.
4. Wan rollout microbatching does not bound policy-recompute activation
   memory. Attempt 1 sampled with `forward_microbatch_size=1` but recomputed
   all four completion rows because `training.policy_recompute` retained its
   `row_microbatch_size=None` default. All four canonical Wan route configs
   now set row microbatch and transition window sizes to one.
5. The original 3D service environment itself can enter NFS D-state before
   Gunicorn writes its first log line. Attempt 3 uses a clean tmpfs venv built
   from the frozen service requirements plus tmpfs source/Qwen/DA3/LPIPS
   artifacts. It passed the strict runtime and gsplat SM 12.0 import gates and
   completed the real four-frame DA3 reconstruction, Gaussian Splatting render,
   and Qwen3-VL scoring chain.
6. Declaring a remote reward binding as `fp32` did not prevent the old HPS
   implementation from entering CUDA autocast internally. The old score was
   visibly FP16-quantized and could collapse within-group score differences.
   Revision `world-r1-8e46b1b63498` removes that autocast; its real smoke and
   checkpoint/health/score hashes are bound by the new receipt. A separate
   one-step Flow/Wan run must still establish whether this fixes the observed
   zero-gradient failure.
7. The default code artifact resolver originally rooted its identity at
   `visual_rl/composition`, so changes under model, algorithm, or runtime code
   could retain the same materialized recipe identity. It now roots at the
   complete `visual_rl` package, with a direct regression test. Runs created
   before this fix remain engineering diagnostics and cannot prove a frozen
   full-source identity.
8. Wan's model scheduler defaults to `stochastic_sampling=False`. The old
   Dynamics replay factory inherited that model setting, so standard and
   conditioned policy rollout stored `action == mean`; replaying the same
   action under the unchanged policy makes the score-function gradient exactly
   zero. Sampling mode is now an explicit Dynamics config/contract field,
   canonical Wan GRPO projections bind it to `true`, replay/factory identities
   include it, and runtime binding fails closed on mode drift. The focused
   local regression set passed 158 tests, including a detached sampled-action
   replay that produces finite nonzero prediction gradients.
9. A World-R1 release-surrogate diagnostic committed its optimizer step but
   then spent about 40 minutes in uninterruptible I/O while writing the 284 MB
   rank checkpoint into the NFS coordinator staging directory. It eventually
   recovered to exit 0 and SUCCESS, but final training outputs still use a
   dedicated `/dev/shm` root; terminal artifacts are archived to NFS only after
   SUCCESS instead of putting NFS on the optimizer commit path.
