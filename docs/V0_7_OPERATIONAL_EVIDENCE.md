# VisualRL v0.7 operational C20 evidence

Updated: 2026-08-01

This document records bounded, real-model engineering evidence for the four
v0.7 recipes. It is intentionally separate from the clean-candidate release
envelope in [V0_7_ACCEPTANCE.md](V0_7_ACCEPTANCE.md).

The runs below prove only the stated operational properties. The later
clean-wheel Flow FP32 native-parity result is recorded separately and does not
upgrade these dirty-candidate C20 runs into final release evidence. Q100
reward-improvement evidence, MG1/NCCL evidence, and the final same-commit
release envelope remain incomplete.

## Candidate and environment

- Source HEAD: `cdd7dae05b46c24c178cde799963898473b8631d`
- Branch: `feature/minimal-readonly-callback`
- Candidate kind: dirty engineering worktree, not a release candidate
- Installed wheel SHA-256:
  `3d99c9767b0ce4016da84af8646afe9e33b6a288cf6281cc643e87fa7785d56b`
- Remote host: `zju-lab-rvpn`
- Training GPUs: NVIDIA GeForce RTX 5090, 32,607 MiB each
- Precision: BF16 operational runs
- Public verification surface: `visual_rl.inspect_run()` and
  `visual_rl.audit_run()`

All four C20 configurations enable gradient checkpointing and
`offload_frozen_modules_during_update`. The Wan configurations additionally
enable VAE tiling. Frozen text encoders and the VAE move to CPU before
policy recompute/backward; the trainable transformer/LoRA remains on the
training GPU.

## Flow native FP32 parity

The standalone Flow-GRPO native oracle passed from a clean committed source and
an isolated wheel installation:

- source commit: `67e7b1704c8fea732bba89f946d60913ea877b9b`;
- source archive SHA-256:
  `abcb604a268036ae2a61ea41a6e70dea8aad4d37b246da3aa5a4b6c9e95fd914`;
- installed wheel SHA-256:
  `2ca088fc955c717c551055e22077ac766879bcc6440debfb1ba976119418e29a`;
- frozen native reference commit:
  `63e4def7159940ba7d60e4e6250eee868342388c`;
- report SHA-256:
  `4b27527de8f5ccd0d4cbbe15bc0e77fd7cda1de5c39004ec6a983170fbbaae4c`;
- precision: FP32; result: 14 / 14 items passed;
- gradient tensors: 764 / 764 passed;
- parameter deltas: 382 / 382 passed, maximum absolute delta
  `6.811126240791054e-07`;
- checkpoint/resume invariants: 7 / 7 passed;
- peak GPU memory: 24,821 MiB; monitored duration: 719 seconds.

The durable canonical report is:

```text
/mnt/data/v-qiaoqifan/visual_rl_runs/flow_pickapic_20260801/
native_parity/67e7b17-clean-wheel/report.json
```

No `PYTHONPATH` or source-tree import was used for the formal run. This proves
the frozen one-shot tensor, gradient, parameter-update, and fresh-resume
contracts against the native reference. It does not prove BF16 long-horizon
quality improvement or the final all-gate candidate identity.

## Completed recipes

| Recipe | Continuous/resume | Public audit | Checkpoint tree SHA-256 | Semantic projection SHA-256 | Manifest / metrics | Peak GPU memory |
|---|---|---|---|---|---|---:|
| Flow-GRPO / SD3 | 20 / 20 | both pass | `ebb89e7fdc16e077c6b262aa659ec95744e108a8ff6aafe8e15578a6e6ca695f` | `c902c8b76be59b694ec43fc1db4d419de92043d69a40052f40b9b060410e9e2b` | 160 / 20 | 29,803 MiB |
| TempFlow-GRPO / SD3 | 20 / 20 | both pass | `a4c721052db9a2d2b65b01f50d58151344194844a293d2d0723a19cbcd15725a` | `806cfcb7cd46785b443eb6b9572e410aa70f1a8ac701aa9b6c1f20338e3dd8d9` | 120 / 20 | 24,073 MiB |
| Flash-GRPO / Wan | 20 / 20 | both pass | `603d478078821049eaff309726172e61f91e78e68dc4bdc8c298a5584798bff3` | `8252645558edaea58274cd9c9fd7f9644063fda6f636fe3e38841d39b6c58831` | 80 / 20 | 22,435 MiB |
| World-R1 / Wan | 20 / 20 | both pass | `16b0c8ddbd9579a1076828537631013225707fee64e5b03fd619ca40e343e1e9` | `86229d9b20431785e2380c2c584f6fe2574fff4649f67a31b6d990ae55821bee` | 40 / 20 | 30,735 MiB |

For every completed row:

- `inspect_run()` reports 20 committed steps, a resumable head checkpoint and
  zero pending transactions for both continuous and resumed output;
- `audit_run()` reports no audit error for either output;
- continuous and resumed outputs have the same head checkpoint tree digest;
- the full public semantic projection is equal after normalizing only
  `run_id`; the projection includes the complete manifest, all core metric
  rows, the checkpoint digest and the audit result;
- the saved peak is below the 32,607 MiB device capacity and the run completed
  without CUDA OOM.

The retained step-20 optimizer checkpoints also prove that these are real
updates rather than commit-only dry runs:

| Recipe | AdamW parameter states | Optimizer step range | Non-zero first-moment tensors |
|---|---:|---:|---:|
| Flow-GRPO | 382 | 20..20 | 380 / 382 |
| TempFlow-GRPO | 296 | 20..20 | 296 / 296 |
| Flash-GRPO | 480 | 20..20 | 480 / 480 |
| World-R1 | 480 | 20..20 | 480 / 480 |

These values were read with `torch.load(..., weights_only=True)` from the
authoritative step-20 `training_state.pt`. The public metrics additionally
report 20 rows and positive active-transition counts at every inspected step.
World-R1 also has 240 / 240 non-zero LoRA-B tensors. Its continuous and resumed
`training_state.pt` and `adapter_state.pt` files are byte-identical.

## Durable artifact locations

Flow-GRPO and Flash-GRPO:

```text
/mnt/data/v-qiaoqifan/visual_rl_runs/four_recipe_native_20260730/
```

TempFlow-GRPO, including controller logs and GPU monitors:

```text
/mnt/data/v-qiaoqifan/visual_rl_runs/four_recipe_closure_20260731/tempflow_final_v3/
```

The copied TempFlow tree contains 512 files and has the verified tree digest:

```text
4bd87c862758fa9023f14b1a6eaeb11dbc7a8885e5766058e0aa3418ab2274ee
```

World-R1, including both runs, the shared content-addressed reward cache,
controller logs, GPU monitors and wheel-only service evidence:

```text
/mnt/data/v-qiaoqifan/visual_rl_runs/four_recipe_closure_20260731/world_final_v4/
```

The World-R1 tree contains 209 files, 636,008,282 apparent bytes and has the
verified canonical tree digest:

```text
bb0c6e7ebc825c713181e6b9d14cc2a494f3251b0daf76283dfc52a530c123dc
```

The copied continuous run, resumed run and 160-file reward cache were each
checked byte-for-byte against their `/tmp` sources, and both copied run
directories passed the public audit again.

The `reward_general` origin imports the installed wheel directly. After both
World-R1 runs completed, the old `reward_3d` process group was terminated and
restarted from `/tmp` with no `PYTHONPATH`; its module origin was the installed
wheel under `site-packages`. A first diagnostic launch without the activated
venv `PATH` failed closed because `ninja` was not visible. Repeating the frozen
command with the reward environment's normal activated `PATH` passed health
and a real four-frame 3D request, returning the finite score
`1.9351264214596995` with `valid_mask=true`. Both the diagnostic failure and
the corrected service log are retained in the durable tree.

## Self-contained runtime boundary

The production package and fixed configurations contain no `reference_repo`,
`reference_code`, runtime `sys.path`/`sys.modules` injection, or dynamic import
from TempFlow-GRPO, Flash-GRPO, or World-R1 source checkouts.

The remote SD3.5 and Wan2.1 checkpoint roots used by these runs are real
directories under the experiment-owned `/mnt/data/v-qiaoqifan/.../checkpoints`
tree, not symbolic links.

The World-R1 reward service remains a separate process and environment by
design, but its required implementation is bundled under
`services/world_r1_strict/native/` with third-party license notices. It is a
versioned companion service, not a runtime link to an external source tree.

All four recipes continue through the same public API, runtime factory,
`ExperimentRunner._execute_step()`, `RolloutBatch`, `UpdateEngine`, shared
clipped-surrogate objective and authoritative commit lifecycle.

## Minimal-closure checklist

| Requirement | Evidence | Status |
|---|---|---|
| One training path and data contract | one `_execute_step()`, `RolloutBatch`, `UpdateEngine` and `PolicyObjective`; architecture tests pass | pass |
| No runtime reference-source dependency | production/source and 156-member wheel scans have zero forbidden hits; service/checkpoint roots checked | pass |
| Frozen-module CPU offload | 24 focused lifecycle/precision/failure tests plus saved sub-32-GB peaks | pass |
| Flow C20 and resume parity | 20-step public audit, matching checkpoint/projection digest, non-zero optimizer moments | pass |
| TempFlow C20 and resume parity | 20-step public audit, matching checkpoint/projection digest, non-zero optimizer moments | pass |
| Flash C20 and resume parity | 20-step public audit, matching checkpoint/projection digest, non-zero optimizer moments | pass |
| World-R1 C20 and resume parity | 20-step public audit, byte-identical checkpoint state and matching projection digest | pass |
| Wheel-only World-R1 service startup | non-source cwd, no `PYTHONPATH`, installed-wheel module origin, health plus real finite 3D score | pass |
| Current local package gate | 676 passed / 7 environment skips; Ruff `E4,E7,E9,F`, compileall, wheel contract, isolated install, `pip check`, outside-repository import | pass |

## Remaining evidence

The following remain incomplete and must not be inferred from the operational
C20 rows or the standalone native oracle:

- Q100 runs and multi-seed reward-improvement verdicts;
- MG1 real NCCL evidence;
- the final same-commit release evidence envelope.
