# APPROVED140 runtime implementation handoff

Implementation in this source snapshot only. **Not GPU validation or launch authorization.**
`BASELINE_SHA256.json` was not overwritten. Existing HPSv3 client/factory/service remain the scoring implementation; no reward replacement or second training controller was added.

## Entry and cadence

Candidate: `experiments/flash_wan_capacity_20261022/approved140-runtime.yaml`, derived from the existing real-entry-smoke-v3 resource identity. Main must verify all server paths, live resource identity, and capacity before use.

The new semantic `training.collection` uses:

```yaml
global_prompt_batch_size: 48
max_optimizer_steps: 140
gradient_accumulation_steps: 1
collection:
  collection_rows_per_commit: 192
  collection_commits: 2
  collection_chunk_rows: 16
flash_evaluation: true
```

Group8 => 384 logical video rows collected under one unchanged behavior policy. Prompt-keyed timestep selection is drawn once for the whole collection. Chunk16 sampling uses original forward4 and decode1; every real media chunk is immediately scored through the existing reward pipeline. Only actual compact transitions/conditioning/rewards remain, with media explicitly absent (not dummy pixels). Re-scoring released media is rejected.

All384 rewards go through the existing batch-std/group-mean normalizer and Flash credit planner **before** the nominal192 masks. `BaseTrainer` reuses the credit pool for the second commit without sampler/rollout/reward advancement. Both commits use `PolicyUpdateKernel.step_slots`, normalizing over the exact active cells in that commit. The Accelerate accumulation=1 guards remain intact: slots, not an implicit Accelerate counter, accumulate one logical commit. Nominal `sample_count` is192; actual active rows can be smaller after the existing credit mask.

70 collections / 140 commits / 26,880 training logical video IDs. The140 candidate refuses wrong384/192/group/forward/replay/decode geometry, wrong20-step/480x832x81/512/tiling configuration, missing evaluation, or wrong author training SHA.

## Parity, EMA and evaluation

- Each collection captures actual CPU trainable-parameter behavior values before collection. Both update parity passes replay against that snapshot in the existing TRAIN numerics context. Current-policy PPO ratios remain real and may differ from one on commit2; the safety thresholds are not silently disabled. Corrupting the behavior snapshot is a tested failure.
- CPU LoRA EMA is updated every **eighth committed optimizer update**, not every forward/slot. Decay follows author helper warmup `min((1+s)/(10+s), .9)` with zero-based committed-update index `s`. This is an explicit commit-based migration of decay.9/interval8 intent, not a claim to reproduce the author's repeated microbatch-level call placement.
- Real evaluation stage uses EMA via a temporary parameter swap (restored in `finally`),300 SHA-verified prompts,50 deterministic ODE steps through the existing model/Dynamics/decode ports, and the bound existing HPSv3 client. It runs before collections at committed counts0,36,76,116, with no additional final evaluation. All81 frames pass to the existing all-frame/top24 client contract in the production candidate. Evaluation does not fabricate policy transitions/log-probs.
- Evaluation global RNG is restored after the stage; training/sample IDs use explicit per-iteration generators. Per-video durable receipts are bound to recipe, data SHA, EMA hash and service identity.

## Recovery and accounting

Existing authoritative atomic checkpoints continue to own model/LoRA, optimizer, LR scheduler, global RNG and sampler state. The data-plane component now also owns CPU EMA, behavior values, pending real compact pool, full-group advantages/credit, commit cursor and completed evaluation cursor. Checkpoint EMA schema/flag explicitly identify this owner. Odd safe points restore the remaining old-policy half without scoring/generating any of its384 logical videos again. Checkpoint next-batch identity reflects the pending half rather than pretending a new sampler window is next.

During collection, durable `collection-cache/collection-{behavior_commit}/rows-*.pt` chunks retain real replay/reward/conditioning and generator-after state. A failure before the first commit can reuse completed chunks from the last authoritative policy/sampler checkpoint; behavior/recipe/row/selection identity mismatch fails closed. Chunks are removed after both commits, not after individual reward calls. Pending odd checkpoints are self-contained and do not require those cache files.

`physical-attempts/collection-*/rows-*.json` and evaluation `attempt-*.json` count starts separately from logical row IDs. A crash between generation/RPC and a durable result can require recomputation/re-scoring with the **same logical IDs**. This is not an exactly-once remote RPC guarantee. Started attempts are not asserted to be completed videos. Keep journals when resuming; do not remove them to force a mismatched cache through.

Use the last complete authoritative checkpoint on optimizer/post-commit failure; do not retry an uncertain optimizer transaction in the same live trainer. Media-bearing traceback frames are sanitized; controller teardown clears the pool and temporary parameter views restore on exceptions.

## Tests and remaining deployment gates

Allowed full-suite command:

```
cd outputs/flash140-runtime-workspace && /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest -q
```

`tests/test_flash_collection_runtime.py` uses explicit fake model/reward leaves with the real compiler/controller/stages/Dynamics/update kernel/checkpoint owners. It covers384->192+192, full384 standard deviation with chunk-varying scores, nonunit second-update ratios, exact continuous-vs-odd-checkpoint model/optimizer/RNG/sampler/EMA continuation, interrupted collection chunk reuse, parity rejection plus media release, interval8 EMA and exception restoration,300x50 ODE evaluation calls with interrupted receipt reuse, typed candidate compilation, and140 manifest quantities. These tests do not prove GPU accuracy, capacity, real HPS scoring or learning quality.

Main/reviewer must independently inspect and merge, then validate the full-size path on GPU. In particular measure peak GPU memory and host RAM (including compact-pool concatenation and checkpoint pickle copies), confirm >=16GiB GPU / >=24GiB host headroom, check disk capacity for compact chunks and checkpoints, verify real BF16 behavior parity and second-update gradients, and run real evaluation/resume. The candidate's server `test.txt` path is a deployment assumption, guarded by the exact approved SHA. No remote browser, service or GPU command was used by this implementation role.

No internal automatic pause policy was added; a main-managed safe-point pause must retain max_optimizer_steps140 and resume from a complete non-terminal checkpoint rather than treating max_steps2 as the approved endpoint.
