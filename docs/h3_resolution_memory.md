# 480x320 two-update memory acceptance

Only `scripts.launch_h3_resolution_memory` admits this fresh max2 diagnostic.
It rejects resume, other dimensions, frames, fps, LoRA rank/alpha, author data
hashes, algorithms and layouts. The compiled profile supplies unchanged AdamW
lr1e-4/betas/precision/FSDP/activation-checkpointing and first5-of10 semantics.
Cadence remains5; lifecycle first paired-layout and terminal checkpoints save
steps1/2. Video observer every1 is an explicitly diagnostic-only exception;
four member0 videos per iteration are PRE-update rollouts, not evaluations.

## Identity and geometry

`H3Config.geometry` / `H3T2VAGeometry` admit480x320 without rounding or resize.
Text cache keys are prompt/tokenizer/template/encoder/layer/content identities,
not image geometry. Reuse is valid only with unchanged actual identity and a
full finite/cache-coverage scan during preflight; no cache identity rewriting.
`VideoContract(124,320,480)` requires exact JPEG dimensions. HPS must start from
this Git-deployed service code with those explicit dimensions and publish its
actual loaded identity. The HPS processor's existing model-native image
processing and all124/frame-microbatch1/top37 aggregation are UNCHANGED; there
is no compatibility resize to384x256. A separate endpoint/manifest is required.
The service fingerprints actual CPU tensors, processor and source again.

## Measurement

Start `scripts.monitor_h3_memory` BEFORE HPS/policy model loading. It runs one
sequential nvidia-smi GPU/process query pair per second, records query duration,
UTC/monotonic time, physical index/UUID, all card bytes/utilization, all compute
PIDs/bytes and `/proc` CPU RSS/rank/visible-device mappings. These are SAMPLED
maxima, not all instantaneous peaks, nor synchronized cross-device snapshots.
512MiB/24h bounds fail closed rather than silently truncating or rotating away
evidence. Its health receipt must have >=2 samples and age<15s at both iteration
boundaries; every rank also checks its allocator journal and CUDA UUID.

`H3_MEMORY_DIR` enables one exclusive JSONL per PID/rank (64MiB bound).
`H3_MEMORY_ROLE=hps` distinguishes the independent scoring process. PyTorch
allocated/reserved/high-water counters are PROCESS/DEVICE-local; they do not
include other PIDs or CUDA/NCCL allocations outside PyTorch. CPU RSS is never
reported as VRAM. Card-minus-allocator is not an exact NCCL measurement.

All resets in this execution path are owned by `runtime.h3_memory.Journal`.
Before EVERY reset, a peak snapshot and cumulative high-water are fsynced.
Nested spans partition execution into non-overlapping segments with ancestor
labels: the parent resumes with fresh counters after a nested child. Direct
phase maxima and ancestor-inclusive maxima may be reconstructed from events;
never subtract nested peaks or present their sum as memory. Raw maxima can be
independently reconstructed across every reset. The legacy decoder/HPS/reset
calls are suppressed only while this journal owns the counters. The old
one-update launcher reset is suppressed too if opted in.

Phases cover runtime init, full cache/graph preparation, model load, build,
each rollout prediction/transition, decode, HTTP reward, video save, anchor,
current differentiable forward, backward, clip, Adam step, zero_grad,
per-update weight/graph audits, checkpoint and terminal validation. Forward
and backward are separate. Each prediction release is logged after backward;
per-rank cumulative counts must be4 grad/44 no-grad per update, no retained
prediction, finite nonzero gradients, changed local LoRA and unchanged frozen
shards. Adam counters must equal1 then2. Audits are persisted before checkpoint
publication and unknown crashes MUST be reconciled, never blindly retried.

CUDA synchronization is used on the main thread at each rollout/anchor/current/
backward/clip/Adam/zero-grad segment boundary, without extra distributed
collectives. Existing rank-local VAE synchronization remains. Event rows report
whether synchronized and its wall seconds; these waits perturb execution and
must not be described as uninstrumented timings. Some wait time overlaps work
that would otherwise be awaited later; not all is incremental overhead.

No new empty_cache calls: existing VAE unused-cache release before decode,
after each video/audio model and in final cleanup; HPS offload-to-CPU cleanup
remain in place and now emit before/after events with reasons. No precision,
sharding, batch, frame or optimizer changes for memory cosmetics. OOM gets an
exception dump with allocation message, allocator stats/summary, phase events
and independent samples; fixed deterministic capacity OOM must not be retried
at reduced settings. Journal failure is fatal, including before update2.

On normal/exception exits journal writers flush. SIGKILL cannot guarantee final
flush, so every reset is durably recorded and card monitor is independently
supervised. CPU fake-counter assertions test logic only; the original seven
CPU tests are not a substitute for actual eight-rank GPU evidence.
