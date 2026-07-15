# E2 results: real SD3 checkpoint/resume equivalence

## Outcome

The mechanical resume path passed: the first process completed step 5 and wrote
a complete checkpoint; a new Python process loaded it, reported
`resume_loaded=true`, resumed from base step 5, executed exactly five additional
steps, reached absolute step 10, passed numerical/parameter/held-out/pixel gates,
and wrote a complete step-10 checkpoint.

The strict equivalence gate failed. The resumed final LoRA adapter did not match
the uninterrupted seed=201 reference byte-for-byte or tensor-for-tensor.

## Final comparison

| Metric | Continuous 10 | 5 + resume to 10 | Difference |
|---|---:|---:|---:|
| Adapter SHA256 | `087d9a6b...d47d8d564` | `0aa20f0a...9feb573b` | mismatch |
| Held-out reward delta mean | +0.001241061 | +0.001551237 | +0.000310176 |
| Step 5-9 max reward-mean delta | — | — | 0.006129429 |
| Step 5-9 max grad-norm delta | — | — | 0.007264318 |

The two adapters contain the same 382 tensor keys. Tensor difference was
`max_abs=0.000127196`, `L2=0.020282`, with 4,663,522 of 4,694,016 elements
different.

## Important diagnosis

The experiment cannot attribute all final drift to checkpoint restore. The new
split run already diverged slightly from the earlier uninterrupted reference
during steps 1-4, before any resume occurred; step 0 matched. Prompt order for
resumed steps 5-9 matched the uninterrupted run exactly, and the checkpoint did
contain optimizer, plugin, RNG, step, config fingerprint, and implementation
state.

Therefore the current evidence is:

- checkpoint serialization/load is complete enough to continue a valid run;
- dataset cursor/prompt order is restored correctly;
- exact independent-run reproducibility is not established on this BF16/CUDA
  runtime;
- because the continuous and split executions were already non-identical before
  resume, this design is insufficient to isolate checkpoint-induced drift from
  underlying GPU/runtime nondeterminism.

## Gate result and next action

E2 fails the pre-registered exact adapter, metric tolerance, and held-out
tolerance gates. E3 TempFlow parity is paused. The next experiment must be an E2b
determinism audit: rerun two independent branches from the exact same step-5
checkpoint, then compare them with deterministic algorithms/settings enabled and
disabled. Only after establishing the runtime reproducibility floor can a fair
resume tolerance be frozen.
