# E2b: SD3 runtime determinism audit

E2 proved that resume mechanically works but failed strict equivalence. Because
the independent split run had already drifted before resume, this follow-up
branches twice from the exact same step-5 checkpoint and RNG state.

Branch A is the completed GPU2 resume from E2. Branch B repeats it on GPU2 and
branch C repeats it on GPU3. No determinism setting is changed in this first
audit; the observed runtime flags are frozen in `recipe.json`.

If identical branches differ, the result establishes the natural drift of the
current BF16/CUDA runtime and blocks an exact resume claim. A later deterministic
mode must be pre-registered and tested separately; this experiment's thresholds
will not be changed after results are observed.

Attempt 2 completed with all three branches execution-valid and identical
prompt/seed sequences, but different final adapters and out-of-tolerance metrics
and held-out values. Exact reproducibility therefore failed; see `RESULTS.md`.

## Current status

Attempt 1 was rejected before training because relocated, content-identical
prompt files changed the v1 config fingerprint. Its logs and PID evidence remain
immutable.

The corrected follow-up is now frozen in `attempt_2_recipe.json`. It reuses the
checkpoint-bound E2 train and held-out absolute paths, changes only output,
resume, target-step, and save-interval fields, and writes to new GPU2/GPU3 run
names. Attempt 2 must run with the pre-fix code; the fingerprint v2 infra change
starts only after the CUDA/BF16 comparison is frozen.
