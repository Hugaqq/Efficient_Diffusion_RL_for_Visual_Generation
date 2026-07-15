# E2b results: runtime is not exactly reproducible

## Overall outcome

The corrected attempt completed all three-branch comparisons. Every branch was
execution-valid, restored the same step-5 checkpoint and RNG state, ran exactly
steps 5-9, used the same prompt/seed sequence, and reached step 10. However,
all three final LoRA hashes differed, and both training metrics and held-out
results exceeded the pre-registered `1e-6` tolerance.

The current PyTorch 2.11/CUDA 13 BF16 execution path is therefore not bitwise
or numerically exact across independent processes. Exact resume equivalence is
not established, and E3 remains paused. This result does not identify the
specific nondeterministic operator; that requires a separately frozen
deterministic-runtime experiment.

## Attempt 1: path-sensitive launch rejection

The first attempt did not execute any optimizer step. Both `repeat_gpu2` and
`repeat_gpu3` loaded the SD3 pipeline, then the checkpoint guard rejected
resume because the resolved training-semantics fingerprint did not match the
step-5 checkpoint:

- checkpoint fingerprint: `39c5f87d3047120019aa45511dd6b0dcf709db29e4a217dd0ad0fef3a76870a5`
- attempted-run fingerprint: `8145f1011c501af898e9e43138d507807e0b8e676fac942436a1a13f6820669f`

The copied E2b prompt files had the same SHA256 content identities as E2, but
their absolute paths changed. Dataset and evaluation paths participate in the
v1 fingerprint. A read-only reconstruction reproduced both hashes exactly.

This was a valid fail-fast under the current implementation, but it was not a
valid determinism run. It also exposed the path/content identity problem now
tracked separately as I0.

## Attempt 2: short-term path correction

Attempt 2 changed no infra code. It reused the original E2 training and
evaluation absolute paths while writing to new output directories. A preflight
reconstruction matched the checkpoint fingerprint exactly:
`39c5f87d...a76870a5`.

Branch mapping:

- A: existing E2 GPU2 resume-to-10 reference;
- B: new independent GPU2 resume from the same step-5 checkpoint;
- C: new independent GPU3 resume from the same step-5 checkpoint.

All branches reported `valid=true`, five metrics rows, `resume_loaded=true`,
base step 5, target step 10, the same config fingerprint, and complete step-10
artifacts.

### Final branch values

| Branch | Final adapter SHA256 | Held-out mean delta | Segment parameter L2 |
|---|---|---:|---:|
| A, existing GPU2 | `0aa20f0a...9feb573b` | +0.001551237 | 0.027440416 |
| B, attempt-2 GPU2 | `a2cd2708...a3d6bad` | +0.001093267 | 0.027509067 |
| C, attempt-2 GPU3 | `f9eb763a...b5a344d` | +0.002206218 | 0.027480540 |

### Pairwise strict comparison

| Pair | Prompt/seed sequence | Adapter max abs | Adapter L2 | Max metric delta | Held-out mean difference |
|---|---|---:|---:|---:|---:|
| A vs B, GPU2 vs GPU2 | exact | 0.000051165 | 0.012349 | 0.016186 | 0.000458 |
| A vs C, GPU2 vs GPU3 | exact | 0.000051000 | 0.009741 | 0.008009 | 0.000655 |
| B vs C, GPU2 vs GPU3 | exact | 0.000048060 | 0.010011 | 0.011257 | 0.001113 |

Each adapter had the same 382 tensor keys, but roughly 4.66 million of 4.69
million elements differed in every pair. The A-vs-B mismatch is especially
important because both were independent GPU2 runs: the issue is not explained
solely by choosing GPU2 versus GPU3.

The observed runtime had deterministic algorithms disabled,
`cudnn.deterministic=false`, `cudnn.allow_tf32=true`, and both GPUs were RTX
5090 cards using driver 580.159.03.

## Gate result and next action

Execution, resume metadata, prompt sequence, and artifact gates passed. Exact
adapter equality, metric tolerance, and held-out tolerance failed. Therefore:

- `eligible_for_exact_reproducibility_claim=false`;
- `e3_unlocked=false`;
- the E2 mechanical resume claim remains valid, but strict equivalence does not;
- the E2 drift cannot be attributed to checkpoint serialization alone.

Do not loosen the thresholds after observing these results. The next legitimate
step is to design and pre-register a deterministic runtime mode, then rerun the
same-checkpoint audit under a new experiment ID. No deterministic-runtime code
change was implemented in E2b.

## Evidence

The original `evidence/runs` directory contains attempt-1 failure logs. The
`evidence/attempt_2` directory contains both corrected run artifacts and the
full comparison JSON. `attempt_2_no_checkpoints.tar.gz` contains 327 files and
no checkpoint. No checkpoint was downloaded.
