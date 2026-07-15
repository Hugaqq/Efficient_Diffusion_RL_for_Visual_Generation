# E1 results: SD3 multi-seed 10-step validation

## Outcome

The experiment completed all six pre-registered runs. All runs were execution
valid, every active run changed LoRA parameters, every zero-learning-rate
control left parameters and paired evaluation exactly unchanged, and all pixel
guardrails passed. The experiment did **not** pass the effectiveness gate.

## Cross-run result

| Metric | Result |
|---|---:|
| Active training seeds | 3 |
| Active seed means | 201: +0.001241; 307: +0.001561; 419: +0.000009 |
| Active aggregate mean | +0.000937 |
| Hierarchical CI95 | [-0.000317, +0.002383] |
| Positive training-seed fraction | 1.0 |
| Zero-LR aggregate mean / RMS | 0 / 0 |
| Active minus control CI95 | [-0.000317, +0.002383] |

Per-color aggregate means were blue -0.001021, green +0.001554, and red
+0.002278. Blue was negative in all three training seeds. The 10-step effect is
therefore small, statistically uncertain, and biased toward red rather than a
balanced RGB improvement.

## Gate result

Passed:

- all runs execution valid;
- at least three independent training seeds;
- positive-training-seed fraction threshold;
- active mean exceeds twice zero-LR evaluation noise RMS.

Failed:

- active hierarchical CI95 lower bound is not positive;
- active minus zero-LR CI95 lower bound is not positive;
- every-color mean positive (blue failed).

`eligible_for_effectiveness_claim` is therefore `false`. The 20/50/100-step
effectiveness scale-up remains locked.

## Visual and guardrail audit

All 432 before/after images were retained in the downloaded evidence. Reviewed
best/worst pairs remained recognizable and showed subtle composition/detail
changes rather than global saturation or under-denoised noise. Across active
runs, saturation ratios were 0.9919-0.9970, luminance ratios 0.9943-1.0050,
spatial-std ratios 1.0003-1.0087, and dynamic-range ratios 0.9959-1.0068.

This supports a narrow infra claim: optimization changes the intended LoRA
state, deterministic controls remain invariant, evaluation artifacts are paired,
and the anti-saturation guardrails work. It does not support a stable training
effectiveness claim.

## Next action

Proceed to E2 real checkpoint/resume equivalence, because resume correctness is
independent of whether this reward produces a useful effect. Do not lengthen the
effectiveness experiment until the blue-reward failure and scorer/task design are
understood.
