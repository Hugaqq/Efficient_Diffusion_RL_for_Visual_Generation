# Flow-GRPO Pick-a-Pic SFW experiment

This experiment tests the existing SD3.5/Flow-GRPO training path with the HPS
v2.1 general reward on realistic prompts. It uses the public Python API only;
`run_with_api.py` is an experiment driver, not a package CLI or a second Runner.

The frozen prompt sources are:

- C20 diagnostic training: `data/prompts/pickapic_sfw_q100_train_v1.txt`
  (100 prompts, preserved unchanged with the C20 evidence);
- Q100 training: `data/prompts/pickapic_sfw_q100_train_v2.txt`
  (100 prompts);
- validation: `data/prompts/pickapic_sfw_heldout_eval_v2.txt` (64 prompts;
  originally named held-out, but now classified as validation because it was
  used for the C20 promotion decision);
- final test: `data/prompts/pickapic_sfw_final_test_v3.txt` (64 prompts,
  reserved until every multi-seed model and decision rule is frozen);
- source and split provenance: `data/prompts/pickapic_sfw_provenance_v2.json`
  and `data/prompts/pickapic_sfw_provenance_v3.json`.

The v2 selection enforces both conditioning budgets before Q100 starts: at
most 128 SD3/T5 tokens and at most 77 HPS/OpenCLIP tokens. This prevents the
reward model from silently evaluating only a prefix of a longer SD3 prompt.

The bounded C20 run is an engineering and learning-signal gate. It validates
that BCHW SD3 images reach HPS, rewards are finite and non-constant within a
group, gradients are finite/non-zero, twenty steps commit, and the final public
status/audit pass. It is not evidence of a quality improvement.

The original seed-17 Q100 is retained as failed evidence. It used one prompt
group of eight samples per optimizer step at learning rate `3e-4`; its held-out
HPS delta was `-0.20302` and its prompt win rate was zero. Do not resume it or
launch the frozen seed-29/43 copies.

`flow_pickapic_c20_stable_v2_seed17.yaml` is the replacement stability gate.
It keeps eight generated samples per optimizer step, but divides them into two
independent prompt groups of four so the update averages two prompt-specific
advantages. It also lowers the learning rate to `1e-4`. All other important
Flow objective and inference settings remain unchanged. This is a new run,
not a mutation or continuation of the failed Q100.

The replacement C20 completed from candidate `b75854029318` on 2026-08-01.
All 20 steps committed, `inspect_run()` and `audit_run()` passed, every step
retained two non-degenerate reward groups, and the maximum sampled frozen-
reference KL was `0.00346581`. The frozen paired validation then produced:

- base mean HPS: `0.2895917892`;
- trained mean HPS: `0.2915687561`;
- mean paired delta: `+0.0019769669`;
- prompt-cluster bootstrap 95% CI:
  `[+0.0003795385, +0.0036850214]`;
- prompt win rate: `56.25%`;
- pre-registered C20 acceptance: pass.

The machine-readable result is
[`evidence/flow_pickapic_c20_stable_v2_seed17.json`](evidence/flow_pickapic_c20_stable_v2_seed17.json).
The complete run, immutable score matrices, paired comparison, runtime logs,
GPU monitor, source archive and installed wheel are retained in the durable
archive named there. This is a bounded single-seed promotion result on a
validation set; it does not replace the three-seed Q100 requirement, the
untouched v3 final test, or an independent perceptual-quality measure.

Promotion beyond this C20 requires all of the following:

- twenty authoritative commits and passing status/audit;
- finite, non-zero within-group reward standard deviation and gradients;
- no late-window reward collapse or runaway frozen-reference KL;
- paired validation HPS point delta above zero using the frozen protocol below.

The C20 result is only a tuning/promotion decision. A final quality claim still
requires the pre-registered Q100 multi-seed gates and the untouched v3 final
test; those gates are not relaxed by this stability run.

The Q100 configurations are frozen before their first launch. Validation may
be read at pre-registered 20-step boundaries to stop an unstable run, but it
must not be used to keep inventing new configurations. A final claim requires
all selected seeds to finish and the v3 final test to be opened only after the
checkpoints and analysis rule are frozen. Training reward alone is not
sufficient.

The paired evaluation protocol is frozen as
`flow_pickapic_paired_hps_v1`:

- all 64 v2 validation prompts;
- inference seeds 1009 and 2027, for 128 paired observations;
- 20 diffusion steps, BF16, 512 px, guidance scale 4.5;
- identical prompt order, batch size 8, and noise seed for base/final;
- no optimizer, backward pass, or training-run mutation;
- primary statistic: mean final-minus-base HPS delta;
- uncertainty: 10,000-replicate prompt-cluster bootstrap, seed 729;
- a seed passes only when the 95% CI lower bound is above zero and more than
  half of validation prompts improve after averaging their two inference
  seeds.

This supports only the claim that validation HPS alignment improved. Because HPS
is also the training reward, it does not by itself establish general human
preference or visual-quality improvement. Saved paired images remain available
for an independent scorer or blinded review.

New candidates also record `update/gradient_norm_pre_clip` and
`update/gradient_norm_post_clip`. In this pure on-policy Flow configuration,
`approx_kl=0` and `clipfrac=0` are expected because rollout collection and the
single update use the same policy. Those fields therefore cannot be used as a
cross-step stability claim; the staged run must monitor gradient norms,
frozen-reference KL, validation HPS, and an independent quality score.

Expected allocation is exactly two RTX 5090 GPUs:

- one physical GPU for the SD3 trainer;
- one physical GPU for the HPS reward service.

Run one configuration after starting the local reward service:

```text
python experiments/flow_pickapic_20260801/run_with_api.py \
  experiments/flow_pickapic_20260801/configs/flow_pickapic_c20_seed17.yaml
```

Run the replacement stability configuration with:

```text
python experiments/flow_pickapic_20260801/run_with_api.py \
  experiments/flow_pickapic_20260801/configs/flow_pickapic_c20_stable_v2_seed17.yaml
```

Run the frozen read-only baseline evaluation before Q100:

```text
python experiments/flow_pickapic_20260801/evaluate_hps.py run \
  --config experiments/flow_pickapic_20260801/configs/flow_pickapic_q100_seed17.yaml \
  --prompts data/prompts/pickapic_sfw_heldout_eval_v2.txt \
  --output-dir experiments/flow_pickapic_20260801/evaluations/base
```

For a final checkpoint, add:

```text
--adapter-checkpoint <run>/checkpoint_000100/adapter
```

Then compare the two immutable score matrices:

```text
python experiments/flow_pickapic_20260801/evaluate_hps.py compare \
  --base-dir experiments/flow_pickapic_20260801/evaluations/base \
  --trained-dir experiments/flow_pickapic_20260801/evaluations/seed17_final \
  --output experiments/flow_pickapic_20260801/evaluations/seed17_comparison.json
```
