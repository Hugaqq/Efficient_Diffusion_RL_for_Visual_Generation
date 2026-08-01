# Flow-GRPO Pick-a-Pic SFW experiment

This experiment tests the existing SD3.5/Flow-GRPO training path with the HPS
v2.1 general reward on realistic prompts. It uses the public Python API only;
`run_with_api.py` is an experiment driver, not a package CLI or a second Runner.

The frozen prompt sources are:

- C20 diagnostic training: `data/prompts/pickapic_sfw_q100_train_v1.txt`
  (100 prompts, preserved unchanged with the C20 evidence);
- Q100 training: `data/prompts/pickapic_sfw_q100_train_v2.txt`
  (100 prompts);
- held-out evaluation: `data/prompts/pickapic_sfw_heldout_eval_v2.txt`
  (64 prompts);
- Q100/evaluation provenance: `data/prompts/pickapic_sfw_provenance_v2.json`.

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

Promotion beyond this C20 requires all of the following:

- twenty authoritative commits and passing status/audit;
- finite, non-zero within-group reward standard deviation and gradients;
- no late-window reward collapse or runaway frozen-reference KL;
- paired held-out HPS point delta above zero using the frozen protocol below.

The C20 result is only a tuning/promotion decision. A held-out quality claim
still requires the pre-registered Q100 multi-seed gates; those gates are not
relaxed by this stability run.

The Q100 configurations are frozen before their first launch. A final quality
claim additionally requires all three seeds and a paired step-0/final
evaluation on the held-out prompts. Training reward alone is not sufficient.

The paired evaluation protocol is frozen as
`flow_pickapic_paired_hps_v1`:

- all 64 v2 held-out prompts;
- inference seeds 1009 and 2027, for 128 paired observations;
- 20 diffusion steps, BF16, 512 px, guidance scale 4.5;
- identical prompt order, batch size 8, and noise seed for base/final;
- no optimizer, backward pass, or training-run mutation;
- primary statistic: mean final-minus-base HPS delta;
- uncertainty: 10,000-replicate prompt-cluster bootstrap, seed 729;
- a seed passes only when the 95% CI lower bound is above zero and more than
  half of held-out prompts improve after averaging their two inference seeds.

This supports only the claim that held-out HPS alignment improved. Because HPS
is also the training reward, it does not by itself establish general human
preference or visual-quality improvement. Saved paired images remain available
for an independent scorer or blinded review.

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
