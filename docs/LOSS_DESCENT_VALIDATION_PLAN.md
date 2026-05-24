# Loss Descent Validation Probe

Purpose: verify that the VisualRL training plumbing can produce a clear,
deterministic loss decrease before spending GPU hours on SD3/FLUX/Wan RL
experiments.

## Why this probe is needed

The normal online GRPO smoke runs often log `loss: 0.0` or a non-monotonic
policy loss because the rollout is sampled from the current policy and
`old_log_probs` initially match `new_log_probs`. That is expected and should
not be used as the first correctness gate.

This probe instead uses a fixed teacher-student tiny diffusion task:

1. A teacher `tiny_diffusion` adapter with a known color-bias vector samples a
   fixed rollout batch.
2. A student `tiny_diffusion` adapter starts from zero bias.
3. The student optimizes a positive logprob-fit loss against the teacher
   rollout.
4. The probe asserts that loss and bias error both fall below strict ratios.

This validates:

- trainable adapter parameters
- rollout tensor shapes and strict `RolloutBatch` validation
- differentiable `recompute_log_probs()`
- optimizer steps
- JSONL metric logging
- checkpoint writing
- GRPO policy-loss diagnostics on the same fixed rollout

## Command

```bash
conda run -n visual-rl python -m visual_rl.cli tiny-loss-probe \
  --output-dir runs/tiny_loss_probe \
  --steps 100 \
  --learning-rate 0.1 \
  --batch-size 4 \
  --num-steps 4 \
  --image-size 8 \
  --seed 123
```

## Success criteria

- `loss_end < 0.1 * loss_start`
- `bias_error_end < 0.25 * bias_error_start`
- `grpo_policy_loss_end < grpo_policy_loss_start`
- `metrics.jsonl`, `summary.json`, and `checkpoint_final/tiny_diffusion.pt`
  are written.

This is the recommended immediate validation before any larger SD3.5 or Wan
run. After it passes locally, run the same command on the server CPU or a
single idle GPU to validate the remote environment.
