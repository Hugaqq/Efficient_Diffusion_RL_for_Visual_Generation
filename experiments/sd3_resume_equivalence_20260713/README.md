# E2: real SD3 checkpoint/resume equivalence

This experiment uses the E1 seed=201 active run as the uninterrupted 10-step
reference. A new process runs to step 5, exits, and a third process loads the
step-5 checkpoint and targets absolute step 10.

Exact safetensors equality is required for the final LoRA adapter. The optimizer
and RNG-containing `training_state.pt` may have serialization-level byte
differences, so it is compared semantically after safe loading. Metrics for
steps 5-9 and paired held-out results use the pre-registered numeric tolerance
in `recipe.json`.

This is an infra correctness experiment. It continues even though E1 did not
pass the reward-effectiveness gate.
