# Current Goal

Make VisualRL a small, readable integration repo for three diffusion-RL sources:
Flash-GRPO, TempFlow-GRPO, and World-R1.

## Mainline

`VisualRLTrainer` is the only active training loop. Keep the train path simple:

```text
prompts -> rollout -> rewards -> advantages -> logprobs -> loss -> optimizer
```

The project should stay on this single train-loop shape while the reference
code is folded in.

## Current Priorities

1. Keep tiny Flash and TempFlow tests as fast regression gates.
2. Move real Flash-GRPO selected-step Wan code into a proper adapter.
3. Keep SD3 TempFlow as the current image-model bridge.
4. Validate World-R1/Wan only through explicit checkpoint loading, sample/logprob
   tensors, and reward-server calls.

## Non-Goals

- Do not add another runner abstraction.
- Do not treat GenRL as a runtime dependency.
- Do not expand FLUX/QwenImage/SD1.5 work until Flash, TempFlow, and World-R1 are
  clean enough to explain and run.
