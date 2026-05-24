# Current Goal

Build VisualRL as a real, usable integration infra for diffusion RL visual
generation. GenRL remains an engineering reference only. The integration
targets remain World-R1, Flash-GRPO, TempFlow-GRPO, and Inferix.

## Mainline Goal

The current mainline is no longer more tiny-infra work. Tiny tests are complete
enough and should now be used only as regression gates.

The small-scale real-model SD3.5 loop now has guarded smoke evidence. The
current goal is to keep extending only from that proven path:

```text
SD3.5 checkpoint
  -> VisualRL SD3 adapter sample
  -> save preview PNGs
  -> reward router
  -> TempFlow/GRPO update
  -> LoRA checkpoint
  -> before/after preview
  -> metrics and blocker documentation
```

## Current SD3.5 Status

- Real SD3.5 CLI smoke passed on an explicitly idle remote GPU.
- `image-preview` saved generated PNG artifacts from the VisualRL adapter.
- Bounded SD3.5 `VisualRLTrainer` runs passed at 1, 5, resume, and guarded
  20-step smoke scale without NaN/OOM.
- Metrics include reward, logprob, KL/clip diagnostics, and parameter/checkpoint
  evidence.
- The 20-step run is short trend evidence only. It does not prove meaningful
  convergence or paper-scale training.
- Tiny loss/reward probes stay green as regression checks, but do not block
  real-model progress unless they fail after touching shared infra.

## Next Bounded Work

- Repeat or extend SD3.5 trend runs only with idle-GPU checks, staged code, fixed
  prompt PNG comparisons, reward/logprob/KL review, and checkpoint/parameter
  deltas.
- Keep FLUX, QwenImage, and SD1.5 real smokes blocked until valid checkpoint
  paths are available; current checkpoint inventory evidence found SD3.5 and
  Wan paths, not FLUX/QwenImage/SD1.5.
- Validate real Wan/World-R1 checkpoint loading and reward-server calls before
  any online video/world training claim. Local Wan planning and World-R1
  endpoint probe contracts are in place, but live sample/logprob tensors and
  successful reward-server calls remain unproven.
- Keep Inferix as dry-run preview/profiling planning until real checkpoint
  execution and logprob/recompute contracts are validated.

## Non-Goals For The Next Phase

- Do not add more tiny-only features unless they protect a real-model bug.
- Do not gate SD3.5 on SD1.5; SD1.5 remains optional until a checkpoint exists.
- Do not start Wan/World-R1 video training before the SD3.5 image loop can
  generate previews and complete a bounded training step.
- Do not run FLUX/QwenImage real smokes until checkpoint paths are available.
