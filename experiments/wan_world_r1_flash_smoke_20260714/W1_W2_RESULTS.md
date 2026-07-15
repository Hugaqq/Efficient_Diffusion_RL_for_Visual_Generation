# W1/W2: real Wan load and PEFT round-trip

W1 and W2 passed on physical GPU2 in the existing `flow_grpo` environment.
The model was the W0-verified 19-file Wan snapshot at revision
`0fad780a534b6463e45facd96134c9f345acfa5b`.

W1 loaded the real `WanPipeline` with a rank-16 LoRA configuration that
*declared* eight upstream target families, and generated one 5-frame 64×64
video over two denoising steps. The archived result records 480 trainable
tensors, but does not record their parameter-name families, so it is not valid
evidence that all eight configured families were actually materialized. The
post-merge W5 rerun therefore treats configured targets, effective parameter
families, tensor count, and parameter count as separate gates. All media,
trajectory, timestep, log-prob, and KL tensors were finite and shape-valid.
Peak CUDA allocated memory was 14,547,411,968 bytes
(13.55 GiB); peak reserved memory was 14,833,156,096 bytes (13.81 GiB), leaving
substantial room on the 32GB card.

The saved adapter contains 480 trainable tensors and 11,796,480 trainable
parameters. Its PEFT artifact is 23,659,220 bytes and contains
`adapter_model.safetensors`, adapter config/metadata, and a README; it contains
no full Wan transformer state.

W2 loaded that PEFT adapter in a fresh Python process. The trainable tensor
hash matched exactly, loading preserved all Parameter object identities, and a
same-prompt/same-seed rollout was bitwise identical for media, latents,
next-latents, timesteps, old log-probs, and KL (`max_abs=0` for every tensor).

W1 attempt 1 is retained as an environment failure: the model loaded, but
Diffusers prompt cleaning raised `NameError: ftfy is not defined`. The missing
runtime dependency was then pinned as `ftfy==6.3.1` (plus `wcwidth==0.8.2`) in
the user's existing Conda environment without root. No algorithm code or
model file was changed by this environment repair.

These experiments prove real model integration and PEFT save/load behavior.
They do not yet prove that backward/update succeeds; that is W3/W4.
