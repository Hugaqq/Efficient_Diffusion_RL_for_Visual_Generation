# W5: deterministic real-Wan checkpoint/resume equivalence

W5 passed exact deterministic checkpoint/resume equivalence for both the
World-R1 and Flash-style paths on physical GPU2.

For each algorithm, a continuous two-step run was compared with an independent
one-step run followed by a fresh-process resume from checkpoint 1 to absolute
step 2. All six segments completed with finite nonzero gradients and the
correct absolute step/status. Deterministic runtime fixed Python hashing,
cuBLAS workspace behavior, TF32, cuDNN, and PyTorch deterministic algorithms
before CUDA work began.

For both algorithms, step-0 and step-1 metrics were exactly equal between the
continuous and split branches. The concatenated SampleManifest records were
semantically exact, and the split final trainable hash exactly matched the
resumed process's initial hash. Final PEFT safetensors SHA256 matched exactly:

- World-R1: `302a2ac817d308b042c59dd4e32cc49258eb6c76f111a5b51dc0442eb6bc69bb`
- Flash-style: `9ddaefcc0b08598a9c7698cfa9d886c3f593abd2b1fc10fd2fa3da0e7cb1bbb1`

The final optimizer, algorithm plugin, RNG state, step, implementation/runtime
identity, config/data fingerprints, PEFT artifact manifest, and all associated
hashes were recursively tensor-exact. No tolerance was used.

W5 establishes correct mechanical resume for this bounded real-Wan recipe. It
does not establish native Flash sampler parity, reference reward parity, or
multi-seed training effectiveness; W6-W8 remain separate gates.
