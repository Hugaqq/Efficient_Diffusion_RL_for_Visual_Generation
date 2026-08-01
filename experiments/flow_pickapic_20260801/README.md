# Flow-GRPO Pick-a-Pic SFW experiment

This experiment tests the existing SD3.5/Flow-GRPO training path with the HPS
v2.1 general reward on realistic prompts. It uses the public Python API only;
`run_with_api.py` is an experiment driver, not a package CLI or a second Runner.

The frozen prompt sources are:

- training: `data/prompts/pickapic_sfw_q100_train_v1.txt` (100 prompts);
- held-out evaluation: `data/prompts/pickapic_sfw_heldout_eval_v1.txt`
  (64 prompts);
- provenance: `data/prompts/pickapic_sfw_provenance_v1.json`.

The bounded C20 run is an engineering and learning-signal gate. It validates
that BCHW SD3 images reach HPS, rewards are finite and non-constant within a
group, gradients are finite/non-zero, twenty steps commit, and the final public
status/audit pass. It is not evidence of a quality improvement.

The Q100 configurations are frozen before their first launch. A final quality
claim additionally requires all three seeds and a paired step-0/final
evaluation on the held-out prompts. Training reward alone is not sufficient.

Expected allocation is exactly two RTX 5090 GPUs:

- one physical GPU for the SD3 trainer;
- one physical GPU for the HPS reward service.

Run one configuration after starting the local reward service:

```text
python experiments/flow_pickapic_20260801/run_with_api.py \
  experiments/flow_pickapic_20260801/configs/flow_pickapic_c20_seed17.yaml
```
