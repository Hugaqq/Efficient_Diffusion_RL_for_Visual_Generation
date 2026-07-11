# SD3 TempFlow 真实验证状态

更新时间：2026-07-12

## 当前结论

VisualRL v0.6 的 SD3 TempFlow 真实主链已经通过一次有硬门槛的端到端验证：真实模型加载、branch rollout、reward、log-prob 重算、backward、optimizer 更新和 checkpoint 均成功。此前的概率轨迹阻断已经解除；这证明 infra 可以实际更新 SD3.5 Medium 的 LoRA，但一次 smoke 还不能证明模型已经学到稳定效果。

Git 状态：`main` 与 `origin/main` 同步在 `f5ddbfa`；SD3 修复位于 `codex/sd3-transition-contract`，远端已同步到 `d860ab4`。

## 已确认的实验现象

- SD3.5 Medium、LoRA rank 8 / alpha 16、256 x 256、单张 RTX 5090 32GB 的预览采样成功，显存约 14–16GB，没有 OOM。
- 1-step old/recomputed log-prob 数值检查完全一致，`max_abs_logprob_delta = 0.0`。
- 2-step 检查在两个 Conda 环境中都出现约 `0.073` 的偏差，超过硬门槛 `1e-5`。
- 失败发生在 backward 和 optimizer step 之前；本轮没有加载历史 LoRA checkpoint，因此不能归因于之前的模型更新。
- 原实验在硬门槛处停止，没有把异常 rollout 用于训练。

## 根因

TempFlow reference pipeline 在每次 SDE 后先保存 FP32 latent，再把实际继续参与下一步 transformer forward 的 latent 转回 BF16。adapter 之前保存的是舍入前 FP32 source state；重算时 transformer 看到了 BF16 视图，但 SDE mean 又使用了舍入前 FP32 source state。第二个 transition 开始，采样与重算不再描述同一条状态轨迹。

TempFlow branching 还有一个独立风险：采样先按 parent batch 做 transformer forward，再扩展分支；旧重算路径却直接按 branch rows 做 transformer forward。对 batch-size-sensitive kernel，这两条路径也可能产生数值差异。

修复动态 branch kernel 后还暴露出第三个边界：SDE branch target 必须保留 FP32 才能精确重算概率，但 TempFlow 的 child continuation 会把它直接送入 BF16 transformer。adapter 现在只在 transformer forward 边界临时把 `hidden_states` 转为模型 dtype，不改写保存的 FP32 target；作用域结束或异常时 hook 会自动移除。

## 2026-07-12 验证结果

本地：`160 passed`，Ruff、compileall 与 `git diff --check` 全部通过。

远端使用 GPU1、RTX 5090 32GB、SD3.5 Medium、BF16、LoRA rank 8 / alpha 16、256 x 256、seed 711，在独立目录运行：

`/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_transition_contract_dtypefix_d860ab4_20260712_gpu1`

- staged `sd3.py` SHA256 与本地一致：`579bf719...25ba5`。
- full 3-step old/new log-prob：最大绝对差 `0.0`。
- branching 的两个 active transition：各自最大绝对差均为 `0.0`，总体 `clipfrac = 0.0`。
- same-seed replay 的 initial latent、scheduler 与 trainable-parameter fingerprints 全部一致；数值 smoke 期间参数未变化。
- 1-step trainer：`valid = true`，`gradients_finite = true`，`grad_norm = 0.0609598`，非零梯度元素 `2,334,720`。
- optimizer 后参数最大绝对变化约 `1.0e-5`，L2 变化 `0.0152019`，非零变化元素 `2,334,720`。
- checkpoint、training state、metrics、manifest、reward table 和训练前后预览均已生成，remote helper 以状态码 0 结束。

这组证据已经覆盖“真实参数能否安全更新”，但没有覆盖“训练是否带来可泛化的质量提升”。同一 seed 的单张预览 reward 从 `0.672617` 到 `0.671179`，一次变化没有统计意义，也不是上升趋势。

## 当前剩余问题

- smoke 只使用一个 prompt 和颜色 reward，不能用于判断学习趋势、泛化或画质退化。
- 当前 checkpoint 同时保存 PEFT adapter 和完整 `transformer_state.pt`；单个 1-step stage 约 4.7GB。开始正式长跑前应改为可恢复的 LoRA-only checkpoint，或至少明确磁盘预算与保留策略。
- 实验结束时 GPU1 出现了另一个 `vggt-omega` 环境的进程。未查看或终止该进程，也没有继续提交 5/20/50-step 任务。
- 当前 SDE globals patch 和 transformer hook 适用于现有单进程串行 runner；共享同一 adapter 的并发采样仍需单独设计隔离。

## 修复与验收顺序

1. 保存采样真正使用的 canonical source state，并让 transformer 与 SDE 重算共享它。
2. branching 重算先收缩到 parent batch forward，再把 noise prediction 展开到各 branch。
3. sampling/recompute 使用同一 transition kernel，并记录 trajectory contract 版本与调度信息。
4. 增加 BF16 source-state、full trajectory、branching parent-batch、同 parent 一致性与梯度测试。
5. 本地测试、GPU1 full/branch parity 和 1-step backward 已完成。
6. 下一阶段先接入训练集与 held-out prompt，运行 5-step 管线检查，再做 20/50-step 趋势实验。
7. 只有 held-out reward、参数更新、数值稳定性与质量 guardrail 同时成立，才允许继续 100 steps。

## 硬门槛

- 更新前 active transition：`max_abs(new_logprob - old_logprob) <= 1e-5`
- 更新前：`clipfrac == 0`
- gradient finite 且非零
- LoRA parameter delta 大于 0
- 不允许通过覆盖 old log-prob、detach 梯度或放宽阈值绕过检查
