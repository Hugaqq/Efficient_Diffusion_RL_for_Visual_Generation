# SD3 TempFlow 真实验证状态

更新时间：2026-07-12

## 当前结论

VisualRL v0.6 的 SD3 TempFlow 真实主链已经完成 GPU2 上的 5/20/50-step 分阶段验证：真实模型加载、branch rollout、reward、log-prob 重算、backward、optimizer 更新、LoRA-only checkpoint、绝对步数恢复和 held-out 配对评估均成功。infra 已经证明可以稳定训练 SD3.5 Medium 的 LoRA；50-step held-out 置信区间仍跨过 0，因此本轮没有足够证据声称模型获得稳定、可泛化的提升，也没有继续到 100 steps。

Git 状态：`main` 与 `origin/main` 同步在 `9f879cd`；本轮数据集、里程碑和 reward 修复位于 `codex/sd3-dataset-pilot`，远端已同步到 `63572e8`。

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

## 2026-07-12 GPU2 数据集里程碑实验

远端使用 GPU2、RTX 5090 32GB、SD3.5 Medium、BF16、LoRA rank 8 / alpha 16、256 x 256、3 个 diffusion steps、训练 seed 711。训练集为 36 条 GenEval RGB prompt，held-out 为 9 条互不重叠 prompt；held-out 使用 eval seeds 1701/1702/1703，共 27 个固定配对样本。实验目录：

`/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_geneval_rgb_5step_63572e8_20260712_gpu2`

- GPU2 启动前和实验结束后均为空闲；整个实验只使用普通用户权限，没有终止进程或访问其他工作区。
- 3-step full trajectory 与两个 branching transition 的 `max_abs_logprob_delta` 均为 `0.0`，same-seed fingerprints 一致。
- 5 steps：5 次更新的 reward std、finite/nonzero gradient 和 parameter update 全部通过；held-out mean delta `-0.000575`，高于 `-0.02` 淘汰线，允许进入 20 steps。
- 5→6 resume：从 step 5 恢复后只执行 1 次更新，`target_step = 6`、`steps_executed = 1`、`resume_loaded = true`，证明绝对步数恢复契约成立。
- 20 steps：从 checkpoint 5 只执行 15 次新增更新；held-out mean delta `+0.000459`，改善比例 `62.96%`，eval-seed cluster CI95 为 `[-0.000766, +0.002752]`。均值转正但区间仍跨 0，属于弱正向迹象。
- 50 steps：从 checkpoint 20 只执行 30 次新增更新；held-out mean delta `+0.000123`，改善比例 `48.15%`，eval-seed cluster CI95 为 `[-0.001286, +0.002093]`。CI95 下界不大于 0，`reward_trend` 与 `eligible_for_next_milestone` 均为 false，因此停止，不运行 100 steps。
- 50-step 训练期间梯度全部 finite/nonzero，`clipfrac = 0.0`，`logprob_delta_abs_max = 0.0`；LoRA 参数 L2 变化 `0.0936984`，约 466.9 万参数元素发生变化。
- 5/20/50-step pixel diversity guardrail 均通过，没有发现亮度、饱和度、动态范围或空间方差坍缩。
- checkpoint 只保存 PEFT adapter 与训练恢复状态，不含完整 `transformer_state.pt`；每个 checkpoint 约 55MB，而不是旧实现的约 4.7GB。

本轮还验证了硬门槛本身有用：最初 GPU1 的 5-step 尝试中，旧的 clipped color reward 把两个分支都裁成 `1.0`，导致 advantage 与梯度为零，训练在 optimizer step 前被阻止。`63572e8` 改为未裁剪的 target-vs-distractor color margin 后，GPU2 的每一步 reward std 与梯度均恢复为非零。

## 当前剩余问题

- 当前 reward 仍是便宜的 RGB color-margin 代理，只能验证 infra 的训练闭环和一个窄目标，不能代表通用文本图像对齐或感知质量。
- 50-step 只有一个独立训练 seed，held-out 也只有 3 个 eval-seed clusters；统计把握不足，不能把略正的均值解释为可靠提升。
- 进入 100 steps 的预设要求是：50-step held-out cluster CI95 下界大于 0、pixel diversity guardrail 通过、至少 3 个独立训练 seeds。本轮只满足第二项。
- 当前 SDE globals patch 和 transformer hook 适用于现有单进程串行 runner；共享同一 adapter 的并发采样仍需单独设计隔离。

## 修复与验收顺序

1. 保存采样真正使用的 canonical source state，并让 transformer 与 SDE 重算共享它。
2. branching 重算先收缩到 parent batch forward，再把 noise prediction 展开到各 branch。
3. sampling/recompute 使用同一 transition kernel，并记录 trajectory contract 版本与调度信息。
4. 增加 BF16 source-state、full trajectory、branching parent-batch、同 parent 一致性与梯度测试。
5. 本地测试、GPU1 full/branch parity 和 1-step backward 已完成。
6. GenEval RGB train/held-out、5/20/50-step 趋势实验、LoRA-only checkpoint 和绝对步数 resume 已完成。
7. 本轮 50-step 趋势未达到 CI95 下界大于 0，且独立训练 seed 数不足，因此 100-step 被硬门槛阻止。
8. 下一轮应先增加更有意义的 reward/数据任务，并运行至少 3 个独立训练 seeds；只有 held-out reward、参数更新、数值稳定性与质量 guardrail 同时成立，才允许继续 100 steps。

## 硬门槛

- 更新前 active transition：`max_abs(new_logprob - old_logprob) <= 1e-5`
- 更新前：`clipfrac == 0`
- gradient finite 且非零
- LoRA parameter delta 大于 0
- 不允许通过覆盖 old log-prob、detach 梯度或放宽阈值绕过检查
