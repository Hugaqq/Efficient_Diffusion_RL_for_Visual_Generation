# SD3 TempFlow 真实验证状态

更新时间：2026-07-12

## 当前结论

VisualRL v0.6 的通用 runner、rollout、feedback、optimizer、artifact 与 checkpoint 主链已经形成；当前阻断项位于 SD3 TempFlow adapter 的概率轨迹契约，而不是训练框架无法启动，也不是历史 LoRA 更新污染了模型。

## 已确认的实验现象

- SD3.5 Medium、LoRA rank 8 / alpha 16、256 x 256、单张 RTX 5090 32GB 的预览采样成功，显存约 14–16GB，没有 OOM。
- 1-step old/recomputed log-prob 数值检查完全一致，`max_abs_logprob_delta = 0.0`。
- 2-step 检查在两个 Conda 环境中都出现约 `0.073` 的偏差，超过硬门槛 `1e-5`。
- 失败发生在 backward 和 optimizer step 之前；本轮没有加载历史 LoRA checkpoint，因此不能归因于之前的模型更新。
- 当前实验在硬门槛处停止，没有把异常 rollout 用于 1/5/20/50/100-step 训练。

## 根因

TempFlow reference pipeline 在每次 SDE 后先保存 FP32 latent，再把实际继续参与下一步 transformer forward 的 latent 转回 BF16。adapter 之前保存的是舍入前 FP32 source state；重算时 transformer 看到了 BF16 视图，但 SDE mean 又使用了舍入前 FP32 source state。第二个 transition 开始，采样与重算不再描述同一条状态轨迹。

TempFlow branching 还有一个独立风险：采样先按 parent batch 做 transformer forward，再扩展分支；旧重算路径却直接按 branch rows 做 transformer forward。对 batch-size-sensitive kernel，这两条路径也可能产生数值差异。

## 修复与验收顺序

1. 保存采样真正使用的 canonical source state，并让 transformer 与 SDE 重算共享它。
2. branching 重算先收缩到 parent batch forward，再把 noise prediction 展开到各 branch。
3. sampling/recompute 使用同一 transition kernel，并记录 trajectory contract 版本与调度信息。
4. 增加 BF16 source-state、full trajectory、branching parent-batch、同 parent 一致性与梯度测试。
5. 本地测试通过后，在 GPU1 依次运行 full/branch parity、1-step backward、5/20/50-step bounded run。
6. 只有 held-out reward、参数更新、数值稳定性与质量 guardrail 同时成立，才允许继续 100 steps。

## 硬门槛

- 更新前 active transition：`max_abs(new_logprob - old_logprob) <= 1e-5`
- 更新前：`clipfrac == 0`
- gradient finite 且非零
- LoRA parameter delta 大于 0
- 不允许通过覆盖 old log-prob、detach 梯度或放宽阈值绕过检查
