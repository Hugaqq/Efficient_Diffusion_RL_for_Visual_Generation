# VisualRL 当前计划

## 已完成：v0.6 Simplified Core

- 唯一公开协调器：`ExperimentRunner`。
- 统一 image/video `RolloutBatch` 与逐样本 `SampleManifest`。
- 可注册 `FeedbackProvider` 与 `OptimizerPlugin`。
- 公开注册入口覆盖 model、rollout、feedback、reward client、algorithm 和 optimizer plugin。
- 严格 config schema；旧 `trainer`、顶层 `output_dir` 和 v01 preset 已移除。
- 自动保存 config、prompt set、manifest、reward、metrics、report 和完整 checkpoint。
- Resume 恢复模型、optimizer、plugin、step、RNG 和配置指纹。
- Resume 会回滚 checkpoint 之后的 manifest、metrics 与 rollout cache；最终 step 总会保存 checkpoint。
- Full GRPO 自动形成 prompt group；singleton group 会报错。
- TempFlow 使用真实共享前缀 branching，并区分 step index 与 timestep value。
- SD3 branch 候选按真实 transition 数生成，scheduler timestep 保留原 dtype。
- simplified core 明确拒绝 `WORLD_SIZE > 1`，不假装支持分布式写盘。
- 旧 `rewards/algorithms/integrations/trainer/experiments` 运行时目录已合并或迁出。

2026-07-12 合并前本地验收：`136 passed`，Ruff、compileall 与 diff check 通过；此前重新构建的 v0.6 wheel 不包含已删除目录。架构、迁移、训练正确性三位审查者的阻断项均已修复并复核。

## 当前阻断项：SD3 TempFlow 概率轨迹契约

真实 SD3.5 验证已经确认 1-step parity 为 0，但 2-step old/recomputed log-prob 最大偏差约为 `0.073`。失败发生在 backward/optimizer 之前，与历史模型更新无关。根因、修复顺序和继续训练的硬门槛见 [docs/SD3_TEMPFLOW_VALIDATION_STATUS.md](docs/SD3_TEMPFLOW_VALIDATION_STATUS.md)。在该门槛修复并通过 GPU1 实模测试前，不启动 50/100-step 正式训练。

## 下一阶段：小规模真实验证

1. 跑 Controlled Tiny RL 三算法三 seed，检查 reward、KL、clipfrac 和失败样本。
2. 在单张 GPU 上做 SD3 TempFlow LoRA bounded run，验证真实 per-step pipeline 和 checkpoint round trip。
3. 独立验证 World-R1 两个 reward server 的输入、输出、失败率和耗时。
4. 在真实 Wan checkpoint 上做 1-step bounded run，再决定是否开展长训练。
5. 生成算法/reward 对比表和 visual report，形成项目实验章节。

## 学习方式

代码实现已经完成。接下来按 [docs/FILL_IN_LEARNING_PLAN.md](docs/FILL_IN_LEARNING_PLAN.md) 逐题练习；每题控制在 40 分钟内，先完成 `exercises/` 中的填空，再对照正式实现复盘。

`visual_rl/third_party/legacy.py` 仍被 SD3/Wan adapter 使用，在真实 adapter 完成原生迁移前保留。
