# Current Goal

VisualRL v0.6 simplified core 已完成。当前目标从“重构框架”切换为“验证框架”，第一优先级是修复并验证 SD3 TempFlow 的概率轨迹契约。

## 已落地

```text
Config -> RolloutEngine -> FeedbackProvider -> OptimizerPlugin -> ArtifactManager
```

- `ExperimentRunner` 是唯一训练循环。
- Flash-GRPO、TempFlow-GRPO、World-R1/Wan 共用 batch、plugin、checkpoint 和 artifact 约定。
- GenRL 只作为工程参考，不进入主干依赖。
- 本地完整测试不运行大模型和远程服务。

## 当前优先级

1. 按 [SD3 TempFlow 真实验证状态](SD3_TEMPFLOW_VALIDATION_STATUS.md) 修复 source-state 与 branching parent-batch parity。
2. 通过本地回归、GPU1 实模 parity 与 1-step backward 硬门槛。
3. 运行 Tiny 三算法对比，验证可比较性。
4. 只在数值门槛通过后执行 SD3/Wan bounded GPU 实验，不直接启动长训练。
5. 记录真实 reward 延迟、显存、失败率与 resume 行为。

## 非目标

- 暂不恢复 SD1.5、FLUX、QwenImage、Inferix。
- 不新增算法专用 Trainer。
- 不把 Tiny/Fake 测试描述成真实模型性能结论。
