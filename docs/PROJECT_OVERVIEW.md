# VisualRL v0.7 项目概览

更新日期：2026-07-29

VisualRL 的目标是为 image/video diffusion reinforcement learning 提供一条
容易阅读、可恢复、可审计的研究训练主线。v0.7 优先保证训练数学和数据合同唯一，
再以真实实验验证正确性、质量与效率。

## 唯一公开路径

```text
完整 YAML
→ vr.load()
→ resolve()
→ validate()
→ run()
→ RunResult
→ inspect_run() / audit_run()
```

公开配置只有 YAML；Python API 只加载、验证和执行它。项目不提供训练 console
command、preset/profile/recipe 合并、外部插件注册或 public Runner constructor。
单进程和 DDP 共用同一个 `_execute_step()`、UpdateEngine 和 artifact commit
lifecycle。

完整使用方式见 [v0.7 用户指南](V0_7_USER_GUIDE.md) 和仓库
[README](../README.md)。当前实验判定见
[v0.7 验收矩阵](V0_7_ACCEPTANCE.md)。

## 共享训练合同

```text
PromptDataset
→ StepContext
→ RolloutRequest / RolloutBatch
→ RewardBatch
→ AdvantageResult
→ PolicyLossInputs
→ PolicyRecomputeStats
→ ObjectiveOutput / UpdateResult
→ StepMetrics / StepArtifacts / StepResult
→ authoritative commit marker
```

关键约束：

- 同一个 `StepContext` 穿过 rollout、reward、update 和 record；
- ManifestBuilder 是 `SampleRecord` 的唯一 producer；
- GRPO、Flash-GRPO、TempFlow-GRPO 共用唯一 clipped-surrogate objective；
- reference KL 只在 update 时由 current/reference policy statistics 计算；
- DDP 只注入 shard、collective、gradient sync、failure consensus 和
  rank-zero commit 差异；
- authoritative marker 决定 committed step，manifest/metrics 是可重建投影。

## v0.7 支持范围

| 方向 | 组合 | 当前源码状态 | 真实实验状态 |
|---|---|---|---|
| Flow-GRPO | SD3.5 + full trajectory | 已接入统一 objective | `not_run` |
| TempFlow-GRPO | SD3.5 + branching | 已接入统一 objective | `not_run` |
| Flash-GRPO | Wan2.1 + single step | 已接入统一 objective | `not_run` |
| World-R1 | Wan2.1 + strict reward service | 已接入统一主线 | `not_run` |
| Tiny | CPU single/Gloo contract | 本地测试使用 | 不代表真实模型 |

真实 C20/Q100、Flow native CUDA parity、MG1/NCCL、远端执行、上传和最终 wheel
安装尚未运行。源码准备、synthetic fixture 或 Gloo 测试不能替代这些结论。

完整范围与明确排除项见 [V0_7_SCOPE.md](V0_7_SCOPE.md)。

## 代码阅读顺序

1. `visual_rl/api.py`：唯一 public experiment handle。
2. `visual_rl/configs/schema.py` 与 `resolver.py`：完整 YAML 合同。
3. `visual_rl/preflight.py`：环境与 topology 验证。
4. `visual_rl/runtime_factory.py`：唯一 component construction。
5. `visual_rl/runner.py`：共享 step 与 commit lifecycle。
6. `visual_rl/optimizers/objective.py` 和 `clipped_surrogate.py`：唯一策略损失。
7. `visual_rl/artifacts/manager.py`：marker、projection 和 recovery。

内置组件清单只由 `visual_rl/builtins.py` 维护。新增内部组件必须实现已有 ABC、
加入该静态清单，并使用相同 typed contracts；不能创建第二个 registry、factory
或训练循环。

## 当前执行顺序

W06 已准备固定实验 source/config/controller，正式导航位于
[experiments/v0_7/README.md](../experiments/v0_7/README.md)。后续必须按
[EXPERIMENT_PLAN.md](../experiments/EXPERIMENT_PLAN.md) 执行：

1. 每个算法先完成 C20 continuous 与 interrupted/fresh-resume parity；
2. Flow-GRPO 额外通过 14-item native parity；
3. 对应 correctness gate 通过后才能运行 Q100 三 seed；
4. MG1 在真实双 GPU/NCCL 上验证内部 node 和 Tiny C20；
5. W07 才构建、检查并在干净环境安装最终 wheel；
6. 远端执行和 evidence 上传需要单独授权。

任何 `not_run`、缺设备或缺依赖都必须保持未完成状态，不能通过 skip、fake
result 或历史证据晋级。

## 维护原则

- 只维护一套训练流程和一套数据合同；
- 删除旧入口，不保留兼容 wrapper；
- 失败必须在对应 mutation 前同步暴露，marker 后失败保留已提交 head；
- 测试结论按范围陈述，Tiny/Gloo 不外推到真实 CUDA/NCCL；
- 质量结论必须在 evidence completeness 后使用预注册统计规则；
- W06 controller 只编排正式 API，不修补 production lifecycle。

版本变化见 [CHANGELOG](../CHANGELOG.md)。
