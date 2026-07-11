# VisualRL 项目计划

## 项目定位

构建面向 image/video generation 的统一 RL infra，把 TempFlow-GRPO 的优化思想、Flash-GRPO 的 selected-step/video 路径和 World-R1 的 world-aware feedback 放入同一套可复现、可扩展、可比较的实验框架。

项目解决的主要问题不是缺少算法，而是研究代码分散、数据格式不同、reward 与训练流程耦合、实验产物不统一，以及新方法接入成本高。

## 三项工作的工程角色

| 来源 | VisualRL 中的角色 |
|---|---|
| TempFlow-GRPO | branching rollout 与 timestep-aware policy optimization |
| Flash-GRPO | single-step/selected-timestep rollout 与 GRPO kernel |
| World-R1 | Wan video adapter 与 3D/world feedback client |
| GenRL | 工程参考，不是 runtime dependency |

## v0.6 已完成

### 统一接口

- `RolloutEngine`
- `FeedbackProvider`
- `OptimizerPlugin`
- `ExperimentRunner`
- `SampleRecord` / `SampleManifest`

### 单一目录主线

```text
runner
core + configs + datasets
model_adapters + rollout + feedback + optimizers
artifacts
```

旧 `trainer/rewards/algorithms/integrations/experiments` 运行时目录已经删除或迁移，不再存在第二套框架。

### 可复现产物

- resolved config
- prompt set 和 prompt metadata
- sample manifest
- reward table
- metrics JSONL
- visual report
- model + optimizer + plugin + RNG checkpoint

### 正确性边界

- GRPO 每个 prompt 至少两个样本。
- reward 只加权，advantage 独占归一化。
- TempFlow 必须是真实共享前缀分叉。
- branch step index 与 scheduler timestep value 分开。
- Resume 通过 config fingerprint 防止语义不匹配。
- Resume 同时校验实现身份与参数签名，并回滚 checkpoint 后未提交的 artifacts。
- v0.6 simplified core 只声明单进程正确性，显式拒绝多进程共同写盘。

## 第十阶段：Controlled Tiny 实验

目标：在 CPU/Tiny 模型上验证三种算法的可比较性，而不只验证能否运行。

实验矩阵：

```text
GRPO + full trajectory
Flash-GRPO + single step
TempFlow-GRPO + branching
x seeds 7, 11, 23
```

记录 reward、loss、KL、clipfrac、logprob delta、zero-std ratio、运行时间和 artifact 大小。

结束标准：三种配置都生成相同结构的 manifest/reward/metric/report；失败 case 可以按 `sample_id` 回溯。

## 第十一阶段：真实 Image bounded run

目标：验证 SD3 TempFlow LoRA 在单卡环境中完成最小真实闭环。

顺序：

1. checkpoint load 与单次 rollout。
2. per-step branching shape 和 prompt ordering。
3. current-policy logprob recomputation。
4. 1-step update 与完整 checkpoint。
5. 1+1 resume 对比连续 2-step。

不直接运行长训练。每一步先保存显存、耗时和失败日志。

## 第十二阶段：World-R1/Wan bounded run

目标：证明 video rollout、World-R1 reward 和 optimizer 能在同一 Runner 下闭环。

前置检查：

- Wan checkpoint 路径、dtype、device 和 transformer class。
- `reward_general` / `reward_3d` URL 与响应 shape。
- 视频帧数、分辨率、prompt/video ordering。
- `old_log_probs.shape == new_log_probs.shape`。
- checkpoint 保存后退出进程并能恢复。

结束标准：一个真实 1-step run 生成完整 artifacts；这之前只称为 bounded contract support，不宣称完整 video RL training。

## 第十三阶段：进一步接入其他工作

只有前三条主线的真实实验稳定后再开放。新工作必须映射到现有接口之一：

- 新模型 -> `ModelAdapter`
- 新采样方式 -> `RolloutEngine`
- 新 reward/VLM/human preference -> `FeedbackProvider` 或 reward client
- 新 RL 目标 -> `OptimizerPlugin`

禁止通过复制 Runner、另建 Trainer 或引入完整外部 runtime 树接入。

## 持续验证

```bash
conda run -n visual-rl python -m pytest -q
conda run -n visual-rl ruff check visual_rl scripts tests train.py
```

Tiny/Fake 测试证明接口和数值契约；真实性能、显存、吞吐和模型兼容性必须由后续 GPU 实验单独报告。
