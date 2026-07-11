# VisualRL 程序填空学习计划

## 使用方法

每次只做一题，最多 40 分钟：

1. 先读题目开头的数据流，不看正式实现。
2. 搜索该文件中的 `FILL_ME`。
3. 只修改当前练习文件。
4. 激活 `visual-rl` 环境后直接运行文件，直到出现 `PASS`。
5. 最后对照“正式实现”路径，写下三句话：输入是什么、输出是什么、错误在哪里被拦截。

练习文件不属于 `visual_rl` runtime，不会影响项目测试。

## 顺序

| 题号 | 主任务 | 时间 | 运行命令 | 正式实现 |
|---|---|---:|---|---|
| 01 | Dataclass 与 Manifest 一致性 | 25 分钟 | `python exercises/01_manifest.py` | `artifacts/manifest.py` |
| 02 | FeedbackProvider 输入输出 | 20 分钟 | `python exercises/02_feedback.py` | `feedback/base.py`, `provider.py` |
| 03 | OptimizerPlugin 完整 update | 35 分钟 | `python exercises/03_optimizer.py` | `optimizers/algorithm_plugin.py` |
| 04 | Artifact 逐样本写盘 | 30 分钟 | `python exercises/04_artifacts.py` | `artifacts/builder.py`, `manager.py` |
| 05 | Shared-prefix branching | 35 分钟 | `python exercises/05_branching.py` | `rollout/branching.py`, Tiny adapter |
| 06 | 完整 checkpoint / resume | 40 分钟 | `python exercises/06_checkpoint.py` | `artifacts/checkpoint.py` |
| 07 | ExperimentRunner 数据流 | 35 分钟 | `python exercises/07_runner.py` | `runner.py` |

## 每题验收问题

### 01 Manifest

- 为什么 `manifest.run_id` 与 `record.run_id` 必须一致？
- 为什么重复 `sample_id` 要在写盘前报错？

### 02 Feedback

- Provider 为什么接收整个 `RolloutBatch`？
- RewardBatch 为什么不再保存 `normalized_total`？

### 03 Optimizer

- 为什么 optimizer 的创建也属于 plugin，而不属于 Runner？
- `zero_grad -> backward -> step` 的顺序为什么不能换？
- 哪些 metrics 应由 plugin 返回，哪些由 Runner 添加？

### 04 Artifacts

- 为什么一个 batch 不能只写一条 sample record？
- 为什么 tensor 要 detach/cpu 后再序列化？

### 05 Branching

- shared prefix 与“多个独立完整采样”有什么可观察差异？
- step index 和 timestep value 为什么不是一个变量？

### 06 Checkpoint

- 只保存模型参数为什么不能得到等价 resume？
- config fingerprint 应允许哪些字段变化？

### 07 Runner

- Runner 为什么不包含 GRPO 公式？
- ArtifactManager 为什么在 optimizer update 之后调用？

## 建议节奏

```text
第 1 天：01 + 02
第 2 天：03
第 3 天：04 + 05
第 4 天：06
第 5 天：07 + 从头画一遍完整数据流
```

卡住 10 分钟时，先看类型签名和测试断言；仍然卡住再看正式实现中对应函数，不要直接复制整个文件。
