# VisualRL v0.6 架构图

## 结论

当前代码只有一套训练架构。`ExperimentRunner` 协调流程，差异能力通过三个接口表达：

```text
RolloutEngine -> FeedbackProvider -> OptimizerPlugin
                         |
                         +-> ArtifactManager
```

Manifest 是训练旁路产物，不进入 loss 或 backward。

## 运行时数据流

```text
VisualRLConfig
  |
  +-> PromptDataset.batch()
  |     prompts + prompt metadata
  |
  +-> RolloutEngine.sample(ModelAdapter, ...)
  |     RolloutBatch
  |
  +-> FeedbackProvider.score(batch)
  |     RewardBatch
  |
  +-> OptimizerPlugin.step(...)
  |     backward + optimizer.step + metrics
  |
  +-> ArtifactManager.record(...)
        SampleManifest + reward table + metrics + report
```

## 五个核心接口

### RolloutEngine

位置：`visual_rl/rollout/base.py`

输入是 adapter、prompt 和 metadata；输出必须是 `RolloutBatch`。现有实现：

- `FullTrajectoryRollout`：每个 prompt 生成多个完整轨迹，形成 GRPO group。
- `SingleStepRollout`：Flash-GRPO selected timestep。
- `BranchingRollout`：TempFlow 共享前缀后分叉；不允许退化成独立完整采样。

### FeedbackProvider

位置：`visual_rl/feedback/base.py`

输入 `RolloutBatch`，输出 `RewardBatch`。默认 provider 是 `RewardRouterFeedbackProvider`，内部组合 reward client、weight、cache 和 failure policy。

Feedback 不做训练归一化。它只报告：

```text
raw
weighted
weighted_total
valid_mask
metadata
```

### OptimizerPlugin

位置：`visual_rl/optimizers/base.py`

一个 plugin 负责一次完整策略更新：

```text
build optimizer
reward -> advantage -> new logprob -> loss -> backward -> optimizer.step
```

默认 `AlgorithmOptimizerPlugin` 复用 GRPO、Flash-GRPO、TempFlow-GRPO loss kernel。更特殊的算法可以注册新的完整 plugin，不需要改 Runner。

`build_optimizer()` 返回值必须提供 `zero_grad`、`step`、`state_dict` 和 `load_state_dict`；否则 Runner 会在训练前报错，避免产生无法恢复的 checkpoint。

### ExperimentRunner

位置：`visual_rl/runner.py`

它只做协调：构造组件、逐 step 调用、错误边界、checkpoint 和 artifact 旁路。它不包含某个算法专属公式。

v0.6 simplified core 只允许单进程运行；`WORLD_SIZE > 1` 会在创建 run 目录前失败，避免多个进程竞争写同一份 manifest 和 checkpoint。

### SampleManifest / SampleRecord

位置：`visual_rl/artifacts/manifest.py`

`RolloutBatch` 和 `RewardBatch` 是内存中的训练数据；Manifest 是落盘后的实验索引。一个 `SampleRecord` 对应一个 image/video sample，保存：

```text
run_id / sample_id / step / prompt / media_type
seed / rollout_type / timestep_summary
reward_values / media_path / rollout_cache_path / checkpoint_path
model_metadata
```

## RolloutBatch

位置：`visual_rl/core/types.py`

```text
prompts              每个样本对应的文本
metadata             prompt、group、branch 信息
media                最终 image 或 video
latents              每个动作前的状态
next_latents         每个动作后的状态
timesteps            scheduler timestep value
old_log_probs        rollout policy 下的动作 log probability
kl                   可选 reference-policy KL
branch_ids           TempFlow 分支编号
model_metadata       可序列化运行信息
model_tensors        logprob 重算需要的 embedding 等 tensor
```

image/video 不需要两套 schema；差异由 adapter 的 `media_type`、media shape 和 metadata 表达。

## Reward 与 Advantage 的边界

唯一规则：

```text
FeedbackProvider 负责 raw reward 和 weighting
AdvantageComputer 负责训练归一化
```

GRPO 优先按 rollout 写入的 `parent_prompt_index` 分组，prompt 文本只作为没有显式 group id 时的兜底。这样两条文字相同但来源不同的父轨迹不会被合并。每组至少需要两个样本，否则均值减自身后没有学习信号，代码会直接报错。

## TempFlow 的两个时间变量

```text
branch_step_index       轨迹中的第几个 denoising step，例如 1
branch_timestep_value   scheduler 的真实值，例如 600
```

loss mask 使用 step index；日志与复现同时保存真实 timestep value。SD3 per-step adapter 可以把全局 step 映射到只包含选中 transition 的局部 batch。

## Checkpoint

每个 checkpoint 同时保存：

```text
adapter/model state
optimizer state
OptimizerPlugin state
completed step
Python / NumPy / PyTorch RNG state
training config fingerprint
adapter / plugin / algorithm implementation identity
trainable parameter name / shape / dtype
```

改变算法、reward、rollout 语义或实现代码后恢复会被拒绝；只改变输出目录、总步数和保存频率是允许的。checkpoint 先完整写入临时目录，artifact 成功后再更新 `latest.json`；从较旧 checkpoint 恢复时，会截断其后的 manifest、metrics 与 rollout cache。

## Config

位置：`visual_rl/configs/schema.py`

正式配置只接受以下顶层结构：

```text
model / dataset / sample / rollout / algorithm
rewards / optimizer / train / runner / paths
```

旧 `trainer`、旧顶层 `output_dir`、`legacy` 和 `rewards.normalize` 不再静默映射，未知字段会在启动前报错。

## 新算法接入

1. 判断它改变的是 rollout、feedback 还是 policy update。
2. 只在对应模块增加实现。
3. 通过注册表增加 config name。
4. 复用 Runner 和 ArtifactManager。
5. 添加一个纯 tensor 单元测试和一个 Tiny 串联测试。

不要增加新的 Trainer，也不要复制训练循环。

公开注册函数位于 `visual_rl/plugins.py`；新增模块可以使用 `register_model_adapter()`、`register_rollout_engine()`、`register_feedback_provider()`、`register_reward_client()`、`register_algorithm()` 或 `register_optimizer_plugin()`，无需直接访问注册表内部字典。

各注册点的构造约定也写在该文件顶部。自定义 feedback 的专属构造参数放在 `rewards.provider_params`；ModelAdapter 必须显式实现保存和恢复方法。

## 阅读顺序

```text
1. visual_rl/core/types.py
2. visual_rl/runner.py
3. visual_rl/rollout/base.py + full_trajectory.py
4. visual_rl/feedback/base.py + provider.py + router.py
5. visual_rl/optimizers/base.py + algorithm_plugin.py
6. visual_rl/artifacts/manifest.py + manager.py
7. 一个具体算法与一个具体 adapter
8. tests/test_experiment_runner.py
```

工具脚本位于 `scripts/`，不属于训练主线；`third_party/legacy.py` 只隔离当前 SD3/Wan 参考路径。
