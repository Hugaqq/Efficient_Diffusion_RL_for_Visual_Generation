# VisualRL 项目说明与 Coding 主线

## 项目作用

VisualRL 是面向 image/video generation 的轻量 Diffusion RL 实验基础设施。项目解决的不是“再实现一个训练脚本”，而是把分散在不同研究代码库中的模型采样、轨迹记录、奖励反馈、策略更新、断点恢复和实验产物整理为一条可复现、可扩展、可比较的训练主线。

当前只服务三类研究能力：

- TempFlow-GRPO：共享前缀 branching rollout 与 timestep-aware policy optimization。
- Flash-GRPO：selected-timestep / single-step video rollout 与 temporal rectification。
- World-R1：Wan 视频模型边界、`reward_general` / `reward_3d` 反馈系统。

GenRL 只作为工程参考，不成为 VisualRL 的 runtime dependency。SD1.5、FLUX、QwenImage、Inferix 和其他模型暂时冻结。

## 唯一执行主线

当前实现保持一条训练循环：

```text
VisualRLConfig
-> PromptDataset
-> ModelAdapter
-> RolloutEngine
-> RolloutBatch
-> FeedbackProvider
-> RewardBatch
-> OptimizerPlugin
-> ArtifactManager
```

目标结构在不增加第二个 Runner 的前提下收敛为：

```text
Python API / YAML / CLI
-> ExperimentSpec
-> Resolver
-> VisualRLConfig
-> RuntimeComponents
   |- ModelAdapter
   |- RolloutEngine
   |- RewardExecutor
   |- AdvantageFunction
   |- PolicyObjective
   |- UpdateEngine
   |- DistributedStrategy
   `- ArtifactManager
-> ExperimentRunner
```

核心约束：

- `ExperimentRunner` 始终是唯一训练协调器。
- Python API、YAML 和 CLI 必须进入同一个 Resolver 和 Runner。
- 算法差异由 rollout、reward、advantage、objective 或完整 optimizer plugin 表达。
- 异步与多卡只能替换内部执行策略，不能产生第二套训练循环。
- 兼容逻辑只能位于入口 adapter，不能进入 Runner、artifact 或新组件内部形成长期分支。

## 当前能力与边界

| 路径 | 已集成能力 | 当前验证边界 |
|---|---|---|
| GRPO | full-trajectory rollout、group advantage、policy update | Tiny/Fake contract |
| Flash-GRPO | selected timestep、single-step rollout、rectification、loss kernel | Tiny 路径；真实 Wan/Flash 尚未闭环 |
| TempFlow-GRPO | shared-prefix branching、SD3/SD3.5 LoRA、checkpoint/resume | 真实 SD3 correctness 已推进到 E3a 数学对齐 |
| World-R1/Wan | Wan adapter、视频 rollout、reward client、完整 transformer checkpoint | MockWan/local contract；真实 Wan LoRA 与 reward 闭环尚未完成 |

截至 2026-07-14：

- v0.6 simplified core 已收敛为唯一 `ExperimentRunner`。
- deterministic runtime 已实现并进入 checkpoint implementation identity。
- checkpoint fingerprint v2 已区分训练语义、数据内容身份和数据来源路径。
- 真实 SD3 相同内容换路径 resume 已通过；v1 checkpoint 继续按旧规则 fail-closed。
- SD3 默认 BF16/CUDA 的首次非确定性已定位到 backward；显式确定性模式下连续训练与 resume 可严格对齐。
- E1 三 seed 效果门槛失败，禁止用更长训练掩盖当前 reward/effect 问题。
- E3a 原生 TempFlow 与 VisualRL 共享 rollout 数学对齐 recipe 已冻结，是当前第一项工作。

真实实验状态、冻结 recipe、失败证据和硬门槛只维护在 [`experiments/EXPERIMENT_PLAN.md`](../experiments/EXPERIMENT_PLAN.md)。本文档只维护 Coding 路线，不复制动态实验日志。

## Coding 计划原则

- 不再设置独立学习阶段；个人练习材料不进入项目提交，也不阻塞 Coding 主线。
- 不再设置独立测试阶段；每个 Coding 阶段内部包含自己的正确性门槛。
- 正确性优先于 API，美观优先级低于训练语义。
- 单卡真实闭环优先于异步和多卡。
- Artifact 事务与稳定 sample identity 是并发执行的前置条件。
- 抽象必须由至少两个真实使用场景证明，避免为未来假设增加目录和接口。

## 优先级

| 优先级 | 阶段 | 含义 |
|---|---|---|
| P0 | C0-C1 | 当前 correctness 与算法语义 |
| P1 | C2-C7 | 简洁入口、PyTorch-style API 和研究扩展点 |
| P1 | C8-C10 | 真实视频代码闭环、可恢复性与安全 |
| P2 | C11-C12 | 异步 reward 与 DDP 正式扩展 |
| P3 | C13-C14 | 由 profiling/显存需求触发的条件阶段 |

## Coding 主线

```text
C0  收口 deterministic runtime / fingerprint v2
-> C1  修正 TempFlow 数学语义
-> C2  简洁配置 Resolver
-> C3  Preflight 与 CLI
-> C4  PyTorch-style 可组合组件 API 与 image/video 数据契约
-> C5  外部 Reward 插件
-> C6  Advantage / Objective / UpdateEngine
-> C7  Experiment API、Evaluator 与轻量 Callback
-> C8  Wan LoRA
-> C9  Flash 与 World-R1 单卡真实代码闭环
-> C10 Artifact、安全与单卡性能基础
-> C11 异步 Reward v1
-> C12 DDP 单机多卡
-> C13 Rollout / Reward / Train 分卡（条件阶段）
-> C14 FSDP2 / Distributed Checkpoint（条件阶段）
```

## C0：deterministic runtime 与 fingerprint v2

状态：实现与真实验证已完成，当前工作区整理为稳定 baseline 后不再混入 C1 的算法修改。

已完成能力：

- 显式 deterministic runtime 配置与运行身份记录。
- checkpoint 保存 optimizer、plugin、RNG、实现身份和数据身份。
- fingerprint v2 使用实际 train/evaluation 内容 hash，而不是把绝对路径当作数据语义。
- 相同内容移动路径允许 resume；内容、split、sampling、reward、算法或 LoRA 改变继续拒绝。
- 无版本 checkpoint 视为 v1，不静默迁移，不提供通用 `--force-resume`。

## C1：TempFlow 数学语义

状态：当前阶段。E3a 共享-rollout recipe 已冻结，先获得差异证据，再修改算法代码。

目标是对齐原生 TempFlow 与 VisualRL 的：

```text
reward
-> advantage normalization
-> selected-transition credit
-> temporal weight
-> PPO objective
-> backward / clipping
-> AdamW state
-> single-step LoRA delta
```

当前必须核对的候选差异：

- `AdvantageComputer` 当前 epsilon 为 `1e-6`，原生 TempFlow 为 `1e-4`。
- 当前 SD3 路径使用位置平方根近似权重，原生语义为 `2.25 * std_dev_t`。
- 当前 SD3 adapter 默认 transformer `eval()`，原生训练脚本使用 `train()`。
- 当前通用 update 路径没有可配置 `max_grad_norm`，原生脚本使用 `1.0`。

预计修改边界：

```text
visual_rl/optimizers/advantages.py
visual_rl/optimizers/tempflow_grpo.py
visual_rl/optimizers/algorithm_plugin.py
visual_rl/model_adapters/sd3.py
visual_rl/configs/schema.py
```

修复要求：

- epsilon 成为算法配置，不改变所有算法的全局语义。
- rollout/recompute 保存真实 transition `std_dev_t`，objective 不再凭位置猜权重。
- transformer mode、gradient clipping 和 objective contract 进入 resolved config 与实现身份。
- 算法语义变化必须有版本，旧 checkpoint 不得静默套用新公式。
- C1 不顺带修改 Runner、CLI 或高层 API。

## C2：简洁配置 Resolver

新增轻量用户配置和纯 Resolver：

```text
preset
+ recipe
+ machine profile
+ user YAML
+ CLI/Python overrides
-> ExperimentSpec
-> VisualRLConfig
```

固定覆盖顺序：

```text
preset < profile < user config < --set < explicit API/CLI arguments
```

要求：

- preset 不保存本机 checkpoint 路径、token 或固定 server 端口。
- 相对路径相对于声明它的配置文件解析，不依赖当前 shell 目录。
- 字典深度合并、列表替换和 `None` 语义固定且可解释。
- 旧完整 YAML 只经过入口转换，随后进入同一个 resolved config。
- Resolver 无模型加载、网络访问和输出目录副作用。

## C3：Preflight 与 CLI

实现两级检查：

- 静态 preflight：配置结构、路径格式、组件 target、依赖声明和兼容关系，不导入不可信插件。
- trusted component load：在模型加载前显式导入本地组件，校验接口、构造参数、版本和代码身份。

CLI 第一版只保留：

```text
visual-rl validate
visual-rl run
visual-rl inspect
```

Resume 使用 `run --resume`，不增加语义重叠的独立命令。CLI 需要稳定退出码、stderr 和可选 JSON 输出。

## C4：PyTorch-style 可组合组件 API 与数据契约

这里的 PyTorch-style 明确定义为：实验者通过普通 Python 构造函数组合模型、rollout、reward、advantage 和 objective，然后调用统一的 `Experiment.validate()` 与 `Experiment.run()`。

它不等于“所有组件都继承 `nn.Module`”，也不允许建立第二套 Trainer。

目标用户代码：

```python
experiment = vr.Experiment(
    model=vr.models.Wan(checkpoint="..."),
    rollout=vr.rollouts.Flash(selected_steps=4),
    reward=vr.rewards.WorldR1(general_url="...", geometry_url="..."),
    advantage=vr.advantages.GroupNormalize(epsilon=1e-4),
    objective=vr.objectives.TempFlow(clip_range=0.01, temporal_scale=2.25),
    train=vr.Train(steps=20, lr=1e-5),
)

experiment.validate()
result = experiment.run(prompts)
```

Python API 与配置必须汇合：

```text
Python constructors -|
                     |-> ExperimentSpec -> Resolver -> VisualRLConfig -> Runner
YAML / CLI ----------|
```

组件契约：

- `ModelAdapter` 暴露明确的训练 `nn.Module`、`parameters()`、`named_parameters()`、`train()/eval()` 和 checkpoint state。
- 有状态组件统一实现 `state_dict()/load_state_dict()`。
- `RolloutBatch/RewardBatch` 提供 `batch_size`、`to(device)`、`detach()` 和明确 Tensor shape。
- 新增不可变 `StepContext`，保存 step、seed、rank、world size 和 policy version。
- 不再通过修改 `rollout.config` 传递当前 step、seed 或 rank。
- `sample_id`、`prompt_id`、`group_id`、`branch_id` 和 `transition_mask` 成为正式字段。
- image/video layout 显式声明为 `BCHW/BFCHW`；metadata 只保存 provenance，不表达关键训练语义。
- prompt group 是不可拆分的数据单位，为 GRPO 和未来 DDP 准备 group-aware sampler。

完成标准：

- baseline Python 实验约 15-20 行，不要求 YAML。
- 替换 reward/objective 只替换一个构造参数。
- 用户不导入 `ExperimentRunner`，不修改 Registry 或核心源码。
- 构造 `Experiment` 时不加载模型、不联网、不创建输出目录。
- Python API 与 YAML 生成相同的 resolved config、fingerprint 和 artifacts。

## C5：外部 Reward 插件

支持函数或 `module:attribute` 形式的外部 reward：

```text
RolloutBatch
-> user reward
-> function/provider adapter
-> RewardBatch
```

要求：

- Python API 直接传入对象时不要求手动 Registry 注册。
- YAML/CLI 通过稳定 target 加载同一个组件。
- 返回数量、顺序、shape、finite 和 valid mask 在进入 optimizer 前验证。
- cache key 包含 sample identity、reward version 和必要输入 hash。
- target、构造参数、代码 hash 和版本进入 resolved config 与 checkpoint identity。
- 外部组件被明确视为可信本地代码，不宣传为沙箱。

## C6：Advantage、Objective 与 UpdateEngine

将普通算法路径拆为：

```text
RewardBatch
-> AdvantageFunction
-> AdvantageResult
-> PolicyObjective
-> ObjectiveOutput
-> UpdateEngine
```

职责边界：

- `AdvantageFunction` 只负责 reward 到学习信号的转换。
- `PolicyObjective` 只负责 loss、KL、clipfrac 等数学公式。
- `UpdateEngine` 统一负责 zero-grad、backward、finite gate、gradient clipping 和 optimizer step。
- `ObjectiveOutput` 明确包含 loss、policy loss、KL、clipfrac 和扩展 metrics。
- 复杂算法保留完整 `OptimizerPlugin` 逃生口，不强迫所有算法拆成三个插件。

## C7：Experiment API、Evaluator 与轻量 Callback

`Experiment` 只保存用户意图并惰性调用 Resolver、Preflight 和唯一 Runner。`RunResult` 返回轻量路径和访问器，不把所有视频一次性载入内存。

训练 reward 与最终评价分开：

- `FeedbackProvider` 产生训练信号。
- `Evaluator` 使用 held-out 数据评价质量、prompt adherence、运动性、多样性或独立偏好分数。

只提供四个 Callback 事件：

```text
on_run_start
on_step_end
on_checkpoint
on_run_end
```

Callback 用于 WandB、额外日志、定期评估和可视化；默认只能观察或执行附加操作，不能修改 reward、loss、gradient 或参数。不要建设复杂事件总线。

## C8：Wan LoRA

补齐真实视频训练的首个阻断项：

- 向 Wan transformer 注入 LoRA。
- 冻结 base transformer，只向 optimizer 暴露 LoRA 参数。
- `named_parameters()` 与 trainable parameter manifest 精确对应。
- 实现 LoRA-only save/load/resume，不再把完整 base transformer 作为低成本主线 checkpoint。
- 零学习率参数不变；非零学习率只允许 LoRA 改变。

## C9：Flash 与 World-R1 单卡真实代码闭环

先在单进程单卡完成两条互不替代的代码路径：

```text
Flash/Wan:
checkpoint -> selected-step rollout -> rectification/log-prob -> LoRA update -> resume

World-R1/Wan:
checkpoint -> full trajectory -> reward_general/reward_3d -> LoRA update -> resume
```

Flash selected-step 不能由 World-R1 full trajectory 顺带证明。两条路径都必须使用统一 Batch、Reward、Update 和 Artifact 契约。

## C10：Artifact、安全与单卡性能基础

这是异步和多卡的前置阶段。

Artifact：

- step 先写 staging，完成 manifest、metrics、reward 和 checkpoint 后再写 commit marker。
- resume 只认最后一个 committed step。
- output directory 增加单写者约束；已提交 checkpoint 不静默覆盖。
- config、manifest、reward、metric 和 checkpoint 使用明确 schema version。
- 增加 checkpoint 保留数量、rollout cache 和磁盘预算策略。

安全：

- 优先将 legacy pickle reward 协议迁移为安全结构化协议。
- 必须显式开启 unsafe legacy 模式并限制可信 host。
- 增加响应大小、timeout、retry/backoff、认证和 secret 脱敏。
- checkpoint 只加载可信来源，未知格式和身份 fail-closed。

单卡性能基础：

- 记录 rollout、reward、recompute、backward、video IO 和 checkpoint 时间。
- 记录峰值显存、samples/s、cache 命中率和 reward p50/p95。
- 支持 gradient accumulation、rollout/reward microbatch 和明确 precision policy。
- 是否进入异步、多卡或 FSDP 由 profiling 结果决定，而不是预设结论。

## C11：异步 Reward v1

第一版只并发同一步中的 reward 工作，不引入跨 policy version 的异步训练：

```text
RolloutBatch
-> RewardExecutor.submit()
-> parallel clients / batch shards
-> RewardExecutor.collect()
-> current-step update
```

实现：

- `SyncRewardExecutor` 保持当前语义。
- `AsyncRewardExecutor` 处理远程 I/O 和多个 reward/client 的并发。
- bounded queue、最大并发数和背压。
- 基于 `sample_id` 的幂等请求、cache、timeout、retry 和取消。
- 记录队列等待、服务延迟、首次失败率和重试后失败率。
- 当前 step 只有拿到完整有效 RewardBatch 才能 update 和 artifact commit。

默认不在等待 reward 时用已经过期的模型预生成下一训练 batch，因此不会悄悄改变 on-policy 语义。

## C12：DDP 单机多卡

第一版使用原生 `torchrun` 与 PyTorch DDP，不编写自定义进程启动器。

实现：

- `DistributedContext` 从环境读取 rank、local rank、world size 和 device。
- `SingleProcessStrategy` 保持当前路径；`DDPStrategy` 包装 Adapter 暴露的训练模块。
- strategy prepare 在 optimizer 构建之前完成。
- 一个完整 prompt group 分配给同一个 rank，不能拆散后再局部计算 advantage。
- metric 按样本数全局归约，不能平均各 rank 的平均值。
- rank 0 负责总 manifest、metrics、checkpoint index 和进度输出。
- media/rollout cache 使用 rank-local shard，再由 rank 0 汇总记录索引。
- checkpoint 保存每个 rank 的 RNG、sampler cursor 和 runtime identity。
- 第一版 resume 要求 world size 相同，变化时明确拒绝。
- 任意 rank 在一个 step 失败时，所有 rank 一起放弃该 step，禁止部分参数更新。

## C13：Rollout、Reward、Train 分卡（条件阶段）

只有 profiling 证明单纯 DDP 无法解决 rollout/reward 等待时才进入该阶段。

```text
Rollout GPU
-> Reward GPU/server
-> Trainer GPU
```

所有 rollout 必须携带 `policy_version`。默认 `max_staleness=0`；若某个未来算法允许 stale rollout，必须由该算法显式声明和验证，不能成为 infra 默认行为。

## C14：FSDP2 与 Distributed Checkpoint（条件阶段）

只有 Wan 单卡无法容纳，或 DDP 复制 base model 成为明确显存瓶颈时才实施。

要求：

- 先证明 Adapter 不持有会被参数分片破坏的陈旧 parameter reference。
- FSDP wrapping、LoRA、optimizer 构建和 checkpoint 顺序固定。
- 分片 checkpoint 使用 PyTorch Distributed Checkpoint 或等价的可审计机制。
- 支持多 rank 保存和恢复；是否允许改变 world size 必须单独声明。
- FSDP 不作为 C0-C12 的完成条件。

## 当前执行顺序

1. 将 C0 当前 correctness 改动整理为稳定 baseline，不与算法修复混合。
2. 执行已冻结的 E3a；若公式门槛失败，只在 C1 边界内修复并重新冻结算法版本。
3. 按 C2 -> C7 完成 Resolver、Preflight、可组合 API 和研究扩展点。
4. 按 C8 -> C10 完成 Wan LoRA、两条单卡视频路径以及并发前的 artifact/security 基础。
5. C11 异步 reward 与 C12 DDP 成为正式后续扩展，各自保持同步单卡路径作为 reference。
6. C13/C14 只有 profiling、显存和真实吞吐证据满足触发条件时才启动。

## 目录职责

```text
visual_rl/
  runner.py          唯一训练循环
  core/              Batch、context、注册表、随机性和运行身份
  configs/           schema、preset、recipe、profile 和 Resolver
  datasets/          prompt 数据与 group-aware sampling
  model_adapters/    Tiny、SD3/SD3.5、Wan 模型边界
  rollout/           full、single-step、branching rollout
  feedback/          provider、client、cache 和 RewardExecutor
  optimizers/        advantage、objective、update 和完整 plugin
  artifacts/         manifest、metrics、report、checkpoint 和 step commit
  evaluation/        独立 evaluator 与跨 run 统计
  third_party/       可选参考仓库的惰性导入边界
scripts/             数据准备、诊断和受控实验工具
tests/               公共行为与聚焦 contract 验证
train.py             最小 config-driven 兼容入口
```

不要为每个新类建立一个目录。`RuntimeComponents`、`StepContext` 等小对象优先放在少量清晰模块中。

## 可复现性与产物

每次 run 维护：

```text
config.resolved.json
prompt_set.json
sample_manifest.json
reward_table.json
metrics.jsonl
visual_report.md
rollouts/
checkpoint_*/
latest.json
```

Checkpoint 保存模型或 LoRA、optimizer、plugin、step、各 rank RNG、配置指纹、数据身份、实现身份、运行身份和可训练参数签名。Resume 拒绝语义不兼容的配置，并清理最后一个 committed step 之后的未提交产物。

## 正确性边界

- C12 完成前只声明单进程正确性，`WORLD_SIZE > 1` 继续提前失败。
- GRPO 每个 prompt 必须形成至少两个样本的有效 group。
- Reward 只负责打分和加权，advantage normalization 只有一个 owner。
- TempFlow branching 必须共享父轨迹，不能退化为多次独立完整采样。
- old/new log-prob 必须描述同一状态转移和同一 policy version。
- 异步 reward 默认不允许 stale-policy update。
- Fake/Tiny 只证明接口和训练契约，不能替代真实 SD3/Wan/reward server 结论。

## 非目标与延后项

当前不做：

- 新增 SD1.5、FLUX、QwenImage、Inferix 等模型主线。
- 在 C10 前实现异步、DDP 或多进程 artifact 写入。
- 未经 profiling 直接实现 FSDP、多机或复杂调度系统。
- 允许 stale rollout 的默认训练语义。
- 自动部署 reward server。
- Web UI/dashboard 和复杂事件总线。
- 用更长训练掩盖 E1 或 E3 的正确性/效果门槛失败。

项目当前最重要的完成标准是：实验者能够通过简洁 Python API 组合模型、rollout、reward 和算法，并在同一个正确、可恢复、可比较的 image/video RL 内核中运行；异步和多卡是在这个核心成立后扩大吞吐与规模。
