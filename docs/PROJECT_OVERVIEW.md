# VisualRL：Goal、当前状态与执行计划

更新日期：2026-07-15

本文档是 `docs/` 中关于项目目标、当前任务和后续计划的唯一事实来源。它回答四个问题：VisualRL 要解决什么问题、目前做到哪里、当前正在做什么、接下来按什么标准推进。

当前实验 recipe、attempt、执行顺序和机器可读证据索引维护在 [`experiments/EXPERIMENT_PLAN.md`](../experiments/EXPERIMENT_PLAN.md)。[`experiments/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md`](../experiments/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md) 与 [`experiments/INFRA_VALIDATION_WORKLOG.md`](../experiments/INFRA_VALIDATION_WORKLOG.md) 是合并阶段的历史快照，不再作为当前状态来源。本文档不复制逐次运行日志。

## 1. 项目 Goal

VisualRL 的目标是成为面向 image/video diffusion reinforcement learning 的轻量研究基础设施：像 PyTorch 组合模型、损失和优化器一样，让研究者用少量 Python 代码组合模型、rollout、reward 和算法，同时得到可复现、可恢复、可审计的训练流程。

当前长期 Goal 是：

> 完成 VisualRL/framecode 的稳定 mainline；持续审计和修复实现，以本地代码审查、真实 GPU 实验和并行开发提升可运行性、训练正确性、研究者易用性以及训练效率与质量。只把成熟内容提交到 GitHub `main`，不提交 `exercises/`、`.codex/` 或未成熟实验草稿。

项目只保持一条训练主线：

```text
Python API / YAML / CLI
-> ExperimentSpec
-> Resolver / Preflight
-> VisualRLConfig
-> RuntimeComponents
   |- ModelAdapter
   |- RolloutEngine
   |- FeedbackProvider / RewardExecutor
   |- Advantage / Objective / OptimizerPlugin
   |- DistributedStrategy
   `- ArtifactManager
-> ExperimentRunner
```

约束：

- `ExperimentRunner` 是唯一训练协调器，不为新算法复制第二套训练循环。
- Python API、YAML 和 CLI 必须汇入同一个 Resolver、配置身份和 artifact 合同。
- 算法差异由 rollout、reward、advantage、objective 或 optimizer plugin 表达。
- image/video 的 batch、sample identity、reward、checkpoint 和 resume 合同必须显式、可验证并 fail closed。
- 新抽象必须由至少两个真实使用场景证明，不能只为未来假设增加层级。

## 2. 成功标准

项目按四级能力验收，不能用较低层证据替代较高层结论。

| 等级 | 要回答的问题 | 完成标准 |
|---|---|---|
| L0 可运行 | infra 能否稳定跑通？ | 安装、`validate/inspect/run`、Tiny/Fake smoke、真实模型有界运行和错误诊断都可重复 |
| L1 正确训练 | 是否真的按声明的算法更新模型？ | active/control、reward/log-prob/gradient/update、checkpoint/resume、故障回滚和数据身份均有严格证据 |
| L2 低成本使用 | 是否明显减少研究代码构造量？ | 常见实验约 15–20 行 Python；替换 reward/objective 只改一个组件；不要求修改 Runner 或 Registry；错误在模型加载前可解释 |
| L3 效率与质量 | 是否提高速度、显存效率或训练质量？ | 在相同语义、相同数据和可比较硬件下获得稳定 A/B；性能收益达到预注册门槛且质量不回退；质量结论需要多 seed 和独立评价 |

当前判断：L0 和 L1 的核心合同已有较强基础，但仍需扩大真实 GPU 覆盖；L2 已有高层 API，尚未完成系统性的代码量与新用户任务验证；L3 已获得固定 Wan 单步配置下的 gradient-checkpointing 显存收益证据，但仍没有可泛化的稳态吞吐或质量提升结论。

## 3. 当前状态

### 3.1 已发布稳定基线

GitHub `main` 已包含 gradient-checkpointing 产品提交 `ee4a44f`，其基线包括：

- 合并后的唯一 VisualRL 主线和兼容入口；
- Resolver、Preflight、CLI、Experiment 高层 API 与 Wan/Flash/World-R1 packaged preset；
- deterministic runtime、fingerprint v2、checkpoint/resume、artifact transaction、rollout cache 和安全边界；
- Wan LoRA、Flash/World-R1 有界路径、异步 reward 和 DDP 的本地合同；
- World-R1 packaged preset 修复。
- Wan/SD3 gradient checkpointing 的显式 on/off、effective-state 验证、artifact/checkpoint provenance 和 resume drift 拒绝。

发布树明确排除 `exercises/` 和 `.codex/`。

### 3.2 已完成的合并、审计与 correctness 基线

两份 Framecode 副本已经完成语义合并，不再存在“等待合并”的任务。合并后已修复并验证的关键问题包括：

- artifact 审计在摘要和权威 marker 校验前不得进行不安全反序列化；
- committed checkpoint tree digest、checkpoint v4 training-state digest 和 rollout-cache generation 必须重新验证；
- authoritative JSON 拒绝 duplicate key、`NaN/Infinity`、危险 symlink 和未知 schema；
- Wan recompute 保留 SDE latent 的原始 FP32 语义，避免提前 BF16 量化；
- DDP 可捕获 update failure 的 model、optimizer、plugin 和 GradScaler 原子回滚合同；
- 相同数据内容换路径允许 v2 resume，数据内容或训练语义变化继续拒绝；旧 checkpoint 按旧规则 fail closed。

真实 GPU 的窄边界证据已经覆盖：

- SD3 continuous 与 resume 的确定性比较；
- Wan World-R1/Flash 单步或短程更新及 resume correctness；
- Flash selected-step sampler 的状态、log-prob、loss 和梯度对齐；
- HPS reward direct/reference/HTTP/infra 分数一致以及 reward 服务中断时不提交错误 step。

这些证据只证明对应冻结配置，不代表长程收敛、任意模型、任意尺寸或生产吞吐。

### 3.3 近期优化与实验

| 项目 | 状态 | 当前准确结论 |
|---|---|---|
| Gradient checkpointing 合同 `ee4a44f` | 已验证并发布 | Wan/SD3 显式 on/off、effective 检查、artifact/checkpoint provenance 和 resume drift 已实现；产品回归、独立 review 与 P6 真实 Wan A/B 均通过 |
| P5 HPS call-coalescing 微基准 | 已完成窄基准 | 两张同 prompt 图片的一次 list call 与两次单图 call reward 一致，调用时间约 `2.084x`；HPSv2 1.2.0 内部仍逐图 forward，因此这不是 tensor/GPU batch，不能外推为端到端训练加速 |
| P6 Wan GC A/B | 已通过（固定配置） | 同一 RTX 5090 上 2 次 warm-up + 6 次 measured 独立进程全部有效；三对 on/off 的 14 项训练摘要全部 exact；update peak 中位数从 `18,157,285,888` 降至 `14,966,723,072` bytes，减少 `3,190,562,816` bytes（`17.57%`） |
| P7 World-R1 prompt-group call-coalescing patch | CPU manager contract 通过，尚未晋级 | 13 个 mock tests 覆盖分组、scatter、错误传播和单卡并发拒绝；仍需直接执行真实 `GeneralRewardInstance` 的测试、强制单可见 GPU，并完成真实 Pillow/HPS 与端到端 GPU A/B |

### 3.4 仍然存在的主要缺口

- Gradient checkpointing 的收益目前只在固定 `128×128`、5 帧、4 denoising step、单 GPU、单步 Wan LoRA 配置中成立；10–100 step 稳态、其他尺寸和真实 reward 共驻仍未验证。
- World-R1 reward server 仍以逐样本调用为主；call-coalescing patch 未完成直接 instance、真实 HPS/O4 和异常恢复验证，也不能称为真实 GPU batching。
- P7 继承的上游 pickle 请求协议只能在可信 loopback wrapper 内使用；不得把原始服务绑定到不可信网络。
- 10–100 step 稳态吞吐、长程 resume、GPU/NCCL 成本和多 seed 质量仍不足。
- 真实双卡 DDP/NCCL correctness、group-aware data shard/cursor、rank failure 和相同 effective batch 对照尚未完成；CPU/Gloo 通过不能替代该门槛。
- 现有高层 API 尚未用统一任务衡量代码行数、配置负担、错误恢复时间和组件替换成本。
- doctor/依赖诊断、可信插件错误解释和远端运行前依赖门禁仍需收口。
- `runner.py`、artifact/checkpoint 和 Wan adapter 体量较大，仍需按职责拆分，但不能在 correctness 实验中同时做大重构。
- 现有 TempFlow/Wan 效果实验多次未达到预注册质量门槛；不得用延长训练或更换评价指标掩盖失败结果。

## 4. 三条并行工作线

### A. 学习与理解项目

目标是让维护者能够解释一次实验从用户输入到 optimizer update 和 committed artifact 的完整路径，而不是记住所有文件。

建议顺序：

1. 从高层入口构造一个 Tiny 实验，阅读 `ExperimentSpec -> Resolver -> Preflight`。
2. 沿 `ExperimentRunner` 跟踪 prompt、rollout、reward、advantage、update 和 artifact commit。
3. 分别替换一个 reward、一个 objective、一个 model adapter，观察哪些合同保持不变。
4. 手动制造数据身份变化、reward 失败和 resume drift，理解 fail-closed 行为。
5. 阅读 P6/P7 的冻结实验，学习如何把“功能存在”转换成“语义、资源和性能均可比较”的证据。

学习练习可以保留在本地 `exercises/`，但不进入 GitHub main，也不成为产品测试依赖。

### B. 产品与架构优化

近期重点：

- 把常见 Wan/Flash/World-R1 实验压缩为少量 Python 构造代码，同时保持 YAML/CLI 等价；
- 增加 `doctor` 或等价的早期依赖、模型文件、reward endpoint 和 GPU 能力诊断；
- 稳定 `ExperimentSpec`、Batch、sample identity、reward、update、checkpoint 和 artifact schema；
- 把 legacy 兼容限制在入口 adapter，禁止兼容分支进入核心训练循环；
- 在测试保护下按 step execution、resume coordination、artifact projection 和 model boundary 拆分超大模块；
- 保留完整 `OptimizerPlugin` 逃生口，不用过度抽象阻碍新算法研究。

### C. 实验与效果验证

实验必须按“先 correctness，再性能，最后质量”推进：

1. 单步 exact gate：固定输入、reward、gradient、update、checkpoint 和资源身份。
2. 10–100 step 稳态：吞吐、峰值显存、resume、cache 和 artifact 成本。
3. 多 seed active/control：独立 evaluator、置信区间、像素/时间护栏和失败记录。
4. 只有单卡语义稳定后，才使用 4–5 张 GPU 并行独立 seed 或验证 DDP/NCCL；不能为了速度混入不同 GPU 的 A/B 偏差。

## 5. 当前任务与下一步顺序

| 顺序 | 任务 | 产出与晋级门槛 |
|---|---|---|
| 1 | P6：Wan gradient-checkpointing 严格 A/B | **已完成。** 8 个独立进程和三对 measured exact gate 全部通过；固定配置的 update peak median 下降 `3.19 GB / 17.57%` |
| 2 | 发布 gradient-checkpointing 合同 | **已完成。** `ee4a44f` 进入 GitHub `main`；只声明固定 P6 配置的单步 update-phase 显存收益 |
| 3 | P7：World-R1 prompt-group call coalescing | **当前任务。** 先补真实 instance 测试、异常后锁恢复、单可见 GPU fail-closed 和可信 loopback wrapper；再冻结上游/HPS 身份，做固定 payload 数值/顺序 A/B 与完整 O4 one-step 的故障、显存和端到端吞吐 A/B |
| 4 | 高层 API 与 doctor | 用相同研究任务比较旧入口和新 API 的代码量、配置字段、失败定位时间；补依赖/路径/模型/reward/GPU 的早期诊断 |
| 5 | 稳态单卡 correctness 与效率 | 10–100 step continuous/resume、gradient accumulation、cache/checkpoint 开销；保留单卡 reference，先完成语义和资源基线 |
| 6 | MG1/MG2：真实多卡门槛 | 先做双卡 DDP/NCCL correctness：相同 effective batch、group 不拆分、无重复样本、rank 一致失败/回滚和 resume；通过后才比较吞吐、通信占比与峰值显存。多张卡各跑独立 run 不算扩展效率 |
| 7 | 质量验证 | World-R1/Flash 多 reward、多 seed active/control、held-out evaluator 和至少两名真实人工评审；只有中间门槛通过才扩成长跑 |
| 8 | 最终架构审计与发布 | 检查单一 Runner、数据契约、兼容边界、错误解释和模块职责；完成干净环境重建、冻结材料端到端复跑、长期资源泄漏检查和最终验收报告；只提交成熟代码并推送 GitHub main |

当前正在执行第 3 项。P7 在进入真实 GPU 之前必须先收紧 direct-instance、单可见 GPU、异常恢复和可信 loopback 合同；未达到这些门槛前不发布 patch，也不把 call coalescing 描述成 GPU batching。

## 6. Goal 关闭门槛

只有以下条件同时满足，长期 Goal 才能标记完成：

- GitHub `main` 从干净环境可以按文档安装、validate、运行最小实验并恢复 checkpoint；
- 冻结源码、配置、数据、模型/reference 和环境身份可以端到端复跑关键 SD3/Wan 路径；
- 单卡和真实双卡的 correctness 门槛通过；性能结论有 matched reference，质量结论有多 seed、独立 scorer 和真实人工评审；
- 中长程 workload 没有持续无界的 GPU/CPU 内存、文件句柄或进程泄漏；
- 最终 acceptance 报告明确区分“能跑、机械正确、可恢复、低成本使用、效率提升、质量提升”，失败项和不支持边界没有被省略；
- 发布提交不包含 `exercises/`、`.codex/`、模型、原始运行产物或未成熟实验草稿。

## 7. Git、实验与安全边界

- GitHub `main` 只接收通过本地回归、独立 review 和与风险相称的真实验证的内容。
- `exercises/`、`.codex/`、大体积 output、模型、数据、原始视频和未成熟 experiment attempt 不提交。
- 失败 attempt 不覆盖、不删除；修正后创建新的 attempt，并保留失败原因。
- 不使用通用 `--force-resume` 绕过数据、配置或实现身份检查。
- 远端 GPU 启动前检查空闲、持有自有锁并只管理本次启动的进程；不得终止其他用户任务。
- P6 的 on/off A/B 已在同一张 GPU 顺序完成；后续 4–5 GPU 只用于可独立并行且不会破坏比较设计的任务。
- P7 只能在单可见 GPU、可信 loopback 服务中验证；当前 HPS 实现是逐图 forward，除非进一步实现并验证 `torch.stack` 后的一次 batched forward，否则不得称为 GPU batching。
- 效率结论必须区分微基准、update phase、单步端到端和稳态训练；质量结论必须区分训练 reward 与独立 evaluator。

## 8. 文档职责

`docs/` 只保留两类文档：

- 本文：唯一的项目 Goal、当前状态和执行计划。
- [`DETERMINISTIC_RUNTIME.md`](DETERMINISTIC_RUNTIME.md)：deterministic runtime 的稳定技术说明。

实验动态状态、冻结 recipe、attempt 和结果不写回本文的细节段落，只在 `experiments/` 证据账本中维护。每次发布候选、计划顺序或能力边界发生变化时，更新本文的“当前状态”和“下一步顺序”，避免再次产生暂停交接、合并计划和项目路线三份互相冲突的文档。
