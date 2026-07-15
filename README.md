# VisualRL v0.6 Simplified Core

VisualRL 是一套面向 image/video generation 的轻量 Diffusion RL infra。项目不再维护多套训练框架，而是把三类研究能力放进同一条可复现主线：

- TempFlow-GRPO：timestep-aware optimizer 与共享前缀 branching rollout。
- Flash-GRPO：selected-timestep / single-step rollout 与策略更新。
- World-R1：Wan video adapter、3D/world feedback client 与 reward server 边界。
- GenRL：仅作为参考代码来源，不是 runtime 依赖。

## 唯一主线

```text
VisualRLConfig
-> PromptDataset
-> ModelAdapter
-> RolloutEngine
-> RolloutBatch
-> FeedbackProvider
-> RewardExecutor
-> RewardBatch
-> AdvantageFunction / PolicyObjective / UpdateEngine
-> OptimizerPlugin escape hatch
-> DistributedStrategy
-> ArtifactManager
```

`ExperimentRunner` 是唯一训练协调器。算法差异不通过新增 Trainer 表达，而是放在 `OptimizerPlugin`、`RolloutEngine` 和 `FeedbackProvider` 中。

## 目录职责

```text
visual_rl/
  runner.py          唯一训练循环
  core/              Batch、StepContext、注册表和确定性运行身份
  configs/           schema、Resolver、preset、recipe 和 profile
  datasets/          prompt 数据准备
  model_adapters/    模型加载、采样和 logprob 重算
  rollout/           full / single-step / branching rollout
  feedback/          provider、client、cache 和同步/异步 RewardExecutor
  optimizers/        advantage、objective、UpdateEngine 和完整 plugin
  artifacts/         事务 commit、manifest、metrics、checkpoint 和 report
  evaluation/        held-out evaluator
  distributed.py     单进程与原生 PyTorch DDP strategy
  third_party/       SD3/Wan 仍需要的参考仓库路径隔离
scripts/             probe、remote smoke 和旧诊断工具
train.py             最小 config-driven 入口
```

## 扩展方式

- 新模型：实现 `ModelAdapter`，声明 `media_type`，调用 `register_model_adapter()`。
- 新 rollout：实现 `RolloutEngine`，输出统一的 `RolloutBatch`，调用 `register_rollout_engine()`。
- 新 reward：Python API 使用外部函数/对象 descriptor，YAML 使用可信
  `module:attribute` target；复杂场景再实现 `FeedbackProvider`。
- 新算法：优先组合 `AdvantageFunction`、`PolicyObjective` 与 `UpdateEngine`；
  不能自然拆分的算法使用完整 `OptimizerPlugin` escape hatch。
- 不需要修改：数据准备、主训练循环、manifest、metric、checkpoint 和 report 保存。

自定义 feedback 的构造参数放在 `rewards.provider_params`。自定义 optimizer 必须支持 `state_dict/load_state_dict`，ModelAdapter 必须实现 `save_pretrained/load_checkpoint`，这些契约都会在训练前或 checkpoint 时检查。

## 启动方式

最小 Python API 与 CLI 使用同一条 preflight、恢复和 Runner 主线：

```python
import visual_rl as vr

experiment = vr.Experiment(
    model=vr.models.MockWan(),
    rollout=vr.rollouts.FullTrajectory(batch_size=1),
    reward=vr.rewards.Mock(),
    advantage=vr.advantages.GroupNormalize(),
    objective=vr.objectives.GRPO(),
    train=vr.Train(steps=2, lr=1e-3),
)
experiment.validate()  # 纯静态；可信组件核验使用 trusted_components=True
result = experiment.run(["a red cube on a white table"])
# experiment.run([...], resume_from=result.latest_checkpoint)
```

```bash
visual-rl presets
visual-rl validate preset:flash_tiny_single_step
visual-rl inspect preset:tempflow_tiny_branching
visual-rl run visual_rl/configs/presets/world_r1_wan_v02_mock.yaml
visual-rl status runs/example
visual-rl audit runs/example
```

`preset:NAME` 明确引用安装包自带的 preset，因此可以从任意工作目录使用；不带
`preset:` 的 `CONFIG` 始终按文件路径处理，不会把同名本地文件与 packaged preset
混淆。`visual-rl presets` 列出当前安装版本提供的全部名称，以上命令也都支持
`--json`。

`validate` 和 `inspect` 默认只执行静态 preflight；加
`--trusted-components` 才显式加载并核对本地 built-ins。`run` 固定执行
Resolver -> static preflight -> trusted component load -> `ExperimentRunner`。
三个命令均支持按顺序应用的 `--set KEY=VALUE` 和单一 JSON envelope 输出；
resume 使用 `visual-rl run CONFIG --resume PATH`。

`status RUN_DIR` 通过 marker-aware lifecycle API 判断 run 是否已完成并可用于聚合；
`audit RUN_DIR` 通过 authoritative commit marker 审计 checkpoint 与派生记录。两者
都支持 `--json`，不会使用宽松 JSON 解析，也不会在终端转述底层未脱敏异常。

CLI 稳定退出码为：`0` 成功，`2` usage/YAML/Resolver/static，`3` trusted
组件，`4` resume/checkpoint，`5` execution，`6` run status/artifact audit
未通过，`1` internal。`status` 仅在完成态及 authoritative marker 有效时返回 0；
运行中、failed、stale、missing、篡改或 audit invalid 均返回 6。诊断与训练进度写入
stderr。仓库内 `python train.py --config CONFIG` 仅保留兼容转发，并会输出
deprecation warning。

运行时同时保留同步单进程 reference path 与原生 PyTorch DDP path。DDP 由
`torchrun` 提供 `RANK/LOCAL_RANK/WORLD_SIZE`，不建设自定义 launcher；一个完整
prompt group 固定在同一 rank，rank 0 管理全局事务产物，各 rank 使用本地
rollout/reward cache。当前自动验证覆盖单机 CPU/gloo 与相同 world-size resume；
GPU/NCCL、多机和弹性 world-size 不在当前正确性声明内。

DDP 对可捕获且所有 rank 仍能参与 collective 的 optimizer-step 异常执行参数、
optimizer 和 GradScaler 回滚；单进程路径不创建这些快照。进程硬退出、进程组中断、
collective 超时或通信后端失效无法由存活 rank 在内存中协调回滚，只能重启并从最后一个
已持久化 commit marker 恢复，不能把这一边界描述为任意故障下的原子更新。

每次 run 自动维护：

```text
config.resolved.json
trigger_decision.json
prompt_set.json
sample_manifest.json
reward_table.json
metrics.jsonl
visual_report.md
commits/
checkpoint_*/training_state.pt
latest.json
```

checkpoint 还保存 optimizer/plugin/RNG 状态、实现身份和可训练参数签名。`commits/commit_*.json`
是 authoritative commit log；`latest.json`、manifest/metric/report projection、retention audit
和 runtime sidecar 都是 marker 持久化后的可恢复派生产物，失败不会把已提交 step 改写为失败。

## 本地验证

```bash
conda run -n visual-rl python -m pytest -q -m "not distributed"
conda run -n visual-rl python -m pytest -q -m distributed
conda run -n visual-rl ruff check visual_rl scripts tests train.py
```

本地测试只运行 Tiny/Fake runtime，不下载模型、不连接远程 reward server、不启动 Wan/World-R1 heavy training。当前代码支持 bounded contract 验证；真实 SD3/Wan 训练结论必须以独立 GPU 实验为准。

第一条命令运行默认离线 suite；第二条单独运行需要本机 loopback socket 的 CPU/gloo
多进程测试。测试由 `tests/test_visual_rl.py` 的三条轻量主线与聚焦 contract 测试共同组成，
覆盖 Resolver/CLI、reward executor、artifact transaction、checkpoint/resume、
Wan LoRA、分布式归约和两进程 CPU/gloo。它不声明覆盖真实大模型、远程 reward
server、GPU/NCCL 的数值与效果验证。

项目定位、三阶段计划、完成/未完成状态、World-R1 硬门槛和能力边界见
[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)；真实实验的当前状态、冻结 recipe
和晋级门槛见 [experiments/EXPERIMENT_PLAN.md](experiments/EXPERIMENT_PLAN.md)，阶段性结果
与修复方向见
[experiments/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md](experiments/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md)。
