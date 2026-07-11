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
-> RewardBatch
-> OptimizerPlugin
-> ArtifactManager
```

`ExperimentRunner` 是唯一训练协调器。算法差异不通过新增 Trainer 表达，而是放在 `OptimizerPlugin`、`RolloutEngine` 和 `FeedbackProvider` 中。

## 目录职责

```text
visual_rl/
  runner.py          唯一训练循环
  core/              RolloutBatch / RewardBatch 与注册表
  configs/           严格 schema 和正式 preset
  datasets/          prompt 数据准备
  model_adapters/    模型加载、采样和 logprob 重算
  rollout/           full / single-step / branching rollout
  feedback/          provider、reward router、client 和 cache
  optimizers/        advantage、GRPO 变体和 OptimizerPlugin
  artifacts/         manifest、metrics、checkpoint 和 report
  third_party/       SD3/Wan 仍需要的参考仓库路径隔离
scripts/             probe、remote smoke 和旧诊断工具
train.py             最小 config-driven 入口
```

## 扩展方式

- 新模型：实现 `ModelAdapter`，声明 `media_type`，调用 `register_model_adapter()`。
- 新 rollout：实现 `RolloutEngine`，输出统一的 `RolloutBatch`，调用 `register_rollout_engine()`。
- 新 reward：调用 `register_reward_client()`，或实现并注册新的 `FeedbackProvider`。
- 新算法：实现 loss kernel，或注册一个完整的 `OptimizerPlugin`；plugin 同时负责创建 optimizer 和完成一次 update。
- 不需要修改：数据准备、主训练循环、manifest、metric、checkpoint 和 report 保存。

自定义 feedback 的构造参数放在 `rewards.provider_params`。自定义 optimizer 必须支持 `state_dict/load_state_dict`，ModelAdapter 必须实现 `save_pretrained/load_checkpoint`，这些契约都会在训练前或 checkpoint 时检查。

## 启动方式

```bash
python train.py --config visual_rl/configs/presets/flash_tiny_single_step.yaml
python train.py --config visual_rl/configs/presets/tempflow_tiny_branching.yaml
visual-rl --config visual_rl/configs/presets/world_r1_wan_v02_mock.yaml
```

v0.6 simplified core 明确只支持单进程；`WORLD_SIZE > 1` 会在创建输出目录前报错。分布式训练属于后续扩展，不在本轮正确性声明内。

每次 run 自动维护：

```text
config.resolved.json
prompt_set.json
sample_manifest.json
reward_table.json
metrics.jsonl
visual_report.md
checkpoint_*/training_state.pt
latest.json
```

checkpoint 还保存 optimizer/plugin/RNG 状态、实现身份和可训练参数签名；`latest.json` 只在 artifact 写盘成功后更新。

## 本地验证

```bash
conda run -n visual-rl python -m pytest -q
conda run -n visual-rl ruff check visual_rl scripts tests train.py
```

本地测试只运行 Tiny/Fake runtime，不下载模型、不连接远程 reward server、不启动 Wan/World-R1 heavy training。当前代码支持 bounded contract 验证；真实 SD3/Wan 训练结论必须以独立 GPU 实验为准。

阅读顺序见 [docs/ARCHITECTURE_MAP.md](docs/ARCHITECTURE_MAP.md)，后续实验见 [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)，程序填空学习任务见 [docs/FILL_IN_LEARNING_PLAN.md](docs/FILL_IN_LEARNING_PLAN.md)。
