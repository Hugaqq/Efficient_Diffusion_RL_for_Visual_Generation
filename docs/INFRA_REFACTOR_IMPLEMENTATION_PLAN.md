# VisualRL Infra 重构完成记录

本文档记录阶段 6-9 的最终结果。未完成工作已经移入 `PROJECT_PLAN.md` 的真实实验阶段。

## 最终架构

```text
Config
-> PromptDataset
-> ModelAdapter
-> RolloutEngine
-> RolloutBatch
-> FeedbackProvider
-> RewardBatch
-> OptimizerPlugin
-> ArtifactManager
```

约束：

- `ExperimentRunner` 是唯一训练循环。
- Manifest 是 artifact 旁路，不进入 gradient path。
- Flash-GRPO、TempFlow-GRPO、World-R1/Wan 是当前主线。
- GenRL 只作为参考。

## 阶段 6：ArtifactManager / ManifestBuilder，已完成

- 每个 image/video sample 转成独立 `SampleRecord`。
- batch reward 按 sample index 对齐。
- 自动保存 `config.resolved.json`、`prompt_set.json`、`sample_manifest.json`、`reward_table.json`、`metrics.jsonl` 和 `visual_report.md`。
- tensor 写盘前转为无计算图的普通数据。
- Manifest 与 JSON 文件使用临时文件替换，避免半写状态。

设计决定：不增加 RewardTable/MetricTable class；reward table 使用普通 JSON，step metrics 使用唯一的 `metrics.jsonl`。

## 阶段 7：唯一 Runner 与工具迁移，已完成

- 内置 adapter、algorithm、reward client 集中注册。
- 公开 factory 自行幂等注册 builtins；独立 Python 进程无需先创建 Runner。
- 插件构造契约显式化：feedback 使用 `provider_params`，optimizer 必须可保存状态，adapter 必须实现 checkpoint round trip。
- `ExperimentRunner` 可从干净 Python 进程直接构造。
- 根目录 `train.py` 与安装后的 `visual-rl --config` 共用同一入口。
- probe、remote smoke、checkpoint inventory 和 runtime plan 位于 `scripts/`。
- 大型 `visual_rl/cli.py` 已移出 runtime package。

## 阶段 8：目录与配置收敛，已完成

```text
rewards + feedback                 -> feedback
algorithms + advantages + plugins -> optimizers
integration sampling helpers      -> rollout
checkpoint + logging + manifest   -> artifacts
```

- 删除旧 `trainer`、`rewards`、`algorithms`、`integrations` 和 `experiments` 运行时外壳。
- 删除 v01 preset、未使用类型、sampler 和 adapter helpers。
- Config 只接受 `paths.output_dir`、`runner` 等正式字段；未知键直接报错。
- `third_party/legacy.py` 因 SD3/Wan 仍直接使用而保留。

## 阶段 9：训练正确性，已完成

### Resume

保存和恢复 adapter、Adam state、plugin state、step、Python/NumPy/Torch RNG、实现身份、参数签名和配置指纹。checkpoint 目录先完整写入临时目录再替换；artifact 成功后才提交 `latest.json`。恢复旧 checkpoint 时会截断其后的 manifest、metrics 和 rollout cache。测试覆盖连续 2-step 与 1+1 resume，也覆盖较旧 checkpoint 与较新 artifact 并存的情况。

### GRPO group

Full rollout 按 `samples_per_prompt` 展开；AdvantageComputer 拒绝 singleton group，并优先按 `parent_prompt_index` 分组，避免相同 prompt 文本的不同父轨迹被错误合并。

### Normalization

Feedback 只输出 raw/weighted reward；AdvantageComputer 是训练归一化唯一所有者。

### Branching

- Tiny adapter 验证共享前缀和分叉后差异。
- SD3 adapter 将参考 per-step pipeline 输出整理为 selected-transition batch。
- SD3 候选只覆盖真实 transition，scheduler timestep 保留原 dtype，并按全局 branch 位置生成 noise weight。
- 不支持 `sample_branching()` 的 adapter 立即报错。
- `branch_step_index` 与 `branch_timestep_value` 独立保存。

## 验证边界

本阶段只运行本地轻量测试和静态检查，不下载模型、不连接远程服务器、不启动 Wan/World-R1 heavy training。真实硬件验证按 `PROJECT_PLAN.md` 继续。

v0.6 simplified core 仅支持单进程；`WORLD_SIZE > 1` 会被显式拒绝，避免多进程共同写同一套 artifacts。

最终验收结果：`136 passed`；Ruff、compileall、`git diff --check`、shell syntax、五个正式 preset 和 wheel 内容检查全部通过。三个只读审查角色分别检查架构边界、迁移清理和训练正确性，提出的 P0/P1 已全部修复后复核通过。

## 学习交接

实现已经完整保留在主线。面向学习的简化程序填空位于 `exercises/`，顺序与时间预算见 `FILL_IN_LEARNING_PLAN.md`。
