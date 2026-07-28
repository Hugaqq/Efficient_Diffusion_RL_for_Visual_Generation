# VisualRL v0.7 发布范围

本文档是 `Plan_7_27.md` 阶段 0 的范围冻结产物。它只记录
“支持/排除/唯一负责阶段/现有源码路径/目标源码路径”，不代表生产实现、
真实 GPU/NCCL、C20/Q100、质量提升或 wheel 验收已经完成。

除特别注明外，下列路径均相对于 `framecode/` 仓库根目录。`目标：删除`
表示该能力在 v0.7 runtime 中没有替代实现，不允许保留 alias、兼容 wrapper
或隐藏配置入口。最终支持结论与实验证据由后续实验与发布文档维护。

## 冻结矩阵

| 维度 | v0.7 冻结结论 | 类型 | 唯一负责阶段 | 现有源码路径 | v0.7 目标源码路径 |
|---|---|---|---|---|---|
| 图像模型 | SD3/SD3.5 | 支持 | 阶段 2 | `visual_rl/model_adapters/sd3.py`；`visual_rl/builtins.py` | `visual_rl/model_adapters/sd3.py`；`visual_rl/builtins.py` |
| 视频模型 | Wan2.1 | 支持 | 阶段 2 | `visual_rl/model_adapters/wan.py`；`visual_rl/builtins.py` | `visual_rl/model_adapters/wan.py` 中的 `WanFlashAdapter`、`WanWorldR1Adapter`；`visual_rl/builtins.py` |
| Flow-GRPO | SD3.5 + full trajectory | 支持 | 阶段 4 | `visual_rl/model_adapters/sd3.py`；`visual_rl/rollout/full_trajectory.py`；`visual_rl/optimizers/grpo.py`；当前无完整公开配置 | `visual_rl/model_adapters/sd3.py`；`visual_rl/rollout/full_trajectory.py`；`visual_rl/optimizers/objective.py`；`visual_rl/optimizers/clipped_surrogate.py`；`visual_rl/optimizers/grpo.py`；`configs/flow_grpo_sd3.yaml` |
| TempFlow-GRPO | SD3.5 + branching | 支持 | 阶段 4 | `visual_rl/model_adapters/sd3.py`；`visual_rl/rollout/branching.py`；`visual_rl/optimizers/tempflow_grpo.py`；`visual_rl/configs/presets/sd3_tempflow_adapter.yaml` | `visual_rl/model_adapters/sd3.py`；`visual_rl/rollout/branching.py`；`visual_rl/optimizers/objective.py`；`visual_rl/optimizers/clipped_surrogate.py`；`visual_rl/optimizers/tempflow_grpo.py`；`configs/tempflow_sd3.yaml` |
| Flash-GRPO | Wan2.1 + single step | 支持 | 阶段 4 | `visual_rl/model_adapters/wan.py`；`visual_rl/rollout/single_step.py`；`visual_rl/rollout/rectification.py`；`visual_rl/optimizers/flash_grpo.py`；`visual_rl/configs/presets/flash_wan_reference.yaml` | `visual_rl/model_adapters/wan.py` 中的 `WanFlashAdapter`；`visual_rl/rollout/single_step.py`；`visual_rl/optimizers/objective.py`；`visual_rl/optimizers/clipped_surrogate.py`；`visual_rl/optimizers/flash_grpo.py`；`configs/flash_wan.yaml`；目标中不存在 `visual_rl/rollout/rectification.py` |
| World-R1 | Wan2.1 + GRPO + general/3D reward | 支持 | 阶段 2 | `visual_rl/model_adapters/wan.py`；`visual_rl/rollout/full_trajectory.py`；`visual_rl/optimizers/grpo.py`；`visual_rl/feedback/world_r1_rewards.py`；`visual_rl/feedback/provider.py`；`visual_rl/configs/presets/world_r1_wan_bounded.yaml` | `visual_rl/model_adapters/wan.py` 中的 `WanWorldR1Adapter`；`visual_rl/rollout/full_trajectory.py`；`visual_rl/optimizers/grpo.py`；`visual_rl/feedback/world_r1_rewards.py`；`visual_rl/feedback/provider.py`；`services/world_r1_strict/`；`configs/world_r1_wan.yaml` |
| 公开调用入口 | Python API | 支持 | 阶段 3 | `visual_rl/experiment.py`；`visual_rl/__init__.py` | `visual_rl/api.py`；`visual_rl/api_types.py`；`visual_rl/__init__.py`；`visual_rl/runtime_factory.py` |
| CLI/console command | 不提供 | 排除 | 阶段 2/3 single cutover | `visual_rl/cli.py`；`visual_rl/entrypoint.py`；`train.py`；`scripts/legacy_cli.py`；`scripts/remote_smoke.py`；`tests/test_cli.py`；`pyproject.toml` 的 console entry | 目标：删除全部 CLI/console/legacy command 路径；唯一替代入口为 `visual_rl/api.py`，`pyproject.toml` 不含 `[project.scripts]` |
| 单卡 | 保留统一 Runner 代码路径 | 支持 | 阶段 5 | `visual_rl/runner.py` 的单卡 inline loop；`visual_rl/distributed.py` | `visual_rl/runner.py` 中唯一 `ExperimentRunner.run()` 与 `_execute_step()`；`visual_rl/distributed.py` 中 `SingleProcessStrategy` |
| 双卡 DDP | 保留统一 Runner 代码路径；正式支持结论由实验文档 MG1 决定 | 条件支持 | 阶段 5 | `visual_rl/runner.py` 的 `_run_distributed()`/`_run_distributed_phase()`；`visual_rl/distributed.py` | 与单卡相同的 `visual_rl/runner.py::_execute_step()`；`visual_rl/distributed.py` 中 `DDPStrategy`；真实支持状态不写入本文件 |
| 多机、弹性 world size | 不支持 | 排除 | 阶段 3 | `visual_rl/distributed.py` 的环境解析；`visual_rl/preflight.py` | `visual_rl/preflight.py` 在重资源初始化前拒绝；`visual_rl/distributed.py` 只消费已验证的单机 runtime 快照 |
| MinWM | 不包含 | 排除 | 阶段 1 | `visual_rl/model_adapters/minwm_transition.py`；`visual_rl/model_adapters/minwm_wan.py`；`visual_rl/model_adapters/minwm_wan_native_backend.py`；`tests/test_minwm_rl.py`；见下方交叉路径 | 目标：上述 MinWM production/runtime/test/config 路径全部删除，无兼容 Adapter、alias、preset 或 feature flag |
| FLUX/Qwen | 不包含 | 排除 | 阶段 2 | 当前 production component 路径不存在 | `visual_rl/builtins.py` 的固定 builtin manifest 不包含 FLUX/Qwen；不新增 Adapter、配置或依赖 |
| 质量提升 | 不作正面声明 | 声明边界 | 阶段 0 | `experiments/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md`；`experiments/EXPERIMENT_PLAN.md` | `docs/V0_7_SCOPE.md` 只保留“不作正面声明”；后续证据不能反向扩张本轮生产范围 |
| 性能提升 | 只保留已经严格验证的窄结论 | 声明边界 | 阶段 0 | `docs/PROJECT_OVERVIEW.md`；`experiments/EXPERIMENT_PLAN.md`；历史实验结果 | `docs/V0_7_SCOPE.md` 保留该边界；未完成的 P7 call-coalescing 声明按下表处理，不改写已有 correctness 实现 |

## 特别决策的删除清单

下列能力已经确定不进入 v0.7，不再以“未来可能接入”为由保留 runtime
抽象。每行只有一个实施 owner。

| 排除项 | 唯一负责阶段 | 现有源码路径 | v0.7 目标源码路径 |
|---|---|---|---|
| MinWM transition Adapter | 阶段 1 | `visual_rl/model_adapters/minwm_transition.py` | 目标：删除，无替代源码 |
| MinWM Wan Adapter | 阶段 1 | `visual_rl/model_adapters/minwm_wan.py` | 目标：删除，无替代源码 |
| MinWM Wan native backend | 阶段 1 | `visual_rl/model_adapters/minwm_wan_native_backend.py` | 目标：删除，无替代源码 |
| MinWM 测试入口 | 阶段 1 | `tests/test_minwm_rl.py` | 目标：删除，不迁为 v0.7 runtime 测试 |
| MinWM 交叉修改 | 阶段 1 | `visual_rl/builtins.py`；`visual_rl/configs/schema.py`；`visual_rl/preflight.py`；`visual_rl/feedback/world_r1_rewards.py` | 同路径只保留非 MinWM 逻辑；MinWM 名称、schema、Preflight、metadata 和专用错误零残留 |
| PickScore runtime reward | 阶段 2 | `visual_rl/feedback/pickscore.py`；`tests/test_pickscore_reward.py` | 目标：删除 runtime 与测试；必要实验结论只进入仓库外 evidence/archive |
| Video-HPSv3 runtime reward | 阶段 2 | `visual_rl/feedback/video_hpsv3.py`；`scripts/serve_video_hpsv3.py`；`tests/test_video_hpsv3_reward.py` | 目标：删除 runtime、server 脚本与测试，无 Runner 可构造 client |
| Reward schedule | 阶段 2 | `visual_rl/configs/schema.py`；`visual_rl/feedback/router.py`；`visual_rl/feedback/provider.py`；`visual_rl/artifacts/builder.py`；`tests/test_reward_schedule.py` | 聚合、加权和 cache 协调只进入 `visual_rl/feedback/provider.py`；删除 `visual_rl/feedback/router.py` 及全部 schedule 字段、artifact 投影和测试 |
| 未完成的 P7 call-coalescing 性能声明 | 阶段 0 | `docs/PROJECT_OVERVIEW.md`；`experiments/EXPERIMENT_PLAN.md` | 删除未验证的性能声明；不得为此改写已经由 correctness 测试覆盖的普通实现，也不得写入 v0.7 正面性能结论 |

## 阶段 0 验收边界

- 上述 15 个冻结矩阵项和 9 个特别决策项均有且只有一个负责阶段，并列出
  当前与目标路径。
- Python API 是唯一目标公开入口；不存在 CLI 兼容任务。
- 排除项没有“未来可能接入”、兼容层、隐藏配置或新增核心抽象。
- 阶段 1 完成后，阶段 2–5 不再包含 MinWM 实现任务。
- 本文档不把计划通过、Tiny 测试或历史窄实验外推为真实 GPU、DDP、质量或
  性能支持结论。
