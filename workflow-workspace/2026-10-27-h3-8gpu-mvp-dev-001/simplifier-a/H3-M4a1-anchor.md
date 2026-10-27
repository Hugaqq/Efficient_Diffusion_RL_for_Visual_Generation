# H3-M4a1-anchor — Simplifier A / W1

verdict=approved

仅批准 A 的 rank-local 冻结 replay 锚原子项；不是 B/C 审核、W1 整体批准、完整 MVP 或 GPU 验收。返工清单：无。

## 审核身份与材料

- 当前审核 HEAD：`467c8fd929804aa291bf01a5fa7cda372cc507db`。
- 当前分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-20b11ec1-66a8-45bb-9e7c-b990f96a8f83`；审核开始 status 干净。只在本引擎 worktree 操作。
- 已重新读取 task.md、plan.md、assignments/a.md、decisions/wave-01.md、Coding A 对应交付、完整 h3_anchor.py 和 feature 测试，以及当前 CPS 与既有 objective。采用最新并行/feature 测试补充，不采用 task 的旧单文件门禁限制。
- 当前已有 A/B/C 三组 feature 测试；本次只审 A 实现及其依赖/回归。未修改源码、测试或他人报告。

## 实现与简化审查

1. `visual_rl/algorithms/optimization/h3_anchor.py::H3ReplayAnchor._preflight`：明确 expected sample IDs、N 步、唯一键与精确数量联合保证完整 sample×index 覆盖，含末步 q0；缺整条样本轨迹同样拒绝。数值合同、CPU B1、最大格数与输入 tensor payload bytes 在任何模型调用前检查。没有按 prompt 合并样本或补值丢格。
2. `capture`：首先消耗一次性资格；身份、preflight、device 和 callable 校验后，在 no_grad 内按输入固定顺序逐格重建临时 CPS record、调用 predict 和真实 evaluate。未把 rollout_score 当锚；读取公开复制接口，无下划线 buffer 访问。每格结束释放临时 record/prediction/score，不持有模型或输入轨迹。
3. `capture` 的 pending 仅在完整成功且 behavior 前后版本一致后发布；中途非有限/版本错误无法读取部分 scores，失败 owner 不可重试。结果为 detached CPU FP32，公开读取 clone；更新 current 参数不刷新 old。
4. `score`：核对 collection/behavior、sample/index，未知键及非法索引拒绝。无 re-anchor/update API、旧权重副本或序列化/池框架。
5. `tests/test_h3_mvp_cpu_anchor.py`：六项实测涵盖篡改 rollout diagnostic 仍重算真实 CPS、全覆盖固定顺序/no_grad、临时 record weakref 释放、缺格/重复/未知键/合同/B2/预算提前拒绝、失败原子性、版本错误、输入与读取 mutation、无记录或 callback 保留。实际既有 `ClippedSurrogateObjective` / `PolicyLossInputs` beta0 接收 old/current 后 backward，双流参数得到有限非零梯度，未另写 PPO。
6. 无需删除的重复数学或额外 runner/factory/manifest；owner 的身份、预算、完整性与失败保护均属于本项必要边界。没有 import B/C 同波模块。

## 独立真实 CPU 门禁

在当前汇合代码 worktree 原样执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

- Flash：exit0，**7 passed in 11.96s**。
- H3 通配符：exit0，**44 passed in 2.71s**（既有22+A6+B6+C10）；不是以 Coding A 分支28项代替汇合目标集。测试通过不替代 B/C 配对审查。
- H3 通配符文件搜索无 skip/xfail；未跑全量。
- `git diff --check` exit0。
- 授权历史比较 `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` 无输出、exit0：原七项完整文件及断言与历史基线一致。
- 累计 `git diff --name-status d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD` 显示仅新增 H3 源码/测试和 workflow-workspace 资料；旧源码与旧配置没有净修改。该累计比较不用于归属各并行 snapshot，具体 ledger 由 orchestrator 记录。

## 后续必须保留的边界

- records 本身不带 sample/collection/behavior 来源，当前显式标签由未来 pool 提供可信绑定；前后版本检查依赖每次 policy mutation 更新版本。不是未申报权重修改的检测机制，也不是跨 rank 冻结门。
- 模型 EVAL、credit 完成、固定 behavior 与 collective 同序由后续阶段保证。callback 不得保留临时设备 record；同波定义的上层闭包可提供样本条件。
- 预算为输入 trajectory tensor payload，不是完整进程 RSS；后续 bounded pool 仍须执行整体资源上界。
- 当前源码搜索尚无生产阶段调用本 owner；这符合本次分配，但后续 OptimizeStage 必须真实消费并补全 world 门、评分、tail/checkpoint/fresh 恢复。不能将本项 approved 当作完整 MVP。

无服务器/GPU/部署/下载/环境升级/push；未验证 CUDA 搬运、FSDP2/NCCL、H3 权重或音视频质量。**GPU_PENDING**。
