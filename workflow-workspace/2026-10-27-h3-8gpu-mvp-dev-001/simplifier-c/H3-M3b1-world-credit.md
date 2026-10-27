# H3-M3b1-world-credit — Simplifier C / W2

- verdict: **approved**
- 审核 HEAD：`ced9a871df0ee6bfe8b560e90da03bc12d4dda0e`（W2 暂存汇合）。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-6d50ea27-b5d2-4bcc-bf26-636b1386a9b8`。
- 仅批准 Coding C 当前原子项，不代替 A/B 配对审查或完整 MVP 验收。初始工作区干净；本次只写本报告。

## 实际审查

读取当前 task.md、plan.md、assignments/c.md、decisions/wave-02.md、Coding C 本项交付；完整检查 `visual_rl/algorithms/optimization/h3_world.py`、`tests/test_h3_mvp_cpu_world.py`、当前 `visual_rl/models/lifecycle/h3_fsdp.py` 和 `tests/test_h3_mvp_cpu_prepare.py`；对照既有 `advantage.py` 与 `objective.py`。

1. **准入与身份**：WORLD8/NCCL 前置检查明确，不拥有 PG。每 rank 有界正样本表、完整 occurrence 组成员、全局样本唯一、组不跨 rank、严格等量及公共 collection/behavior/N/geometry/schedule/score 标识检查成立。返回深层 tuple 的 frozen plan，保留 rank-major/local 原序，global_count 包含全部 N 格。
2. **协作拒绝**：本地表/配置/reward 预检错误编码为固定有限错误 receipt，仍进入同一次 all_gather_object；汇合后统一拒绝。有效输入只编码基本类型 JSON bytes，含样本/身份长度及序列化字节上限，不传媒体、latent 或模型。PG/通信故障不被伪装成可恢复错误；本项也没有宣称提供实际模型阶段健康门。
3. **完整 credit**：reward 必须完全覆盖本 rank 计划，三元组和 plan/collection/behavior 一致、valid 全真、finite float；行重排按 sample_id 对齐，不丢样重归一化。组均值与完整唯一 world population std 在 CPU FP64 计算，epsilon=1e-4，输出拥有副本的 detached CPU FP32；无额外 clip，同值奖励精确零。没有为复用旧 processor 伪造生产 trajectory/condition 身份；短小单一数学实现与真实旧 processor、独立手算交叉验证合理。
4. **loss/reducer**：slot local mean 的系数严格为 d*n_slot/N_world_cells，没有第二次 GA/world 除法。实际 prepare 的每个 FSDP block 显式调用公开 set_gradient_divide_factor(8.0)，缺 API 在 prepare 前置能力检查拒绝，prepared 暴露同一 divisor；没有仅凭 world_size 推测默认平均。此次 prepare 改动限于该能力/设置/property，没有改变既有 AC、dtype、mesh 或物化顺序。
5. **真实 CPU 验证范围**：world 八项仅替 WORLD/collective 叶子，以实际函数生成八 rank receipt，再向各模拟 rank 重放，验证同次汇合、统一拒绝、下游调用零次及身份/预算边界。真实 ClippedSurrogateObjective/PolicyLossInputs、八个 tiny trainable 副本逐格累积、显式模拟归约与单进程 global mean 的 loss/gradient/SGD update 对照通过；含末格、零优势图连接 backward/step 和优势大于5不裁剪。新增 reducer 测试真实执行公开 setter，仅替 state leaf，并验证生产 _shard_block 接线及缺能力拒绝；不是实际 CUDA collective。
6. **范围与简化**：无同波 pool/decode import、无第二套 PPO/通用 normalizer/phase runner，也未修改旧 advantage/objective。当前 plan/credit 尚未接实际 stage/controller，属于明确后续必需工作，不能以 helper 数量称闭环完成。

未发现本原子项必须退回的正确性、边界或过度开发问题。返工清单：无。

## 独立 CPU 门禁与历史核验

在本审核 worktree 执行以下原样命令，均 exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.88s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 65 passed in 3.45s（原44 + W2 A6/B6/C9）
```

H3 目标文件 grep 无 skip/xfail；未跑全量。原 Flash 七项文件相对 `d818b8a231b52a9a024fa2f522057acdbb7bf099` 完整历史 diff 无输出、exit0，含断言不变。获准累计 stat 只有 H3 新源码/测试及 workflow-workspace 资料，旧源码/配置没有净改动；此累计证据不冒充逐 coding snapshot 精确归属。`git diff --check` exit0。

## 接收与后续边界

Orchestrator 登记本项 snapshot/批准，结合 A/B 各自裁决亲跑汇合门禁。后续生产阶段必须将 geometry/schedule/score 标识绑定实际内容，消费共同 plan 与 prepared.reducer_divisor，保证每格恰消费一次，并补齐实际同序健康门、global norm、真实评分、controller、完整 tail/checkpoint/fresh 恢复。

本次无源码/测试修改，无 GPU、PG 初始化、网络、服务器、下载、升级、push。真实 FSDP2/NCCL 除数与八卡执行、H3 权重和质量均未验证；CPU setter/state probe 和模拟平均不是这些证据。**MVP 未完成，GPU_PENDING。**
