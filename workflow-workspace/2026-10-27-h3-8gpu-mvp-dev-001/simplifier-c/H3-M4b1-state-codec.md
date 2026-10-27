# H3-M4b1-state-codec — Simplifier C / W4

- verdict: **approved**
- 审核 HEAD：`fb085fb35fb239cdd4c151d39decbd7fae268c9d`（W4 暂存汇合）。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-29df6a46-e199-4bd7-a9cf-6a9aa0b3d761`。
- 仅批准 Coding C 当前 state codec 原子项，不批准兄弟任务，也不代表完整 checkpoint 或整 run fresh 恢复已完成。初始工作区干净，本次只写本报告。

## 实际审查

读取当前 task.md、plan.md、assignments/c.md、decisions/wave-04.md 和 Coding C 本项交付；完整检查 `visual_rl/artifacts/checkpoint/h3_state.py`、`tests/test_h3_mvp_cpu_checkpoint_state.py`，核对既有 CheckpointSafePoint 的 assert_ready/violations 和 CheckpointProgress 的计数/身份合同。

1. **save 前置强门**：实际 WORLD8/NCCL 与最终 prepared/optimizer 集合检查，既有 safe-point proof 和 progress 一致性、step>0、梯度全部 None。每个预期 trainable 必须已有精确 step/exp_avg/exp_avg_sq，正训练步与 progress 相同，moments shape/dtype/local finite 合法；cold、partial、旧参数对象或不安全 proof 均在 `_apis`/optimizer getter 前共同拒绝。没有零梯度 dummy step。
2. **完整 state 与纯度**：所有 rank 在 writer 外进入公开 model/optimizer getter 与 DCP save；使用 strict、nonfull、不丢 frozen、不 CPU offload 的 state options。保存冻结 base、LoRA、持久 buffer，独立保存并绑定 rope.inv_freq；精确区分 None/constant scheduler。测试实际比较 save 前后 model/buffers/Adam moments-step-groups/scheduler/grad/RNG 一致，禁止 step spy 未触发。
3. **真正 fresh Adam**：load 要求实际空 optimizer.state，未调用可能初始化空 Adam 的 DCP optimizer getter/setter。独立 dict 分配接收 moment/step，不预装 optimizer 状态；真实 DCP 文件填充后校验 FQN/shape/dtype/step/moments/groups，再公开 strict model setter 与 AdamW.load_state_dict 应用。FQN 顺序预先严格核对，未靠不受检查的位置映射错绑。公开 AdamW setter 的 CPU 证据证明不 step、不改模型权重；新模型/空 Adam/新 scheduler 的恢复与固定下一步逐值对照通过。
4. **文件与协作边界**：本地 metadata、keys/shape/dtype、存储路径与长度检查在 DCP load 前；有界健康 receipt 只交换共同路径和 manifest 摘要。rank0 仅写小 manifest；save/load/getter/setter 均不是 rank0-only callback。生产 writer `overwrite=False,sync_files=True`；仅新建调用者目录内 h3-state，不发布 complete/latest。getter/collective 内部失败交上层结束 attempt，不用 finally 补 collective 或伪回滚。
5. **信任与恢复范围**：DCP pickle metadata 的信任前提写明为外层 manager 已验证 artifact；本 codec 不冒充不可信文件沙箱或全树 hash manager。不抢先恢复 RNG/data/behavior，safe-point 是调用者提供的既有 proof，不伪称 codec 自己证明 trainer 完整 tail。
6. **CPU 证据与简化**：七项测试执行真实 tiny loader/PEFT、checkpoint wrapper、prepared guard、AdamW/LambdaLR、磁盘 .metadata/.distcp 和公开 state APIs；只替明确 WORLD/CUDA/分片观测及 no-dist 传输包装。包含模拟 peer 的真实文件 load、非零 moments/step2、冻结 base/两类 buffer/不同初始 lr 恢复、下一步 step3、缺字段/错误形状/损坏 storage/manifest 的提前拒绝。没有第二 manager、旧 manager/prepare 改写或同波 A/B 模块依赖。

未发现本原子项必须返工的正确性或过度开发问题。返工清单：无。

## 独立真实 CPU 门禁

在本审核 worktree 原样运行，均 exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.68s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 108 passed, 9 warnings in 7.23s（原93 + W4 A4/B4/C7）
```

9条 warning 实际来自 DCP 明确单进程/no-dist 保存读取，以及损坏 fixture 故意覆盖已有 tmp_path checkpoint；没有过滤。目标 grep 无 skip/xfail，未跑全量。原 Flash 七项完整文件相对 `d818b8a231b52a9a024fa2f522057acdbb7bf099` 历史 diff 无输出、exit0，含断言不变；`git diff --check` exit0。未把本工作区 clean 或累计基线比较当作单个 coding snapshot 精确范围审计。

## 后续必需工作 / 未验证

- Orchestrator 登记本项 snapshot/approved，结合 A/B 各自裁决亲跑 W4 汇合门禁；本报告不裁决 A 的三对象组合测试。
- 本 codec 尚未接原 AtomicCheckpointManager 共享 staging/单 writer 完整发布；后续必须绑定真实模型/拓扑/资源/cache/双 schedule/数据语义，提供 BaseTrainer 完整 tail 返回后的真实 proof，保存/恢复 rank RNG、data、behavior 并做下一步整 run 对照。raw h3-state 目录不是 complete checkpoint。
- CPU no-dist 文件恢复不证明 CUDA DTensor 的 DCP 规划、AdamW setter、FSDP2/NCCL 多 rank 保存/恢复；这些必须另行授权实测。当前没有 GPU、真实 PG、网络、服务器、模型下载、环境升级、push 或手工提交。**MVP 未完成，GPU_PENDING。**
