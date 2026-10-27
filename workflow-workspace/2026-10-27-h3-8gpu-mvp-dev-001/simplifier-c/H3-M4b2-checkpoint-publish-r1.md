# H3-M4b2-checkpoint-publish-r1 — Simplifier C

- verdict: **approved**
- issue: **C-PUBLISH-01 已关闭（实现与 CPU 证据）**
- 原任务：H3-M4b2-checkpoint-publish；本复审解除该任务此前唯一返工项，原 changes 报告保留，不改写历史。
- 审核 HEAD：`c8422b3a780d933564e94dd58c52a5c8efd8f9dc`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-9cf19e5a-1f6c-4f8c-bf92-58e7e78572be`。接收工作区干净。

## 复审核验

重读当前 task/plan/assignment C、原 changes 报告和 Coding C r1 交付；读取最新完整 h3_transaction.py、两项新增测试及受影响的既有失败/恢复测试，核对 manager._fsync_directory 和唯一发布体。

1. `h3_transaction.py::checkpoint.publish` 仅 rank0 执行 `_fsync_directory(path / _RANKS)`；位置在 rank-written、verify-ranks 全部成功之后，事务 manifest、manager complete/rename 之前。直接复用现 manager helper，没有新 publisher、额外框架或八 rank 重复同步。
2. 目录 fsync 位于既有 `_phase('publish', ...)` 本地 action 内。OSError 进入失败 receipt，所有存活模拟 rank 共同拒绝，由 `_attempt` poison；没有 finally collective，也不继续写 manifest/complete/latest。
3. 顺序测试包装真实附件写入和完整 hash/身份校验，在实际目录 fsync 时确认八 rank written/verified 完成、八附件齐全、尚无 manifest/complete/final；调用真实 helper 后严格观察 `rank-dir-synced → h3_transaction.json → complete.json → after_complete_marker → after_checkpoint_rename`，目录同步仅 rank0 一次。
4. 定点目录 fsync OSError 测试保留 owner 原异常 cause；确认八 rank 全部失败/poison、无 manifest/complete/latest/final、manager 找不到 complete，重试无新 collective 或第二次同步。非目标目录仍执行真实 fsync。
5. 原 rename 后 latest IO 失败保留合法 complete、附件/DCP 失败不发布、fresh 空 Adam/两恢复门/RNG 最后及下一固定更新测试继续通过。修复未改变事务公开接口或恢复语义。

无新增必须返工项。结合原审查已核验的 manager 复用、全 rank codec、RNG/data 身份及两阶段恢复，本 publisher 原子项可交 orchestrator 接收；不代替同波 A/B 裁决或完整 MVP 验收。

## 独立真实门禁

在本审核 worktree 原样重跑，均 exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.66s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 135 passed, 18 warnings in 16.58s（此前126 + A4/B3/C2）
```

18条 warning 为明确 no-dist DCP 保存/读取及既有损坏 fixture 覆盖提示，未过滤；目标 grep 无 skip/xfail，未跑全量。原 Flash 七项完整文件相对 d818b8a231b52a9a024fa2f522057acdbb7bf099 的历史 diff 无输出、exit0，断言不变；git diff --check exit0。

## 边界及待办

由 orchestrator 登记 r1 snapshot/批准并关闭 issue，结合各配对裁决继续汇合验收。真实 data/profile/binding/publisher 与唯一 RunController 的默认装配、整 run fresh 恢复/下一步对照仍须完成。

本次仅写审核报告，未修改源码/测试/原 changes 报告。未执行断电或崩溃实验，不宣称断电恢复已实测；CPU 文件顺序/失败测试不证明真实 CUDA/DCP/NCCL、多卡保存恢复或权重/音视频质量。无 GPU、真实 PG、网络、服务器、下载、环境升级、push。**MVP 未完成，GPU_PENDING。**
