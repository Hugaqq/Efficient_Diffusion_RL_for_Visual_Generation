# H3-M4b2-checkpoint-publish — Simplifier C / W5

- verdict: **changes**
- 审核 HEAD：`230696472b735966d3ac17102120a444f4ff8697`（W5 暂存汇合）。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-770eef22-ff89-4046-9afa-ad24fb5d5a1f`。
- 仅审 Coding C 本项；本步骤 done 不表示批准。未改源码、测试或他人报告。

## 必须返工：C-PUBLISH-01 — rank 附件目录未持久化

定位：
- `visual_rl/artifacts/checkpoint/h3_transaction.py:122-135`，`_write_rank` 创建 `h3-ranks/` 和 rank-N.pt，仅 flush/fsync 每个文件。
- 同文件 `:273-285`，rank-written/verify-ranks 后直接写事务 manifest 并调用 manager 发布，没有同步 `h3-ranks/` 目录。
- `visual_rl/artifacts/checkpoint/manager.py:613-619`，`_build_state_tree` 遇目录直接 continue，仅对文件 fsync；`:255-258` 只同步 staging 根目录、rename、同步 checkpoint root。

问题：fsync 文件保证的是文件内容/相应 inode，不代替其父目录中新建文件名条目的 fsync；同步 staging 根目录也不递归同步 `h3-ranks/`。因此当前路径不能保证崩溃后已发布 complete 的八个 rank RNG/data 附件目录条目仍完整。文件 hash 和正常读取测试不能关闭这一持久性缺口。这里是新事务自行创建的子目录，不要求重写原 manager 或推测 DCP 内部同步行为。

最小修正（回 Coding C）：
1. 在八份附件全部写完、校验成功之后且调用现 manager 发布之前，显式 fsync `h3-ranks/` 目录；可复用现 `_fsync_directory`，由主 writer 在受控本地发布阶段完成。或者每个 rank 写完文件后同步父目录，并确保全部成功才发布。保持唯一 manager 发布体，不引入第二 publisher。
2. 目录同步异常必须进入现协作错误路径，poison attempt，不能继续创建合法 complete/latest；不在 finally 补 collective。
3. 扩展本 feature CPU 测试：观察真实目录 fsync 位于附件完成之后、complete/rename 发布之前；注入该目录 fsync 的 OSError，证明所有模拟 rank 拒绝、无完整发布且不可重试。原 rename 后 latest 失败保留 complete 的测试必须保留。
4. 最后源码/测试修改后重新运行原 Flash 七项及完整 H3 通配符集，再交本配对复审。

此发现依据实际代码的持久化顺序和文件/目录 fsync 语义；没有执行断电/崩溃实验，也不冒称已复现数据丢失。

## 已核对的实现与边界

已读取当前 task.md、plan.md、assignments/c.md、decisions/wave-05.md、Coding C 交付，完整 manager.py、h3_transaction.py、六项 publish 测试及现 RankRNGSnapshot 实现。

- 新 create_staging/commit_existing 使用本 manager 创建的对象能力、step/root/inode 检查；原 commit(writer) 与新入口共用 `_publish_staging`，未复制第二套 complete/tree/progress/rename/latest 发布器。
- 共同 staging 后，所有 rank 的真实 codec getter/save/load/setter 位于单 writer 发布之外；rank0 仅发布文件。测试 spy、真实 DCP no-dist 文件、manager inspect_complete 和故障路径实际执行。
- 八个 rank 的私有 RNG 身份组成有序全局 rng_state_id，不要求私有 RNG 相同；rank 附件校验 hash、rank/world/proof/data/behavior，数据 envelope 有限 JSON 且拥有副本。
- restore_prepared 先检查完整树/contract/附件并恢复空 Adam；restore_bound 重新读取完整树与必需 live callback，恢复并回读 data/cursor/behavior，最后调用原 RNG restore。两恢复门 once-only，失败 poison，无伪 rollback。
- CPU fresh 对照验证实际 RNG 最后恢复、下一 Python/NumPy/Torch 随机数及固定下一次 Adam/scheduler 更新一致；live/data 仍是本项接口 fixture，不是同波真实 H3PromptSource/profile 或默认 RunController 整 run 装配证据。
- 除上述持久性缺口，未发现需额外框架或大范围重构的问题。保留既有防护、故障注入及 no-dummy-step 边界。

## 独立真实 CPU 门禁

在本审核 worktree 原样执行，均 exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 17.40s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 126 passed, 16 warnings in 14.00s（原108 + W5 A6/B6/C6）
```

16条 warning 为明确 no-dist DCP 保存/读取与损坏 fixture 覆盖提示，未过滤。目标 grep 无 skip/xfail，未跑全量。原 Flash 七项完整文件相对 d818b8a231b52a9a024fa2f522057acdbb7bf099 历史 diff 无输出、exit0；git diff --check exit0。测试通过不覆盖 C-PUBLISH-01，不能据此批准发布持久性。

## 后续

先由 Coding C 修复 C-PUBLISH-01 并重测复审。随后 orchestrator 再决定接收，安排真实 data/profile/publisher 与唯一 RunController 的组合、真实 tail proof/contract/G3/live身份及整 run fresh 恢复对照。真实 CUDA DTensor/DCP/NCCL、模型/HPS资源与质量仍未验证。本次无 GPU、真实 PG、网络、服务器、下载、环境升级、push。**MVP 未完成，GPU_PENDING。**
