# H3-M4b2-checkpoint-publish-r1 — C-PUBLISH-01

- `implementation_ready: true`，仅本次最小返工，等待原 simplifier-c 复审；不自行将原 publisher 的 changes 改为 approved。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-1554f542-75f4-46a7-9d47-b77adebddef5`。
- 开始/结束 HEAD：`75b80699e0380a69c4b7068d19cdba62d420eaab`，接收时干净；未切分支或手工提交。
- 已读当前task/plan/assignment C和原changes报告，核对实际transaction发布顺序及manager目录同步helper。

## 最小修复与逐条证据

1. `visual_rl/artifacts/checkpoint/h3_transaction.py` 仅导入现manager `_fsync_directory`，在rank0的受控 `publish()` 内同步 `path / 'h3-ranks'`。该块位于所有rank附件写完、`verify-ranks`共同校验之后，事务manifest/complete/rename之前。没有8rank重复同步，没有另造publisher或修改manager。
2. 同步异常自然由既有 `_phase('publish', ...)` 转成失败receipt；所有存活rank共同拒绝并由 `_attempt` poison。不继续manifest/complete/latest，不加finally collective。
3. 在原feature追加两项测试，旧六项及全部断言保留：
   - 真实文件write/fsync和完整附件校验完成后记录各rank；目录同步时断言8份文件与8rank校验均完成、manifest/complete/final均不存在。调用真实目录fsync后，再观察真实JSON写入和原manager故障观察点，精确顺序为 `rank-dir-synced → h3_transaction.json → complete.json → after_complete_marker → after_checkpoint_rename`，目录同步恰一次且owner为rank0。
   - 仅对h3-ranks目录同步注入OSError；非目标目录控制执行真实fsync成功。八个模拟rank共同失败、owner异常cause保留、全部poison，最终publish receipt失败，无manifest/complete/latest/final，原manager找不到complete。重试没有新collective或第二次fsync。
4. 原rename后latest写失败仍保留合法complete、真实DCP文件/fresh空Adam/两恢复门/RNG最后等测试全部原样重跑通过。

本轮只改上述transaction与 `tests/test_h3_mvp_cpu_checkpoint_publish.py`，另新增本报告。源码diff为导入扩展和3行同步/注释；测试追加92行，合计96 insertions/1 deletion（import行替换）。旧manager/codec/RNG/protocol/runtime、A/B源码及原tests只读。未覆盖原交付或changes报告。

## 最后源码/测试编辑后的真实门禁

以下原样命令均exit0，本轮重新执行，不沿用旧114/126项记录：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.11s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 128 passed, 18 warnings in 12.45s（本分支基线126 + C新增2）
```

warnings为明确no-dist DCP及既有损坏fixture覆盖提示，未过滤；无skip/xfail或全量测试。已核对实际git diff；git diff --check exit0。原Flash完整历史diff相对d818b8a231b52a9a024fa2f522057acdbb7bf099无输出，七项及断言保持不变。

## 交付边界

交原simplifier-c复审C-PUBLISH-01的实现/顺序/失败证据，再由orchestrator决定关闭issue和接收publisher。引擎snapshot待生成；汇合仍需完整门禁。新测试沿用明确CPU通信/设备叶子与真实磁盘DCP/manager，不证明CUDA/NCCL多卡。

未做断电/崩溃实验，不声称已复现数据丢失或实测断电恢复。无GPU、真实PG、网络、服务器、下载、环境升级、push；完整默认RunController和整run恢复仍待后续装配。**GPU_PENDING。**
