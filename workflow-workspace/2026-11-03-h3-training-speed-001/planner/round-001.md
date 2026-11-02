# Planner round-001 — 首次审计与门禁前置

## 已完成的实际检查

- 完整读取TASK、workflow.json、AGENTS与六份inputs。历史source-AGENTS只作历史，不执行Mac/SMALL40/resume命令。
- `git log -12 --oneline`：HEAD为`86a628c docs(h3): define hardware-aware atomic performance workflow`，父为生产`3db2a05`。
- 初始`git status --short`为空；相对生产baseline的授权源码diff为空。
- 实读采样/梯度路径：`h3_dual_rollout.py`、`h3_dual_stages.py`、`h3_stages.py`。10步、stateless组级first-five选择、单样本选中转移、anchor复验、逐条backward释放、WORLD优势与梯度缩放仍为基线实现。
- 实读FSDP与同步：`h3_fsdp.py`（1D WORLD8、AC、逐block reshard、BF16 compute/FP32 reduce、明确divisor8）、`h3_coordination.py`（gate、单owner工作线程、主线程心跳、aux deadline、1D shard norm）、`h3_file_phase.py`、`h3_session.py`。没有修改这些路径，也没有以CPU模拟证明GPU通信性能。
- 实读VAE/HPS：`h3.py`、`h3_decode.py`、`h3_hps.py`、`serve_hpsv3_video.py`。Decoder辅助对象复用但每视频/音频权重每样本加载/释放；CPU媒体单lease；GPU40GiB准入/16GiB安全边界；HPS CPU模型长驻、逐请求onload/offload、124单帧评分。评分/保存回调`observed_score`需在media lease内完成。
- 实读校验：`h3_binding.py`、`h3_lifecycle.py`、`h3_text_cache.py`。checkpoint_boundary及紧邻lifecycle均调用validate_live，含全模型hash/全缓存finite；记录为待审候选，未删除强门禁。conditioning与policy内也有重复复制/设备同步候选。
- 实读 `launch_h3_resolution_memory.py` 与 `launch_h3_media.py`，确认独立两更新审计与观看副本路径，原两步不是新增授权。

## 实测CPU结果：未通过

固定授权原七项命令实际输出：

```text
FAILED tests/test_flash_collection_runtime.py::test_approved140_candidate_compiles_typed_cadence
1 failed, 6 passed, 3 warnings in 46.53s
```

关键栈：测试line1321 `launch.prepare_output(tmp_path / 'owned', identity=identity)` → `scripts/launch_flash140.py:233`，错误 `runtime outputs must stay outside the clean source checkout`。

实际tmp_path位于当前worktree `.pi/workflow-tmp/pytest-of-user/pytest-0/test_approved140_candidate_com0`。门禁把合法的生产源码树防护触发了；需修夹具隔离，不是放松生产规则。已继续读测试1300..1524与`flash_small40_repair.py`：后段还有prepare_output，且夹着真实Git/config历史证明，局部模拟ROOT必须避免污染后者。

`git diff --check`通过（空输出）。测试后`git status --short`为`?? .pi/`；保留其内容、不访问/删除、不改ignore，不声称树clean。相对生产的授权源码diff仍为空。Planner只写自己的计划报告。

## 工具异常与处理

`grep`工具报`spawn rg ENOENT`，未安装依赖/改PATH，改用目录列表与read读取真实文件。猜测测试文件`tests/test_h3_decode.py`不存在，已列tests目录确认实际fixture名`test_h3_mvp_cpu_decode.py`（未将未读fixture当证据）。不重放有副作用命令，不访问workspace外目录。

## 决策

- 本轮独立单元G0：只修测试输出夹具的隔离并增加真实拒绝断言；详见PLAN.md。
- V1复用、P1流水、I1同边界重复验证、F1通信profile仍明确开放；详见CANDIDATES.md。因真实门禁失败优先必要前置，不在一个commit夹带性能优化。
- 无代码commit，没有新性能数据，不能称已提速/2倍达成。保存硬件证据支持否决标准DDP，但不支持把D2 mesh直接替换为已可用方案。
- 不创建FINAL/VALIDATION-PLAN：首次代码循环未开始、多个高收益方向仍待实现/审核。
- 返回implement；下游Coder→Simplifier→Planner复审，原Euler-B继续暂停，GPU/服务操作均未执行。
