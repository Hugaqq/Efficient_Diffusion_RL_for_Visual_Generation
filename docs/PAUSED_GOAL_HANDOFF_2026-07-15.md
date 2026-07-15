# VisualRL Coding Goal 暂停交接文档

更新日期：2026-07-15

本文档用于把当前暂停的 VisualRL C0-C14 Coding Goal 交给新的 Codex 对话继续审阅和收口。它记录的是暂停时的实际代码、测试和审查状态，不代表项目已经最终验收。

## 1. 当前结论

核心功能已经基本实现，整体完成度约为 85%-90%。当前剩余工作不是继续增加功能，而是：

1. 修正一项因 checkpoint v4 完整性校验而过时的测试。
2. 重新完成 C10 Artifact/Checkpoint/Cache 的独立安全审查。
3. 重新完成 C12 DDP/多卡原子更新的独立审查。
4. 跑最终全量离线测试、三个 CPU/gloo 双进程测试、全量 Ruff 和 diff 检查。
5. 根据最终审查结果修复问题，然后更新项目文档和阶段状态。

暂停时没有已确认的新 P0/P1/P2 缺陷，但 C10 和 C12 的最终独立审查都没有完成，因此不能声明 Goal 完成。

## 2. 工作区与安全边界

### 2.1 实际修改目录

所有 Coding 主线修改都在隔离副本中：

```text
/Users/qvanium/.codex/visualizations/2026/05/29/019e733b-5d45-7581-bbf4-ea87605ba6e4/framecode-coding-mainline
```

Git 分支：

```text
codex/c0-c1-mainline-20260714
```

当前修改尚未提交，工作区包含大量预期内的 tracked/untracked 变更。后续对话不得使用 `git reset --hard`、`git checkout --` 或清理未跟踪文件，也不得回退不属于当前修复的改动。

### 2.2 必须保持只读的实验主目录

```text
/Users/qvanium/Desktop/Efficient_Diffusion_RL_for_Visual_Generation/framecode
```

该目录仍可能承载用户实验。后续所有编辑必须继续发生在隔离副本，不能把隔离副本直接覆盖回主目录。

### 2.3 运行限制

- 不下载模型或数据。
- 不访问远程 reward server。
- 不启动 Wan、World-R1 或其他重训练。
- 不使用 GPU。
- 只运行 CPU/offline 单元测试和有界的本机 gloo 双进程测试。
- macOS 沙箱会阻止 loopback socket；三个 gloo 测试需要通过 `require_escalated` 运行。

Python 解释器：

```text
/opt/homebrew/Caskroom/miniconda/base/envs/visual-rl/bin/python
```

## 3. C0-C14 阶段状态

| 阶段 | 内容 | 暂停时状态 |
|---|---|---|
| C0 | deterministic runtime / fingerprint v2 | 已实现并完成前序验证 |
| C1 | TempFlow 数学语义 | 已实现并完成前序验证 |
| C2 | 简洁 Config Resolver | 已实现并完成前序验证 |
| C3 | Preflight 与 CLI | 已实现并完成前序验证 |
| C4 | 最小 image/video 数据契约 | 已实现并完成前序验证 |
| C5 | 外部 Reward 插件 | 已实现并完成前序验证 |
| C6 | Advantage / Objective 拆分 | 已实现并完成前序验证 |
| C7 | Experiment 高层 API 与轻量 Callback | 已实现并完成前序验证 |
| C8 | Wan LoRA | 已实现并完成前序验证 |
| C9 | Flash / World-R1 有界代码闭环 | 已实现；未运行重模型 |
| C10 | Artifact、Checkpoint、Cache 与安全加固 | 最新修复已落地且聚焦测试通过；最终独立审查未完成 |
| C11 | 异步 Reward 执行 | 已实现并完成前序验证 |
| C12 | DDP、多卡与原子 optimizer update | 最新修复已落地；最终独立审查未完成 |
| C13 | scaling policy | 已通过独立复审 |
| C14 | scaling decision 持久化与 resume | 已通过独立复审 |

学习侧练习和笔记已经按用户要求冻结，不属于当前收口阻塞项。

## 4. 暂停前最新实现

### 4.1 C10：ArtifactManager 权威提交语义

主要文件：

```text
visual_rl/artifacts/manager.py
visual_rl/preflight.py
visual_rl/runner.py
tests/test_artifact_transactions.py
tests/test_preflight.py
tests/test_c10_runner_artifacts.py
```

最新修改包括：

1. `ArtifactManager._load_commit_markers()` 改为失败关闭。
   - `commits` 为 symlink 或非目录时抛错。
   - `commit_*.json` 为 symlink、非普通文件、损坏 JSON 或 schema/identity 不合法时抛错。
   - 不再静默跳过损坏的权威 marker。
   - 不同 marker 覆盖相同 artifact step 时抛错。

2. ready journal 中的 checkpoint 摘要成为恢复承诺。
   - 新增 `_ready_checkpoint_expectation()`。
   - 新增 `_validate_recovery_checkpoint()`。
   - ready transaction 重试或 `recover()` 时，重新计算得到的 checkpoint tree SHA256 必须与 journal 中已持久化的值相同。
   - checkpoint 在 journal ready 后、marker 发布前被篡改时，恢复会 quarantine transaction，不会写 commit marker。

3. post-commit 异常边界收紧。
   - `ArtifactManager._recoverable_post_commit()` 只捕获 `Exception`。
   - `ExperimentRunner._run_post_commit_bookkeeping()` 只捕获 `Exception`。
   - `KeyboardInterrupt` 和 `SystemExit` 不再被吞掉，但已经持久化的 commit marker 仍然保留。

4. `ArtifactManager._write_json()` 使用 `allow_nan=False`。

5. `preflight._commit_marker_step()` 对损坏的权威 marker 抛出 `ResumePreflightError`。
   - `latest_committed_step()` 和 `_resolve_committed_checkpoint()` 会先验证全部 matching marker，不再跳过坏 marker 后退到旧 checkpoint。

新增故障注入测试覆盖：

- ready 后 checkpoint 被篡改。
- 损坏 marker 失败关闭并释放 writer lock。
- marker step 重叠。
- marker 持久化后的 `KeyboardInterrupt`。
- NaN、Infinity、负 Infinity JSON。
- preflight 不得跳过损坏 marker。
- Runner post-commit `KeyboardInterrupt`。

这些测试已包含在最近一次 108 项聚焦测试中并通过。

### 4.2 C10：Checkpoint v4 完整性

主要文件：

```text
visual_rl/artifacts/checkpoint.py
tests/test_checkpoint_security.py
```

最新实现：

- `CHECKPOINT_FORMAT_VERSION = 4`。
- `checkpoint.json` 保存 `training_state_sha256`。
- `read_and_validate_training_state()` 在任何 `torch.load`、optimizer/plugin/RNG 状态变更之前校验 `training_state.pt` 摘要。
- v3 继续按安全格式兼容。
- v1/v2 继续保留显式 unsafe-legacy 边界。
- checkpoint JSON 写入使用 `allow_nan=False`。
- 新增 raw-byte tamper、optimizer-state tamper、非自引用摘要和 strict JSON 测试。

实现该部分的 worker 报告：34 项 checkpoint 聚焦测试和 5 项 resume/preflight probe 通过，Ruff 通过。父线程随后把它与其他 C10 测试合并运行，108 项聚焦测试通过。

### 4.3 C10：Rollout cache v2 generation 协议

主要文件：

```text
visual_rl/rollout/cache.py
tests/test_rollout_cache.py
```

最新实现：

- `CACHE_VERSION = 2`。
- 每次 save 产生新的 generation id。
- tensor payload 和 media payload 都包含 generation。
- metadata 保存 tensor/media SHA256，并且最后原子发布，作为当前 generation 的权威记录。
- load 时要求 metadata、tensor、media 的 version、generation 和 digest 全部匹配。
- 并发写同一步最多导致某次读取失败关闭，不允许返回跨 generation 混合 batch。
- 保留 v1 和无版本 legacy cache 读取。
- 保留 `weights_only=True`、symlink/path 防护和 truncate 行为。

worker 聚焦测试 23 项通过；与其他 C10 测试合并后 108 项聚焦测试通过。

尚未做独立的 power-loss/fsync fault injection。它是最终审查应评估的残余风险，不应在没有证据时宣称已经覆盖。

### 4.4 C12：DDP 原子更新与 GradScaler

主要文件：

```text
visual_rl/distributed.py
visual_rl/runner.py
visual_rl/optimizers/update_engine.py
visual_rl/optimizers/algorithm_plugin.py
tests/test_distributed.py
tests/test_distributed_runner.py
tests/test_distributed_update.py
tests/test_grad_scaler_state.py
```

已经落地的修复：

1. Runner 用 `strategy.atomic_optimizer_step(...)` 包住整个外部 `OptimizerPlugin` update，而不是依赖 plugin 主动调用内部 rollback callback。
2. snapshot/restore 覆盖 model parameters、optimizer state、GradScaler state 和任意 stateful plugin state。
3. 任一 rank 出现可捕获的 optimizer/update 失败时，所有 rank 同步失败并回滚。
4. collective 在执行前验证 operation 和 root rank 的跨 rank consensus。
5. FP16 GradScaler 改为跨 step 持久化，并进入 plugin checkpoint state。
6. 同目录 in-place resume 不允许使用比最新权威 marker 更旧的 fallback checkpoint；分支到新 output directory 仍可显式使用旧 checkpoint。

前序验证：

- 相关非 loopback 测试曾达到 70+ 通过，loopback 3 项曾全部通过。
- 最新 C10 修改不直接触碰 distributed/update 文件，但最终验收仍必须重新跑三项 loopback 测试。
- C12 最终独立 reviewer 在返回结论前按用户要求被终止，所以当前不能签署 PASS。

### 4.5 C13/C14

已修复并经过独立复审：

- out-of-place branch resume 从 source run 读取并复制 scaling decision，而不是错误读取 destination。
- 在配置 runtime、构建 model/optimizer、写 output 前先验证 source decision。
- 同目录旧 checkpoint fallback 不得越过更新的 marker；分支输出仍允许旧 checkpoint。

独立复审证据：21 项聚焦测试和 Ruff 通过，无待处理 finding。

## 5. 最新测试状态

### 5.1 已通过

暂停前最新 C10 聚焦命令结果：

```text
108 passed in 2.45s
```

覆盖文件：

```text
tests/test_artifact_transactions.py
tests/test_checkpoint_security.py
tests/test_rollout_cache.py
tests/test_preflight.py
tests/test_c10_runner_artifacts.py
```

最新 scoped Ruff 结果：

```text
All checks passed!
```

### 5.2 全量测试的唯一已知失败

最近一次非 loopback 全量结果：

```text
677 passed, 1 failed, 3 deselected in 12.86s
```

失败测试：

```text
tests/test_visual_rl.py::test_checkpoint_fingerprint_versions_fail_closed
```

失败原因不是产品实现回归，而是测试仍按旧 checkpoint 语义直接改写 `training_state.pt`。checkpoint v4 会先检测到 `training_state_sha256` 不一致，因此测试无法继续走到它原本想验证的“未知 config fingerprint version”分支。

建议的最小测试修复：

1. 在 `tests/test_visual_rl.py` 导入 `hashlib`。
2. 每次该测试执行 `torch.save(state, state_path)` 后，重新计算：

   ```text
   hashlib.sha256(state_path.read_bytes()).hexdigest()
   ```

3. 把结果写入内存中的 `metadata["training_state_sha256"]`。
4. 重新写 `checkpoint.json`。
5. 第一段继续期待 `Unsupported config fingerprint version`。
6. 第二次把 state version 改回 2 后，也必须重新更新 SHA256 并写 metadata；保留 metadata version 为 99，才能继续验证 state/metadata version 不一致。

不要降低或绕过 checkpoint v4 摘要校验来迁就旧测试。

### 5.3 暂停前没有完成的测试

- 修复上述测试后的单项重跑。
- 修复后的完整非 loopback suite。
- 三个双进程 CPU/gloo 测试的最终重跑。
- 全仓 `ruff check`。
- `git diff --check`。
- 可选的 `compileall`。

## 6. 独立审查状态

### 6.1 C10 reviewer

Agent：`019f6180-5c70-7b13-94db-399667b21f8a`

用户要求暂停后，该 reviewer 立即停止。它的当前结论是：

- 暂无已确认 P0/P1/P2 finding。
- 初步确认严格 marker、ready digest、`training_state_sha256`、`allow_nan=False`、cache generation/digest 和 `weights_only=True` 等机制存在。
- 尚未逐行审查故障路径、测试覆盖、兼容性，也没有独立运行测试和 Ruff。
- 因此不能给 C10 PASS。

### 6.2 C12 reviewer

Agent：`019f6180-811a-7611-a858-2eada388c324`

该 reviewer 在形成报告前被停止，没有可用结论。后续对话必须重新发起一份完整、只读、独立 C12 审查。

## 7. 建议的恢复顺序

### 步骤 1：确认隔离工作区

```bash
cd /Users/qvanium/.codex/visualizations/2026/05/29/019e733b-5d45-7581-bbf4-ea87605ba6e4/framecode-coding-mainline
git branch --show-current
git status --short
```

预期分支为 `codex/c0-c1-mainline-20260714`。不要清理 dirty worktree。

### 步骤 2：只修复已知的过时测试

编辑：

```text
tests/test_visual_rl.py
```

按第 5.2 节更新两次被改写 state 对应的 `training_state_sha256`。先运行：

```bash
env PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  DIFFUSERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_ADDOPTS='-p no:cacheprovider' \
  HF_HOME=/tmp/visualrl-hf-cache \
  XDG_CACHE_HOME=/tmp/visualrl-xdg-cache \
  /opt/homebrew/Caskroom/miniconda/base/envs/visual-rl/bin/python -m pytest -q \
  tests/test_visual_rl.py::test_checkpoint_fingerprint_versions_fail_closed
```

### 步骤 3：重跑完整非 loopback suite

```bash
env PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  DIFFUSERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_ADDOPTS='-p no:cacheprovider' \
  HF_HOME=/tmp/visualrl-hf-cache \
  XDG_CACHE_HOME=/tmp/visualrl-xdg-cache \
  /opt/homebrew/Caskroom/miniconda/base/envs/visual-rl/bin/python -m pytest -q tests \
  --deselect=tests/test_distributed.py::test_ddp_two_rank_cpu_smoke \
  --deselect=tests/test_distributed.py::test_ddp_object_collective_contract_consensus \
  --deselect=tests/test_distributed_runner.py::test_two_rank_runner_resume_artifacts_metrics_and_failure
```

### 步骤 4：重新发起独立 C10 审查

审查范围：

```text
visual_rl/artifacts/manager.py
visual_rl/artifacts/checkpoint.py
visual_rl/rollout/cache.py
visual_rl/preflight.py
visual_rl/runner.py 的 post-commit 路径
tests/test_artifact_transactions.py
tests/test_checkpoint_security.py
tests/test_rollout_cache.py
tests/test_preflight.py
tests/test_c10_runner_artifacts.py
```

必须给出按 P0/P1/P2 排序的 findings 或明确 PASS。重点检查：

- 坏 marker 是否在 constructor、recover、preflight 全部失败关闭。
- ready journal 是否可能被重新计算的摘要覆盖。
- checkpoint digest 是否确实在任何反序列化/状态变更前验证。
- cache 并发写是否只会返回完整 generation 或失败。
- symlink/path traversal/TOCTOU 边界。
- `KeyboardInterrupt` 与普通 post-commit 异常的不同处理。
- v1/v2/v3/v4 checkpoint 和 unversioned/v1/v2 cache 兼容边界。
- power loss 和 directory fsync 是否需要补强，或应作为明确残余风险记录。

### 步骤 5：重新发起独立 C12 审查

审查范围：

```text
visual_rl/distributed.py
visual_rl/runner.py 的 distributed/update 路径
visual_rl/optimizers/update_engine.py
visual_rl/optimizers/algorithm_plugin.py
tests/test_distributed.py
tests/test_distributed_runner.py
tests/test_distributed_update.py
tests/test_grad_scaler_state.py
```

必须验证：

- 外部 plugin 完全绕过内部 callback 时，整个更新仍被 outer atomic boundary 保护。
- model、optimizer、scaler、plugin state 的 snapshot 和 restore 顺序正确。
- rank-local failure 能在所有 rank 上形成一致错误，不会死锁。
- operation/root consensus 不引入额外 collective 次序不一致。
- GradScaler 跨 step 和 checkpoint resume 持久化。
- single-process path 不改变原有行为。

### 步骤 6：运行三项 loopback 测试

需要沙箱外 loopback 权限：

```bash
env PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  DIFFUSERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_ADDOPTS='-p no:cacheprovider' \
  HF_HOME=/tmp/visualrl-hf-cache \
  XDG_CACHE_HOME=/tmp/visualrl-xdg-cache \
  /opt/homebrew/Caskroom/miniconda/base/envs/visual-rl/bin/python -m pytest -q \
  tests/test_distributed.py::test_ddp_two_rank_cpu_smoke \
  tests/test_distributed.py::test_ddp_object_collective_contract_consensus \
  tests/test_distributed_runner.py::test_two_rank_runner_resume_artifacts_metrics_and_failure
```

### 步骤 7：最终静态检查

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/visual-rl/bin/python -m ruff check .
git diff --check
```

如需 compile check：

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /opt/homebrew/Caskroom/miniconda/base/envs/visual-rl/bin/python -m compileall -q visual_rl tests
```

### 步骤 8：最终广域审计

在 C10/C12 专项 PASS 后，再安排一个新的只读 reviewer 检查整个 C0-C14 diff 的跨模块契约，尤其关注：

- config resolver -> preflight -> runner 的顺序。
- RolloutBatch/RewardBatch/sample identity 对齐。
- async reward 失败/取消与 artifact transaction 的关系。
- optimizer plugin state 与 checkpoint v4 的关系。
- out-of-place resume 的 source/destination artifact 边界。
- public API 是否仍然只有一条主线，legacy 是否只是兼容层。

只有专项审查、广域审计、全量测试、三项 gloo 和静态检查全部通过后，才能更新计划为全部完成并关闭 Goal。

## 8. 暂停时不得宣称的事项

在完成第 7 节前，不得声称：

- C10 已通过独立安全审查。
- C12 已通过独立多卡审查。
- 全量 test suite 为全绿。
- checkpoint/cache 已覆盖真实断电或所有 TOCTOU 场景。
- Wan/World-R1 重模型训练已验证。
- 整个 C0-C14 Goal 已完成。

当前准确表述应为：核心 Coding 主线基本完成；最新 C10/C12 修复已落地，108 项聚焦测试通过；非 loopback 全量测试仅剩一项需要随 checkpoint v4 更新的测试；最终专项审查和 gloo 回归尚未完成。

## 9. Goal 状态

Goal ID 对应当前 Codex task：

```text
019e733b-5d45-7581-bbf4-ea87605ba6e4
```

暂停时 Goal 仍为 active，未标记 complete，也不应标记 blocked。用户只是要求暂时停止并生成交接文档。
