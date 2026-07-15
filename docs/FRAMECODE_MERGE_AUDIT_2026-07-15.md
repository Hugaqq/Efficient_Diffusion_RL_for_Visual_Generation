# 两份 Framecode 合并审计

更新时间：2026-07-15

## 审计对象

- 实验工作区：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_for_Visual_Generation/framecode`
- C0-C14 Coding 副本：`framecode-coding-mainline`
- 共同 Git 基线：`ff5ab70`

实验工作区是实验脚本、recipe、结果和远端证据的超集；Coding 副本包含 C0-C14 的新 CLI、Resolver、Preflight、Experiment API、artifact transaction、checkpoint v4、cache v2、异步 reward、DDP 与 scaling 主线。两侧有三十余个共同修改文件，因此不能目录覆盖。

## 合并所有权

| 范围 | 合并基线 | 处理方式 |
|---|---|---|
| C0-C14 产品代码 | Coding 副本 | 作为主线导入，再移植实验侧经过验证的语义 |
| `experiments/` | 当前实验工作区 | 整体保留；只选择性提交 recipe、validator、摘要和小型结果 |
| `tests/test_visual_rl.py` | 当前实验工作区 | 保留实验合同；适配 checkpoint v4 与新安全边界 |
| `evaluation/cross_run.py` | 当前实验工作区 | 保留 execution 与 pixel guardrail 分离 |
| Prompt/TempFlow/Flash/Wan 细节 | 语义合并 | 保留 Coding 安全合同，并回放实验已验证行为 |
| `exercises/` | 不进入合并 | 保持整个目录忽略，不提交文件或依赖测试 |

## 审计发现与修复状态

### P0：不安全 artifact 反序列化（已修复）

实验侧旧 `visual_rl/artifacts/audit.py` 会在可信根、权威 commit marker 和摘要校验前，对 artifact 调用 `torch.load(..., weights_only=False)`。被篡改的 artifact 可执行任意 pickle 载荷。

已实现：审计入口只解析严格 JSON；先限制路径位于 run root、校验权威 marker 与 checkpoint tree digest，再调用受信训练状态 reader。普通不受信 pickle load 已从 audit 路径移除，并用“篡改时零 state/load 调用”回归锁定顺序。

### P1：权威 marker 摘要未在 resume 时重新比对（已修复）

Coding 副本会把 checkpoint tree SHA256 写入 ready journal 和 commit marker，但恢复已提交 checkpoint 时只校验摘要格式，没有重新计算实际 tree。checkpoint 与内部 metadata 被一致替换时，marker 的承诺没有真正生效。

已实现：preflight、resume、audit 和 status 对 committed checkpoint 都重新计算 tree digest 并 fail closed；合法的 marker 前崩溃由 transaction recovery 恢复，post-commit tree tamper 则拒绝。

### P1：异步 reward timeout 只是软超时（已修复合同边界）

运行中的线程无法被 Future cancellation 强制停止，同一个 provider/session 也没有统一线程安全合同；永久阻塞的外部插件可能阻止进程退出，网络层和 executor 层重试还可能发生乘法叠加。

已实现：provider 默认串行，只有显式声明并发安全才并行；线程 provider 不允许声明 hard timeout，要求时 fail-fast；每个 handle 有协作取消信号，timeout/close 后排队 shard 不再调用 provider，重试预算与 timeout guarantee 显式化。真正无法协作取消的第三方 scorer 仍需后续进程隔离。

### P1：状态文件不是 artifact 权威性证明（已修复）

旧 `run_status.json` 依赖 PID 存活，存在 PID 复用、DDP 多 rank 竞争、原始异常文本泄露 URL，以及 completed 状态未验证 marker/artifact 的问题。

已实现：只由 rank 0 写状态，异常先脱敏；聚合 readiness 同时验证 completed 状态、权威 commit marker 与 marker 对应 tree digest；PID 只保留为诊断信息。

### P2：文件系统与解析边界（已修复）

- `recover()` 在遍历 `.staging` 前拒绝 staging-root symlink。
- 权威 JSON 读取拒绝 duplicate keys 与 `NaN/Infinity`，关键替换后执行父目录 fsync。
- checkpoint v1/v2 到 v4、manifest v1 到 v2 使用显式 migration；未知版本 fail closed，不静默降级。
- CLI 恢复顺序调整为：静态可信校验、共享 recovery、最终校验、Runner resume，避免在合法 crash-before-marker 状态上过早失败。

### P2：DDP 原子回滚成本与能力边界（本地合同已修复，远端待测）

C12 每 step 会快照 trainable parameters、optimizer、plugin 和 GradScaler state。LoRA 可能可接受，但 full-model 或大 optimizer state 会增加显存/主存和 step latency；rank 硬退出或 collective hang 也不能由 Python rollback 普适处理。

已实现 snapshot bytes/time 与默认 1 GiB fail-fast 预算；参数、optimizer、plugin、GradScaler 校验处于同一原子边界，跨 rank 错误会同步后回滚。本地真实双进程 Gloo 5 项通过；非末尾有效 microbatch 使用 `no_sync()`，通信次数与 full-batch oracle 一致。远端仍需测 GPU/NCCL 正确性、峰值显存和通信成本；保证只覆盖“可捕获且所有 rank 能到达同步点”的更新失败。

### P1：Wan recompute 提前量化 SDE latent（已修复并远端验证）

旧实现先把 rollout current/next latent 转成 transformer BF16，再 `.float()` 交给 SDE，导致原始 FP32 状态先发生不可逆舍入。`2adfbfd` 将 transformer model input 与 SDE 状态分离：transformer 继续接收 BF16，SDE current/next latent 保留 rollout 的原始 FP32 值；非 BF16 精确可表示输入的单元回归已锁定该合同。

远端 W7b 严格异构 Flash parity 已通过 sample/tensor、selected index、grouped seed、scheduler、recomputed log-prob、loss 和全部 480 个 gradient 精确门槛。W5 attempt_1 因实验 harness 在 Runner 配置 deterministic runtime 前初始化 CUDA 而在训练前失败，证据已保留；修正 harness 顺序后的 attempt_2 中，World-R1 与 Flash 的 continuous 2-step 对 split 1-step + resume-to-2 exact comparison 均全部通过。该结论只覆盖 Wan correctness，不覆盖 SD3、HPS 训练效果、质量或速度。

同一提交上的 reward_general attempt_1 也严格通过：HPS direct reference、loopback HTTP 与 infra 三路分数逐项相同，无效 payload 和内部故障均返回 500，且未发生 silent fallback。其 `legacy_pickle` 兼容协议被限制为 loopback 可信主机；这只证明受控 legacy score path 的正确性和 fail-closed 边界，不证明公网协议安全、训练效果或服务性能。

### P2：测试与可维护性（回归已闭合，拆分仍待办）

- 合并前 Coding 非 loopback 基线的 checkpoint v4 过时测试已修复。
- `2adfbfd` 上完整 non-distributed 为 883 passed、2 skipped、5 deselected；真实双进程 Gloo 为 5 passed、885 deselected。
- Ruff、compileall 与 `git diff --check` 全部通过。
- `runner.py`、artifact manager、checkpoint 和 Wan adapter 均已接近 1.7k-1.9k 行，后续需按 step execution、resume coordination、artifact projection 和 identity 拆分。
- 仍缺静态类型检查基线、统一 CI/coverage/timeout 配置；distributed marker 已建立，GPU/NCCL marker 与远端调度仍需标准化。

## 合并与验证计划

1. 保护实验树和合并前 tracked patch，创建独立 integration 分支。
2. 导入 Coding 主线，明确排除 `experiments/**`、`exercises/**` 和 learning-scaffold test。
3. 并行完成 artifact/runner 安全修复、实验语义白名单移植、C11/C12 review。
4. 修复当前公共合同与 checkpoint v4 过时测试。
5. 运行 artifact/C11/C12 聚焦测试、完整 non-loopback、三项 gloo、Ruff、diff check 和 compile check。
6. 本地全绿后运行远端 correctness gate：Wan W5/W7b 与 reward_general HPS direct/HTTP/infra parity 已完成；SD3 deterministic resume 仍待完成，再按 P3/O4/D3、P2/P5 和新高语义 bounded effectiveness 顺序推进。
7. 提交拆分为：C0-C14 主线导入；安全/一致性修复；精选实验材料；文档与实验总结。所有提交排除 exercises 与大体积运行产物。

实验层完整结论与远端队列见 [`EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md`](../experiments/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md)；Wan `2adfbfd` 的证据边界与机器可读索引见 [`WAN_RESULTS_2adfbfd.md`](../experiments/postmerge_validation_20260715/WAN_RESULTS_2adfbfd.md)。
