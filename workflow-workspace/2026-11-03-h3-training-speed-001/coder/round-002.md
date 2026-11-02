# Coder round 002 — I1 同边界一次完整 live 校验

## 意图与实际实现

起点 `47453a2fe236392e6fdc5bd2e86c82cc166fb724`。按当前 planner/PLAN.md，仅修改 h3_binding.py、h3_lifecycle.py 和原七项测试内部必要回归；未混入 VAE/流水/FSDP 优化。

- 新 `H3RuntimeBinding.checkpoint_snapshot(stages, trainer)` 自己调用一次完整 `validate_live()`，返回 `(原四元组boundary, 该次live)`；没有调用者 verified/token 输入，没有持久缓存或跨边界复用。
- 原 `checkpoint_boundary` 委托新 API 丢弃 live，仍每次完整自校验；保留全部 Adam、scheduler、selection、RNG、proof 与进度构造。
- 长校验前 deepcopy tail/source 状态并记录 next batch；返回前复核 source/prepared 对象所有权、真实 stage tail/source 状态和 next batch，拒绝推进、残留梯度、新 reservation。没有新增 full scan。
- 仅 run 的正常 checkpoint 分支改用新 API，保持原 _phase、memory span、保存条件、transaction 接口与立即同步消费。restore/prepared/bound/finalize 路径无改动。
- `validate_live` 及完整模型/代码 hash、cache finite/身份、HPS/source/model/numerics 验证实现原样保留。没有改变科学协议、保存恢复数学或生命周期所有权。

## CPU 回归与真实检查

在原 `test_approved140_manifest_counts` 内扩展，不增删/skip 七个原函数，不删原断言。

真实 tiny binding/source/stages 完成两轮 CPU 迭代，用委托真实 validate_live 的 spy 检查：每个 snapshot 一次、legacy boundary 一次、同边界重复调用和下一更新均重验；返回 live 是本次真实结果对象。完整 boundary 等价，模型/buffers/Adam/scheduler/gradients/source 及 Python/NumPy/Torch CPU RNG 不变。

覆盖非法 pre-tail、foreign stages、残留梯度、scheduler/Adam counter 错配。真实追加 VAE artifact 字节后下一次 snapshot 被 full model hash 拒绝；还原测试文件后重新验证成功。真实验证成功后注入 cursor 推进和真实 source.reserve(2)，分别被后置一致性门拒绝；reservation 仅测试 teardown abort 并保持 poisoned，不在生产自动回滚。

真实 H3LifecycleBackend.run 的两个保存分支使用 CPU trainer/transaction/transport 叶子，检查一次 snapshot/边界、精确 live/progress/proof/data/contract/identity 传入 transaction；任何额外 validate_live 会直接使测试失败。snapshot 异常经实际 _phase 传播且没有新 transaction 调用。

共享旧 binding fixture 不含生产后来加入的心跳/显存门，初期四次固定门禁分别在未替换 broadcast（53.23s）、tiny CPU 权重被生产24GiB host reserve 拒绝（50.49s）、aux CUDA stream guard（50.85s）及 decoder CUDA-only guard（50.96s）失败，均为 6 passed/1 failed；未改变生产。限于本测试局部补齐 broadcast/file_phase、已有 CPU aux 协调叶子及已有 CPU decoder，并对 tiny 权重预算断言 <10MB 后替代 host telemetry 门。真实 hash/preflight/finite/binding 实现未 mock；未创建 GPU/服务/PG。

首次全通过：7 passed, 3 warnings in 53.61s。随后清理单元素测试循环并加强为真实 reservation 后，最终提交前固定原门禁：**7 passed, 3 warnings in 53.30s**。三条警告仍为原 DCP 单进程保存/加载及 TypedStorage。原当前 worktree 导入断言、真实 Git/config 证明保持通过。

已查看实际累计生产基线 diff：除已提交 G0 外，本轮仅上述三个代码/测试文件。`git diff --check` 空输出通过。原未跟踪 `.pi/` 保留、不读内部、不清理、不提交；不宣称整树 clean。

## 收益：假设而非实测

已重读 inputs/baseline-MEMORY-REPORT.md：第二保存轮 validation 1007.942s，未提供单次 validate_live 计时。若两次 full scan 等时且主导该段，条件估算省约8.4min/保存轮；127.73→119.33min、约1.07× 的分母含采样/评分/优化/校验/保存，不含初始化。正式 cadence5 仅假设摊1.68min/update，含优化与摊销校验保存的111.08min分母约1.5%。这些均沿用 Planner 假设，代码/CPU没有测出实际收益，不能声称 speedup 达成。

没有 GPU、SSH、HPS 请求、服务变更、部署或真实训练。原 Euler B 继续暂停。未建立文件系统原子快照/锁，不能防止任意并发写盘；仅消除紧邻重复扫描，保留每个边界的新鲜完整检查和 transaction 原证明。

## 交接

原子提交标题/正文见 coder/COMMIT_MESSAGE.txt；提交后 SHA 单独记入 coder/round-002-commit.md（避免自引用）。交 Simplifier 独立审查，再由 Planner 审核实际 diff/commit/测试。VAE复用、流水等仍开放；最终 integration/perf发布及授权GPU两更新验收未执行。
