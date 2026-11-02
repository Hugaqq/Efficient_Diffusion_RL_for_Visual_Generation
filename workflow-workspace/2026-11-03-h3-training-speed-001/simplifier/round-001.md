# Simplifier round 001 — G0 独立复审

## 结论

G0 测试夹具隔离符合本轮计划；未发现需要修复的正确性缺口或有明确维护收益的等价简化。**不改源码/测试，不造 Simplifier commit**。交 Planner 复审并选择下一独立单元，不表示整个工作流完成。

已读 TASK.md、AGENTS.md、planner/PLAN.md、Coder 报告/收据，以及 inputs/baseline-FINAL.md、baseline-MEMORY-REPORT.md、sharding-estimates.json、bootstrap-cpu7.txt。硬件/阶段成本按存证理解，不根据 GPU 型号猜互联；标准 DDP 静态已超 64GiB。本轮无性能改动，无 speedup 声明。

## 实际源码/diff 检查

- 当前 HEAD b8f4ade5b4896d1e9ba24349994e80e705ad5499 为工作流快照；git log 确认 Coder 原子提交 e324aaf（父 a71ea2f），标题为 `fix(tests): isolate launcher output fixtures in managed worktrees`。
- 实际执行相对生产 3db2a05e1c464bd8cbd97c0290a094d869a9d1fd 的授权源码 diff：仅 tests/test_flash_collection_runtime.py 原 test_approved140_candidate_compiles_typed_cadence 改动，visual_rl/ 与 scripts/ 无 diff。不是仅依据 Coder 总结。
- 实读测试相关完整函数及 scripts/launch_flash140.py 的真实 prepare_output：ROOT 禁令先于目录创建/owner 发布；生产安全门未改。
- 模拟源码根与 owned/run-002 输出均为 tmp_path 下兄弟路径；patch 仅覆盖真实 prepare_output 调用，退出 context 后断言真实 ROOT 恢复。真实 preset_config、Git/config 证明不在模拟 ROOT 下运行；原有 fake Git 故障注入仍保留。
- 原 fresh/retry、owner mismatch、显式 resume、latest、orphan 保留、terminal 拒绝，以及 owner 升级前后调用/审计拒绝断言均保留。新增真实 ROOT 内路径、模拟 ROOT 本身及子路径拒绝且不创建目录的覆盖。无需为三段局部 context 抽象 helper：现有作用域直接可见，额外抽象无必要收益。
- 资源驻留、队列上界、错误/取消/排空：本 diff 不引入队列、线程、异步任务或设备驻留；未改生产生命周期/异常传播。局部 ROOT 通过 monkeypatch.context 在异常时亦恢复，不吞异常。
- 样本身份、dtype/RNG、冻结行为策略、梯度/更新边界与 checkpoint 数学：生产实现零改动，本单元只隔离输出准入夹具，不影响上述协议。此结论不等于已经验证未来流水线的资源安全或 GPU 数值等价。

## 独立实际验证

运行规定原七项 CPU 命令，实际结果：**7 passed, 3 warnings in 45.90s**。无 skip/替换测试；警告为 DCP 单进程保存/加载及 TypedStorage 弃用。实读 setup 内当前 worktree visual_rl/collection 导入路径断言，整套运行通过；真实历史 Git/config 证明通过。

`git diff --check` 实际空输出通过。初始 `git status --short` 仅 `?? .pi/`；保留该未知/工具临时资料，不读取内部、不清理、不暂存。本角色只新增本报告，未执行 git add/commit。最终状态另以实际命令检查，不宣称 clean。

## 剩余事项交 Planner

1. 独立复核 e324aaf 与本轮原七项证据；G0 是必要测试前置，不是性能优化。
2. 按计划选择 VAE 复用等下一独立显著收益单元；未来有并发时仍需明确队列有界/背压、样本绑定、错误取消排空及释放、参数更新屏障，不以本轮 CPU 通过替代。
3. 最终 integration/专用 perf 分支发布尚未执行；真实 GPU 性能/两更新验证仍需另行授权，无训练/推理/SSH/HPS/服务操作；原 Euler B 不恢复。
