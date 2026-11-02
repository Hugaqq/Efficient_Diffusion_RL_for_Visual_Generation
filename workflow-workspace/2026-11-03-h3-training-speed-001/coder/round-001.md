# Coder round 001 — G0 测试夹具隔离

## 意图与范围

按 planner/PLAN.md 仅修复 managed worktree 内 tmp_path 位于真实源码树时的输出准入夹具冲突。起点 a71ea2f9c91aef79f4a37fb13d6a9fe339078076；本轮不实现训练性能优化。

仅修改 tests/test_flash_collection_runtime.py 的原 test_approved140_candidate_compiles_typed_cadence：
- 在输出生命周期调用区间用 monkeypatch.context 临时将 launch.ROOT 指向 tmp_path/source-checkout，输出为其兄弟目录；仍调用真实 prepare_output。
- owned fresh/retry/resume 生命周期及 run-002 的两次 owner 升级恢复调用均隔离。
- 新增真实 ROOT 内路径拒绝且不创建目录、模拟 ROOT 本身及子路径拒绝且无目录/owner 副作用；每段 patch 后确认真实 ROOT 恢复。
- 保留全部原断言；preset_config 和历史 Git/config 证明在真实 ROOT 下执行，既有显式 fake Git 故障测试保持不变。

生产 visual_rl/、scripts/、依赖和安全门未改；七个原测试函数数量未变。所有测试创建均使用获准 worktree 临时目录；真实 root 拒绝用例不创建输出。

## 实际验证

固定原七项 CPU 命令实际执行结果：**7 passed, 3 warnings in 46.53s**。警告为原有 DCP 单进程保存/加载及 TypedStorage 警告。原 setup 内当前 checkout 的 visual_rl/collection 导入路径断言仍执行通过；真实历史 Git/config 对象证明亦通过。

`git diff --check` 通过（空输出）。相对生产基线的授权源码 diff 仅上述一个测试函数；git status 在测试后为测试文件修改及原有 `?? .pi/`，后者保留、不读取内部、不清理、不提交，不宣称树 clean。

辅助 grep 工具因 `spawn rg ENOENT` 不可用，改用 read 查看测试导入路径断言和真实源码；未安装依赖或更改环境。

## 收益与边界

本轮恢复必要 CPU 门禁，没有训练耗时或 speedup 测量，不给端到端倍数估算。硬件/成本依据已读 TASK、baseline-FINAL.md、baseline-MEMORY-REPORT.md 与 bootstrap-cpu7.txt；本单元不涉及 GPU 驻留/通信优化。未运行 GPU、训练/推理、SSH、HPS 请求或服务变更，Euler B 保持暂停。

提交标题/正文见 COMMIT_MESSAGE.txt；原子提交后的 SHA 将记录在 coder/round-001-commit.md（避免自引用 SHA）。交 Simplifier 独立复审，再由 Planner 选择下一独立优化；GPU 性能仍待另行授权验收。
