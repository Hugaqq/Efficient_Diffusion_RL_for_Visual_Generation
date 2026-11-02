# Simplifier round 002 — I1 独立复审

## 结论与提交

审查当前真实实现后，I1 符合本轮 PLAN 的窄范围与校验合同；未发现需要修复的正确性缺口或值得另做的必要等价简化。**不改代码/测试，不创建 Simplifier commit**。保留 deepcopy 和后置边界复核，不将安全门当作冗余删除。交 Planner 复审，非工作流完成声明。

当前 HEAD 为工作流快照 7b1a49fbee603deefd54534b99dc07bc855ca32d；实际 git log 确认 Coder 原子提交 a211cbf，父提交 47453a2。生产基线 3db2a05e1c464bd8cbd97c0290a094d869a9d1fd 的真实累计 diff 已检查：G0 测试夹具之外，I1 仅 h3_binding.py、h3_lifecycle.py 和原 test_approved140_manifest_counts 内必要回归，没有 transaction/schema/配置/依赖修改。

## 真实审查

重新读取 TASK.md、当前 planner/PLAN.md、Coder round-002 报告、存证 baseline-MEMORY-REPORT.md，以及当前 binding、lifecycle、stage tail、dual stage 继承关系、source state 和 transaction 数据/身份校验源码；不只依赖上游总结。

- checkpoint_snapshot 自己调用完整 validate_live 一次，返回该次 live 对象；无 caller token/verified 开关，无持久缓存。旧 checkpoint_boundary 委托并保留原四元组、自校验，不复制构造实现。
- validate_live 的完整内容/hash、资源/source/model/numerics 门未改。Adam/scheduler、selection provenance、RNG capture、progress/proof/data 构造仍保留。返回前后所有权检查及 stage tail/source/next batch 比较有效，独立 deepcopy 不会把可变嵌套引用当历史证据。
- 实读 H3IterationStages.checkpoint_boundary：检查所属 trainer、无 active/poison、已完成 collection、step/cursor/behavior 和清空 gradients；dual stage 继承此实现。source.state_dict 拒绝 pending/poison，返回独立 JSON 状态。因此长验证后新 reservation 或推进仍被拒绝。
- lifecycle 仅在原保存分支、原 memory span 和 _phase 内改为新 API；立即向原 transaction 传 live/progress/proof/data。保存 cadence、pause、first/terminal 特例及 restore/finalize 路径未改。失败通过原 _phase 传播，不新增保存调用或吞异常。
- 资源驻留/队列/取消/排空：没有新线程、queue、异步或 GPU 常驻对象；额外内存仅小型边界 JSON 的局部副本，无视频/模型复制。未改变租约、取消、pool 释放或 PG 生命周期。
- 样本身份、dtype、RNG、冻结策略与更新边界：模型/评分/采样/更新源码无 diff；RNG 为原只读 capture，未引入随机调用或额外 optimizer/scheduler step。transaction 原 JSON 大小/身份/进度及参数证明保留。
- 本实现只面向同步立即消费，不是任意并发文件修改的原子快照/锁；不扩大其保证。两次全扫描移除一次并不等于已经测得 IO 或 GPU 收益。

## 必要测试复核与独立执行

实际 diff 中回归保留原七函数与所有原断言。真实 tiny binding/scan spy 覆盖两轮边界、同边界重复、新旧 API 次数和返回对象；检查 prepared/optimizer/source 与 Python/NumPy/Torch CPU RNG 不变。实际 VAE 文件字节漂移拒绝后恢复成功；非法 tail/foreign/grad/counter/scheduler 拒绝；真实校验后 cursor/reservation 漂移拒绝。真实 lifecycle.run 两次保存验证精确对象传递、禁止第二次 validate_live，snapshot 错误不保存。测试叶子替换只用于 tiny CPU/CUDA/transport，不把这些结果当分布式 GPU 证明。

独立固定原七项 CPU 命令实际：**7 passed, 3 warnings in 52.76s**。警告仍为 DCP 单进程保存/加载与 TypedStorage。当前 worktree 导入路径断言和真实 Git/config 证明包含于通过的原门禁。`git diff --check` 实际空输出通过。

一次读取 h3_source.py 使用超出文件末尾的 offset，工具明确返回文件共278行；已改读190起至末尾，确认 source 边界实现，无副作用重试。

初始 git status 仅 `?? .pi/`，保留不读取内部、不清理、不暂存。本角色仅新增本报告，最终状态以命令复核，不宣称 clean。

## 剩余事项

- Planner 独立审核 a211cbf、本轮实际 diff 与原七项结果，再决定下一独立优化。VAE 复用/有界解码评分流水仍开放，不因 I1 完成而宣布收益耗尽。
- 存证 checkpoint_validation 第二轮1007.942s 是整个阶段，不是单次扫描计时；未实测本改动收益，不给已达成 speedup 或新耗时预测。
- 标准 DDP 静态超64GiB，reserved 不代表活跃张量，不据此猜并发驻留余量；硬件按已有证据，不按型号推互联。
- 最终 integration/perf 发布及 GPU 两更新验证未执行，后者需另行授权。本角色未运行 GPU、SSH、HPS/服务、训练/推理/部署；原 Euler B 保持暂停。
