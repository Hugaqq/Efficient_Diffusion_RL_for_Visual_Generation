# Assignment C — W7 真实binding/data与publisher组合验收

- active: true
- task_id: H3-M4b3-bound-checkpoint-integration
- pair: coding-c → simplifier-c
- execution: parallel W7；decisions/wave-07.md
- integration_parent: `d7c75b75210782a63a4fffcae117f2ace19fa760`
- dependencies: 已批准profile/source/stages/codec/transaction（C-PUBLISH-01已关闭）；W6 binding合法constant-scheduler公开接口已实现但因A-BIND-01尚未批准。本项只验证合法输入组合，不替代A复审。

## 唯一目标与路径

新增 `tests/test_h3_mvp_cpu_bound_checkpoint.py` 与本task_id交付报告；**本波不改任何生产文件或已有tests**。测试真实H3RuntimeBinding + H3PromptSource + H3IterationStages/BaseTrainer完整tail产生的contract/identity/progress/data，接已批准H3CheckpointTransaction/AtomicCheckpointManager，fresh对象两恢复门后实际下一轮。

这是关闭“publisher仍用fixture contract/dummy data、binding从未publish”的组合证据缺口；不是默认RunController测试或另造runner。只能使用当前公开binding API，不能等待/import A本波新增符号，不改A测试fixture签名，不mock binding/source/checkpoint逻辑来绕过问题。若发现生产组合缺陷，列精确复现及owner/最小路径需求回orchestrator，不越界修代码。

## 冻结输入与测试要求

1. 使用真实tiny模型完整文件/内容hash、cache/profile/source、真实最终prepare/Adam/合法constant scheduler、已acquire的真实HPS wrapper。沿用已审查重模型/VAE/HTTP/通信/CUDA叶子替身，仅叶子可替；不能用旧 `_contract/_identity/_data` fixture代替绑定生成的身份/完整source state。允许复用已存在只读低层测试helper；本文件自己持有组合fixture。
2. 至少一轮真正stages/trainer更新并完整提交tail，调用binding.checkpoint_boundary获得实际progress/proof/data；八rank协议必须以对应rank的真实source/边界资料构造。检查所有rank共同身份一致，rank-specific私有RNG表有独立来源而非强令相同。
3. 实际transaction.checkpoint及真实磁盘DCP/manager inspect_complete执行，保留allrank codec在单writer之外证据。CPU无法真实分布式DCP时，可沿用既有明确no-dist文件传输叶子（rank0写相同CPU副本，其余参与协议），但须断言所代表模型状态一致；不能把IO变成内存dict或把副本当真实分片。
4. 构造全新adapter/prepare/空Adam/source/binding/transaction/trainer；调用restore_prepared恢复模型/moments/scheduler但不恢复数据/RNG；restore_bound重新调用真实binding.validate_live，source.load_state_dict与实际回读，RNG最后。不得先恢复source来跳过第二门，不拿旧binding/live缓存冒充fresh身份，不dummy step。
5. 比较连续路径和fresh恢复路径下一轮的真实request IDs/prompt/condition/初态/显式双RNG、完整stages返回、参数/Adam/scheduler/data/behavior/next-step。继续运行真实下一轮而不是仅固定手工loss更新；真实全局RNG恢复可另用串行rank case，避免八线程争抢单进程全局RNG。清晰区分模拟私有进程状态与真实restore_current证据。
6. 至少一个两门之间实际cache/source或完整checkpoint文件漂移拒绝用例，以及cold/未完成tail不会进入codec的组合检查。原保护不削弱；合法路径不依赖尚待A修的错误scheduler。

## 生命周期接口问题须显式暴露

现RunController顺序是prepare→restore_prepared→transform→bind→restore_bound。当前transaction第一门需要expected_contract/identity/live，binding构造又需要实际prepared/source/scorer。测试须记录这些真实对象何时可建立，不用伪G3 receipt解决顺序。第二门read_data必须从source实际state形成envelope，不能调用仅允许新trainer已完成tail的checkpoint_boundary来伪装fresh恢复tail。如果现API无法合法组合，报告具体失败与最小owner修复请求；不通过手改私有trainer/source计数或删检查使测试通过。

## 门禁/复审

最后编辑后plan Flash七项/H3完整通配符，两集真实执行；无skip/xfail/全量/过滤warning。报告branch/HEAD、命令/exit/数量/时间/diff/check、实际组合与替身边界和下游唯一controller顺序建议。简化审查只审本测试及其所证明事实，不能据合法组合成功批准A-BIND-01。发现其他owner缺陷回orchestrator配对修复。GPU_PENDING。
