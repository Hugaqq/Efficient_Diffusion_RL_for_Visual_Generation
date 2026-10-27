# Assignment C — W4

- active: true
- task_id: H3-M4b1-state-codec
- pair: coding-c → simplifier-c
- execution: parallel W4；decisions/wave-04.md
- integration_parent: `7cf28dced4c8193ccf9321d62b5b2e479bd904f7`
- dependencies: 现H3PreparedModel/coordination与W3六阶段已配对approved，代码汇合5397c46662db30ef535afc0fb4b192df019c4d62；不依赖B新adapter或A组合测试。

## 单一目标/范围

实现**H3已prepare模型/Adam/scheduler/必要buffer的全rank DCP state codec及fresh optimizer恢复**，给现checkpoint manager下一项整合使用。它只写调用者提供的共同staging内状态，**不自己发布complete/latest/终态、不造第二checkpoint manager**。

仅新 `visual_rl/artifacts/checkpoint/h3_state.py`、新 `tests/test_h3_mvp_cpu_checkpoint_state.py`、W/coding-c/H3-M4b1-state-codec.md。旧manager/protocol/transaction/state/RNG、prepare/coordination/旧tests/A/B模块只读。若必须修旧API先给具体请求，不能自造备用manager或整段vendor。

## 真实codec合同

- 生产仅调用者已初始化NCCL WORLD8、same topology；prepared是实际H3PreparedModel、最终Adam对象/参数集合严格匹配。使用真实公开 `torch.distributed.checkpoint` save/load 与model/optimizer state_dict API，lazy能力检查。全rank都进入getter/save/load/set，不能把潜在collective藏在rank0 writer callback。
- 接收共同staging路径、明确expected topology/模型/语义身份以及现CheckpointSafePoint/CheckpointProgress或同样严格的已有proof；验证完整committed step>0、无open data reservation/dynamics/reward、非poison、GA0/梯度已清，safePoint与progress一致。scope只模型状态，caller后续必须在原BaseTrainer整轮返回/next-step推进后供真实proof；不能凭optimizer.step已调用就声称完整tail安全。
- **cold save必须在任何可能有副作用的optimizer helper之前拒绝**：每一个期望trainable Adam state都已有有限step>0/exp_avg/exp_avg_sq及正确全局shape/dtype/参数组。部分缺state/错误FQN/旧参数对象/未清梯度拒绝；不得人工零梯度step填moments，不strict=False或跳过missing。
- 保存完整model state（含冻结base和持久buffer）作为最小正确版本，不为节约空间引入LoRA-only恢复。分片保留分片，不allgather成完整每rankCPU/GPU权重。非持久rope.inv_freq必须单独保存/核验或绑定可靠重建值，不漏；scheduler state与None-schema精确区分。
- capture/getter必须无训练副作用：参数、Adam moments/step/组超参、scheduler、梯度以及RNG不因snapshot改变。同步DCP写完后才返回小codec receipt；不把DTensor塞普通torch.save来冒充DCP。返回receipt/元数据只供外层现manager做树hash与complete，raw staging不能直接称可恢复checkpoint。
- **fresh load是独立强门**：新prepare模型+真正空state新Adam，读取保存的非零moments/step，严格恢复每个FQN、param groups/lr/scheduler/model/buffer，下一优化步应与不中断对照一致。不能用已热optimizer.load来代替fresh；不得调用dummy optimizer.step或把零state“先初始化后伪称恢复”掩盖API缺口。
- 注意PyTorch某些get_optimizer_state_dict在cold optimizer会初始化/step；不能未经检查在fresh load前调用。选择支持真实fresh materialization的公开DCP路径或明确构造目标state容器（不step、不改权重），记录精确API依据。若当前API无法在权限内实现，报告blocked和复现，不改环境或写stub。
- 本项不还原全局训练RNG/data cursor/behavior，但接口必须不抢先restore RNG；外层bound恢复门最后恢复现RankRNGSnapshot。checkpoint完整进度/资源/双schedule/缓存/拓扑contract由后续manager/runtime绑定。本codec的模型恢复测试不是整个run恢复完成。
- 兼容/路径/metadata/shape错误在潜在state collective前通过有界健康receipt让全部存活rank统一拒绝；getter/collective内部死亡仍由launcher退出，不在失败finally补collective或回滚已写complete。读取严格绑定same-world/topology/model schema；不支持弹性/mid-collection。

## CPU验证

实际tiny loader/PEFT/guarded prepared与真实Adam训练至少一步形成非零moments；**真实磁盘DCP保存/读取与公开state APIs**在CPU测试执行，只有CUDA/PG/分片传输叶子替换或明确no-dist测试包装。不mock整个codec/save/load为内存dict；不运行真实PG/网络/GPU。

保存前后比较全部状态与RNG/grad不变；cold/部分Adam状态/未清grad/unsafe proof提前拒绝且getter/optimizer.step计数为0。保存至tmp_path真实DCP文件，再新建模型+空state Adam+新scheduler，fresh load后逐FQN state/value/shape/dtype/step/group/lr/buffer严格检查，固定下一步数据对照不中断结果。spy证明save/load没有额外optimizer.step。损坏/缺文件、wrong topology/model/schema/shape/scheduler及部分state拒绝。

测试中CPU参数不称DTensor；生产依然真实DCP allrank入口。明确哪些proof/collective为测试leaf，所有被测控制/数学/文件代码实际执行。最后源码编辑后跑plan固定Flash7/H3通配符，无skip/xfail/全量；报告branch/HEAD、命令/exit/数量/时间、diff/check与给现manager的精确staging接口。

Simplifier C重点审cold getter纯度、fresh空Adam物化、state完整覆盖、全rank参与/非持久buffer及不伪造完整checkpoint；changes回C重测。无服务器/GPU/下载/升级/push，GPU_PENDING。
