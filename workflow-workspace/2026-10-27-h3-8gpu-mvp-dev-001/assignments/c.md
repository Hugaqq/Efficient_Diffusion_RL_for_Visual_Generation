# Assignment C — W5

- active: true
- task_id: H3-M4b2-checkpoint-publish
- pair: coding-c → simplifier-c
- execution: parallel W5；decisions/wave-05.md
- integration_parent: `6f5e46fce212e00d547a3ca07fa5524018bef816`
- dependencies: W4真实DCP state codec approved；现AtomicCheckpointManager/CheckpointContract/Progress/SafePoint/RankRNGSnapshot只依赖稳定类型；代码汇合fb085fb35fb239cdd4c151d39decbd7fae268c9d。

## 单一目标/允许写入

让现**AtomicCheckpointManager发布allrank H3完整状态**，补齐shared staging、rank RNG/data附属状态及分阶段恢复。不能另造第二checkpoint manager/publisher。

允许精确扩展 `visual_rl/artifacts/checkpoint/manager.py` 共享staging创建/commit-existing窄入口（旧commit语义/断言保持）；新 `visual_rl/artifacts/checkpoint/h3_transaction.py`、新 `tests/test_h3_mvp_cpu_checkpoint_publish.py`、W/coding-c/H3-M4b2-checkpoint-publish.md。现h3_state.py/旧protocol/state/RNG/transaction/runtime及其他tests只读；A/B本波新data/profile不import。若需改codec缺陷先报告给本orchestrator，不能顺手混大改。

## 冻结整合合同

- 接受实际H3PreparedModel、现CheckpointContract/Progress/SafePoint/H3StateIdentity、既有已初始化WORLD8、调用者真正完整tail后的小rank data/behavior状态。data payload要求JSON-safe有限小schema和显式content identity，不能包含Generator、tensor轨迹、model或活跃reservation；C不伪造真实data cursor，未来runtime从A的state_dict传入。
- **主writer创建共同staging**，广播相同路径/step/身份，所有rank检查位于自己的声明checkpoint root。所有rank在writer callback以外调用真实save_h3_state，然后各写自己的rank附属文件（RankRNGSnapshot现capture/codec复用、data/behavior、rank/topology与state摘要）。模型get/save/load/set必须全rank，不藏在manager单writer里。
- Manager增加最小 `create_staging`/`commit_existing`等功能，复用原complete/state_tree/progress/rename/latest实现，不复制完整发布逻辑。严查receipt由本manager创建、step/root/路径/无symlink/未complete与现final不存在；禁止接管未知临时目录。旧commit(writer)可复用内部helper但原异常/文件格式/故障注入语义不变；不要改变Flash旧checkpoint布局。
- 全rank成功、预期8rank附属状态恰齐且内容hash一致后，只有rank0对完整DCP文件树+rank状态调用现manager纯文件发布。返回现CommittedCheckpoint或可解析的小broadcast receipt。rename后latest失败保留合法complete，不倒删已发布目录；rank死亡/collective内部失败不finally barrier补checkpoint，不把半staging作为恢复来源。
- global progress跨rank相同：`rng_state_id`应为**有序8rank RNG身份表的全局摘要**，不能要求8个各自rng hash相同、也不能直接删RNG身份。rank私有RNG在对应文件严格校验；data的共同cursor/next_global_batch_id保持全局含义，局部样本状态保各rank。小表含rank、rng/data身份、同topology/schema即可，不gather整个RNG tensor或媒体。
- proof仍必须来自原BaseTrainer完整tail返回后（含data/behavior/pool/next-step）；保存拒step0/cold/mid-collection/poison/未清梯度。不能以caller传True就跳过现SafePoint.assert_ready及codec强门。本模块不声称自己执行trainer tail，future runtime须给真实proof。
- **restore_prepared**：先manager.inspect_complete验证整个树/expected contract/same-topology和rank附件，再各rank真实load_h3_state到fresh空Adam；返回待bound处理的rank state/RNG，不立即restore RNG，不读raw staging冒充complete。
- **restore_bound**：重新核验实际complete树与caller提供的live资源/data/缓存/score/双schedule/模型/topology绑定（不能只复用prepared阶段的旧hash）；调用必需data恢复callback，校验behavior/cursor；最后用现RankRNGSnapshot.restore_current恢复本rankPython/NumPy/TorchCPU/CUDA状态。两阶段恢复各自只可一次，部分失败poison不就地重载。当前API不需import同波A/B，只接受实际序列化payload与外部验证/恢复能力，不提供noop默认值。
- 本步骤保存完整codec+rank RNG/data/behavior；最终run是否completed和统一继续分支由未来RunController处理，不让rank0单独提前退出。保留code/model/score/schema身份，拒elastic/world重编号，禁止把LoRA export当恢复。

## 真实CPU验收

真实AtomicCheckpointManager完整目录/manifest/tree hash/complete/latest和真实h3_state/DCP磁盘路径执行；8rank控制可用有界内存collective叶子或单进程明确transport包装，生产逻辑不mock。重模型/CUDA/PG/DCP通信叶子明确标注，禁止真实PG/网络/GPU。已有fixture可只读复用。

证明model getter/save所有rank在单writer外，rank私有RNG不同仍产生相同global progress，预期rank文件覆盖；检查完整checkpoint可被原manager.inspect_complete读取。真实fresh模型/空Adam及新data对象恢复，data/behavior/scheduler/各RNG与下一随机数/下一固定更新对照不中断。spy证明RNG最后恢复、两gate各刷新文件/live身份；损坏DCP/缺rank文件/错误合同/私有rank/topology/data身份/midcollection提前拒绝，无optimizer dummy step。

故障覆盖写附件/DCP失败不发布、rename后latest失败不倒删合法complete、未知staging/重复step拒绝；原Flash七项实际回归确认旧manager语义未变。最后编辑后跑plan Flash7/H3通配符，无skip/xfail/全量/warning过滤；记录branch/HEAD、命令/exit/数量/时间、diff/check和future runtime精确方法。

Simplifier C审是否真正复用原manager、allrank codec/纯writer、全局与rank身份层次、两恢复门/RNG最后和fresh对照，changes回C。CPU no-dist不是CUDA DCP/NCCL证据；整run controller装配仍后续，GPU_PENDING。
