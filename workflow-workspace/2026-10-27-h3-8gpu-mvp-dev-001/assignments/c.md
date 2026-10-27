# Assignment C — W6 原配对返工

- active: true
- task_id: H3-M4b2-checkpoint-publish-r1
- original_task: H3-M4b2-checkpoint-publish
- issue_id: C-PUBLISH-01
- pair: coding-c → simplifier-c（同原配对）
- execution: parallel W6，decisions/wave-06.md
- integration_parent: `a54d296a7635fe3d552b0fe9beb03717e34249a2`
- review_dependency: W/simplifier-c/H3-M4b2-checkpoint-publish.md verdict=changes；代码汇合230696472b735966d3ac17102120a444f4ff8697。原publisher尚未批准。

## 唯一最小修复

实际 `_write_rank` 仅fsync每个rank文件，checkpoint/verify-ranks之后直接publish；manager只同步staging根，不递归同步h3-ranks目录条目。因此补**父目录fsync持久化顺序**，不能仅添加文档或哈希检查。

允许写 `visual_rl/artifacts/checkpoint/h3_transaction.py`、`tests/test_h3_mvp_cpu_checkpoint_publish.py`，新W/coding-c/H3-M4b2-checkpoint-publish-r1.md。旧manager/codec/RNG/protocol/A/B/runtime/原tests只读；不覆盖初次交付或changes报告，不做其他checkpoint重构。

- 8份附件写完且完整校验成功后，在rank0受控纯文件发布阶段、事务manifest/complete/rename发布前，复用现manager `_fsync_directory` 同步 `path/'h3-ranks'`。
- 同步OSError必须进入现 `_phase` 失败receipt，所有存活rank拒绝、transaction poison；不得继续发布complete/latest，不在finally补collective。不新增publisher，不把同一目录重复8rank同步作为必要框架。
- 保留完整目录覆盖/hash/identity检查、两恢复门/RNG最后、cold/fresh强门及原rename后latest失败保留合法complete行为。

## 新测试与实测

在本feature追加真实顺序观察：附件已全写完/校验→h3-ranks目录fsync→manifest/complete/rename；不是仅assert helper被调用。对该目录fsync注入OSError，证明所有模拟rank共同失败、无complete/latest、不能重试，其他目录同步不误触发故障。原latest失败保留complete、真实DCP/publisher/fresh测试断言不得删改弱化。

最后源码编辑后必须重新原样跑plan Flash七项和完整H3通配符；不沿用中断前/上一轮126项通过，无skip/xfail/全量/warning过滤。报告实际branch/HEAD、命令/exit/数量/时间、diff/check以及C-PUBLISH-01逐条关闭证据；引擎提交，禁止手动Git操作。

Simplifier C复审只审本修复及原项受影响边界，明确approved/changes，不以done代替批准。未实做断电测试，不冒称崩溃恢复已验；CPU不是GPU/DCP多卡证据。GPU_PENDING。
