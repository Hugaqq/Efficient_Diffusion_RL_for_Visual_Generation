# Assignment C

- active: false
- status: idle
- task_id: H3-IDLE-C-002
- pair: coding-c → simplifier-c
- source_dependency_commit: `8a6763cf3defcb07415dfeda501204e630b42f5a`
- depends_on: M1a coding snapshot `7b29f4bf8d990794d4b9bc826a82bd390861d530` 已获批准；CPS 与模型 block/FQN 接口仍待稳定并获配对批准。
- allowed_source_files: []
- permitted_action: idle，不修改源码/配置/测试，不运行无任务测试、不制造交付文件。若引擎要求回执，仅如实回 idle。
- interface_lock: 不提前改 runtime/lifecycle/prepared/checkpoint/world 数学，也不另建 PG/Runner。
- acceptance: 无修改，无伪测试/GPU声明。
- non_goals: 本轮不审 A、不启动多进程或 GPU、不做服务器/网络/环境操作。

后续候选职责是 FSDP2 prepare/world 数学/完整 tail checkpoint，必须按小原子项逐次正式分派；此处不是提前授权。
