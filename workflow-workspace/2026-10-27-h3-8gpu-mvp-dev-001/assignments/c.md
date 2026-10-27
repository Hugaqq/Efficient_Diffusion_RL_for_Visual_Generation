# Assignment C

- active: false
- status: idle
- task_id: H3-IDLE-C-003
- pair: coding-c → simplifier-c
- source_dependency_commit: `8ed954f2562cf9d1f5432a7bb84ada181cdae6a9`
- depends_on: M1a及M1b（CPS snapshot `725f48f4d67c67d90c60cc3536f4bc40f12802f2`）已获配对批准；本轮B做forward，模型加载/block/FQN接口仍待稳定。
- allowed_source_files: []
- permitted_action: idle，不修改源码/配置/测试，不运行无任务测试、不制造交付文件。若引擎要求回执，仅如实回 idle。
- interface_lock: 不提前改 runtime/lifecycle/prepared/checkpoint/world 数学，也不另建 PG/Runner。
- acceptance: 无修改，无伪测试/GPU声明。
- non_goals: 本轮不审 B、不启动多进程或 GPU、不做服务器/网络/环境操作。

后续候选职责是 FSDP2 prepare/world 数学/完整 tail checkpoint，必须按小原子项逐次正式分派；此处不是提前授权。
