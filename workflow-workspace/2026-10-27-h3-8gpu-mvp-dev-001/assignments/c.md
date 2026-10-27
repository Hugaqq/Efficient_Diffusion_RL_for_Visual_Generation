# Assignment C

- active: false
- status: idle
- task_id: H3-IDLE-C-004
- pair: coding-c → simplifier-c
- source_dependency_commit: `63fe34c99f96a115ef3cb743965aa4338e3653d6`
- depends_on: M1a/M1b和M2a1 forward（snapshot `edd5c874f10ffa917a71bd9f2fa9085247075a6e`）已配对approved；本轮B实现loader，等待meta/blocks/FQN/CPU初态接口稳定后再派prepare。
- allowed_source_files: []
- permitted_action: idle，不修改源码/配置/测试，不运行无任务测试、不制造交付文件。若引擎要求回执，仅如实回 idle。
- interface_lock: 不提前改 runtime/lifecycle/prepared/checkpoint/world 数学，也不另建 PG/Runner。
- acceptance: 无修改，无伪测试/GPU声明。
- non_goals: 本轮不审 B、不启动多进程或 GPU、不做服务器/网络/环境操作。

后续候选职责是 FSDP2 prepare/world 数学/完整 tail checkpoint，必须按小原子项逐次正式分派；此处不是提前授权。
