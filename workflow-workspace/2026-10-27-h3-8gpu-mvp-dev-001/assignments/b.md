# Assignment B

- active: false
- status: idle
- task_id: H3-IDLE-B-001
- pair: coding-b → simplifier-b
- source_dependency_commit: `076635a61263ffd5e7c5be4cc64d77ef99242c51`
- depends_on: H3-M1a-joint-schedule 及后续 CPS 合同审核批准；当前没有已批准功能提交。
- allowed_source_files: []
- permitted_action: idle，不修改源码/配置/测试，不运行无任务测试、不制造交付文件。若引擎要求回执，仅如实回 idle。
- interface_lock: 本轮 joint packing/dual schedule 由 A 单独定义；B 不提前写 adapter/recipe 或另一个合同。
- acceptance: 无修改，无伪测试/GPU声明。
- non_goals: 本轮不审 A、不实现 H3 loading/registry、不访问服务器/网络/GPU。

后续候选职责是 adapter/真实依赖与独立 recipe 接入，具体任务、允许文件和验收必须等 orchestrator 新 assignment；此处不是提前授权。
