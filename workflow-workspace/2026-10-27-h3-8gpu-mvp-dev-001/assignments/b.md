# Assignment B

- active: false
- status: idle
- task_id: H3-IDLE-B-002
- pair: coding-b → simplifier-b
- source_dependency_commit: `8a6763cf3defcb07415dfeda501204e630b42f5a`
- depends_on: M1a coding snapshot `7b29f4bf8d990794d4b9bc826a82bd390861d530` 已获批准；等待本轮 H3-M1b-cps 合同审核。
- allowed_source_files: []
- permitted_action: idle，不修改源码/配置/测试，不运行无任务测试、不制造交付文件。若引擎要求回执，仅如实回 idle。
- interface_lock: joint packing/dual schedule 已稳定；本轮 CPS 由 A 单独实现；B 不提前写 adapter/recipe 或另一个合同。
- acceptance: 无修改，无伪测试/GPU声明。
- non_goals: 本轮不审 A、不实现 H3 loading/registry、不访问服务器/网络/GPU。

后续候选职责是 adapter/真实依赖与独立 recipe 接入，具体任务、允许文件和验收必须等 orchestrator 新 assignment；此处不是提前授权。
