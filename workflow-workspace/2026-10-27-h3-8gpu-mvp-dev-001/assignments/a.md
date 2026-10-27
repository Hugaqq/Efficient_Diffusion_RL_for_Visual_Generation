# Assignment A

- active: false
- status: idle
- task_id: H3-IDLE-A-004
- pair: coding-a → simplifier-a
- source_dependency_commit: `63fe34c99f96a115ef3cb743965aa4338e3653d6`
- completed_dependencies: H3-M1a-joint-schedule snapshot `7b29f4bf8d990794d4b9bc826a82bd390861d530`；H3-M1b-cps snapshot `725f48f4d67c67d90c60cc3536f4bc40f12802f2`；均配对approved，当前无changes/失败门禁。
- allowed_source_files: []
- permitted_action: idle，不写源码/测试/无意义报告，不运行无任务测试；引擎要求回执时仅返回idle。
- interface_lock: 已批准joint/schedule/CPS保持不动；本轮B独占H3目标测试文件并实现本地meta/LoRA加载边界；forward已配对approved。若后续接入暴露A接口问题由orchestrator单独返工给A。
- acceptance: 无修改，无伪测试/GPU声明。
- non_goals: 不提前做collection/旧锚/评分，不审B，不操作GPU/服务器/网络/环境。

原交付与审查记录保留；后续A任务须新assignment才授权。
