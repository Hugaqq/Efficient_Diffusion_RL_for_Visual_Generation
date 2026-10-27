# Assignment A — W8 idle（已关闭返工）

- active: false
- task_id: H3-W8-A-idle
- pair: coding-a → simplifier-a
- integration_parent: `a5fef751a88cad751a36240ffe1f5731986e1b1e`
- completed_dependency: H3-M4c1-runtime-binding-r1已获原simplifier-a approved；A-BIND-01关闭，原binding接收。
- allowed_files: 无源码/测试写入；不改B/C工作区或报告。

## 真实依赖与并行重评

本配对的joint/CPS/anchor/pool/stages/source/binding均已有实现与批准，不再制造重复helper或无目标重构。剩余入口级六阶段/生命周期验证需要B本轮H3-M4c2-controller真实backend/default dispatch和资源owner顺序，当前没有该实现；独立提前写测试只能mock整backend，不构成所需证据。此idle不是共享测试单文件限制。

本轮B唯一拥有runtime/h3_lifecycle.py、composition.py、新controller feature；A binding/stages/source公开接口只读冻结，B若发现缺口应返回orchestrator，下一步交原A小修，禁止B越界改A owner。B交付后可再派A独立入口/失败边界补验或其所属生产缺陷，不能现在凭假接口开工。

验收/非目标：idle不宣称新代码/门禁，不新建角色或调度器，不执行GPU/服务器/网络/安装/push。既有146项汇合测试历史不是本idle测试。当前默认入口未完成，GPU_PENDING。详见decisions/wave-08.md。
