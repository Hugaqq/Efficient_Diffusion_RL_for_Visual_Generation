# Assignment C — W8 idle（组合验收已接受）

- active: false
- task_id: H3-W8-C-idle
- pair: coding-c → simplifier-c
- integration_parent: `a5fef751a88cad751a36240ffe1f5731986e1b1e`
- completed_dependency: H3-M4b3-bound-checkpoint-integration已approved；实际binding/source/tail→publisher→fresh下一轮CPU组合接受，C-PUBLISH-01已在W6关闭。
- allowed_files: 无源码/测试写入；不改B/A文件或报告。

## 真实依赖与并行重评

prepare/world/coordination/codec/publisher及真实对象fresh组合均已批准。当前尚缺的是默认RunController对第一恢复门前真实binding建立、第二门source回读、same-run输出准入及最终RunResult的装配，均由B本轮唯一lifecycle owner负责。继续独立写一个checkpoint包装器会重复该owner；入口恢复/终态故障补验须等待实际B backend/output接口，不能用mock backend制造并行。

保持现transaction/codec/manager只读冻结；B如发现其所属缺陷，应报告最小复现回orchestrator→原C修复，不顺手改publisher。C之后可接真实入口恢复/fault补验或本owner缺陷；不因当前idle增加角色、第二manager或纯文档代替源码。

验收/非目标：不宣称本轮改动或测试；146项为既有汇合证据，非本idle执行。无GPU/服务器/真实PG/HTTP/下载/升级/push。当前默认入口未完成，GPU_PENDING。详见decisions/wave-08.md。
