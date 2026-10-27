# Workflow 工作区

每个任务命名为 `YYYY-MM-DD-功能-NNN`，同日期同功能从001递增；不复用已完成目录。

| 工作区 | 用途 |
|---|---|
| [2026-10-27-h3-8gpu-mvp-dev-001](2026-10-27-h3-8gpu-mvp-dev-001/README.md) | 原多Agent开发；在入口集成缺口处failed，保留历史记录 |
| [2026-10-28-h3-mvp-manual-finish-001](2026-10-28-h3-mvp-manual-finish-001/final.md) | 主会话接管：初始化错误门、正式入口和CPU端到端收尾；GPU待验收 |

从本源码仓库根目录启动开发workflow。定义中的工具读写路径必须带对应工作区前缀，不把task/报告散放源码根目录。工具状态/session/worktree保存在`.pi/workflows/`，不手动搬动或提交。旧设计审查和reports位于外层项目，保持原样；本目录的reference只读，不修改旧结论。
