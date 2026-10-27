# H3 MVP 主会话接管收尾

原工作流 `3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc` 在B发现 C-PREPARE-02 后走入failed出口；该历史状态保留，不改写为工具成功。用户要求主会话直接收尾，以下工作在独立 `finish/h3-mvp-20261028` 分支完成。

## 实现

- 初始化owner读权重及完整覆盖检查失败，在mesh/wrapping/state collective之前与全部rank交换状态；保留原异常，没有失败后的补救collective。
- 原create_default_run_controller按recipe选择H3生命周期绑定，仍由同一个RunController和原GRPOTrainer六阶段执行，没有第二Runner。
- 完成真实profile/preflight/session、模型/数据/评分构造、FSDP prepare、观察到的prepared合同、两阶段恢复、训练/周期checkpoint与终态发布。
- 输出目录独占锁；拒绝覆盖已有fresh输出、拒绝非latest恢复和将输出写入模型/cache/source；诊断先于checkpoint发布，终态checkpoint之后中断可恢复发布而不重做更新。
- 补充正式CLI、连续更新、durable checkpoint后中断、fresh恢复继续，以及终态checkpoint恢复的CPU端到端测试。
- 使用说明：`docs/h3_mvp.md`；完整结构模板：`configs/v2/h3_t2va_mvp.yaml.example`，需填真实资源，不能直接运行占位模板。

## 实测及边界

主会话完整指定CPU门禁：**159 passed, 27 warnings in 50.80s**，即原7项＋152项H3。警告为CPU no-dist DCP及既有负例覆盖写入提醒，未屏蔽。只有模型/VAE/HTTP/CUDA/通信叶子被替换，真实默认factory、controller、CLI、训练阶段、公开状态API和磁盘checkpoint路径均执行。CPU线程中的meta构造加测试专用锁，模拟真实进程的独立性，不修改生产模型初始化语义。

此结果发生在合入远端新增SMALL40恢复/备份两提交之前；之后用户明确要求跳过追加CPU测试，直接转GPU，故没有对合并后树重跑整套CPU门禁。保留远端改动，不覆盖或强推。

原子代码提交：
- `675422c`：初始化失败协同退出及回归。
- `7da735d`：H3默认controller/CLI与端到端测试。

尚未在真实H3权重和8卡NCCL上验收。新终态kind为`visual_rl_h3_run_success`，不声称旧单进程inspect_run/audit_run已支持；无音频质量reward、无样本视频默认持久化、无跨拓扑恢复。首次GPU测试需足够空闲且获准的8卡、可加载的模型格式、严格离线文本缓存和124帧HPS服务。

用户已明确提供并授权SSH到 `ubtroot@192.168.102.193`；首次连接超时，未到达服务器，未操作进程或GPU。后续状态以主会话进度文档为准，不将提交/推送视为GPU通过。
