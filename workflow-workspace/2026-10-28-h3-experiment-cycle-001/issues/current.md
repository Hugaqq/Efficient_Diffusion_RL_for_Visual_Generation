# 当前交接：H3-EXP-001 open；网络恢复后先补环境材料

第三轮补充：已按主会话09:47恢复确认继续，但exp003-host-001与最小hostname查询exp003-connectivity-minimal-002再次均port22连接超时code255，无stdout。再次needs_input暂停，详见experimenter/environment-003-connectivity.md。没有取得新的converter/配置/源hash材料，无新源码问题可交Coder修复。

第二轮：Coder已核查既有分支并返回need_experiment（coder/H3-EXP-001.md）。Experimenter补材料时固定SSH连续两次code255，现请求needs_input暂停，详见needs-input.md及experimenter/environment-002-connectivity.md。没有新的源码失败或修复验收；不重复派Coder在缺相同材料时盲改。网络恢复后先只读核查，再补完整converter/原生配置/安装源hash/既有converted根信息。

以下为首轮修复请求，保持有效：

请Coder先读task.md及`issues/H3-EXP-001-artifact-admission.md`，修复/提供最小原生H3 artifact适配准备路径。真实GPU环境probe已执行，生产build_h3_meta对原缓存报`artifact file missing or path escape: config.json`；原文件是HF blob链接，且实读旧schema/融合QKV键与当前新类严格loader不同，不能仅解引用或改类名冒称兼容。

必须优先查已有origin/h3-pytorch-training、origin/h3-readable-integration的加载/映射。只本地修改并由Git部署，Experimenter负责所有服务器操作；不删guard、不改原权重、不追加CPU回归。

SHA d5c10c9f00e44a816e1f0881c7e28ba8accbcdae；job artifact-admission-probe-001 已退出code1，无活跃训练/未知操作。PID765033及完整日志/命令/实测资源见不可变issue报告。Flow/Flash真实更新0/0。

后续必需single-step缺口另存`issues/H3-EXP-002-flash-single-step-gap.md`；先模型准入再逐阶段推进。text cache producer/HPS124准入尚未完成，不称ready。正式步数缺失不在此阶段请求暂停。
