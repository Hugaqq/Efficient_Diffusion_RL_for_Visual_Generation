# 当前交接：repair — H3-EXP-001

请Coder先读task.md及`issues/H3-EXP-001-artifact-admission.md`，修复/提供最小原生H3 artifact适配准备路径。真实GPU环境probe已执行，生产build_h3_meta对原缓存报`artifact file missing or path escape: config.json`；原文件是HF blob链接，且实读旧schema/融合QKV键与当前新类严格loader不同，不能仅解引用或改类名冒称兼容。

必须优先查已有origin/h3-pytorch-training、origin/h3-readable-integration的加载/映射。只本地修改并由Git部署，Experimenter负责所有服务器操作；不删guard、不改原权重、不追加CPU回归。

SHA d5c10c9f00e44a816e1f0881c7e28ba8accbcdae；job artifact-admission-probe-001 已退出code1，无活跃训练/未知操作。PID765033及完整日志/命令/实测资源见不可变issue报告。Flow/Flash真实更新0/0。

后续必需single-step缺口另存`issues/H3-EXP-002-flash-single-step-gap.md`；先模型准入再逐阶段推进。text cache producer/HPS124准入尚未完成，不称ready。正式步数缺失不在此阶段请求暂停。
