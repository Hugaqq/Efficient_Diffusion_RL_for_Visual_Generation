# Round 005 — 等待M0已存原始证据，必要时另批M1测量

## 本轮决定

`waiting_external`，**不是exhausted/代码阶段结束**。当前没有要交Coder的新代码计划，也没有已知测试失败/未知副作用等待处理；不重放GPU或工具。等待用户明确恢复并补充本轮可核成本输入，详见 `planner/MEASUREMENT-CARD.md`。

优先M0：补齐已保存的第二轮32样本decode/VAE load/HPS/export/owner事件、来源SHA及可用权重header/拓扑/host事实，新增GPU工作/更新=0。Planner没有越权访问工作区外归档。

若M0不足，再由用户**单独批准**M1：一次全新输出、同协议fresh两更新、P1＋F0 opt-in测量，训练最多新增2个global更新/64样本，绝不自动启动/重跑旧run；该测量不等于最终端到端速度或严格GPU数值配对验收。执行者/最终SHA/设备/服务归属/资源/目录/时窗必须先明确。

## 已实施项的独立复审

- G0 `e324aaf`：CPU夹具隔离，生产安全门不变。
- I1 `a211cbf`：同一checkpoint边界fresh完整校验返回boundary/live，无跨边界cache。
- P1 `9e183bb`：默认off的owner内双缓冲、Queue1/总许可2、handoff/关闭/错误排空和CPU日志隔离。
- F0 `9b0a6b3`：默认off，每rank固定第二更新/第二本地样本/index0一次CPU+CUDA window，rank/start/stop错误门、停止后CPU导出/受控trace及receipt。

本轮实读F0完整recorder/launcher/collect/dual stage接线及全部F0回归，核真实累计diff/log/HEAD stat和Coder/Simplifier报告。当前HEAD `0eb2a5b6703fc726a063918eaaacdf63672e876f`为工作流快照；无Simplifier代码commit。未发现须先修复的F0正确性缺口。不可用CUDA/CUPTI的测试叶子合成活动不是GPU实测；export真实兼容性与资源峰仍待测。

Planner实跑原固定七项短入口：**7 passed, 3 warnings in 44.91s**；当前worktree导入、原Git/config证明和G0/I1/P1回归保持通过。CPU耗时不作训练speedup证据。

## 为什么现在需要外部输入

当前inputs实际只有摘要/峰值/分片估算/CPU门禁，没有逐样本原始计时或CUDA operator trace。

- V1：用户8s加载/37s解码/50s评分尚非已核边界；GPU load计时本身含CPU加载和H2D。缓存pair会新增驻留/offload、RNG/失效合同；P1后也不一定省墙钟。没有实际D/H就进一步缓存，不能保证显著收益。
- P2：跨owner流水可能减少fill/drain，但涉及跨rank/owner工作存活与取消/心跳/媒体传递。应先确认P1真实GPU并发、D/H与资源安全，不在未测方案上堆更复杂并发。
- F1/F2：42.600min采样未分compute/NCCL/memcpy/host；56.49GB是字节估算，不是带宽。DDP静态超64GiB，D2仍要1D梯度/checkpoint重设计；F0入口已具备，先测再选择。
- H1/T1：HPS已CPU持久加载、每请求上/下卡，真正搬运/模型/预处理比例未知；重复小搬运/检查有代码迹象但不能无占比就删门或承诺显著收益。

因此仍有开放高收益机会，不是“没有任何优化”。本轮为正常等待明确外部证据/授权，不是预算触顶，不生成FINAL/VALIDATION-PLAN，也不声称全部候选处理完。

## 成本/不变量/验收边界

存证第二轮42.600+51.318+12.994min，加校验16.799/保存4.021≈127.73min（**含优化/校验/保存，不含初始化等**）；前两段93.918min（**不含优化/保存**）；全两步294.31min含初始化/审计/保存等，不能除二当稳态。之前P1/I1估算仍只是条件假设，不叠加成已实现收益；F0本身无直接speedup。

原科学协议/精度/RNG/梯度/奖励124/top37/保存恢复/安全门保持；本角色未运行GPU/torchrun/训练/推理/SSH/HTTP/HPS/服务/部署，Euler-B继续暂停。所有后续代码commit仍需原七项全通过与diff检查、功能测试同原子commit、Coder/Simplifier分开。

本轮无预期代码commit；仅Planner报告。最终integration、专用perf发布、GPU严格等价/最终两步速度保存恢复验收**均尚未完成或获本轮授权**。候选发布/测量前也须由获准integration重新核真实门禁；Planner不越权push/部署，不把等待当完成。
