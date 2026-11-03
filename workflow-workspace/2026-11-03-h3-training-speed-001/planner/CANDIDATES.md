# H3候选表

基线：生产 `3db2a05e1c464bd8cbd97c0290a094d869a9d1fd`；工作流起点 `86a628cb58ffe558fc91da9daf9fcff2946398a9`。当前round-006恢复HEAD `e5cfeb446bb20d1222a1e8df09516f4682572eee`只比上一已审HEAD多Planner报告快照，无新源码。G0/I1/P1/F0已通过前轮独立审查；用户已补M0并明确恢复，原成本等待解除，**本轮选H1 implement**。P1/F0仍默认off，GPU/端到端收益未验证。

## 成本/硬件口径

证据为 TASK 与 inputs 保存报告，不是本轮硬件查询。10×CMP170HX 64GiB，训练物理0/1/2/3/6/7/8/9，HPS独立物理4；不占用/推断GPU5。PCIe Gen2×4、无NVLink、PIX/PHB/SYS跨NUMA为任务已存事实，当前inputs未附完整原始拓扑矩阵；禁止凭型号替代链路测量。

`inputs/baseline-MEMORY-REPORT.md`第二轮：采样42.600min，decode+reward51.318min（含观看副本），optimize12.994min，checkpoint validation1007.942s，save241.260s。三大计算阶段约106.91min（**含优化，不含校验/保存/初始化**）；加checkpoint校验和保存约127.73min（**含优化/校验/保存，仍非全生命周期**）。前两段93.918min不含优化/保存。两步监控294.31min含初始化、审计和退出，不可直接除二当稳态。

allocated峰19.708GiB不代表可额外常驻任意模型；reserved约62.87GiB/整卡约63.39GiB。标准DDP静态68941191680bytes≈64.21GiB，**尚未加激活即超容量**。D2静态35332218880bytes≈32.91GiB仅估算。

|ID|候选/实际证据|收益依据（非已实现）|状态/下一决定|
|---|---|---|---|
|G0|CPU门禁夹具隔离。原6通过1失败；修复后仅测试原函数内局部ROOT夹具与拒绝断言，生产保护不动|无训练提速；恢复后续commit必要门禁|**完成并复审**：Coder `e324aaf`，Simplifier无必要改动/无commit；Planner实跑7 passed/47.76s|
|V1|VAE有界复用，原每sample两模型重载|M0实测load均值video7.258＋audio1.181=8.439s，32次合计270.054s（4.501min，含构造/读盘/H2D）。owner内保留一次可去24次load约197.290s=3.288min serial上界，未扣offload/hash/驻留成本|重新列高优先：当前P1评分较慢可遮蔽后3次reload，但H1后decode预计成瓶颈，VAE更值得后续单独实施。容量/RNG/失效合同及headers仍待设计，不用此缺项阻塞H1|
|P1|单owner四样本双缓冲，保留8个owner顺序|M0旧serial D/H反事实：主体3072.563→1984.139s，条件省18.140min；完整保存轮127.73→约109.59min（**含优化/校验/保存、不含初始化**）。不是P1 GPU实测，不叠I1|**已实现并复审**：Coder `9e183bb`，Simplifier无commit；Planner实际7/7/51.64s。默认off、总许可2/Queue1、显式storage handoff、原score/export/GPU门、CPU日志隔离与join/drain保持；真实GPU等价/资源/收益待授权|
|P2|跨owner/全32连续流水|M0后若H1令decode成为瓶颈，owner间可隐藏的前7个评分尾部约179.100s≈2.985min（辅助段口径，不含优化/保存），小于原H50假设的5.25min|仍开放，先更直接的H1；跨rank心跳/存活/媒体/单HPS顺序与取消复杂度不能和H1同commit。旧D/H反事实不是新GPU测量|
|I1|基线同checkpoint边界连续两次live验证；现已由fresh snapshot返回boundary/live并后置复核，全部强检查覆盖保留|实测整个validation约16.8min/保存轮，不是单次调用耗时。若两次扫描各占约一半可省约8.4min/保存轮，127.73min含优化/校验/保存分母条件约1.07×；cadence5稳态仅摊约1.68min/update，111.08min含优化及摊销校验/保存分母约1.5%。未实测、不能给已达speedup|**已实现并复审**：Coder `a211cbf`，Simplifier无代码commit；Planner实际7/7/48.32s。fresh snapshot同时返回boundary/live，旧API自校验，后置独立source/tail复核；restore/最终门未改。实际节时仍未验证|
|F0|采样通信/算子观测前置：固定第二更新/第二本地样本/index0的一cell，每rank一次、默认关闭CPU+CUDA trace|无直接speedup。42.6min占保存轮127.73min约33.4%（含优化/校验/保存、不含初始化），该段没有可靠算子/通信分解，不能凭Gen2带宽猜瓶颈|**已实现并复审**：Coder `9b0a6b3`，Simplifier无commit；Planner005实际7/7/44.91s。默认off、固定once/rank、主线程启停、停止后CPU导出/receipt，异常失败门保留；合成CUDA测试不是真实CUPTI，M1需另批|
|F1|采样FSDP通信。每block WORLD8、reshard_after_forward=True，BF16 compute/FP32 reduce，AC外包FSDP；每rank40 rollout root前向/轮，另4 anchor+4 current|估算每次遍历allgather接收约56.49GB；不是PCIe实测时间。42.6min含计算/通信/门禁/检查；约63.9s/forward只是阶段平均|005待外部M1真实F0 trace区分NCCL/compute/memcpy/CPU等待；gate自身也有NCCL不能全部归FSDP。入口已就绪，不再盲改reshard/prefetch/mesh或用理论带宽当测量|
|F2|D2四副本混合分片|估算接收32.28GB/遍历、静态32.91GiB；不含工作集、聚参峰、VAE|暂缓架构修改。实际`_local_gradient`与checkpoint/binding写死1D WORLD8 Shard(0)、除数8；改mesh不是独立安全一行优化。需梯度数学、组拓扑、checkpoint兼容和获准测量|
|F3|标准全GPU DDP|64.21GiB静态已超64GiB，未含激活|当前协议下排除；不能用allocated峰小为依据切DDP|
|H1|每POST模型.to(cuda)与.to(cpu)重复搬运；M0有32条细分及原摘录|实测搬运790.792s=13.180min，占hps_score父段约50.3%；不是整个49.112s都搬运。与P1重叠后M0反事实增量约320.363s=5.339min/轮，保存轮约109.59→104.25min，约1.05×（**含优化/校验/保存、不含初始化**）；无已达收益|**006唯一选定**：默认off，单UUID专卡单模型复用、原40GiB/16GiB门、低余量真实offload/cold回退、60s idle和退出释放、新manifest身份策略；只代码＋CPU，绝不操作现有服务/请求GPU|
|T1|条件搬运/重复校验。每步conditioning clone→to(device)，`_policy`重复模型inventory/rope.cpu比较/Adam counters；采样结果gate+同步|实际源码冗余可见，但没有阶段占比。测量显式同步每rank合计41s不能8倍相加，也非可归因新增开销|次级候选；不删安全门、不改变RNG/precision；先定位成本再决定是否值得独立单元|

## 进度与限制

- 001：G0已实现并复审。Coder提交前7/7（46.53s），Simplifier独立7/7（45.90s），Planner复审实跑7/7（47.76s）；累计授权源码diff仅测试。`.pi/`未跟踪保持不动。
- 002：I1已实现，Coder提交前7/7（53.30s），Simplifier独立7/7（52.76s），Planner003实跑7/7（48.32s）。CPU耗时波动不作为训练speedup证据。
- 003：P1已实现，Coder恢复后提交前7/7（51.68s），Simplifier独立7/7（51.79s），Planner004实跑7/7（51.64s）；已读取恢复失败/应用一次/resolved记录，不将历史5/7当成功。媒体和日志必要改动同功能commit；不从CPU重叠推GPU收益。
- 004：F0已实现，Coder提交前7/7（51.52s），Simplifier独立7/7（53.75s），Planner005实际7/7（44.91s）。窗口仅CPU合成活动逻辑证明，无GPU profile或speedup。
- 005：曾waiting_external等待M0，原CPU7/7/44.91s已执行，非完成。
- 006：用户补M0完整32行/四rank原摘录/来源SHA并明确恢复；已存M0-ROWS.csv/M0-ANALYSIS.md，搬运成本足够选择H1。HEAD stat仅Planner报告，源码未变，不重复门禁/恢复旧补丁，不重索同一M0。完整拓扑/VAE headers/CUDA trace仍NA但不阻塞H1。M1/GPU/服务操作仍未授权；不写FINAL。
- I1预计收益只发生于保存边界：旧两步每步诊断保存，不外推成正式cadence5每步省8.4min；294.31min两步全生命周期仍不可直接平均当稳态。
- 原Euler-B继续暂停；未发任何GPU/HTTP/SSH请求、未运行训练/推理、未改服务或依赖。
- Planner完成不等于硬件最优；后续每轮复审真实diff/commit/七项输出，最终integration与perf分支发布尚未发生。
