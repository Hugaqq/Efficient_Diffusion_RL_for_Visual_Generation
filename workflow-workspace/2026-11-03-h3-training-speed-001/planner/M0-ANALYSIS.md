# M0已接收：第二轮32样本成本归因与下一优化排序

## 来源/证据层级

用户在本次恢复消息直接提供M0：旧run已保存日志的32行完整数据、SUMMARY、4个训练rank/相同sample的HPS原始摘录、源文件SHA。新增GPU调用=0。来源路径只作索引；Planner未读取工作区外文件，也未重复索取M0。完整480条原文由主会话归档，**本角色没有独立重hash整个归档**，下表SHA是交接提供的来源指纹，不伪称本角色读取了480条。

选用的32行数值、sample/rank/member/selected index和memory源行号原样保存于 `planner/M0-ROWS.csv`。阶段context.step=2对应optimizer_step=1；8rank各4样本、4组各8member，selected indices为4/4/3/3。跨文件识别据完整decode结束与原串行decode_begin..score_end区间；1220条非JSON行未被臆造为事件。

|来源basename|交接SHA256|
|---|---|
|train-rank0-pid3997145.jsonl|ab704dec8a076adda5a434cc76098a9975a80d2666d83a9b20fbc4413bea3361|
|train-rank1-pid3997146.jsonl|72d12d7eb8949a5899c1607b4c3010d8592fa02a650cc305d54c5c8e429f2beb|
|train-rank2-pid3997147.jsonl|aa87f06601be539eaa9828dfd9a995061a7b0207bb7ffaf9d1c67b7f7ab61b32|
|train-rank3-pid3997148.jsonl|0afeae77bdc79331cfe70eb3a103308c00eb9a18a481ee66e583b8b02b21e23b|
|train-rank4-pid3997149.jsonl|cfbbaac3d000302a8f0cb657f49abc54a84cbc3906bc7865e949c1bb983d0412|
|train-rank5-pid3997150.jsonl|31b975d2a5f6b94d1fcfb67df0ca4fa1d5d7eb63fc5ea75acc8d1d6d631ccb04|
|train-rank6-pid3997151.jsonl|a616a1950df7cc8b40a7d5f42dafe038e985f4246f17e3191d3886fe5cb3bf49|
|train-rank7-pid3997152.jsonl|410a07b7f61357c187391a9fabc451a949317e9927c233b9c9c0735650053377|
|train.log|5c3abf6342fd54d6b619978f12b9db8b3d2071ab24665bac5fa161861d95801b|
|hps-rank-1-pid3991487.jsonl|1098fb33457c2f4c887ff3919222afb03fb3746c39a78b469987b4bc19c83ad9|

逐项对照所给原始摘录：训练rank0/2/4/6的member0分别train.log 1252..1274、1573..1589、1873..1889、2178..2194，load/full/score与CSV相同；相同sample HPS段为882/890/896、1098/1106/1112、1314/1322/1328、1530/1538/1544行。HPS role rank=-1/PID3991487/UUID `GPU-0e1cd165-126d-e889-2b87-240d8faddddb`，不是训练rank4（物理6）。

## 互斥成本分解（旧串行，实测）

|阶段|32条合计s|平均s/样本|边界|
|---|---:|---:|---|
|decode_full|1459.611|45.613|完整CPU媒体就绪；包括两VAE/搬运/格式等|
|其中video load|232.262|7.258|CPU构造/读权重＋GPU上卡，不是纯磁盘|
|其中video decode|1175.089|36.722|原decode流，不能加到decode_full之外|
|其中audio load|37.792|1.181|同样含上卡|
|其中audio decode|4.557|0.142|原audio decode流|
|其余decode边界/清理等残差|9.911|0.310|由父减子，非独立kernel观测|
|score_full|1612.952|50.405|原score_one含JPEG/HTTP及观看副本|
|其中hps_score父段|1571.599|49.112|服务端父阶段，含onload/offload|
|其中hps_onload|386.598|12.081|模型.to(cuda)观察段|
|其中hps_offload|404.194|12.631|模型.to(cpu)观察段|
|hps_score减上/下卡残差|780.807|24.400|包括原124帧处理/前向及其它开销，**不是纯GPU kernel时间**|
|score_full减hps_score残差|41.354|1.292|客户端/协议/导出及边界差，不能全称JPEG|

关键结论：上/下卡合计 **790.792s=13.180min，24.712s/样本，占hps_score约50.3%**；两个VAE load合计270.054s=4.501min，不能把两者等同优先级。

`sum(decode_full+score_full)=3072.563s=51.2094min`，与旧宏段51.318min差约6.52s（外层gate/日志等边界）；不能父子重复相加。旧GPU采样42.600min/优化12.994min/校验16.799min/保存4.021min仍保留。

HPS原摘录上卡后allocated约15.50GiB、保存报告全服务峰15.730GiB，支持“独立64GiB评分卡上可考虑单模型复用”的量级；不是实时可用内存/独占证明。不得拿训练rank allocated峰19.7GiB替代HPS准入；运行时仍需完整UUID、真实free及原40GiB/16GiB门。

## 与P1的临界路径：不是简单加13.18min

以下是**用旧串行实测D/H构造的反事实调度估算**，未执行新GPU/CPU模拟器；假设重叠不导致CPU/PCIe竞争、keep状态数值/工作区相同，无失败回退/额外I/O代价。不是P1/H1已实现收益。

记每owner四条为D_i/H_i（CSV原值）。本组每个旧H均大于下一个D，单owner双缓冲的理想完成时间为 `D_first + sum(H_i)`：
- 8个D_first合计约371.186s；所有H合计1612.952s。
- P1理想主体 `T_P1≈1984.139s=33.069min`（不含采样/优化/校验/保存，也未加约6.52s外层边界）。对旧serial主体51.209min，条件节约18.140min；非实测。

若H1 warm请求移除上/下卡，令`H'_i=H_i-onload_i-offload_i`，其余工作不变，所有H'低于D。首个cold onload约12s仍可被下一decode覆盖；每owner理想主体改为 `sum(D_i)+H'_last`：
- 八个H'_last约25.462/25.674/24.954/26.415/25.570/25.577/25.449/25.065s，合计204.165s。
- `T_P1+H1≈1459.611+204.165=1663.776s=27.730min`。
- **H1相对P1增量约320.363s=5.339min/轮**，而非13.180min。cold启动、idle TTL卸载、headroom回退、监控/导出/争用会影响实际收益。

分母明确：旧保存轮约127.73min（含采样/评分/优化/完整校验/保存，不含初始化等）；只代入上述调度假设，P1后约109.59min、再加H1约104.25min，H1增量约1.05×，P1+H1相对旧保存轮约1.23×。前两段93.918min（不含优化/保存）条件约降至70.44min、约1.33×。不叠加I1未测收益，不以旧两步294.31min直接除二，不声称达到2倍。

## 重新排序/可实施决定

1. **本轮H1：默认off、UUID约束的专卡单HPS模型有界驻留，原准入/低余量offload回退/idle TTL/退出清理**。明确50.3%服务段在重复搬运，源代码确有每POST完整.to(cuda)/.to(cpu)，无需再等同一M0或先跑GPU才写这个可选策略。只代码/CPU验证，不操作服务。
2. V1仍有空间：H1后decode预计成为瓶颈，owner内复用若只保留一次load，旧32行中可去掉的24次load约197.290s=3.288min（serial上界，不含offload/hash/安全代价）。现在应恢复高优先候选，而不是永久以“P1隐藏load”排除；具体实现仍需容量/失效/RNG设计，headers NA不阻碍H1。
3. P2跨owner：按上述H1调度尾部约204.165s，理论可再隐藏前7个尾部约179.100s≈2.985min，须保留跨rank错误/资源合同；先H1，不同单元不混做。
4. F1/F2真正compute/NCCL占比、拓扑矩阵与1D norm/checkpoint变更仍NA。F0已就绪，未来获准实测再选，不把这个缺项当H1的阻塞。

M0已满足本轮H1成本选择；M1/任何GPU执行仍未授权。原已实施源码未变，不重跑上轮已执行CPU门禁或旧恢复操作，继续Coder→Simplifier原子循环。
