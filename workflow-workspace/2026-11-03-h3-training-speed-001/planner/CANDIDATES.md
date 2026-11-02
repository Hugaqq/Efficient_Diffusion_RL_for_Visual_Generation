# H3候选表

基线：生产 `3db2a05e1c464bd8cbd97c0290a094d869a9d1fd`；工作流起点 `86a628cb58ffe558fc91da9daf9fcff2946398a9`。当前轮次003复审HEAD `266a25c43d5b7c76c0caed63da0ca737b35484c1`。G0 `e324aaf`、I1 `a211cbf`已通过独立代码/CPU复审；尚未实测任何GPU/端到端提速。

## 成本/硬件口径

证据为 TASK 与 inputs 保存报告，不是本轮硬件查询。10×CMP170HX 64GiB，训练物理0/1/2/3/6/7/8/9，HPS独立物理4；不占用/推断GPU5。PCIe Gen2×4、无NVLink、PIX/PHB/SYS跨NUMA为任务已存事实，当前inputs未附完整原始拓扑矩阵；禁止凭型号替代链路测量。

`inputs/baseline-MEMORY-REPORT.md`第二轮：采样42.600min，decode+reward51.318min（含观看副本），optimize12.994min，checkpoint validation1007.942s，save241.260s。三大计算阶段约106.91min（**含优化，不含校验/保存/初始化**）；加checkpoint校验和保存约127.73min（**含优化/校验/保存，仍非全生命周期**）。前两段93.918min不含优化/保存。两步监控294.31min含初始化、审计和退出，不可直接除二当稳态。

allocated峰19.708GiB不代表可额外常驻任意模型；reserved约62.87GiB/整卡约63.39GiB。标准DDP静态68941191680bytes≈64.21GiB，**尚未加激活即超容量**。D2静态35332218880bytes≈32.91GiB仅估算。

|ID|候选/实际证据|收益依据（非已实现）|状态/下一决定|
|---|---|---|---|
|G0|CPU门禁夹具隔离。原6通过1失败；修复后仅测试原函数内局部ROOT夹具与拒绝断言，生产保护不动|无训练提速；恢复后续commit必要门禁|**完成并复审**：Coder `e324aaf`，Simplifier无必要改动/无commit；Planner实跑7 passed/47.76s|
|V1|有界VAE复用。`h3_decode.py::_decode_stream`每次`_load`并finally drop；GPU版每次CPU加载后上卡；32样本各视频/音频各加载一次|用户约8s/样本加载边界尚未证实。若8s是两VAE合计，当前总256s；每rank四样本、仅owner turn内复用可最多省24×8=192s≈3.2min/轮（不计额外offload/hash成本）。占127.73min含优化/校验/保存分母约2.5%；不是倍数提升|003再读本地保存参考VAE两源码：decode不采样，video保留默认tiling，audio有weight-norm hooks/filters；不等同实际安装loader的无RNG证明。复用需pair驻留或offload、预算/身份失效/构造RNG合同，8s未拆CPU/H2D。仍高优先开放，本轮先流水现有decode，不新增长驻模型/改变加载调用序列|
|P1|单owner四样本decode/HPS双缓冲：保留8个owner顺序，在原owner worker解码，增加一个CPU消费者；旧decoder lease和共享Journal stack均需必要适配|条件D45s/H50s：8*(D+H+3max)=1960s vs串行3040s，理想省18min；映射保存轮127.73→109.73约1.16×（**含优化/校验/保存、不含初始化**），前两段93.918→75.918约1.24×（**不含优化/保存**）。非实测，不叠加I1估算|**003唯一选定**，默认off的opt-in；总在途许可2、Queue1、显式CPU storage所有权移交、失败取消排空join、保留原score/export和所有GPU安全门；日志CPU隔离scope禁止重置另线程GPU峰|
|P2|跨owner/全32样本连续流水，消除每四条的fill/drain|D45/H50假设：D+H+31max≈1645s主体，辅助段约27min的理想口径；相对P1理论再省315s≈5.25min。含优化保存完整收益低于前两段倍数|开放但暂缓：需重新设计owner心跳/媒体跨rank传递或request交接、单HPS顺序、资源排空；不是P1自动已实现收益|
|I1|重复checkpoint live验证。当前源码`checkpoint_boundary`内`validate_live()`；`H3LifecycleBackend.run`立即再调用一次；期间仅只读counter/progress/proof/RNG快照构建，无模型/优化/数据更新|实测整个validation约16.8min/保存轮，不是单次调用耗时。若两次扫描各占约一半可省约8.4min/保存轮，127.73min含优化/校验/保存分母条件约1.07×；cadence5稳态仅摊约1.68min/update，111.08min含优化及摊销校验/保存分母约1.5%。未实测、不能给已达speedup|**已实现并复审**：Coder `a211cbf`，Simplifier无代码commit；Planner实际7/7/48.32s。fresh snapshot同时返回boundary/live，旧API自校验，后置独立source/tail复核；restore/最终门未改。实际节时仍未验证|
|F1|采样FSDP通信。每block WORLD8、reshard_after_forward=True，BF16 compute/FP32 reduce，AC外包FSDP；每rank40 rollout root前向/轮，另4 anchor+4 current|估算每次遍历allgather接收约56.49GB；不是PCIe实测时间。42.6min含计算/通信/门禁/检查|高收益候选，先可关闭有界profile（算子、NCCL、搬运、CPU等待）或已存事件细分；禁止推导已达speedup|
|F2|D2四副本混合分片|估算接收32.28GB/遍历、静态32.91GiB；不含工作集、聚参峰、VAE|暂缓架构修改。实际`_local_gradient`与checkpoint/binding写死1D WORLD8 Shard(0)、除数8；改mesh不是独立安全一行优化。需梯度数学、组拓扑、checkpoint兼容和获准测量|
|F3|标准全GPU DDP|64.21GiB静态已超64GiB，未含激活|当前协议下排除；不能用allocated峰小为依据切DDP|
|H1|HPS驻留/搬运。`serve_hpsv3_video.py`CPU模型初始化一次，不每请求重载；每请求上CUDA，124个单帧前向，finally CPU offload+empty_cache|每样本约50s是用户估算；保存报告未给onload/forward/offload细分，不可把50s全部视为搬运|候选待细分。独立物理4不自动允许改服务；若将来写可选驻留/批处理，保留健康identity、准入/回退及逐帧数值排序，部署仍另授权|
|T1|条件搬运/重复校验。每步conditioning clone→to(device)，`_policy`重复模型inventory/rope.cpu比较/Adam counters；采样结果gate+同步|实际源码冗余可见，但没有阶段占比。测量显式同步每rank合计41s不能8倍相加，也非可归因新增开销|次级候选；不删安全门、不改变RNG/precision；先定位成本再决定是否值得独立单元|

## 进度与限制

- 001：G0已实现并复审。Coder提交前7/7（46.53s），Simplifier独立7/7（45.90s），Planner复审实跑7/7（47.76s）；累计授权源码diff仅测试。`.pi/`未跟踪保持不动。
- 002：I1已实现，Coder提交前7/7（53.30s），Simplifier独立7/7（52.76s），Planner003实跑7/7（48.32s）。CPU耗时波动不作为训练speedup证据。
- 003：选P1单owner双缓冲。当前lease锁跨消费、Journal共享stack是已确认的流水接线障碍，作为同一优化必要改动一起测试/提交，不能直接套executor。V1/P2/F1等仍开放；不判耗尽。
- I1预计收益只发生于保存边界：旧两步每步诊断保存，不外推成正式cadence5每步省8.4min；294.31min两步全生命周期仍不可直接平均当稳态。
- 原Euler-B继续暂停；未发任何GPU/HTTP/SSH请求、未运行训练/推理、未改服务或依赖。
- Planner完成不等于硬件最优；后续每轮复审真实diff/commit/七项输出，最终integration与perf分支发布尚未发生。
