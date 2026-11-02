# H3候选表

基线：生产 `3db2a05e1c464bd8cbd97c0290a094d869a9d1fd`；工作流起点 `86a628cb58ffe558fc91da9daf9fcff2946398a9`。当前轮次002复审HEAD `78fb5c544442df51369b727b00f7677bc38daba3`。G0原子代码commit `e324aaf`已通过独立复审；尚无性能代码commit，未实测任何提速。

## 成本/硬件口径

证据为 TASK 与 inputs 保存报告，不是本轮硬件查询。10×CMP170HX 64GiB，训练物理0/1/2/3/6/7/8/9，HPS独立物理4；不占用/推断GPU5。PCIe Gen2×4、无NVLink、PIX/PHB/SYS跨NUMA为任务已存事实，当前inputs未附完整原始拓扑矩阵；禁止凭型号替代链路测量。

`inputs/baseline-MEMORY-REPORT.md`第二轮：采样42.600min，decode+reward51.318min（含观看副本），optimize12.994min，checkpoint validation1007.942s，save241.260s。三大计算阶段约106.91min（**含优化，不含校验/保存/初始化**）；加checkpoint校验和保存约127.73min（**含优化/校验/保存，仍非全生命周期**）。前两段93.918min不含优化/保存。两步监控294.31min含初始化、审计和退出，不可直接除二当稳态。

allocated峰19.708GiB不代表可额外常驻任意模型；reserved约62.87GiB/整卡约63.39GiB。标准DDP静态68941191680bytes≈64.21GiB，**尚未加激活即超容量**。D2静态35332218880bytes≈32.91GiB仅估算。

|ID|候选/实际证据|收益依据（非已实现）|状态/下一决定|
|---|---|---|---|
|G0|CPU门禁夹具隔离。原6通过1失败；修复后仅测试原函数内局部ROOT夹具与拒绝断言，生产保护不动|无训练提速；恢复后续commit必要门禁|**完成并复审**：Coder `e324aaf`，Simplifier无必要改动/无commit；Planner实跑7 passed/47.76s|
|V1|有界VAE复用。`h3_decode.py::_decode_stream`每次`_load`并finally drop；GPU版每次CPU加载后上卡；32样本各视频/音频各加载一次|用户约8s/样本加载边界尚未证实。若8s是两VAE合计，当前总256s；每rank四样本、仅owner turn内复用可最多省24×8=192s≈3.2min/轮（不计额外offload/hash成本）。占127.73min含优化/校验/保存分母约2.5%；不是倍数提升|已复审现有loader与CPU fixture：现状仅一个VAE模型在内存，复用可能增加CPU pair驻留或GPU offload搬运；8s未分离磁盘/构造/H2D，且loader RNG/可变buffer/缓存失效未定。仍高优先开放，不以allocated低推断可常驻；本轮先做无需新驻留/加载假设的I1，后续继续V1|
|P1|decode/HPS有界流水。`H3DualIterationStages.reward`逐样本decode→score；`aux_turns`整个rank四样本逐owner执行；独立HPS卡无法重叠|若辅助段51.318→27min，则前两段93.918→69.6约1.35×（**不含优化/保存**）；127.73→103.4约1.24×（**含优化/校验/保存、不含初始化**）。均为条件理论估算|高优先后续单元，不能与V1/G0混做。需单解码生产者/单评分消费者、极小队列、lease生命周期、跨rank样本顺序、背压/取消/错误/排空；不留32条FP32视频|
|I1|重复checkpoint live验证。当前源码`checkpoint_boundary`内`validate_live()`；`H3LifecycleBackend.run`立即再调用一次；期间仅只读counter/progress/proof/RNG快照构建，无模型/优化/数据更新|实测整个validation约16.8min/保存轮，不是单次调用耗时。若两次扫描各占约一半可省约8.4min/保存轮，127.73min含优化/校验/保存分母条件约1.07×；cadence5稳态仅摊约1.68min/update，111.08min含优化及摊销校验/保存分母约1.5%。未实测、不能给已达speedup|**002唯一选定**：新鲜边界一次完整验证同时产出boundary+live，立即交原transaction；保留旧boundary API与全部校验覆盖，保留跨边界/restore/最终文件检查，无持久缓存/verified逃逸|
|F1|采样FSDP通信。每block WORLD8、reshard_after_forward=True，BF16 compute/FP32 reduce，AC外包FSDP；每rank40 rollout root前向/轮，另4 anchor+4 current|估算每次遍历allgather接收约56.49GB；不是PCIe实测时间。42.6min含计算/通信/门禁/检查|高收益候选，先可关闭有界profile（算子、NCCL、搬运、CPU等待）或已存事件细分；禁止推导已达speedup|
|F2|D2四副本混合分片|估算接收32.28GB/遍历、静态32.91GiB；不含工作集、聚参峰、VAE|暂缓架构修改。实际`_local_gradient`与checkpoint/binding写死1D WORLD8 Shard(0)、除数8；改mesh不是独立安全一行优化。需梯度数学、组拓扑、checkpoint兼容和获准测量|
|F3|标准全GPU DDP|64.21GiB静态已超64GiB，未含激活|当前协议下排除；不能用allocated峰小为依据切DDP|
|H1|HPS驻留/搬运。`serve_hpsv3_video.py`CPU模型初始化一次，不每请求重载；每请求上CUDA，124个单帧前向，finally CPU offload+empty_cache|每样本约50s是用户估算；保存报告未给onload/forward/offload细分，不可把50s全部视为搬运|候选待细分。独立物理4不自动允许改服务；若将来写可选驻留/批处理，保留健康identity、准入/回退及逐帧数值排序，部署仍另授权|
|T1|条件搬运/重复校验。每步conditioning clone→to(device)，`_policy`重复模型inventory/rope.cpu比较/Adam counters；采样结果gate+同步|实际源码冗余可见，但没有阶段占比。测量显式同步每rank合计41s不能8倍相加，也非可归因新增开销|次级候选；不删安全门、不改变RNG/precision；先定位成本再决定是否值得独立单元|

## 进度与限制

- 001：G0已实现并复审。Coder提交前7/7（46.53s），Simplifier独立7/7（45.90s），Planner复审实跑7/7（47.76s）；累计授权源码diff仅测试。`.pi/`未跟踪保持不动。
- 002：选I1同边界校验结果复用。V1/P1/F1等仍开放；不因本轮较小风险的IO优化而宣布已没有高收益机会。
- I1预计收益只发生于保存边界：旧两步每步诊断保存，不外推成正式cadence5每步省8.4min；294.31min两步全生命周期仍不可直接平均当稳态。
- 原Euler-B继续暂停；未发任何GPU/HTTP/SSH请求、未运行训练/推理、未改服务或依赖。
- Planner完成不等于硬件最优；后续每轮复审真实diff/commit/七项输出，最终integration与perf分支发布尚未发生。
