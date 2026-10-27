# CMP170HX 拓扑补充与单节点8卡设计影响

日期：2026-10-27。来源：用户提供的终端输出转录，本会话未执行服务器命令、GPU测试或资源清理。此文件是已完成workflow的后续补充，不修改或重写当时的final.md与inputs快照；最终报告中“NVLink/NUMA未知”的部分现在可按下述证据更新。

## 1. 新增事实

- 用户可见命令为 `nvidia-smi nvlink --status`，各GPU均显示 `Device does not have or support Nvlink`。
- 上方另有GPU拓扑矩阵；矩阵格式与topology查询相符，不将其误写成仅由nvlink --status命令输出。
- 全部GPU对均无NV#，当前没有可用NVLink连接。
- GPU0—4关联NUMA0，CPU affinity `0–19,40–59`。
- GPU5—9关联NUMA1，CPU affinity `20–39,60–79`。
- 上述是亲和关系，不证明进程已绑核，也不推出物理核数。
- 同NUMA内大部分为PHB；跨NUMA全部为SYS。
- PIX配对为(0,1)、(2,3)、(6,7)、(8,9)。PIX路径短不等于P2P可用或带宽充足。
- 前次用户观测每卡65536MiB=64GiB、10卡高占用仍保留；本次拓扑不证明占用已释放。

## 2. 用户提供的矩阵

|GPU|0|1|2|3|4|5|6|7|8|9|
|---|---|---|---|---|---|---|---|---|---|---|
|0|X|PIX|PHB|PHB|PHB|SYS|SYS|SYS|SYS|SYS|
|1|PIX|X|PHB|PHB|PHB|SYS|SYS|SYS|SYS|SYS|
|2|PHB|PHB|X|PIX|PHB|SYS|SYS|SYS|SYS|SYS|
|3|PHB|PHB|PIX|X|PHB|SYS|SYS|SYS|SYS|SYS|
|4|PHB|PHB|PHB|PHB|X|SYS|SYS|SYS|SYS|SYS|
|5|SYS|SYS|SYS|SYS|SYS|X|PHB|PHB|PHB|PHB|
|6|SYS|SYS|SYS|SYS|SYS|PHB|X|PIX|PHB|PHB|
|7|SYS|SYS|SYS|SYS|SYS|PHB|PIX|X|PHB|PHB|
|8|SYS|SYS|SYS|SYS|SYS|PHB|PHB|PHB|X|PIX|
|9|SYS|SYS|SYS|SYS|SYS|PHB|PHB|PHB|PIX|X|

## 3. GPU UUID证据

|GPU|UUID|
|---|---|
|0|GPU-4a964c9b-4342-56a1-e7d7-84067d4a83ab|
|1|GPU-044b9d5f-9bfc-ab71-461e-031111b5188d|
|2|GPU-88d2a221-fcaf-bc5a-06b8-e938bc4c6091|
|3|GPU-951ff2c5-249f-de07-3781-cd373572e3e7|
|4|GPU-0e1cd165-126d-e889-2b87-240d8faddddb|
|5|GPU-074b8279-72a3-3719-7b1f-edf28df18632|
|6|GPU-e6524512-4d1d-1f46-706a-386454902227|
|7|GPU-f41d9e4a-a366-4a7b-59d6-f0607512e4d9|
|8|GPU-fc48c168-0c86-cda9-8749-153b7ee6e7b4|
|9|GPU-8567aab4-abe1-c4fc-d800-c9c818497463|

这些UUID仅是识别证据，不构成用户授权占用任意卡。实际训练须确认允许集合、可用时段和当前资源。

## 4. 对已完成架构报告的修订建议

### 4.1 FSDP2依然是容量/工程候选，但不能默认吞吐最佳

block级FSDP2+reshard_after_forward=true会在每次完整模型forward逐block聚参。扩散采样每个去噪步都执行模型；old-policy replay和有梯度replay又增加完整前向，backward与AC还可能带来额外聚参。

如果被分片参数总量为S、world=8，忽略实现细节，一次完整前向每rank逻辑接收量约为S*(7/8)。N个去噪步约N*S*(7/8)；这是逻辑参数接收量，不是实际跨NUMA链路字节，不是所有rank物理流量总和，也不能直接换算墙钟时间。实际NCCL算法、分层通信、PCIe共享上行和计算重叠都会影响结果。

单节点不等于NVLink环境。8卡必跨当前两个NUMA域，FSDP聚参会遇到跨域通信。因此应把通信准入测量提前，保持“FSDP2是先减少工程变量的基线”而非“最快架构”的表述。

### 4.2 不凭这份拓扑把reshard关闭

若为减少采样聚参而在各rank保留整模型参数，需要额外完整权重驻留。33B量级BF16权重约66GB（十进制）/61.5GiB，已经接近64GiB单卡容量；还未包括FP32岛、激活、VAE、encoder、通信buffer。实际参数数/量化/组件load决定真值。不能把reshard_after_forward=false当无代价加速，也不能假设仅训练LoRA就不需要冻结基座权重。

### 4.3 SP保持关闭，TP2推理候选局限在PIX配对

- 首版SP=1仍合理。SP一般每层attention需要通信，既有性能成本也有正确性适配；不会自动解决PCIe瓶颈。
- 仅当激活容量实测要求且P2P/通信通过时，考虑把SP2组放在PIX配对上。若FSDP仍为WORLD8，局部SP不会消除跨NUMA聚参。
- 若训练侧rollout聚参成为主瓶颈，可提前比较同8卡分阶段vLLM-Omni TP2×4副本；候选配对为(0,1)、(2,3)、(6,7)、(8,9)。这不是已经验证的优选：TP同样有每层通信，H3跨引擎策略score/packing/LoRA映射、显存切换和额外副本都有成本。
- 不引入HSDP，不把两个4卡FSDP组再同步副本梯度偷偷包装成别的名称来绕过用户范围。不默认4卡训练+4卡采样。

### 4.4 候选物理卡分配

若全部获准且空闲，拓扑上可首先测量物理卡 `0,1,2,3,6,7,8,9`：NUMA0四卡、NUMA1四卡，保留四对PIX。该集合仅是benchmark候选，不是已确认可用GPU或最优配置。用UUID固定映射；设置可见设备后local rank编号变化，不能混用物理编号和逻辑编号。未选择的4/5不自动变成可占用reward卡。

### 4.5 CPU/RAM也需要NUMA放置

每个rank的数据预处理、轨迹CPU池、pinned memory及H2D/D2H尽量匹配GPU的NUMA域。若CPU fallback reward很重，按节点规划CPU线程和内存，避免所有任务挤在NUMA0或oversubscription。需要验证实际affinity/cpuset，不把截图中的CPU affinity当已完成绑定。

## 5. 下一步证据与决策门（仅计划，未执行）

1. 用户确认获准8卡UUID、空闲时段、host RAM/CPU以及目标画布/采样步数。
2. 只读核对实际/最大PCIe link generation与width，训练torch/CUDA/capability/attention库版本，CUDA P2P能力矩阵。
3. 获授权后在空闲卡运行有界通信基准：PIX pair、同NUMA PHB pair、跨NUMA SYS pair；测小/大payload单向/双向以及多对同时传输。
4. WORLD8 NCCL all-gather/reduce-scatter及LoRA相关all-reduce，以真实block/gradient大小分档；看最慢rank、共享链路、是否host staging，不仅看单对峰值。
5. 真实模型最小目标工作量分阶段profile：模型加载、no_grad denoise、anchor、replay/backward、decode、reward、checkpoint。控制采样语义和batch不变后再比较FSDP与TP2推理。
6. 报告固定任务的AV/s、GPU-hours/有效样本、峰值GPU/CPU、失败重做率与质量，不以8卡总显存或0%利用率推断速度。

## 6. 结论

用户的拓扑解读成立，并关闭了“是否有NVLink”和“NUMA映射”的证据缺口；PCIe/P2P实效及可用资源仍未知。当前设计应升级为“PCIe+双NUMA通信敏感的单节点8卡方案”，提前比较通信成本，不承诺FSDP2或TP2谁更快。无需为了这个变化立即重跑全部八个Agent或修改框架源码。
