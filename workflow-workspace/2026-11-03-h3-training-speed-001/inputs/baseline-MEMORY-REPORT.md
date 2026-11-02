# 480×320两次更新：显存与耗时报告

**全程allocator峰：8个训练rank均19.708485GiB，发生于第2轮VAE decode；reserved峰62.805–62.871GiB，发生于第1轮backward或第2轮current_forward。两种峰不能混称“图占用”。**

## 全程每rank峰值
|rank / 物理GPU|allocated峰GiB|reserved峰GiB|整卡采样最大GiB|本PID采样最大GiB|CPU RSS采样最大GiB|
|---|---:|---:|---:|---:|---:|
|0 / GPU0|19.708|62.871|63.376|63.363|63.37|
|1 / GPU1|19.708|62.805|63.386|63.373|6.75|
|2 / GPU2|19.708|62.871|63.376|63.363|6.58|
|3 / GPU3|19.708|62.805|63.362|63.350|6.58|
|4 / GPU6|19.708|62.871|63.376|63.363|6.66|
|5 / GPU7|19.708|62.805|63.362|63.350|6.53|
|6 / GPU8|19.708|62.871|63.376|63.363|6.81|
|7 / GPU9|19.708|62.805|63.362|63.350|6.88|

HPS独立GPU4/PID3991487：allocated峰15.730GiB、reserved峰15.977GiB，整卡/PID采样峰16.348/16.336GiB，CPU RSS采样峰31.87GiB。它不是训练rank4（训练rank4是物理GPU6）。

## 时间曲线

[全10物理卡SVG](final-analysis-v6/memory-curves.svg) · [PNG](final-analysis-v6/memory-curves.png)

![rank0与独立HPS预览](final-analysis-v6/rank0-and-hps-preview.png)

## 阶段表（rank0示例）
完整8rank/逐step/direct与ancestor-inclusive版本见[GiB CSV](final-analysis-v6/phase-peaks-GiB.csv)。下表内存数值均GiB，是含子阶段的最大值，**峰值不可相加**。

|step|phase|allocated峰|reserved峰|整卡采样峰|PID采样峰|采样点数|
|---|---|---:|---:|---:|---:|---:|
|1|rollout|12.961|51.744|52.237|52.225|2555|
|1|vae_decode|19.538|51.744|52.237|21.080|184|
|1|anchor_forward|12.972|51.969|52.474|52.461|256|
|1|current_forward|15.993|62.848|63.353|63.340|256|
|1|backward|15.856|62.871|63.376|63.363|262|
|1|gradient_clipping|9.308|42.877|N/A|N/A|0|
|1|optimizer_step|9.539|43.102|43.606|43.594|3|
|1|zero_grad|9.462|51.969|N/A|N/A|0|
|1|checkpoint_validation|9.384|43.102|43.606|43.594|1006|
|1|checkpoint_save|10.103|43.102|43.659|43.646|269|
|2|rollout|13.128|62.494|63.052|63.039|2555|
|2|vae_decode|19.708|62.494|21.401|21.389|181|
|2|anchor_forward|13.139|52.156|52.722|52.709|254|
|2|current_forward|16.148|62.805|63.370|63.357|257|
|2|backward|16.011|62.525|63.091|63.078|264|
|2|gradient_clipping|9.471|39.422|N/A|N/A|0|
|2|optimizer_step|9.547|39.490|N/A|N/A|0|
|2|zero_grad|9.470|52.156|N/A|N/A|0|
|2|checkpoint_validation|9.392|39.490|40.056|40.043|1002|
|2|checkpoint_save|10.110|39.490|40.056|40.043|239|

采样阶段缓存可到51–62GiB，而解码前原有unused-cache释放会下降。VAE峰约19.54→19.71GiB；第二轮已有持久optimizer/runtime状态，差额未作精确分解，不等于链式留图。8rank每次4个单选中转移grad root前向，反向后预测对象全部释放；梯度累积不等于留图：每样本10步采样无grad，只重建所选1步，4样本依次反向后释放，未保留采样链图。clip/Adam/保存均独立记录。标准DDP或更小显存卡是否能跑不能由此直接推断：当前是WORLD8/FSDP且保留原40GiB admission/16GiB安全边界。

## 两轮阶段耗时
|rank0宏阶段（instrumented）|第1轮min|第2轮min|
|---|---:|---:|
|rollout|42.628|42.600|
|decode+reward（含4个观看副本保存）|51.815|51.318|
|optimize全部|13.014|12.994|

以下单位秒，来自entry-event至同步末事件；含子阶段，不能与父阶段重复相加。源宏阶段计时还包含完整span进出日志成本。

|细分阶段|第1轮s|第2轮s|
|---|---:|---:|
|anchor_forward|256.144|254.950|
|current_forward|255.817|256.249|
|backward|264.920|265.359|
|gradient_clipping|0.212|0.208|
|optimizer_step|2.357|0.122|
|zero_grad|0.014|0.018|
|checkpoint_validation|1009.135|1007.942|
|checkpoint_save|269.872|241.260|

instrumented optimizer_step测量段约2.36s/0.12s（含标记、日志与必要同步），不是纯Adam kernel微基准；整个optimize约13min不是该函数单独耗时。checkpoint validation约16.8min/轮，包含原有全缓存/内容身份验证；保存约4.50/4.02min，15029个文件/2.026GB每份。没有为缩短时间跳过强校验。

## 观测方法与边界
- 所有内存单位GiB=2^30 bytes。allocator是每PID/指定CUDA device的PyTorch统计；**allocated是活跃张量占用，不等于仅autograd图；reserved包含活跃张量与缓存/碎片，不等于活跃图**。
- 8rank完整事件序号/统一reset前后持久化/high-water重建全部通过，训练进程均正常process_end。每rank 458–462次reset的旧峰值均先落盘。没有靠新增empty_cache美化曲线。
- 独立nvidia-smi整卡/进程原始采样 **17621条**，覆盖服务切换前、准备、两轮全部工作、保存和退出，时长294.31min。目标1s，实际最大间隔 **4.868s**，超过1.5s共28次。采样最大值不能冒充捕获所有瞬时峰。
- 每阶段NVML值按单调时间关联最近进程阶段事件；GPU/进程查询顺序存在时间偏差，边界归因近似。**N/A=该短阶段未碰到采样点，不是0内存**；如第二轮Adam仅约0.12s，没有直接NVML点，但allocator峰仍保留。
- 图用5s observed-max bins；allocator事件以点标记、超过15s无观测不画斜坡。线段只是连接观测，不是连续瞬时曲线；分段peak在读取时刻标记，实际峰发生于对应段内。
- NVML减allocator错时峰值不能得到精确CUDA/NCCL开销。HPS在物理GPU4独立进程，不能加进训练rank；旧HPS仅切换前短暂出现，按external单列。CPU RSS是CPU观测值，不是GPU峰值，且不是完整ru_maxrss。
- 每rank显式测量cuda.synchronize等待合计41.12–41.15s，互相并行，不能8rank相加当墙钟额外延迟。它不含所有原有同步，且部分等待只是把稍后本来要等待的GPU工作提前等完。未执行无测量对照，因此**不能精确归因增量测量开销**；阶段耗时已受同步/日志影响。
- 外部监控顺序查询耗时合计5277.1s、最长1.874s，是并行监控开销观测，不可直接加到训练耗时。原有VAE unused-cache release共256次、HPS offload cleanup共64次，已逐次记录；新增调用0，计数见FINAL-AUDIT.json。

## 与旧384×256的关系

像素数确为1.5625倍，当前width480/height320均未缩回。旧run decoder会重置peak，19.37GiB是末段口径；旧reserved62.5–62.9GiB不等于allocated或图。**无可靠旧全程allocated基线，不计算显存改善/增长比例，也不把历史约63GiB单卡reserved当保留整条采样轨迹的证据。** 新HPS服务还与训练卡分离，不能当严格同条件旧/新A/B。
