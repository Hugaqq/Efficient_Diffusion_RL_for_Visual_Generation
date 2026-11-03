# M0/M1 最小外部证据卡 — 待输入/待批准，禁止自动执行

状态：Planner round-005 `waiting_external`。这不是FINAL，不是预算耗尽，也不是已有失败/不明副作用。当前已审代码和CPU门禁没有未解决的已知失败；等待的是原始成本证据或明确的新GPU实验授权/结果。

## 先选M0：补齐已存证据，新增GPU更新=0

优先由用户/获准证据持有人提供旧run的**已保存文件**到本任务inputs下新子目录，保留来源/文件SHA/原始时间/源码身份。不请求启动任何模型/服务、SSH或benchmark；Planner不能自行跨工作区取档。

现有inputs实列仅基线FINAL/MEMORY摘要、分片估算、阶段峰值JSON、旧CPU结果/历史规则/CPU短入口；没有逐样本原始时间事件、完整拓扑矩阵或CUDA operator trace。MEMORY报告链接的原phase-times/raw archive不在当前inputs。

最小优先包：
- 同协议**第二轮32样本**的逐rank train JSONL/日志中 `h3_reward_decode_begin/end`、`h3_vae_load_begin/end`、`h3_vae_decode_begin/exit`、`h3_reward_score_begin/end`、`video_save`、`h3_aux_begin/end`及其sample/时间关联原文。提供有界切片即可，但要保留bookends、rank、sample、component与来源SHA，不只再给“8/37/50秒”平均数。
- 对应HPS已有 `hps_score/hps_onload/hps_offload` JSONL或原日志；无法提供则标NA，不因缺失重启服务/新增请求。
- 已保存真实VAE配置及weight header/文件大小清单（不需权重payload），用于计算FP32权重/缓存容量；已有host可用内存/RSS样本及保存拓扑/UUID映射若可用一并提供。不存在则标NA，不凭型号或allocated推断。

可附32行CSV以方便核查，但原始片段仍需可查：
`rank, sample_id, owner, group, member, decode_begin/end, video_load_s, audio_load_s, complete_decode_s, score_begin/end, export_s, hps_onload_s, hps_offload_s, source_file_sha`。

边界必须写清：当前GPU decoder的load计时含CPU构造/读权重**及H2D上卡**，不是纯磁盘；decode流计时含既有格式/搬运步骤。主D应取完整decode lease进入至CPU媒体就绪，H应含原score_one/JPEG/HTTP及该样本观看副本；不要将细分与父阶段重复相加。

M0可帮助选择下一代码项，但不能证明新P1 GPU并发已安全/提速或F0 CUPTI兼容。收到M0后用户明确恢复Planner，重新审证据；**不自动升级到M1**。

## 仅M0不足时申请M1：一次新fresh两更新测量

**未获授权，下面只是请求卡。** 用户须明确批准M1、执行者、最终SHA、全新输出目录、设备/服务/日志归属和资源时窗；当前角色不执行/部署/请求HPS。若没有批准，则继续等待，不重放工具或恢复旧run。

### 目的与上限

仅测候选GPU兼容、P1真实D/H关键路径与FSDP/compute/搬运证据。最多**新训练2个global optimizer更新**，按原入口每轮32样本，总64条/640转移；原HPS总64次POST、7936评分帧、无POST重试，健康准入最多9次GET（8rank原准入＋至多1次只读核验）。不开额外warmup/冷加载/推理/评分/评估/基线A/B/第三更新。

使用已提交两步入口 `launch_h3_resolution_memory.py` 的 `--reward-pipeline` 与 `--profile-rollout`；固定one-shot为每rank第二更新、第二本地sample、index0，共8份trace。不开任意schedule/全程profile。flag不是授权，本文不提供可直接启动的GPU命令。

申请单次6小时**墙钟上限**，只是安全/授权上限、不是预测耗时或速度承诺；不得据此扩工作流8小时活动预算。若用户不同意时窗/请求数，应先改单而不是私自缩协议/换benchmark。

### 必须先满足的前提

- 先由获准integration/主会话核候选提交、真实原七项与diff/import、专用perf分支安全交付；部署只能Git精确匹配clean SHA，不能复制源码热修。当前功能顶为`9b0a6b3`，Planner已测快照`0eb2a5b6703fc726a063918eaaacdf63672e876f`；实际批准/部署SHA若不同必须重新验证。**当前没有最终integration/发布成功的证据**，不能虚称已发布。
- 输出/monitor/证据目录均由用户明确指定、全新且不在源码/原始模型/缓存/旧run内，不resume、不覆写原3db2a05 run/旧checkpoint/旧证据；Euler-B保持暂停。
- 核实完整物理→UUID→rank映射：训练仅0/1/2/3/6/7/8/9；HPS必须实际在独立物理4；GPU5不占用。保存已有或获准只读拓扑/链路/NUMA事实，不按型号猜NVLink/有效带宽；不额外跑PCIe/NCCL微基准。
- 现有HPS的权重/processor/配置/逐帧124/top37 identity和endpoint健康必须匹配；不得改reward服务源码、重启/替换/驻留/批处理/占用别人的服务。服务与日志sink须可在不改写冻结旧run证据的前提下使用；不满足则停止申请，不擅自服务变更。没有现成HPS细分日志时标NA，不通过改服务补齐。
- 原科学协议全部不变：480×320、124@24、10步first5选1其余Euler、4×8、原模型/数据/缓存/精度/LoRA32/64/lr1e-4/优势/Flash/缩放/单cell反向释放；formal cadence5保留原两步诊断保存例外。模型/数据/HPS/源码完整核验不省略。
- CUPTI/public CPU+CUDA支持、host与磁盘容量必须真实准入；trace发布上限每rank128MiB不是临时磁盘/Kineto RAM硬界。已有24GiB host reserve、40GiB decode准入/16GiB余量和monitor/journal健康门不动；额外采集开销不能用降低安全门换取。

### 返回给Planner的最小数据

1. 实际部署SHA/clean/import路径、完整配置及模型/缓存/数据/HPS身份、设备/进程映射、fresh/Adam空状态证据。
2. 每rank两个update审计：累计root grad8/nograd88、单随机index0..4、frozen不变/LoRA改变、finite非零norm、Adam1→2、anchor/初始ratio日志、tail/cursor/成功/退出状态；两份保存完整性与8个member0视频/receipt真实核查。完整冷加载未执行不得称reload成功。
3. 每rank原journal、P1 CPU-consumer events、stdout/VAE细分、HPS可用原日志、NVML/CPU RSS监控。按sample对齐D/H/观看副本，画decode-score重叠与owner切换的fill/drain；未记录的许可等待只能标“推断gap”，不能伪造精确Queue等待时间。
4. 8份原始profile trace和小receipt，独立流式重验trace SHA（代码末尾仅检查大小，不能冒充再次全hash），确认真实CPU/CUDA活动而非测试合成。报告compute/NCCL/memcpy/CPU空隙及重叠；区分gate NCCL与FSDP聚参，窗口不覆盖row初始H2D/全rollout，不能直接外推为42.6min的完整比例。
5. 每阶段及完整wall：初始化/采样/解码评分导出/优化/完整校验/保存/收尾分别给出，含哪些父子阶段明确；profile窗口/导出开销单列。P1 payload最多2条不是整RSS界，报告真正host/GPU峰及monitor缺口。

### 失败规则与不作的声明

原准入/完整性/identity/finite/图/余量/monitor/profile门失败、HTTP错误、日志超限或deadline时按现有整attempt失败路径退出；无自动重试/resume/重新采样/删除部分checkpoint/再更新。记录真实已提交次数及仍在途的服务操作，不把不可强杀请求/worker叫已排空；确认所有owned进程/线程退出前不能续更，不碰未owned服务。用户主会话核查后才可另批恢复。

此M1有profile扰动，旧294.31min也非严格同条件对照，**不能据一次结果声称端到端speedup/2倍**。CPU已证明相同JPEG请求字节，不等于真实GPU逐帧数值/排序严格配对证明；本卡不额外请求HPS、不用有损MP4回读冒充原评分张量。严格原生输入配对的逐帧/top37容差与最终同协议两步速度/保存恢复验收，仍须另行具体实验卡/授权（代码若改变更不能复用本次结果冒称最终）。

## 恢复条件

用户明确恢复，并提供M0可核原始证据；或明确批准M1且由获准执行者返还上述可审核结果。Planner再读真实材料，更新候选表并选择一个代码单元/继续等待。**没有正常结束、没有自动执行、没有自动重放。**
