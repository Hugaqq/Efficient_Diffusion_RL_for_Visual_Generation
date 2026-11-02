# 480×320两次真实更新与完整显存记录：最终报告

**L1工程验收完成：fresh run恰好2次真实Adam更新，step1/step2均保存并独立完整性检查通过，真实train/supervisor/monitor exit0，SUCCESS committed_steps2。新任务累计实际更新2，验证额外更新0；未启动200或其他实验。**

**L2不作全面通过：8片文件/几何合格，但物理教学动画prompt的样帧生成了真人吉他弹唱，水下男孩衬衫颜色也未满足蓝色。当前两步不是本H3协议的L4固定评估收益证明。**

## 问题与固定合同
主问题：在480×320、原训练/解码/评分方案下，能否完成两次真实更新并可靠记录全阶段内存？相对前一384×256单更新任务，本次改分辨率和更新上限，并加入观察性日志/同步；不是严格速度/质量A/B。

- 480×320（3:2，像素153600），124帧24fps；10次转移，每prompt组从0..4均匀选一个随机转移，其余Euler、terminal index9非随机。
- WORLD8/SP1/TP1、4prompt×8=32/轮、single-cell microbatch1；LoRA rank32/alpha64、AdamW lr1e-4、原beta/eps/clip/混精度/FSDP方案。无新增empty_cache/降帧/降batch/降精度/改分片。
- 正式cadence5保留；本2步在update1/2各保存checkpoint和4个member0视频（诊断例外）。**视频均来自该次更新前rollout，不是更新后固定评估。**
- fresh pretrained底模/LoRA/Adam/RNG/游标，从未加载旧14步或昨日step1。step1后同PID/Adam状态继续到step2；未为了记账从零重做。
- 最终生产源码 **3db2a05e1c464bd8cbd97c0290a094d869a9d1fd**，VPS→原7项CPU7通过30.02s→commit/push→GPU独立clean/HEAD/import验证。CPU/4KiB零更新测量探针只验证记录逻辑；真实GPU证据另见本run。
- train19700 SHA `610e16e6b0e8a30bbb1e75cf87413bf592fe491faa74d918d57d914d865e0016`，test300 SHA `96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5`。本次只消费train前8个prompt occurrence（每个8样本），不把test用于梯度。
- 底模SHA `0b5fbbdf39278f666984b4a5692da50834c978e7effd050f3145ddafa7dcbc0e`；完整20000条真实文本缓存身份 `0ef2a6300c2ebe64d5590615e1e1efd24f046b66ef8619598ea907e53cc2e14c`。文本模板/encoder/layer50与geometry无耦合；未伪造新缓存身份，原覆盖/finite/文件校验保持。
- HPS真实模型/processor/配置/包与旧服务相等，契约124帧480×320/top37；identity `b07250238b58fce2bccf15359e6b898732deeab35e8b2c8022ac165a7598f801`。只在原生HPS processor内按既有流程处理，不在接口外resize回384×256。获明确授权后优雅替换旧实例，保留其配置/身份/回退命令，新HPS在独立物理GPU4。

## 实际验收
|项目|结果|
|---|---|
|真实更新|8rank同一live run的Adam counter1→2，恰好2个global更新；norm **0.0001499729925061204 / 0.00007273578240725677**均finite非零|
|参数|两个更新所有8rank LoRA本地分片指纹都改变；冻结参数分片指纹均不变。step2 before是携带的step1-after审计基线，不冒充额外独立pre-step2快照|
|图/前向|每rank每步4grad/44nograd root调用；合计8grad/88nograd，8次prediction反向后释放均true。内部activation-checkpoint层重算不计入root调用数|
|采样|64条trajectory×10=640转移，逐条只选一个0..4随机index，terminal Euler；两轮选择分别[3,3,1,3]、[4,4,3,3]|
|一致性|64个rollout/anchor old-score检查最大差0；64个初始current ratio均1.0，不等于严格官方双流Flash无偏证明|
|真实评分/视频|64个finite HPS视频评分，7936帧实际评分；每轮全部32条/index存档，8个member0 MP4及receipt取回，SHA与独立ffprobe480×320/124帧/24fps/32kHz双声道通过|
|保存|两份checkpoint各15029文件/2,025,683,882bytes，既有manager全树完整性检查；next_optimizer_step1/2、data cursor4/8、accumulation0、RNG身份有效，step1未被step2覆盖|
|冷加载口径|本run做完整性/保存进度检查，**没有额外GPU冷加载**。不把它称作已实际reload；历史同格式冷加载证据仍按旧run范围有效|
|退出|train exit0@13:31:54Z，监督/监控exit0@13:32:29Z。8训练rank、torchrun、预检、监督、监控PID均退出；仅新HPS3991487 ready，物理GPU4约390MiB|

## 显存与阶段耗时
[完整显存/方法报告](MEMORY-REPORT.md) · [全曲线SVG](final-analysis-v6/memory-curves.svg) · [PNG](final-analysis-v6/memory-curves.png) · [逐rank/阶段GiB表](final-analysis-v6/phase-peaks-GiB.csv) · [每rank总峰表](final-analysis-v6/rank-summary-GiB.csv) · [全部阶段计时](final-analysis-v6/phase-times.csv)

全8rank allocated峰均 **19.708485GiB（第二轮VAE decode）**；reserved峰 **62.805–62.871GiB（前向/backward缓存）**，整卡采样最大约**63.386GiB**。这不是“63GiB活跃图”。HPS独立allocator峰15.730GiB，不混成训练rank。raw **17621**样本、最大间隔**4.868s**，8rank所有reset前旧峰持久化并可独立重建。短Adam阶段NVML可能N/A，不能补造瞬时值；详细说明见报告。

训练数据准备/权重读取/强验证与GPU计算均计入全程，监控覆盖约**294.31min**（4h54min）；不是“Adam更新函数用了几小时”。每轮采样42.6min、解码/评分约51min、全部优化约13min；optimizer_step测量段约2.36s/0.12s（含日志/同步，不是纯kernel微基准）。测量同步等待每rank约41.1s，但不是可归因的新增墙钟开销；没有无测量对照，不报告虚构增量或speedup。

## 训练奖励不是固定评估曲线
|关联更新|真实样本|mean|population std|min|max|
|---|---:|---:|---:|---:|---:|
|1|32|6.5634|1.4754|2.5247|9.4554|
|2|32|4.4978|3.3862|-3.2803|10.3595|

两批prompt不同，且均为更新前训练rollout；均值升降不能说明学习收益。原始64行见[CSV](rewards-all64.csv)，含group/member/随机index；每组8人，非member0的其他56视频没有按本协议要求额外保存，不能重生成冒充原样本。

## 画质/L2观察
- 第1轮：士兵白背景队列、厨房背影、地球上人物、太空战舰等主场景可辨；gamer身份/178cm/zombie等细节不能完整确认。
- 第2轮：猫看雪窗、俯视球场、水下男孩可辨；**物理教学动画对应样帧是吉他弹唱，明确视觉语义不符**；男孩衬衫是白色而非蓝色。
- 8片是有损观看副本，人工只检查少量1s/4s帧（首片另有4帧）。未判断全片时序/音频；没有用照片主体清晰替代严格native fidelity对齐。当前没有严格同输入384×256配对，项目已存在的其他native/A100定性对照仍在其原范围有效。
- 两步支持本固定WORLD8协议L1，提供有限L2观察与L3原始评分，不证明本H3新协议L4学习收益。历史Wan SMALL40正向结果不被抹掉，也不直接外推为本run收益。

## 失败/接续与证据
attempt001在HPS初始化因LOCAL_RANK环境错误退出、训练0更新；attempt002全预检通过但双HPS CPU驻留不满足模型加载门禁，训练0更新。原始材料保留。明确授权替换旧HPS后attempt003 fresh完成2更新。CUuuid前缀测量缺陷已在启动前按CPU7/Git流程修复。最终仅离线分析修正退出后/proc空字段覆盖rank、断开长无观测区间斜线，原始数据和生产模型源码未改。

- [独立完整审计](FINAL-AUDIT.json)：640转移/64评分/64重放与ratio、8rank graph/参数、媒体与退出。
- `attempt003-final-raw-evidence.tar.gz`及`final-evidence/`：原始整卡/进程、逐rank/独立HPS JSONL、train/HPS日志、config/真实身份/退出/控制脚本。约203MB原NVML解压数据，完整保留，不把摘要替代原文。
- `attempt003-checkpoint-metadata.tar.gz`与`final-checkpoint-metadata/`；完整checkpoint tensor文件仍在远端原址 `/home/user/user/h3-resolution-memory-20261102-001/attempt-003/output/checkpoints/step-1`、`/home/user/user/h3-resolution-memory-20261102-001/attempt-003/output/checkpoints/step-2`，未删除/篡改。`attempt003-final-checkpoint-process-verification.json`给出真实检查与文件SHA。
- `attempt003-media/step1-all4`、`step2-all4`及各自index/review：8片真实MP4/receipt与稀疏预览。
- [证据文件SHA/路径索引](EVIDENCE-INDEX.json)。产物保留旧尝试与旧run，不把旧更新/旧不完整峰值算入本次。

**本任务完成后等待用户；不自动200、不执行研究计划中的新基线/并行方案/推理对照。** 后续实验需新目录/实验卡及明确授权。
