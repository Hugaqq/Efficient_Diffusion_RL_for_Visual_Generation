# Assignment A — W5

- active: true
- task_id: H3-M4a5-data-source
- pair: coding-a → simplifier-a
- execution: parallel W5；decisions/wave-05.md
- integration_parent: `6f5e46fce212e00d547a3ca07fa5524018bef816`
- dependencies: 现H3IterationRequest/stages、注册H3Adapter、W4组合均approved；代码汇合fb085fb35fb239cdd4c151d39decbd7fae268c9d。无changes。

## 唯一目标/文件

实现供现H3IterationStages真实消费的**本地prompt数据预留/提交与same-topology边界恢复**，取代测试Data类作为未来生产来源。不另造通用dataset框架。仅新 `visual_rl/data/h3_source.py`、新 `tests/test_h3_mvp_cpu_source.py`、W/coding-a/H3-M4a5-data-source.md；所有现源码/tests只读，不import B/C本波新模块。

## 固定最小行为

- 构造接受本地train/eval prompt文本路径、显式expected文件SHA256、已加载真实H3Adapter、现H3CPSDynamics、rank/world8、K>=2、base seed、target device、显式sample/cell/trajectory预算。真实读取文件/核验字节与hash，不下载、不编造dataset。明确UTF8逐行规则、保持精确prompt内容（只剥行终止符，不strip文本），空行/坏编码/缺样本清楚拒绝；train至少8条，eval非空且覆盖cache。source identity绑定原文件内容/解析版本，不用路径当内容身份。
- 前置adapter.verify_coverage(train_prompts,eval_prompts)必须实际执行；model/condition维度/cache身份与geometry/dynamics layout吻合，不能随便给stage共同字符串。共同geometry_id可从真实geometry/cache与initial_state版本canonical payload派生，rank无关。
- 最小确定性采样：每collection全world8个prompt occurrences，每rank一个完整K组；global occurrence=step*8+rank，按train原序循环读取（明确循环策略版本，不宣称随机shuffle）。相同文本不同occurrence仍独立group_id。sample_id由source/occurrence/member稳定派生；固定world/placement/SP1/TP1/B1/N不随余额改变。
- `reserve(step)->H3IterationRequest`：仅next-step、无既有reservation/poison；绑定实际IterationIdentity、collection/behavior/next_behavior与prompts。实际调用adapter.encode(H3PromptInput)，conditions填其CPU embeddings，并保留可供后续predict绑定的真实cache/prompt身份；不要伪造H3Conditioning或将缓存miss当零embedding。
- 初态实际adapter.prepare_latents，用CPU明确video/audio H3Generators；rollout generators在target device。初态与rollout、模态、sample各独立SHA-derived seed命名空间，种子算法/版本明确、63bit域；不污染全局Python/NumPy/Torch RNG。CUDA generator仅生产调用，测试只CPU叶子。N步顺序流复用既有CPS约定，不每步意外重seed。
- reserve不推进cursor/behavior；commit只接受同一个活跃request、恰推进一次global collection/next-step、data cursor/behavior。abort放弃reservation并标记attempt失败，不回滚参数、不原地重抽补样；后续须新对象从完整checkpoint恢复。不存全数据媒体或历史轨迹，committed request引用及时释放。
- 暴露只读read_behavior_id、共同next_global_batch_id（world下一批occurrence IDs摘要，不是本rank独有摘要）和data identity，供future runtime构造现CheckpointProgress。`state_dict()`只允许无活跃reservation、非poison边界，返回小JSON-safe schema（source/parse/sampling/stream版本、seed、rank/world/K/N/geometry/相关cache identity、next step/cursor/behavior）；不存模型/tensor/Generator/活跃pool。
- `load_state_dict`只到fresh未使用数据对象，精确检查文件内容、rank/world/布局/stream与模型条件身份，不允许elastic/重编号/热恢复/mid-collection。恢复next请求应与不中断完全一致（IDs、prompt、embeddings、初态及显式RNG状态）。默认global RNG由外层RankRNGSnapshot最后恢复，本对象不要抢先改全局RNG。
- 可合理限制max_steps避免无限训练，但不能隐式截断K/全N；耗尽目标只由controller max_steps决定，不用data空batch触发单rank退出。

## CPU验收

真实tmp_path train/eval与safetensors cache，真实H3Adapter加载/encode/prepare初态（重transformer叶子tiny）。证明8个rank有不同样本/独立完整组、重复文本不并组、两次commit与reserve/abort严格行为、cache coverage/文件漂移/shape/预算及错误身份拒绝。保存边界data状态→fresh同rank对象恢复→下一request和不中断逐值对照，global RNG未改。

至少一个case将真实新data对象提供给现H3IterationStages/BaseTrainer跑完整一轮（使用已批准W4组合测试低层叶子或明确tiny；不能mock data.reserve/commit）。验证tail真实推进、pool释放、nextstep与恢复状态一致。不改原108项断言。

最后编辑后plan固定Flash7/H3通配符真实全过，无skip/xfail/全量；交付branch/HEAD、命令/exit/数量/时间、diff/check与state schema/精确runtime接线接口。Simplifier A审真实数据来源/预留原子性/确定流和无模型图保留，changes回A。GPU_PENDING。
