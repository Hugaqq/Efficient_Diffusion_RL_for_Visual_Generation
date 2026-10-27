# H3-M4a5-data-source — Coding A / W5

## 状态与实际身份

- `implementation_ready: true`：本原子项源码及指定CPU门禁已完成，交simplifier-a；不是W5汇合批准或默认RunController完成。
- worktree：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-08fdc555-6975-494c-b44c-d4f5700316d9`。
- branch：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-08fdc555-6975-494c-b44c-d4f5700316d9`。
- 开始HEAD/基线：`d2a6b8563f5bcc2d5f95bf9e24458dc25d257b91`，初始干净；assignment integration parent `6f5e46fce212e00d547a3ca07fa5524018bef816`，当前基线含W5协调更新。
- 已读task/plan/assignment A/wave-05、前项approved审查、当前H3Adapter/config/interface/cache、stages请求/数据端口及对应真实测试helpers。无前置changes。
- 引擎snapshot SHA pending；无暂存、手工commit/merge/push或切分支。

## 仅新增

1. `visual_rl/data/h3_source.py`
2. `tests/test_h3_mvp_cpu_source.py`
3. 本报告

未修改所有既有源码/测试/断言、B/C同波文件、task/reference/AGENTS或工作流。没有同波新模块import、新runner或通用dataset框架。

## 实际数据/身份与解析

`H3PromptSource`必需参数为本地train/eval路径及各自expected lowercase SHA256、已加载H3Adapter、精确H3CPSDynamics、rank/world_size=8、group_size K、base_seed、device和sample/cell/trajectory预算。

- 真正读取文件并检查原始字节hash；每文件最多16MiB，缺失/叶子symlink/坏hash拒绝。不下载、不生成数据。路径不进入source内容身份；移动同内容文件仍可精确恢复。
- `h3.utf8-lf-crlf.v1`：严格UTF-8，仅LF或CRLF为记录终止符；末行可无换行；恰剥终止符，不strip prompt首尾空格，不把Unicode文本分隔符当换行。零长度记录、BOM、孤立CR和坏编码明确拒绝。非空空白字符本身不被擅自删除。train至少8条，eval非空。
- 构造真实调用adapter.verify_coverage(train,eval)，再真实encode首条取得完整cache index identity；校验已加载model text_dim5120和geometry.joint_layout与dynamics一致。
- source_id由解析版本及train/eval原文件SHA派生；geometry_id由真实geometry、cache index身份、meta config SHA、adapter.initial_state_version派生，rank无关。data_identity进一步绑定sampling/stream版本、完整双grid/eta、seed、WORLD8/SP1/TP1/B1、K/N、device type、预算和video-only score profile。模型权重/资源本身的真实hash准入仍由loader/runtime/checkpoint语义绑定负责，本模块不另hash巨大模型。
- reserve/commit/state/restore重查原文件与真实cache覆盖/内容漂移；构造和状态不靠mtime或路径替代内容身份。

## 确定性与预留提交

- `h3.world8-ordered-cycle.v1`：global occurrence=`step*8+rank`，每rank一完整K组；prompt取train[occurrence % len(train)]。没有shuffle、丢样、余额缩K或空batch退出。K2..512、N<=4096对齐已有world准入；step上限2**31-1仅作整数域保护，controller仍决定训练max_steps。
- group_id由source_id+occurrence派生，sample_id再加member；重复文本的不同occurrence仍不同group/sample。collection/behavior/next_behavior版本由共同data_identity+step派生，所有rank相同，不是权重fingerprint。
- `reserve(step)->H3IterationRequest`只允许next-step且无活跃reservation/poison。逐member实际调用adapter.encode(H3PromptInput)，保留原H3Conditioning，request.conditions使用其CPU拥有副本；从不伪造condition或cache miss零embedding。
- 实际调用adapter.latent_spec_for_batch和prepare_latents，初态为CPU显式双H3Generators产生的FP16 joint。rollout generators放target device，GPU只在未来生产reserve执行，测试仅CPU。
- `h3.sha256-seeds.v1` canonical JSON元组为 `(stream_version, base_seed, source_id, occurrence, member, purpose, modality)`；SHA256前8bytes大端取低63bit。purpose为initial/rollout，modality为video/audio；四个命名空间和sample/occurrence独立，不使用Python hash，不改Python/NumPy/Torch全局RNG。rollout持有每sample/模态顺序流，不按步重新seed。
- reserve不推进counter/behavior；commit只接受同一个active request，重查输入后恰推进step/global cursor/behavior一次并清除request/typed条件引用。abort清除active并poison，不回滚参数、不就地重抽。reserve失败同样poison；commit漂移错误不推进且清引用/poison。
- 数据对象借用adapter，不负责关闭其模型/aux；只保文本/小身份及当前K条请求，不保历史轨迹/媒体。完整trajectory预算公式沿pool：`K*(N*(4*D+4)+2*D)`，不声称包括embedding/caller副本/模型workspace的总RSS。

## 未来runtime可直接使用的端口

```python
source = H3PromptSource(
    train_path=..., eval_path=..., train_sha256=..., eval_sha256=...,
    adapter=loaded_adapter, dynamics=dynamics, rank=rank, world_size=8,
    group_size=K, base_seed=seed, device=device,
    max_samples=..., max_cells=..., max_trajectory_bytes=...,
)
# H3IterationStages绑定：
# data=source, read_behavior_id=source.read_behavior_id
request = source.reserve(step)
conditioning = source.conditioning_for(sample_id)  # 仅active，真实adapter产物
# typed predictor使用该conditioning及其condition_identity，不能重造伪身份。
source.commit(request)  # 原BaseTrainer完整tail调用
state = source.state_dict()
```

只读`source_id`、`data_identity`、`geometry_id`、`next_global_batch_id`和`read_behavior_id()`可用于runtime/CheckpointProgress。next_global_batch_id是共同data identity与下一批8个global occurrence IDs的摘要，不是rank-local样本摘要。

## JSON-safe边界状态

schema_version=`h3.prompt-source.v1`，顶层精确字段：

- `binding`：上面的解析/采样/stream/source/文件hash与counts、geometry/cache/model config/初态版本、world/placement/SP/TP/B1/K/N/精确双grid/eta、seed/device type/预算/profile。
- `data_identity`、`rank`、`device`。
- `next_optimizer_step`、`global_cursor=step*8`、`behavior_id`、`next_global_batch_id`。

不保存模型、tensor、Generator、request或pool；返回深层JSON拥有副本。仅无active reservation且非poison时可state_dict。

`load_state_dict`只允许fresh未使用对象一次；对当前真实文件/cache/模型条件合同和精确schema/类型/所有字段进行核对，进度必须内部一致。错误rank/world/布局/stream/seed/eta/预算/条件身份、额外字段、bool step、热恢复和mid-collection拒绝；失败对象poison。恢复不碰全局RNG，外层RankRNGSnapshot必须仍最后恢复。此处same-topology指固定逻辑rank/WORLD8/placement/device/SP1/TP1/B1合同，实际硬件拓扑资源身份由上层checkpoint contract提供。

## 六项真实CPU测试

1. tmp_path真实CRLF/LF train/eval及safetensors cache，真实registry H3Adapter/session meta加载（仅重transformer类tiny）。观察coverage回调仍调用真实方法；8rank不同sample和独立完整组、重复文本不合并、精确空白/中文、共同IDs；独立手算SHA种子和torch.randn双流初态；两次commit/循环策略及全局三类RNG不变。
2. active重复reserve、错误request commit、重复commit、mid-collection state拒绝；typed condition与request tensor mutation隔离；commit后weakref证明source不保已提交request/condition；abort poison不可补样。
3. 两次commit后真实JSON往返→fresh同rank data object加载→下一request的identity、prompt、embeddings、初态、显式generator状态逐值相同；同内容不同文件路径仍一致；不抢先改变全局RNG。
4. schema/rank/world/K/stream/cache/schedule/seed/eta/预算/进度字段及热恢复负例，全部拒绝，不移除旧断言。
5. 空/短train、空eval、坏UTF8/BOM/孤立CR/空行、缺文件/symlink、train/eval cache miss、hash/缓存漂移、错误layout/device和预算拒绝；commit时文件漂移无进度伪提交。
6. 真实新source作为现H3IterationStages.data，实际reserve/commit未mock；真实H3Adapter.predict使用source原H3Conditioning、真实tiny loader/PEFT CPU物化、prepared guard/Adam、真实双VAE decode与现BaseTrainer一轮执行。tail推进global cursor8/step1、pool release、Adam step1，边界JSON恢复后next request与连续路径一致。

第6项明确使用已批准stage feature的CPU `_world_leaf` 与 `_CoordinationLeaf`、media-dependent评分叶子；本项验证真实新数据端口与adapter/stages消费，不冒称重新取得真实八rank协调/真实HPS组合或GPU证据。既有W4真实stages+HPS+Coordination八线程组合测试原样保留并在本轮通配符重跑通过。

## 最后源码编辑后门禁

在上面worktree根目录原样执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3 exit0：**114 passed, 9 warnings in 8.65s**（基线108+本项6，不含未汇合W5 B/C新增项）。
- Flash exit0：**7 passed in 12.71s**。
- 9条warning来自既有checkpoint no-dist DCP与损坏fixture重写提示，未过滤；无skip/xfail、无全量。
- git diff无tracked变动；git diff --check exit0。新源码/测试untracked不在该diff内，已读回实际源码并真实import/执行，不冒称snapshot diff已审核。
- 原Flash完整文件相对授权d818b8a231b52a9a024fa2f522057acdbb7bf099历史diff为空，exit0；本轮未改任何旧测试。

## 待审与未完成边界

交simplifier-a审核真实来源/hash、精确解析、预留提交、确定流及fresh恢复语义；changes回A重测两集。引擎形成snapshot后由orchestrator记录，W5汇合仍需完整门禁与data/profile/publisher实际组合。

本项不装配唯一RunController，不写checkpoint manager，不证明模型权重/缓存生成过程、CUDA generator/FSDP2/NCCL/HPS124资源或质量可用。无GPU/服务器/HTTP/PG初始化/下载/环境升级/push。**GPU_PENDING，MVP未完成。**
