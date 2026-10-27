# H3-M4a4-object-integration — Coding A / W4

## 判定 / 实际身份

- `implementation_ready: true`：新增组合验收测试和本分支门禁已完成，交simplifier-a；测试确实组合W3三真实对象，不是生产controller装配交付。
- 实际worktree：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-4b0c636a-c9a4-48a8-adb5-46a41a065fd6`。
- 实际branch：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-4b0c636a-c9a4-48a8-adb5-46a41a065fd6`。
- 开始HEAD/基线：`477325141c0911aeb3cdd8a85937a6843532be87`，初始干净。assignment integration parent `7cf28dced4c8193ccf9321d62b5b2e479bd904f7`；实际基线包含W4协调更新。
- 已读当前task、plan、assignment A、wave-04、前项stages approved审查，以及当前真实stages/HPS/coordination和已有相关feature helpers。前项无changes。
- 新snapshot SHA pending，由引擎生成；未暂存、手工commit/merge/push、切分支或访问其他worktree。

## 唯一新增

- `tests/test_h3_mvp_cpu_integration.py`
- 本报告

所有生产源码、原93项H3测试、原Flash七项及B/C本波文件均未修改。未发现需要原owner修复的生产组合缺陷；没有用高层mock遮蔽失败，也未删除或放松原断言。

## 真正同时执行的对象

每个模拟rank均构造并执行：

- 现真实`GRPOTrainer/BaseTrainer`与`H3IterationStages`六阶段、原`IterationIdentity/StageValue`。
- 真实pool/CPS/anchor/world准入与credit/ClippedSurrogateObjective，完整K=2、N=2，含末格q0。
- 真实`H3PreparedModel`、`PreparedModelBundle` guard、最终参数集合校验、AdamW与LambdaLR；不是fake optimizer。
- **真实`H3HPSReward.acquire/score_one/close`**、原resource factory/Hpsv3VideoClient、JPEG编码、health与score validators。constructor只声明，worker显式acquire，各rank独占client，finally仅本地close且验证恰一次。
- **真实`H3Coordination.__init__/gate/aux_turns/clip_grad_norm`**；没有使用_CoordinationLeaf，没有以本地clip_grad_norm_替换协调器。
- 真实H3 forward bridge和双VAE decoder控制、反归一化/AV转换与lease；媒体在lease内实际交给HPS wrapper。

8个worker各自真正执行stage与数据工作，不复制rank0回执冒充其他rank。每rank有不同sample/group ID，两个iteration共同collection/behavior连续，conditions显式按sample绑定并在预测callback核对。观察stage调用的wrapper只记录事件并调用原stage，没有替代其返回/控制。

## 明确替身叶子及数学边界

1. **通信/WORLD/CUDA观察叶子**：测试内`_MemoryWorld`，8线程、按rank递增操作序号、有界receipt，只允许内存集合；每次等待20秒、future90秒，kind/参与rank不匹配立即断言。非协作异常唤醒等待者为失败，不死等、不吞错。没有真实PG、socket或CUDA；不修改线程数/共享环境。
2. **重H3模型与分片通信叶子**：toy H3每rank一个真实CPU `[2]` trainable shard，逻辑全参数为8×2。tiny forward使用显式内存weight gather，autograd Function在每格backward真实收集8份梯度、SUM/8后选所属row（toy平均reduce-scatter）。这是测试模型通信叶子，不是第二训练loop；实际stage仍执行真实backward/Adam/控制。
3. **DTensor观察叶子**：只替换`_dtensor_parts`，对本rank真实Parameter/grad给出逻辑WORLD8/Shard(0)/global shape16及真实CPU local值。未修改`_trainables`、`_local_gradient`或norm控制数学；没有创建假DTensor实例或生产CPU fallback。此证据不等于真实H3的FSDP2/AC/DTensor/NCCL行为。
4. **两个VAE重量级类**：复用只读decoder feature的tiny VAE class叶子；真正decode_one/shape/stats/lease代码执行。32×32×124是合法tiny测试画布，不改生产配置。
5. **HTTP transport叶子**：复用只读HPS feature的`_Transport`，不创建socket。读取真实提交的每张JPEG，按像素均值产生124个FP32 frame scores，独立top37参考；原factory/client/response validators未mock。端口是本地manifest/typed request fixtures，不代表实际服务ready。
6. **数据fixture**：显式预留sample/条件/初态/RNG、真实position/commit/abort/version计数，按原tail调用；不是整个trainer/data阶段替身。实际生产数据源/controller仍待后续装配。

零信用case不使用lambda reward或score mutation：给不同ID的样本相同初态/条件和相同seed但独立Generator对象，真实采样/decoder产生相同媒体，真实HPS transport从相同JPEG得到相同分数，真实world信用精确为零。

## 四个新增pytest cases与实证

两个成功case（非零/零信用）都让8rank各完成**两次**完整iteration：

- 每次world admission真实汇总16个不同sample ID；所有rank的coordination identity/event/sequence payload相等，逐collection序号从0连续推进。
- 精确全局事件日志断言：完整rollout→aux rank0..7 begin/end→anchor全格/complete→current全格backward→version完成→唯一norm→optimizer完成→tail ready/complete。每rank也记录stage顺序、sample/index/no_grad及backward。
- aux实际work顺序0,0,1,1,...,7,7，两轮重复；带锁并发计数峰值**1**。只有小JSON receipt进入world对象gather，不gather媒体。每rank4个实际protocol POST，恰每sample一次；每POST124张JPEG、sample/prompt准确，audio保留于decoder但不发送到HPS。
- 每rank每轮rollout/anchor/current各4次真实forward，前两组no_grad，current逐格backward；N=2含最后q0。8rank共同完成每格toy同步，共8次backward集合（每rank8次），每轮一次真实norm SUM。
- 独立参考从每格**归约前**8rank gradient vectors计算toy SUM/8及全步累积，再计算全向量FP64 norm与clip系数；对照真实coordinator返回norm及实际送进Adam的local gradients。没有把norm自己的输入标量当唯一数学参考。
- 独立从JPEG-derived frame scores手算top37→完整world std/组均值信用，逐rank对照stage的真实信用；非零case参数两模态发生有限变化，零case仍正常backward/norm0/Adam state step2/scheduler/tail但参数不变。
- 每rankdata position4/commit2、behavior policy-2、BaseTrainer.next_optimizer_step2，真实Adam state step2；tail提交时池仍ready、返回后池released，媒体weakref已释放。

两个失败case均在rank3注入，8rank都得到真实H3CoordinationError并停止：

- HPS transport返回错误sample_id，原response validator失败；aux到rank3首样本即共同拒绝，rank4..7没有work、所有rank没有anchor/backward/optimizer/tail。
- 模型正常返回后在current第二格注入NaN，真实CPS/score finite校验通过backward前共同gate拒绝；首格backward确实已执行，但无人step/data commit或进入norm。
- 两case均断言所有coordinator poison、data abort一次、next-step0、Adam state仍空；重试trainer/gate立即拒绝，close不添加任何collective。没有失败finally的补救barrier。

raw forward/backward内部失败需launcher整体退出/超时teardown；本组合测试不以制造无限等待证明其可恢复。前项单rank raw-failure测试保留，本项不改生产异常边界。

## 最后编辑后真实CPU门禁

在上述worktree根目录原样执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3通配符exit0：**97 passed in 8.31s**（基线93+本项4；不含尚未汇合W4 B/C新项）。初版组合测试同样实跑97/7.12s；最终强化顺序/norm参考后以上述8.31s为准。
- Flash原七项exit0：**7 passed in 12.58s**。
- 无skip/xfail，无全量测试，无网络/GPU/环境升级操作。
- git diff无输出：全部为允许新增文件；git diff --check exit0（tracked diff检查，不冒称覆盖untracked测试）。新增测试已真实import/执行并读回关键断言。
- 原Flash完整文件相对授权d818b8a231b52a9a024fa2f522057acdbb7bf099历史diff无输出、exit0；本轮未改任何旧测试。

## 交付边界

本证据可供simplifier-a与orchestrator关闭“W3三对象CPU同链尚未执行”的待办；必须经过对应审查与W4汇合门禁，不能以本角色done代替批准。没有新增生产controller装配功能，也不宣称真实H3/FSDP2/CUDA/NCCL/124帧服务资源或视频/音频质量通过。

后续仍需注册模型/算法recipe、唯一RunController/真实数据端口、完整checkpoint manager发布及同拓扑fresh恢复。引擎snapshot归属待生成；无手工Git提交或push。**GPU_PENDING，MVP未完成。**
