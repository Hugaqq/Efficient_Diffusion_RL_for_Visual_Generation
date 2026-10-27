# Assignment C — W3

- active: true
- task_id: H3-M3b2-phase-norm
- pair: coding-c → simplifier-c
- execution: parallel W3；冻结接口见decisions/wave-03.md
- integration_parent: `e436c192778a6bbf24e2bee7ebef777332138f9a`
- dependencies: W1 prepare与W2 world/reducer已配对approved，W2源码汇合ced9a871df0ee6bfe8b560e90da03bc12d4dda0e。无changes。

## 唯一功能/允许文件

实现**现WORLD8已初始化运行的阶段健康/aux轮转与唯一global grad norm**能力，供A六阶段结构接口调用，不是新训练runner或PG owner。

仅新 `visual_rl/models/lifecycle/h3_coordination.py`、新 `tests/test_h3_mvp_cpu_coordination.py` 与W/coding-c/H3-M3b2-phase-norm.md。现prepare/world/helper/旧tests只读；不import A/B同波新模块。接口与A已冻结，不按自己偏好改名；确有必要回报orchestrator。

## 公共能力（必须全部真实可调用）

`H3Coordination(plan, *, process_group, device)`或等价最小类，绑定现H3WorldPlan、实际NCCL WORLD8/当前rank CUDA及共同plan identity，不创建/销毁PG或Accelerator；不访问资源UUID。方法：

1. `gate(event: str, *, local_error: bool=False) -> None`：每次交换有界小receipt（共同plan/collection/behavior、序号、event、rank、status）；不传模型/轨迹。所有存活rank相同event/sequence并且status全成功才返回，否则统一失败并poison本attempt，禁止继续调用训练门。event须有界且不包含rank-specific sample ID，A使用local cell ordinal。非法本地event/status先编码错误receipt共同拒绝，不先本地throw让peer挂住。PG死亡/collective内部错误不承诺协作恢复，也不在finally多加barrier。
2. `aux_turns(work: Callable[[], T]) -> T`：logical rank0..7固定顺序；每轮共同begin/end gate，只有轮到本rank才调用自己的work（其内部逐条decode/score）；其他rank等待，节点heavy并发1。只保留本rank小结果，不gather媒体或跨rank传AV。不允许rank先进入下一模型阶段。可协作work异常转换同轮失败status，让存活rank共同终止；保留本地原异常链，不重试/换样。死亡/中断仍由上层退出。不要把此实现扩展通用actor/异步服务平台。
3. `clip_grad_norm(prepared, *, max_norm: float) -> float`：消费现H3PreparedModel的最终trainable_named_parameters。预检唯一param对象、所有预期trainable且梯度非None，必须实际WORLD8 1D shard DTensor（当前root/refiner冻结，无复制trainable）；不接受未同步普通Tensor或其他mesh/placement冒充。只处理真正local shard元素，不对global DTensor值再重复统计；empty local shard可以合法，None不可当零。
   - local finite/覆盖错误先共同gate或有限status归约，全部成功再执行norm collective。每rankFP64 sum(local_grad²)，WORLD SUM一次，sqrt得到**唯一参数元素global norm**，不再除world8、不mean各rank norm；验证aggregate finite。
   - scale=1当norm0，否则min(1,max_norm/norm)（明确合法max_norm>0）；同一scale原地施加各local grad。零梯度合法且返回0，None/NaN/inf明确失败；不偷注入epsilon、删梯度覆盖或require-nonzero。
   - 该梯度已经由FSDP公开gradient divide factor8正常backward同步，不再额外平均gradient或no_sync。不执行optimizer/zero_grad/scheduler。返回Python float norm，供A日志与一次commit前验收。

模型内部forward/AC/backward失败可能不能到门；本能力不得许诺救活这种collective。A只在model成功返回后对local score/有限性错误使用gate。close若提供仅释放本地引用，无collective/PG teardown。要避免norm中临时保留计算图或复制完整参数。

## 真实CPU验证

不执行GPU、PG初始化或网络。只替WORLD/CUDA/collective/DTensor-local-shard观测等叶子，实际gate/轮转/norm控制与数学不mock。模拟8rank receipts验证event/序号/plan不一致及本地错误统一拒绝/poison；work只在owned turn运行一次、错误后无后续turn或model事件、无媒体gather。

真实CPU tensors模拟8份**不同**local shards，独立单全局向量参考比较SUM-of-squares与clip后值，证明不是平均norm/重复参数；含不等local shard长度、empty local shard、有限零梯度、缺grad/nonfinite/复制trainable/错误mesh/duplicate对象等拒绝。production DTensor检查必须真实存在；fake只用于测试不可运行分布式叶子，不提供生产CPU fallback。测试A冻结签名可用，但不导入同波A新module。

最后源码编辑后跑plan固定Flash7/H3通配符，无skip/xfail/全量；记录branch/HEAD、命令/exit/数量/时间、diff/check、实际公共方法与替身范围。Simplifier C审单owner/同序错误传播/aux并发和global唯一norm，changes回C修复重测。

## 非目标

不改world credit或prepare，不建训练loop/optimizer/tail/checkpoint，不实现controller/资源launcher。该协调能力仍须在A真实stage逐格调用并在后续runtime配置真实PG，不能把8rank CPU模拟当NCCL验收。GPU_PENDING。
