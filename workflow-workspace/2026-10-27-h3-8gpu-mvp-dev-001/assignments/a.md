# Assignment A — W3

- active: true
- task_id: H3-M4a3-stages
- pair: coding-a → simplifier-a
- execution: parallel W3；先读decisions/wave-03.md冻结接口
- integration_parent: `e436c192778a6bbf24e2bee7ebef777332138f9a`
- dependencies: W2 pool/decode/world配对approved，代码汇合ced9a871df0ee6bfe8b560e90da03bc12d4dda0e；W1锚/cache/prepare approved。无changes。

## 单一功能：现六阶段H3迭代接线

本项必须把现pool/decoder端口/world credit/anchor/objective/prepared optimizer实际接入 **原BaseTrainer/GRPOTrainer**，不是再新增独立helper或第二训练loop。不做controller/config/checkpoint，先得到可由现controller binding构造的真实六阶段对象。

允许新 `visual_rl/algorithms/trainer/h3_stages.py`、新 `tests/test_h3_mvp_cpu_stages.py`；仅为每格协作状态hook精确扩展你拥有的 `h3_pool.py`/`h3_anchor.py`及对应pool/anchor feature测试。W/coding-a/H3-M4a3-stages.md。原grpo.py/stages.py/runtime/旧tests及B/C文件只读；若范围过大先报拆分，不越界重写。

## 输入/输出锁

- 复用现IterationIdentity/StageValue/BaseTrainer；所有stage保留同一identity对象。输入一次真实iteration request（source/collection/behavior、local sample三元组、prompt/condition关联、精确dynamics、initial states/显式RNG及预算）；数据与版本由调用者显式给，不产生dummy trajectory/media或假dataset。
- 必需注入已有prepared对象（含optimizer/scheduler/reducer_divisor）、model prediction callable、单样本decode lease callable、score_one、`coordination_factory(plan)`、真实数据commit/abort能力。Prelude取得共同plan后调用factory得到该轮coordinator；未来runtime用真实H3Coordination(plan,PG,device)，不在admission前伪造plan。没有默认常量reward/no-op coordinator/假optimizer；算法不import具体H3模型实现或B/C同波模块。冻结scorer/coordination签名见wave-03。
- Prelude验证请求并调用真实admit_h3_world，绑定准确geometry/schedule/score id与集合；本地预检失败要以local_error参与共同拒绝，模型零调用。stage内返回真实plan/pool上下文，不借Flash collection schema。
- Rollout调用现H3RolloutPool.collect，prepared EVAL、固定behavior，全部N步同rank调用量。可增加小的逐格status hook：pre-forward gate、model返回后CPS local结果gate；需要在local sample/score finite失败时通知其他rank，不能在一rank失败时其他继续下一模型调用。raw forward/AC/backward内部失败不假定能共同到门，不在异常finally新增collective，令上层attempt失败。
- Reward在**完整rollout后**coordinator.aux_turns中逐sample取terminal、进入decoder lease、score_one(sample_id,prompt,media)，离开即释放；不收集全批媒体、不进入下一rank模型阶段。实际结果组装为 `(sample_id,group_id,member_index,float_reward,True)`，无重试/换seed补样。
- Advantage真实gather_h3_credit；Credit将本rank信用映射到所有N格，beta0、algorithm_weight1、无额外clip。仍通过独立advantage/credit stage返回StageValue，不把两stage替成生产占位。
- Optimize先现H3ReplayAnchor.capture一次，完整成功且共同gate后才能current replay；每个B1/index格当前prepared预测→CPS.evaluate→既有ClippedSurrogateObjective(clip.005,beta0)→按plan.loss_scale(n_slot=1,reducer_divisor=prepared.reducer_divisor)立即backward，不能保N个图。旧锚不再刷新、不得直接rollout_score；全N含q0、同序格数，与pool有序样本条件准确关联。
- 每格local score/objective有限性结果在backward前共同gate；backward正常FSDP同步，不no_sync。不另写PPO或再除world/GA/N。全部格完成才coordinator.clip_grad_norm(max_norm=1)，**一次**真实Adam step、一次scheduler、zero_grad。全零信用走图连接backward与普通commit，norm0合法，None梯度仍由coordinator拒绝。
- 通过现trainer已有commit_iteration tail挂接真实data commit、behavior/collection进度更新、pool.release，next_optimizer_step继续由BaseTrainer唯一维护；任一失败不就地重试，不伪rollback optimizer。仅在trainer整轮返回后可成为后续checkpoint候选；不要在optimize末尾冒充完整safe point。首次/后续迭代标识严格连续。
- 引入上下文/结果类型限必要字段，不复制通用runner、执行事务框架或Flash多update/EMA逻辑。controller/数据采样装配后续补齐，不本项绕过现入口。

## 验收

实际BaseTrainer.run_iteration走全部六stage，真实pool/CPS/anchor/world表校验/objective/Adam，至少两次更新；tiny仅替重model/VAE、score transport和CUDA/collective叶子。W3同波B/C未在分支出现时，可给明确测试叶子实现冻结能力，不导入其未完成源码；不得mock整trainer、stage、pool或目标函数。

记录精确call序列、完整含末格、评分在所有rollout后、anchor在首current前、有限非零双流梯度/参数变化、data/版本/next-step每轮恰一次、pool只在tail释放；全相同reward也真实普通step。preflight/评分/old/current finite失败不得step/tail，部分step/tail失败poison不再次更新。原七项与原feature断言保持。

最后编辑后跑plan两条固定命令（Flash7/H3通配符），无skip/xfail/全量；交付branch/HEAD、命令/exit/数量/时间、diff/check、新增hook与后续runtime接口。Simplifier A重点审是否真实复用六stage、旧锚/归一化/每格图释放/全tail及失败边界，changes回A。CPU tiny不证明真实GPU或整个RunController装配完成，GPU_PENDING。
