# W3 — 原六阶段接线 / 现有真实reward / 分布式健康与norm

基线 `e436c192778a6bbf24e2bee7ebef777332138f9a`；W2代码汇合ced9a871df0ee6bfe8b560e90da03bc12d4dda0e，三份对应review均approved。本轮orchestrator亲跑Flash7/12.16s、H3通配符65/3.46s，全exit0。并行不代表发布main；GPU_PENDING。

## 最小reward选择（显式决策，不冒称上游严格复现）

task允许reward用CPU/现有接口；已有生产 `Hpsv3VideoClient` 和 `flash_hpsv3_json.v1`，VideoContract支持124帧（上限128）。本MVP首个显式配方采用 `h3_hpsv3_video_only_v1`：所有124帧真实HPSv3，再按现协议top30%均值，唯一权重1，无伪造音频reward。**它不度量音频文本/AV对齐，不能声称音频质量**。音频状态/采样/解码/联合CPS元素均值/梯度保持，绝不退回video-only latent或Gaussian。

设计reference中的CLAP+ImageBind是候选，非task硬性服务选择；不为首闭环引入未核验ImageBind非商用许可、额外encoder平台或下载。HPS接口仅复用，旧Flash SMALL40/140配方和服务不改。124×256×384所需真实HPS契约/资源仍待另行准入；历史81帧服务不自动兼容。本轮禁止实际HTTP/GPU/服务部署，测试只替transport叶子。新profile须明确该reward选择，不能隐藏成原UniRL reward复现。

## 并行边界

| slot | task | 源码拥有 | 测试拥有 |
|---|---|---|---|
| A | H3-M4a3-stages | algorithms/trainer/h3_stages.py；只为逐格同步hook精确扩展h3_pool.py/h3_anchor.py | test_h3_mvp_cpu_stages.py；必要hook扩展pool/anchor测试 |
| B | H3-M2c-hps-reward | algorithms/rewards/h3_hps.py | test_h3_mvp_cpu_hps.py |
| C | H3-M3b2-phase-norm | models/lifecycle/h3_coordination.py | test_h3_mvp_cpu_coordination.py |

前缀visual_rl/及tests/。三组均不改原grpo trainer、旧stage/runtime/recipe/Flash tests；A不能import B/C新模块，仅接受下面冻结的必需结构接口。B/C不改A代码。各配对同波独立实现；未来composition必须提供真实对象，不允许默认no-op。

## 冻结窄结构接口

- 评分端 `score_one(*, sample_id:str, prompt:str, media) -> float`，media结构已有video/audio/metadata；B实现真实HPS，A只依赖此callable合同。finite Python float，权重1；样本身份由A绑定现world reward五元组。
- A的Prelude完成实际admit_h3_world后调用必需的 `coordination_factory(plan)`，未来runtime以真实 `H3Coordination(plan,process_group,device)` 构造；不要求A提前拥有未生成的plan。
- 协调端 `gate(event:str, *, local_error:bool=False) -> None`：按调用次序有界交换(sequence,event,status)，存活rank一致成功才继续；A使用相同rank-independent event（local cell ordinal，不使用不同sample_id）。异常不能变成继续训练。
- 协调端 `aux_turns(work:Callable[[],T]) -> T`：固定logical rank0..7轮转，只有当前rank运行自己的整段本地逐条评分；每轮前后共同gate，保证节点同时只有一个rank heavy aux。非owner等待，不gather媒体；失败统一拒绝。
- 协调端 `clip_grad_norm(prepared, *, max_norm:float) -> float`：严格WORLD8分片trainable梯度的有限性/覆盖检查、唯一global norm和in-place clip。零norm合法，None/非有限拒绝；返回实际global norm。
- A的prediction、decoder来自现模型域能力，协议由实际H3 helper接口封装；不在算法模块import具体模型实现，不另建模型/optimizer或PG。

这不是新runner/服务协议：A必须调用现BaseTrainer/GRPOTrainer的六stage控制，B只现HTTP client叶子adapter，C只已初始化PG的健康/norm能力。A输入中的协作接口是强制依赖，不提供生产noop或CPU fallback。

## 接收与后续

每组独立分支重跑Flash七项/H3通配符，review分别审。汇合后需额外验证A与B/C对象结构兼容，不能只凭各自测试成功；有接口误差回相应owner。此波不宣称RunController/recipe已接通，后续马上完成ModelAdapter/registry/controller及完整tail/checkpoint/fresh恢复；不能用三个callback替身的tiny成功当生产装配完成。
