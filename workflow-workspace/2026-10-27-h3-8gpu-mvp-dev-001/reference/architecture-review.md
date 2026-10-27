# MiniMax-H3 单节点8卡：最终架构设计与准入门

## 0. 结论与报告边界

**推荐的首版是有条件的同模型训练侧rollout：单节点获准8卡、block级FSDP2、SP1/TP1、模型forward/replay微批1、时间chunk1、一次collection一次全局更新。** 选择它是为了减少参数映射、双引擎数值差异和资源生命周期，不是因为已经证明它最快。完整实现与验收后，若rollout取参通信成为瓶颈，允许改为同8卡分阶段TP推理副本；若单样本激活成为瓶颈，才审SP2/4。

**当前结论不是训练ready：** 用户历史观测为10张CMP170HX、每卡64 GiB、显存高占用；任务只使用其中获准的8张，UUID/时段未给定。真实H3峰值、节点互联、BF16/attention/NCCL、依赖组合及DCP恢复均未验证。不得默认物理0—7、占额外两卡、释放其他进程或承诺加速/画质收益。

算法必须采用显式新profile，不能笼统称“复用joint logprob”：**CPS负MSE代理score、元素均值AV归约、量化固定动作、全10个policy步含零方差末步、冻结replay旧锚**是本文推荐，不是三上游统一标准，也尚未获用户对质量目标的批准。

### 证据与工作范围

- 已首先重读 `inputs/hardware.md`，完整读取 `task.md`、`summary.md`、三份 `round_2/*.md`；第一轮三报告已在同一持久会话前次完整读取。本次重新回查新增及高风险源码。第一轮摘要保留，不覆盖。
- **事实**分为用户历史硬件观测和快照静态代码；**推算**列假设；**建议**是未来实现选择；**待验证**不因三方同意而关闭。
- 本次只写 `final.md`。未运行训练、GPU/服务器命令、源码测试、安装、性能实验或结构脚本；未修改inputs、源码、task、workflow或验证脚本。独立结构检查在本步之后由工作流执行，不能把文件完整性当GPU验证。
- 路径简称：**V/**=`inputs/framecode/visual_rl/`；**U/**=`inputs/UniRL/unirl/`；**O/**=`inputs/verl-omni/verl_omni/`；**F/**=`inputs/Flow-Factory/src/flow_factory/`；**R**=`inputs/UniRL/examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml`。全部相对此报告工作区。固定提交/快照范围见 `inputs/sources.json`，不访问其中记录的旧绝对来源路径。

## 1. 先可行性：容量、通信、加载与软件

### 1.1 硬件事实与不能推出的结论

`inputs/hardware.md`记录用户2026-10-27历史输出：65536 MiB=64 GiB/卡，已用48491—62305 MiB；差额约3.16—16.65 GiB/卡，仅为算术值，不是可立即分配保证。GPU-Util=0%不证明空闲。采用实机显示值，不用常见8GB SKU记忆覆盖，也不猜测显示原因。

8卡名义512 GiB不是统一内存；每rank必须独立通过各阶段峰值门。型号、Bus ID和UMD13.3均不能证明NVLink、P2P、PCIe带宽、BF16、FA3或目标torch运行环境。以下容量讨论只适用于**未来拿到足够空闲的获准8卡**；历史截图没有证明该条件成立。

### 1.2 参数与每rank预算：统一结构口径

【推算】依据A第二轮Q2、B/C第二轮Q2的条件结构核对：U/models/minimax_h3/vendor/transformer_minimax_h3.py:98-125,215-229,320-340,425-520。默认h=5376、attention宽7168、FFN14336、time2688、50主blocks；假设目标Diffusers SwiGLU首投影为2f、输出为f→h，忽略小bias/norm。

| 项 | 条件估算 | 不能据此声称 |
|---|---:|---|
| 主block矩阵 | `4h*7168 + 3h*14336 + 18h*2688 = 645,464,064`参数/block | 真实artifact恰采用全部默认构造值 |
| 50主blocks BF16 | 64.55 GB≈60.11 GiB；八分片约7.51 GiB/rank | 所有模块都除8，或这是峰值 |
| root外冻结残余R | A估两refiner及投影/头/时间MLP等合约1.72 GB≈1.60 GiB/rank | 所有R都FP32；此估算已经完整盘点buffer/临时量 |
| 指定六类targets、r32 LoRA | L≈166.30M；FP32参数/grad/两个moments按16L/8约0.31 GiB/rank | 不同targets/optimizer实现仍相同；低精度工作副本已计入 |
| 上述理想常驻合计 | 约9.43 GiB/rank | 64 GiB已保证可运行 |
| 全block在途取参 | 单block约1.20 GiB，含LoRA量级约1.21 GiB；双block约2.42 GiB | 库内恰只有一份buffer，或不存在共享storage重复计数 |

R的细化是A第二轮新增推算；B/C核对的是主blocks/LoRA条件公式，不能写成三人独立确认9.43 GiB实值。本汇总复读默认构造与精度声明，未执行参数枚举。上游“66 GB”是完整加载量级注释，**不再作为精确可分片权重值，也不与上述数值平均**。

统一准入式：

`M_peak,r = max_phase[S_weights/8 + R_replicated + M_LoRA_state + M_unshard/prefetch + M_precision_work + M_activation + M_aux + M_IO/NCCL/allocator] + margin`。

首版不主动开启跨block预取，不采用保全模型unsharded来省AG；库内缓冲仍须审计。8 GiB余量只能作为待校准工程占位，不能当充分容量证明。真实artifact须逐FQN列shape、dtype、trainable、shard owner及必要buffer；setter、首次Adam、checkpoint等瞬态也计入峰值。

### 1.3 辅助组件、媒体与host RAM

【事实/量级来源】U/models/minimax_h3/config.py:70-105的上游注释给Qwen约64 GB BF16、两VAE约10 GB FP32；不属于CMP测量。32B BF16约59.6 GiB的encoder与训练shards共驻单卡不成立，八进程各载完整CPU encoder也可能耗尽主机内存。

**首版决策：训练及评估prompt的text embedding缓存全覆盖，训练内miss早拒；无实时CPU/GPU编码fallback。** 缓存生成自身的资源、许可和峰值另行授权，不能假定已经存在。身份包括权重、tokenizer/processor、模板、层选择、截断、dtype和prompt；U/models/minimax_h3/text_embed.py:36-77的第50层/去末norm语义不能由普通最终hidden state替代。

【推算】低画布256×384、请求124帧、24fps：37 latent帧、video3552 rows×96、audio414 rows×32，共354240标量；FP16单AV状态约0.676 MiB。每rank8条、10步保存x/action/next三份约162 MiB，未计条件、metadata和临时副本。terminal latent单份很小，不等于单步模型激活小。

- FP32 RGB低画布139.5 MiB/条，64条约8.72 GiB；768×1344约1.43 GiB/条，64条约91.55 GiB。均未含音频、resize、复制和评分模型。
- 队列按bytes和最大在途条数共同限制；首版最多1条已解码AV在途。实际cap须能容纳合法画布及工作区，不能用小cap迫使暗中缩帧、量化或删音频。
- attention若退回显式L²大矩阵，峰值可远超线性像素估计；memory-efficient attention是实际组合准入项，不照搬FA3。
- host预算分初始化和aux阶段：rank0 full state、映射/转换/page cache、rank-local轨迹/条件、VAE/scorer CPU权重及预处理、DCP staging。控制线程和队列并不自动控制所有库缓存，须测RSS/磁盘及释放后的回落。

### 1.4 加载路线：采纳A-15/C-15事实，保留B的有界目标

【本轮汇总复读确认】U/train/backend/sharded_load.py:81-113,301-316普通非EP路径由rank0合并**完整CPU state dict**，其他rank空dict，调用`set_model_state_dict(full_state_dict=True,broadcast_from_rank0=True)`；这不是逐tensor常量内存loader。:528-538仅检查匹配参数名比例至少25%，默认strict=False，不能作为完整H3权重覆盖证明。

**设计决策：首版选meta参数构造→LoRA初始化/精度配置→block分片→rank0单份完整CPU基座输入→最终分片物化的受审计路线。** 不每GPU先加载完整基座，不八份CPU基座常驻。先释放初始化full-state，再加载aux，避免无必要重叠。

准入条件：
1. host可承受完整CPU字典及临时副本；精确target setter实现没有不可容纳的GPU完整副本或其他峰值。
2. 完整base FQN、shape/dtype映射全部覆盖；仅白名单LoRA初始化项、明确可重建非持久buffer可例外。额外/重复键映射冲突拒绝；不搬上游25%阈值作完整性门。
3. 共享base/LoRA初态与必要buffer来源一致；U/models/types/meta_init.py:85-142,157-209的非持久构造态恢复需保留，不能把全部buffer置meta后丢掉。
4. 若host或目标setter不满足，**停止该实现路线并另审真正流式/分片输入**；不能降级为GPU先全载。

B第二轮仍要求有界shard读取/映射，但未提供实际loader闭合证据；A/C并行后出证据明确现参照需要完整CPU字典。本文采纳可追溯事实而不称B已接受该路线；有界host加载仍是替代目标，不是已实现能力。

### 1.5 通信为什么可能推翻同模型默认

【推算】BF16主blocks64.55 GB、8-rank ring AG每遍历每rank接收约`7/8*S_weights=56.48 GB`。冻结基座仍需取参，不只通信LoRA。

P8/K8、每rank8条、N10/J10、每调用reshard的情景：rollout80遍、old replay80遍、训练前向/反向取参约160遍，共约320遍，逻辑接收约18.07 TB/rank。**训练反向一次取参只是情景假设，不是上界**；AC重入、cast、prefetch、实际NCCL算法都会改变流量/重叠。假想20 GB/s只会得到算术时间，不能当CMP测量。

同模型优先减少工程变量；双引擎TP2×4可以减少推理侧反复全层聚参，但每卡推理权重约S_weights/2而非/8，另有训练shards、workspace、睡眠切换和LoRA映射峰值。只有完整collection净收益通过实测才切换，不默认4+4卡池。

### 1.6 软件与来源准入

`inputs/framecode/pyproject.toml:9-25`要求Python>=3.10,<3.12；`inputs/UniRL/pyproject.toml:9-38`要求>=3.12,<3.14及更高transformers/PEFT。整包不能直接兼容同装，但不证明H3局部移植不可能。

锁定Python、torch/CUDA/runtime/driver、NCCL、diffusers、PEFT、transformers、attention、artifact及源码许可；不要继承Global历史torch2.7.1/2.13当本机证据。A倾向优先兼容公开H3实现、B/C倾向窄vendor：**最终规定端口和数学合同，具体公开API或合规vendor选型留M0审计，不默认搬UniRL整包、Ray/EP/VeOmni训练框架。** ImageBind非商业许可提示见R:127-149，生产用途必须审核，不能用技术可行替代许可批准。

## 2. 单节点8卡拓扑、batch与归约

### 2.1 默认拓扑

```text
获准UUID清单 -> 逻辑rank r0…r7（不等于物理GPU 0…7）

参数分片/梯度组 F = [r0 r1 r2 r3 r4 r5 r6 r7]
一个逻辑H3策略版本，主blocks+LoRA分片；root外冻结精度岛复制
独立样本通道      d0 d1 d2 d3 d4 d5 d6 d7
SP singleton      [r0] [r1] [r2] [r3] [r4] [r5] [r6] [r7]
TP=1；一个runtime拥有PG/mesh/设备，禁止重复prepare/wrap
```

不同rank处理不同样本，但共同按同序block执行AG/反向归约；8个分片不是只生成1条视频。forward必须通过prepared handle。LoRA限主blocks，root外所有trainable早拒，冻结残余仍计R并保证初始化/必要buffer一致。

### 2.2 明确batch实例

| 维度 | 首版候选 | 含义 |
|---|---:|---|
| prompt occurrences P | 8 | 独立group身份，不按重复文本合并 |
| candidates K | 8 | 每组独立随机候选；不同于UniRL默认16 |
| collection / optimizer trajectory batch | 64 | 一池一次更新，期间行为参数不变 |
| per-rank placement rows | 8 | local_complete，一rank完整一组 |
| 外层GA | 1 | 不把内部80次loss重复除以GA |
| rollout / replay model microbatch | 1 / 1 | 与placement8不同；不是GPU batch8 |
| N / J / timestep chunk | 10 / 10 / 1 | 全10policy格、一次只保当前chunk图 |
| 每rank loss格 / 全局格 | 80 / 640 | 正常一池一次Adam，不每步Adam |

现V/data/group_placement.py:99-176要求`P*K=W*placement_rows*GA`，local_complete每placement含完整K。不得写placement1/GA8后仍称同一完整本地组合同。增加P、增加K或改变J是显式训练选择，不因总乘积相同就算法等价。

### 2.3 正确global loss与统计

推荐优势：`A_gk=(R_gk-mean_group(R))/(std_collection(R)+1e-4)`，ddof0，完整唯一64行reward，CPU float64统计后FP32信用，无额外优势裁剪。先固定权重合成各reward分量，再归一化。现normalizer可复用数学，不可继续用每rank本地batch std冒充global std。

设backend reducer为`sum_rank(grad)/d`，全局有效loss格`N_valid`，slot本地mean含`n_slot`格，则反传系数：

`w_slot = d*n_slot/N_valid`。

SP1且实际d=8时，单格8/640=1/80，最终每格权重1/640。d必须由目标后端证明。现V/algorithms/optimization/kernel.py:**609**为`slot.active_count/slot.global_active_count`，其中分母受本地plan约束；要分离local/world count及校验，不能只改字段值。

不等量反例：两rank分别2格和6格，local mean后均值给每格1/4与1/12，正确应都1/8。数学上d/N可修权重，**不能修collective次序**。首版拒绝uneven/空rank/不匹配模型调用计划；J固定，未来J可变须另定cell mean还是trajectory mean。

同步后检查finite、唯一参数元素global norm并clip；复制参数去重、DTensor本地norm不当global norm。首版每chunk正常同步并累加分片梯度，不未经审计套no_sync。Adam仅一次；数值归约误差用预先定义容差验证，不承诺逐位一致。

### 2.4 SP2/4仅为后续候选

| SP度s | 独立样本通道D=8/s | B64波次 | 全8rank分片候选中的关键变化 |
|---:|---:|---:|---|
| 1 | 8 | 8 | 首版唯一准入目标，无SP通信 |
| 2 | 4 | 16 | 同样本两rank协作，leader仅统计；额外attention/布局通信 |
| 4 | 2 | 32 | 四rank协作，非分片activation和头不按1/4减少 |

参数组仍F=WORLD8，SP是其逻辑子组；具体卡映射等拓扑证据。O/workers/engine/fsdp/diffusers_impl.py:199-226,561-574证明上游两个mesh独立，不证明本机训练组合已通过。

SP对生成标量的残差sum与真实DOF归约，排除padding/条件rows，不能mean各rank不同长度local mean。所有成员参与必要backward；leader-only统计不等于leader-only loss执行。

若目标SP伴随满足`sum_s h_s = κ*g_unique`，FSDP reducer为SUM/d，每复制样本格的系数应为`d/(κ*N_unique)`。κ可能为s或1，取决于可微SUM/gather backward/内建平均；不是从forward相同或人数猜得。若伴随不是正确梯度的常数倍，标量补偿也修不好。**κ/d及目标内核未证明，SP2/4训练不开放。** 若改为D-rank参数组，常驻变`S_weights/D`并需副本梯度同步，不能沿用7.51 GiB或原AG公式。

## 3. 新H3算法与数值profile

### 3.1 CPS与模态权重：不能偷换成Gaussian

对模态m，当前sigma=s、下一sigma=u、模型已转换成noise-ward预测v：

`q=u*sin(eta*pi/2)`；`x0=x-s*v`；`e_hat=x+(1-s)*v`；
`mu=(1-u)*x0+sqrt(u*u-q*q)*e_hat`；`a=Q_FP16(mu+q*z)`；
`ell_m=-mean_generated((stopgrad(a).float()-mu.float())²)`。

合并为`ell=(Dv*ell_v+Da*ell_a)/(Dv+Da)`。用`rho=exp(ell_current-ell_old)`进入PPO clipped surrogate，但它是代理ratio，不认证为真实重要性比/真实KL。

证据：U/sde/kernels.py:163-180,246-306；U/models/types/diffusion.py:108-119。VeRL实际使用各0.5：O/pipelines/minimax_h3_flow_grpo/common.py:34-37,124-132及同目录diffusers_training_adapter.py:315-335、vllm_omni_rollout_adapter.py:470-489（首轮汇总复读）。低画布元素权重下音频约3.74%，不能宣称等同50%。附近effective std不改变实际score权重。

真正Gaussian包含`1/(2q²)`和`log q`；省去方差缩放改变梯度，不只是常数抵消。**不把两个模态std平均成一个Gaussian std，也不全局放松V的Gaussian保护。**

### 3.2 末步q=0、量化和零优势

候选保N10/J10，末步代理格仍active。固定`a=Q(mu_b)`，`grad ell_theta = 2/D * sum_j[(a_j-mu_theta,j)*grad mu_theta,j]`。

- 无量化、θ=θ_b且计算均值完全一致时，末格score及初始score梯度为0。
- FP16量化残差可使初始score梯度非零，也可能为0或抵消。old/current同式而rho=1并不意味着rho导数为0。
- 即使末格在理想例中贡献0，删它并由J10改J9 mean会改变其余格归约系数；未来多update还可能出现非零末格梯度。
- CPS q0不是可评分的非退化Gaussian。只在声明的CPS终步允许q0；其余异常q0拒绝。保finite、layout、mask、固定action detach。
- U/sde/kernels.py:302-306即使q0仍抽noise；沿用顺序Generator时不得因乘零擅自跳过消费。新member-keyed流也须显式版本化，不能暗称与原流同seed逐位一致。

**全零优势最终选择：正常optimizer commit。** 合法完整数据、beta0、有限零A仍走图连接零loss/backward/同步，允许global norm=0；一次Adam、scheduler、data/pool/版本照常提交，记录该事件。不删样、不注入1e-6、不单rank跳步。已有Adam moment可移动参数，不称no-op。H3显式关闭“梯度必须非零”门；finite与应参与trainable的图/grad覆盖检查保留，None/断图不当合法零梯度。用户若要求零信用冻结Adam和进度，须另版全局skip政策。

### 3.3 主动选择与参照关系

| 项 | 本文推荐 | 说明 |
|---|---|---|
| T2VA、元素均值CPS、replay anchor、全10步、guidance1、shift12/3、eta.7 | 沿所选UniRL式语义 | 非三上游一致，也非精确density |
| 256×384、124帧/24fps | 低画布资格基线 | 121/124不可静默互换；目标高质量画布仍待用户 |
| LoRA32/64、dropout0，主blocks六类target | 沿R:102-117 | 不让suffix误命中root外refiner |
| Adam lr1e-4、betas(.9,.999)、eps1e-12、wd0；constant LR、warmup0；max norm1 | 沿R:87-101,164-168 | 具体DTensor Adam实现选项须锁；不因小grad擅改mean为sum或epsilon |
| PPO clip.005、beta0、无额外advantage clip | 沿本地上游helper的目标方向 | U/algorithms/base.py:179-195不clamp优势；V/optimization/credit.py:175-198会clamp，需显式None能力，旧默认不改 |
| P8/K8；CPU float64、eps1e-4、occurrence/global std | 新profile主动改变 | 相对U的K16、FP32/eps1e-8，不能宣称严格复现 |
| chunk1、SP1、一次更新、无EMA、CPU评分 | 首版执行/状态范围 | 不授权更改现Flash批准EMA、pending池或终态评估 |

本轮A/B明确推荐无额外优势裁剪；C要求该值显式但未收到第二轮新回执。本汇总亲自复读双侧helper/credit，选择None以免无声继承旧±5，不以“两票”证明目标最佳。

### 3.4 旧锚与行为版本

固定默认：**完整reward/credit后、任何参数更新前**，OptimizeStage在现kernel前准备一次全collection replay anchor。θ_b固定，EVAL+no_grad、B1/chunk1、同packing/双schedule/FP16固定动作/FP32评分；所有rank完成后才current update。每rank额外80次forward，无旧权重副本，不留图。

绑定behavior/base/LoRA版本、双精确sigma数组、selection、score/reducer/terminal、layout/masks、存储量化、encoder/condition和数值策略。rollout score可留资格诊断，但loss只消费标记来源为replay的冻结旧分数。未来同池多update不在θ1重新定义old。

U/algorithms/flowgrpo.py:90-177真实生成并消费该锚；V/algorithms/optimization/kernel.py:585-608的绝对step0诊断不是anchor实现。省掉replay pass须在真实artifact、存储往返、内核/AC/geometry下对分模态与joint score、mean、梯度及更新过门，并版本化old来源、拒混池。rho初始1只证明评分关系，不证明另一引擎真实采样分布一致，更不把CPS变成精确概率。

## 4. 单轮数据流、资源默认与执行一致性

### 4.1 阶段与owner

```text
M0授权/软件/artifact/缓存/内存准入
  -> 全rank共享初始化、meta+block分片、final binding、Adam
  -> PRELUDE：rank0广播小计划，各rank合法性/模型事件预检
  -> ROLLOUT：θ_b固定，全8rank按匹配顺序B1 joint采样
                各自存FP16转移和terminal latent；无decode、无图
  -> 全rank模型phase结束、reshard与健康状态确认
  -> AUX：有序logical-rank turn，节点heavy aux并发1
          本rank逐条FP32 VAE解码 -> CPU固定scorer -> 释放AV
  -> REWARD COVERAGE：64行小表，唯一ID/valid完整检查
  -> ADVANTAGE/CREDIT：全collection统计；发回本rank信用
  -> OLD ANCHOR：全rankθ_b no_grad replay，冻结FP32旧分数
  -> UPDATE：matched B1/chunk1 current forward -> 健康诊断门 -> backward
  -> 全局finite/norm/clip -> 一次Adam/scheduler/zero_grad
  -> trainer一次data/pool/next-step tail -> 全rank成功 -> behavior b+1
  -> 释放池；可选全rankcheckpoint/eval，单writer发布文件
```

六stage保留为prelude加rollout/reward/advantage/credit/optimize；anchor在OptimizeStage内单次prepare，不新增通用第七stage。不允许一rank先进入下轮FSDP而其他还在aux。

| 状态/资源 | 唯一owner及职责 |
|---|---|
| PG、UUID映射、phase参与者、共享计划 | runtime；不与Accelerator重复创建/prepare |
| 最终DTensor参数、guarded forward、同步/norm、模型态codec | prepared/backend handle；禁止raw旁路 |
| rank-local转移/终态/条件/score/anchor | 当前H3 iteration的小型pool；stage只拿视图，不持模型/optimizer/带图tensor |
| 非prepared VAE/CPU scorer | 现资源/组件owner的受控lazy load、phase驻留与close |
| optimizer/scheduler/zero_grad | execution，已有操作只调用一次 |
| data/pool消费、next-step | trainer；runtime只汇合完成状态，不再执行一份 |
| checkpoint完整文件树 | 现checkpoint manager；publisher无模型getter回调 |

### 4.2 Aux分歧的最终工程选择

A第二轮建议每rank CPU scorer、节点并发2且副本入host预算；B建议按rank turn并发1、lazy释放；C建议全部8rank本地并发、host不足早拒。**这是未测性能/驻留权衡，不是已达成一致的部署。**

本文选择B的**并发1有序rank-local aux turn作为首个资格基线**，理由是当前host RAM/scorer能力未知，它缩小同时驻留与排错范围且不用跨rank传媒体；不是按票数，也不声称最快。每个turn只该逻辑rank加载/使用所需非prepared辅助组件，video/audio VAE可顺序驻留，CPU scorer按固定组合顺序评分并释放。不能让已经完成的rank进入长等NCCL模型collective；由健康控制phase协调，timeout覆盖批准阶段预算。

目标是lazy生命周期，不假定`close`自动清除全部库缓存；未证明host峰值、CPU实现可用或内存回落即不准入。测量后可改并发2或8，并重新核host峰值、评分数值与resource身份。GPU评分是后续独立placement profile：只能用本获准8卡，预算仍包含训练shards；有界媒体文件/流逐条ACK，禁止聚齐全部媒体或latent到rank0。没有证据就停止，不暗用第9卡或换reward制造加速。

### 4.3 Latent-only不是现有接口已经支持

**采纳并亲查C-16：** V/algorithms/rollout/collector.py:584-619在`_finalize`强制decode，:630-687虽按decode微批仍拼整批媒体；V/data/samples/trajectory.py:734-750只允许已评分single_step释放media。

因此必须增加明确的**latent-only rollout结果/池视图**，reward stage再decode/score，之后生成紧凑已评分H3 replay batch。不能塞假媒体、借single_step标记或仅添加barrier。pool持detached joint状态、terminal、共享condition、ID/behavior、双schedule/score元数据、冻结score/anchor；CPU/disk byte cap预检，超限拒绝或走已声明本地盘路径。磁盘块仅位于获准run目录的attempt/collection/rank命名空间；本任务没有创建运行池。

生命周期：reserve→写latent→逐条媒体score后释放→填anchor→update读→**完整tail成功后**删除消费完的池。失败缓存不冒充可恢复的mid-collection状态。H3不能放宽V/runtime/algorithm_binding.py:1028-1083的Flash/Wan/preset guard，也不复用Flash多commit/EMA/SMALL40合同。

### 4.4 小表、随机流、质量与重试

全局只传有界的`collection/behavior/sample/group/member/score_components/valid/版本`表及必要诊断，不传全部latent、video、audio或conditions。检查恰好PK唯一member后冻结一次score表。local_complete只省组均值通信，collection std仍全局。

共同模型初态与采样RNG分离；sample/member/模态/step流和eval流版本化，固定实际Generator消费顺序。池不拥有任意可变Generator对象；same-topology恢复保存真实rank默认RNG及显式流键/派生版本，不编造不存在的跨iteration cursor。

首版无透明单member重试：异常/非有限reward整collection失败、无更新，退出从最近完整点重做；不换seed补候选、不删失败行减K。有限黑帧/静音样本仍保留，按既定reward政策评分；新增惩罚需版本化。未来bounded scoring retry只在固定媒体/版本/幂等ID/deadline及首次结果冻结合同通过后开放，随机服务同请求不保证同值。

候选训练reward为CLAP+ImageBind(mode=all)、各1，CPU可用性/数值/许可准入后使用。CLAP主要音频文本，audio_video alignment没有完整视觉质量覆盖；独立评估prompt-video一致、视觉保真/黑帧、时序/运动、静音/削波、AV同步、实际帧数/fps/sample_rate/时长与固定heldout可视听样例。不用总reward或GPU利用率替代吞吐和质量结论。

### 4.5 同序模型事件和失败处理

等slot数只是必要条件，partition跳过、reference/CFG、AC重算及实际FSDP child访问也必须匹配（V/algorithms/optimization/recompute.py:550-601）。H3首版guidance1/beta0、固定geometry/J，缩小动态分支；禁止lossy跳block缓存等未审组合。

拟改执行边界：公共计划→slot本地预检与健康READY→prepared forward→本地score/parity/finite结果→**所有rank已完成模型collective且健康时**共同决定backward→正常同步。现kernel本地raise、execution closure后立即backward（V/algorithms/optimization/kernel.py:585-609；execution.py:391-411），需在execution层受控传递诊断状态，不在每个adapter独立加barrier。

| 事件 | 首版处理 |
|---|---|
| 空rank、uneven、缺cache/非法shape或计划 | 模型phase前共同拒绝 |
| 已完成forward后的可协作score/parity错误 | backward前公共失败，无step |
| aux错误、invalid或缺member | 无模型collective在途的健康phase共同失败；不补样 |
| forward/AC/backward内部OOM、collective失配、rank死亡 | 不假定能到后置门；launcher超时终止整组 |
| optimizer或data tail部分成功 | 整attempt poison，不就地重试Adam，不进入下个behavior |

控制PG/心跳是诊断与健康协调，不保证救活挂住的NCCL。失败finally不新增barrier、unshard或临时checkpoint。正常与failure-local close分开审计。

## 5. 最小H3接口、精度与VisualRL改动清单

### 5.1 必须新增的窄类型能力

V/algorithms/dynamics/interface.py:190-229绑定单dt[B]、mean/std与x_t同dtype且std>0；TransitionRecord.validate:483-496再次构造Gaussian stats。本轮复读确认：**仅覆写_log_prob不够**。

推荐封闭typed分支，而非通用多模态图框架：
- `JointLatentLayout`：video/audio offsets、row原shape、生成/条件/padding masks、geometry、packing版本。存储可继续`[B,T,Dv+Da]`；attention输入顺序另显式表示。
- `H3ScheduleSnapshot`：两套精确sigma数组、transition index、eta/selection、预测符号、score/量化版本。原单dt不改义为index，不塞平均dt蒙混。
- `H3CPSStats`及对应record/evaluation分支：FP32两流mean/q/dt、score_active与noise_nonzero、terminal能力、固定动作域。sample/evaluate/make_record在窄public端口合法分派；Gaussian分支保原校验与World post-hook语义。
- `AVMedia`：video、waveform、fps/sample_rate、实际长度/有效范围及独立拥有/释放；H3 latent-only与已评分replay状态可合法无媒体，不假称single_step。

H3 adapter只做一次joint prediction、packing/sign与decode；Dynamics负责CPS数学，不另藏训练loop。score FP32不为迁就FP16状态合同降精度；首版无conditioner/hook，不支持组合早拒。

### 5.2 参数生命周期与精度

主`transformer_blocks`内LoRA targets：attn.to_q/to_k/to_v/to_out.0、ff.net.0.proj、ff.net.2；prefix限制防误命中refiner。root_wrap=false、root外全冻结，未被合法分片/同步的trainable直接拒绝。冻结复制部分仍需要初态及buffer管理。

FP32岛包括输入patch投影、时间MLP、输出heads、rope；AdaLN SiLU等按原实现处理，不整体`.to(bfloat16)`或root autocast覆盖。U/models/minimax_h3/vendor/transformer_minimax_h3.py:425-440明确声明；R:62-117给FP32 master/reduce、cast_forward_inputs=false及LoRA范围。候选采用外部non-reentrant block AC一条路径，关闭内部flag，EVAL+grad，包裹顺序同FSDP一起锁定验证，不叠两套AC。

V/runtime/lifecycle.py:742-805现在先build optimizer再manager.prepare（本轮复读）；V/models/state/parameters.py:155-197是旧Parameter身份协议。改为最终DTensor/FQN绑定后建optimizer/scheduler、更新数值owner，**不是删除身份保护**。backend负责global norm、状态codec和受控close；旧single模型路径保留。

### 5.3 按依赖的源码落点

所有路径相对V/；“新”文件/类型只是拟议，本任务未修改。

| 顺序/最终ID | 最小改动区域 | 当前依据、依赖与验收 |
|---|---|---|
| **F-01 profile** | `composition/config`、recipes、`models/catalog.py` | 显式score/terminal/anchor/advantage/zeroA/资源范围；依赖用户目标与软件门。不得继承Flash隐含参数 |
| **F-02 H3语义** | 新`models/implementations/h3.py`、新`algorithms/dynamics/h3_cps.py`；`dynamics/interface.py,session.py,replay.py`、`runtime/model_binding.py` | B-02/03、C-03/04；单dt/正std/dtype和record冲突真实存在。typed分派与双schedule，旧Gaussian负例不变 |
| **F-03 数据/媒体** | `data/samples/trajectory.py,collate.py`；`models/interface.py`；`rollout/builder.py,collector.py` | C-16，当前finalize强制decode、media释放限single_step。增加latent-only、AV及scored-replay合法状态，保状态链/shape检查 |
| **F-04 模型backend** | `runtime/lifecycle.py`；`models/lifecycle/prepared.py`、`models/state/parameters.py`及窄H3 loader | A-15/C-15、B-06。meta/最终绑定/完整FQN、FP32岛/AC、单PG owner；strict覆盖、host峰值、DTensor Adam/norm准入 |
| **F-05 数据分配与数学** | `data/prelude.py,group_placement.py`；`optimization/advantage.py,credit.py,slots.py,kernel.py,recompute.py,execution.py` | A-04/05、B-05、C-06/07。全局小表、None优势clip、local/world N、d/N、公共事件门、零A普通提交，不另写PPO |
| **F-06 本地池/aux** | 新小型`algorithms/trainer/h3_collection.py`；`trainer/stages.py`；`runtime/resources.py`与非prepared组件驻留 | C-08/09/16。六stage保留，rank-local池无模型owner；heavy acquire延后；不放宽`runtime/algorithm_binding.py:1028-1083`的Flash guard |
| **F-07 提交/恢复** | `trainer/grpo.py`；`runtime/checkpoint_binding.py`；`artifacts/checkpoint/{manager,transaction,protocol,state}.py` | kernel外tail、rank/global身份、共同staging、全rank codec/纯writer、双门/最后RNG；依赖F-04—06 |
| **F-08 独立净简化** | metadata数值policy、相邻统计/host转换、capture和reader边界 | A-14/B-11/C-13。只在owner/有效期已证明的地方去重，不作为所有H3工作前置条件 |

**最小第一批实施内容**：F-01/02/03的profile与typed语义、F-04的加载/绑定准入原型及tiny验证设计；随后接F-05—07形成可恢复闭环。不能先删除single断言、加一个YAML开关就称分布式，也不能先重写整个request/builder/artifact体系。

净简化可做：纯不可变metadata hash移构造/绑定/视图切换/恢复边界；同mask归约复用；已拥有CPU结构不反复从设备取回；同capture临时重复clone在保最终snapshot拥有后去掉。真实文件摘要绑定实际读字节、两门各刷新；可变tensor内容、rollback副本、分片完整性、prepared hooks、固定动作/旧锚、mask覆盖和behavior版本不可删除。old replay不是冗余诊断。保旧SD3/Wan、World surrogate及Flash SMALL40/pending/eval合同。

## 6. 完整提交、checkpoint、恢复和评估

### 6.1 Safe-point覆盖完整tail

本轮复读V/algorithms/optimization/execution.py:439-505已有optimizer/scheduler/可选EMA-reference/zero_grad/logical_commit；V/algorithms/trainer/grpo.py:148-163之后才data、collection消费与next-step。新H3无EMA/reference；汇合现有操作，不能再调用第二份。

执行顺序：全slots反向/同步→finite/global norm/clip→一次Adam→一次scheduler→zero_grad→trainer一次data/pool/next-step→全rank完整成功确认→global behavior版本与safe-point。executor的COMMITTED只是内部子阶段，不提前充当全局安全点。partial失败不承诺分布式原子回滚，从最近完整checkpoint重启。

### 6.2 全rank状态操作与单writer发布

| 操作 | 参与者/限制 |
|---|---|
| 兼容、state coverage、文件和资源预检 | 全rank先本地检查，健康READY在潜在collective之前 |
| 模型/optimizer getter、DCP save/load/set | 全rank同operation/mesh/path参与；禁止隐藏在rank0 callback |
| rank RNG/data/设备绑定/局部shard记录 | 各rank写自己条目 |
| 文件树checksum、manifest/complete、rename/latest | 同一文件manager单writer，不访问模型 |
| FSDP eval、必要权重view切换或export | 对应全rank同序参与；rank0只负责普通文件/小指标不等于只它forward |

当前V/runtime/checkpoint_binding.py:539-557把本rank RNG identity放progress；artifacts/checkpoint/transaction.py:407-425却要求progress跨rank一致。设计分层：
- global contract/progress：artifact/精度/score/layout/计划、W8/SP1、committed step/behavior、共同游标摘要及**有序rank状态表的全局digest**。
- rank binding/state：逻辑rank→获准UUID/local device/mesh/data/RNG、shard文件索引及hash。比较全表而非把各自单行hash当同一个global值；本地绑定严格检查保留，不删除全部numerics。

共同staging供所有codec写入，包含DCP metadata、所有shards和rank附属态；全rank正常完成后单writer校验完整树、complete/原子rename/latest。可为现manager增加begin/commit-existing窄接管，不造第二manager。rename后latest失败可能已经有合法complete，保留它，不倒删。rank死亡时未完成staging不是checkpoint，不能等它再barrier补确认。

### 6.3 状态覆盖、冷态与fresh恢复

保存/校验至少包括：LoRA/master参数和必要模型状态、Adam moments/step/参数组、scheduler、全局step/behavior/collection、data cursor、rank RNG与显式流策略、score/双schedule/selection/quantization/cache/resource身份、eval状态、base artifact不可变来源。冻结base若不重复保存，必须严格绑定可再取得的相同权重；LoRA导出不是训练恢复。

必要buffer分三类：state_dict持久态、锁config/dtype可重建非持久构造态、影响未来计算的可变运行态。H3 rope等不由strict键集自动覆盖，恢复需重建/校验或保存真实可变态。

**首版训练checkpoint不支持step0/cold/部分缺失预期trainable Adam state，进入有副作用helper前全rank早拒。** 不能虚构零梯度step填moments，不能整体strict=False掩盖缺态。只有调用过optimizer.step不足以证明全部FQN已初始化。

**独立强门：已有moments的checkpoint加载到fresh进程的新optimizer。** 验证目标get/save/load/set是否物化及有何副作用，全部state presence/shape/dtype/step、参数组、LR/scheduler/参数/RNG在恢复后正确，下一步与不中断对照一致。禁step0只缩小save范围，不证明fresh load安全。当前无目标准确DCP实现/运行证据，不能称codec已经可用。

恢复顺序：全局artifact/文件/同拓扑预检→构造final shards及optimizer→prepared gate全rank加载/覆盖验证→bound gate新完整性与live data/reward/score/cache检查→各rank训练RNG最后恢复→全rank统一CONTINUE/COMPLETED。保两个时点，不因去重复hash合并。首版只支持pool已耗尽的完整collection边界、same world/rank语义/mesh映射，不承诺重编号、弹性或任意mid-collection精确恢复；旧Flash pending合同独立保留。

### 6.4 评估与退出

eval有独立seed和固定prompt/geometry/score版本，所有必要FSDP rank按同序调用；首版评估样本数不合法整除则提前拒绝。媒体本地发布、小receipt汇合，不all-gather全视频。eval结束恢复训练模式/RNG；初版H3无EMA，未来EMA必须全rank一致且状态完整。

V/artifacts/checkpoint/transaction.py:452-479的failure_gate在operation之后，无法保证collective内部失配的peer到门。内部错误/死亡由launcher结束整组；failure-local close只做已审计本地释放，不隐藏unshard/save/barrier。当前prepared.close清guard/引用，不代表所有custom closer无通信。

终态恢复统一分支，不能rank0提前return而其余继续模型调用。已有durable checkpoint不等于干净退出：有主错保主错并附cleanup错误；无主错cleanup失败仍报告失败/非零退出并携durable身份；不删除合法complete来“回滚成功”。

## 7. 概念配置草图：当前不能直接运行

以下是拟议合同，**H3 family、score family、dual schedule、latent-only、FSDP backend/阶段计划、world归约、None优势裁剪、cold/fresh codec等尚未接入现生产schema**。某些字段有单进程对应物，也不意味着整段YAML已被支持。未知准入字段必须填实值，不以null作为运行默认。

```yaml
profile: h3_t2va_cps_element_replay_v1  # 拟新增，用户目标待确认
hardware:
  num_nodes: 1
  world_size: 8
  authorized_gpu_uuids: null           # 必填，禁止默认为物理0—7
  environment_lock: null              # 必填：依赖/内核/driver证据
  artifact_manifest: null             # 必填：完整FQN/dtype/buffer/许可
backend:
  kind: fsdp2_blocks
  parameter_group: world
  root_wrap: false
  sequence_parallel: 1
  tensor_parallel: 1
  reshard_after_forward: true
  init: meta_then_rank0_cpu_full_state  # 有host/setter峰值及覆盖门
  active_prefetch: false
  parameter_cpu_offload: false         # 非prepared aux驻留另管
  activation_checkpoint: external_nonreentrant_only
  internal_gradient_checkpointing: false
  reducer_divisor: must_verify         # 不能凭W推定d
model:
  task: t2va
  geometry: {height: 256, width: 384, requested_frames: 124, fps: 24}
  lora: {rank: 32, alpha: 64, dropout: 0, prefix: transformer_blocks}
  lora_targets: [attn.to_q, attn.to_k, attn.to_v, attn.to_out.0, ff.net.0.proj, ff.net.2]
  precision: {block: bf16, islands: fp32, trainable_master: fp32, reduce: fp32,
              trajectory: fp16, transition_score: fp32, vae: fp32}
  text_cache: require_full_train_eval_coverage
policy:
  transition: joint_cps
  score: {family: cps_neg_mse, reduction: element_mean, target: stored_sampled_action}
  schedules: {steps: 10, video_shift: 12, audio_shift: 3, eta: 0.7}
  policy_steps: all_10_including_terminal_proxy
  guidance: 1
  hook: none
  old_anchor: {source: replay, timing: after_credit_before_current, frozen: true}
  ppo_clip: 0.005
  beta: 0
  advantage: {center: occurrence_group, std: unique_collection, ddof: 0,
              compute_dtype: float64, output_dtype: float32, epsilon: 1.0e-4, clip: null}
  zero_advantage: ordinary_optimizer_commit
  require_nonzero_gradients: false      # 只针对此profile；finite/图覆盖仍保留
batch:
  prompts_per_collection: 8
  candidates_per_prompt: 8
  placement: local_complete
  per_rank_placement_rows: 8
  outer_accumulation: 1
  forward_microbatch: 1
  replay_microbatch: 1
  timestep_chunk: 1
  updates_per_collection: 1
  unequal_work_or_empty_rank: reject_before_model_phase
optimizer:
  kind: adamw
  lr: 1.0e-4
  betas: [0.9, 0.999]
  epsilon: 1.0e-12
  weight_decay: 0
  max_global_grad_norm: 1.0
  lr_schedule: constant
  warmup_steps: 0
  implementation_policy: must_lock_and_validate
  ema: false
aux:
  phase: after_all_rollout
  order: logical_rank_turns
  node_heavy_concurrency: 1             # 保守资格基线，非性能最优共识
  decode_microbatch: 1
  scorer_device: cpu
  scorers: [{name: clap, weight: 1}, {name: imagebind, mode: all, weight: 1}]
  lazy_load_release_per_turn: true
  automatic_member_retry: false
storage:
  trajectory: rank_local_cpu_or_declared_disk
  max_decoded_av_inflight: 1
  pool_byte_cap: null                  # 按画布/host准入填，不含糊无限
  media_and_workspace_budget: null     # 必填，scorer/resize另盘点
checkpoint:
  boundary: complete_collection_and_tail
  state_codec_participants: all_ranks
  file_publisher: single_writer
  topology: same
  cold_adam: reject_before_codec
  fresh_optimizer_restore: independent_required_gate
  resume_gates: [prepared, bound]
  restore_rng: last
```

运行预算、数据集、eval固定样本/步数、实际audio sample_rate/长度及服务deadline仍须用户和artifact定稿，不凭本草图编造。

## 8. 里程碑、证据验收与性能方法

**以下全部为未来另行授权的计划，本workflow没有执行。** 每门失败应停下，不以删audio、缩J、换reward或加未授权GPU蒙混过门。

| 门 | 交付 | 必过证据/停止条件 |
|---|---|---|
| **M0 资源/目标/依赖** | 8 UUID与窗口；硬件/host材料；用户T2VA、画布/reward/score选择；源码/模型许可；完整环境/artifact/cache清单 | 0%利用率、UMD或SKU不足准入；公开API/vendor和加载路线审计；必需FQN全覆盖，非25%比例 |
| **M1 语义与tiny** | typed joint/CPS/latent-only、独立score family、零A与优势参数、旧锚 | pack/slice/collate、双精确schedule、符号、124 geometry/AV长度、量化q0及梯度、CPS/Gaussian拒绝域、action detach、旧锚不可漂移；tiny不替真H3 |
| **M2 分布式结构** | tiny可容纳模型的8rank/小规模参考、final binding/backend、global表/slots/完整tail/codec | 固定全局数据单/多rankloss-grad-update容差；d、FP32reduce、AC、global norm、零梯度/缺梯度区别；uneven/空rank拒绝；实际模块collective序列；健康前后故障与内部失败整组退出 |
| **M3 真实8卡低画布闭环** | 获准8卡真正H3：load→rollout→aux→credit→anchor→update→save→fresh-process restore→下一步/eval | 分阶段每rankallocated/reserved、host RSS/磁盘/转换峰值；text cache覆盖；真实BF16/attention/AC/DTensor Adam；pure capture、fresh optimizer moments/step与RNG下一步对照；未通过不称训练ready |
| **M4 固定语义性能/质量** | 稳态重复collection、checkpoint/eval摊销、逐项调aux并发/缓存/预取/存储 | 固定artifact/P/K/N/J/画布/reward/score/硬件；端到端AV/s、commit/s、GPU-hours/有效AV、失败重做率和分项质量；报告冷启动和稳态、rank最慢阶段与峰值 |
| **M5 条件扩展** | SP2/4或同8卡阶段TP副本，分开实施 | SP的mask/DOF/随机流/κ/d/伴随与checkpoint拓扑；TP的权重版本ACK、LoRA映射/geometry/score及驻留切换；净收益和质量通过才替默认 |

### 测量口径与切换判据

记录加载/cache、每步forward取参与compute重叠、old-anchor、current forward/backward/RS、optimizer/tail、decode、CPU评分、H2D/D2H、控制/磁盘等待、checkpoint/eval。不能将AG与计算时间简单相加忽略重叠；同时报告实际端到端时间。

- **SP触发**：chunk1+单AC之后单样本激活/workspace仍越预算，且瓶颈不是加载/aux/RAM；SP实现正确并有可接受通信成本。记录总峰值，而非只报某activation约1/s。
- **TP触发**：同工作量实测rollout取参占主导且TP驻留可行；节省rollout时间大于新增sleep/wake、权重同步/布局、资源切换及额外锚成本。最好直接比较完整collection重复测量，避免漏项；无统一预设加速倍数。
- **aux调优**：并发1→2/8先过host峰值和评分等价门；GPU placement另审训练shards+VAE/scorer峰值。
- 增P/K、删J、改变轨迹dtype/目标画布/reward是独立配方变化，不能作为同工作量speedup。固定时间质量预算单独报heldout视觉、运动、audio与AV质量，不以reward上涨证明画质。

## 9. 辩论裁决、修订与保留分歧

三个第二轮并行完成，没有读取彼此第二轮。因此以下后出发现由本汇总核验/裁决，**不虚称其他角色已逐项回应**。

| 问题/追踪 | 最终处置 | 依据与仍缺项 |
|---|---|---|
| Q1 CPS/权重/terminal；A-07、B-03、C-03 | 采纳元素均值CPS新profile、J10 terminal proxy；保Gaussian原检查 | 三方源码一致且汇总回查。VeRL .5/.5差异保留；用户质量基准未定，不称最佳或统一标准 |
| Q1 优势/零A；B-05、A/C Q1 | 显式float64/eps1e-4/K8、零A普通commit；无额外优势clip | 前者是主动改变；None clip由A/B提出并经汇总双侧代码复核，C未后续回执；保旧recipe默认 |
| Q2 容量；A-02、B/C Q2 | 用主blocks/LoRA条件结构式，不用66 GB加载峰互证；R约1.60 GiB单独标A推算 | 真实artifact/外部SwiGLU、复制/工作副本、激活与aux峰值仍缺 |
| Q2 加载；新增A-15/C-15 | 采meta+rank0 full CPU输入的受审计候选；完整FQN覆盖 | 汇总亲查full字典/25%弱guard。B有界shard目标未闭合；host不够即换路线，不宣称已有流式loader |
| Q3 SP；A-03/B-08/C-06 | 全8 FSDP+SP子组仅候选，SP1首版 | 共同归约图不能替目标adjoint/内核证明；κ/d未知，后续实现阻塞 |
| Q4 同序/归约 | d*n/N与实际模型事件并审，uneven/空rank早拒 | kernel缩放行统一609；同slot数不充分，公共门不能救死亡rank |
| Q5 H3依赖 | 只定窄adapter合同，拒整包搬入；公开H3或vendor在M0审计 | A偏公开API，B/C偏vendor；Python范围不交叠属事实，不证明局部迁移不可行；许可/版本未闭合 |
| Q6 latent-only；新增C-16 | 必须纳入首批实质改造 | 汇总亲查强制decode及media释放限制；不是只加phase barrier，A/B未读取此第二轮新发现 |
| Q6 aux并发 | 本文选并发1lazy rank turn作为资格基线 | A=2、B=1、C=8均未测；保留替代方案，性能/host证据决定以后并发，不称共识最佳 |
| Q7 旧锚 | credit后current前一次frozen replay，首版不省pass | 真算法输入，不是step0诊断；rho1不证明采样分布等价 |
| Q8 恢复 | 全tail/全rank codec/纯writer、cold早拒、fresh-load独立强门 | 目标codec未证明；两个恢复时点与必要buffer均保留；DCP名义不关闭问题 |
| Q9/10 最小实现与切换 | 六stage/现owner窄扩展，独立局部去重；实测可推翻同模型 | 不新增actor/参数服务器，不复制trainer；无实际吞吐或质量收益证据 |

### 否决或暂不纳入首版的方案

1. **直接套wrapper/删single断言**：参数身份、optimizer次序、world数学、phase/恢复均有实质缺口。
2. **把CPS改Gaussian、静默删末步/音频或继承Flash信用**：改变目标，不属于工程加速。
3. **先GPU完整加载、CPU八份基座/Qwen、训练时cache miss临时上卡**：加载/host/单rankphase风险未通过预算。
4. **只rank0跑FSDP eval或capture、rank死亡后barrier补保存**：潜在collective失配；post-operation gate不能救挂死。
5. **首版SP开关、4+4固定池、异步旧策略、第二推理engine常驻、弹性恢复**：需要不同容量/数学/版本/恢复证明，无必要证据即不增加复杂度。
6. **先造大型actor/factory/receipt框架或全局去保护**：不增加已证明能力；旧锚、真实内容、snapshot/rollback、hook、完整组及提交边界必须保留。

## 10. 需要用户补充的资料与决策

按阻塞优先级：

1. **资源授权**：具体8张GPU UUID、允许使用时段、现有进程归属和足够空闲窗口；不能由历史0%决定。
2. **节点内材料**：获准GPU的topology、P2P、PCIe实际代际/宽度、NUMA/CPU、host RAM、盘空间/吞吐与当前占用。可提供 `nvidia-smi topo -m`、`nvidia-smi -q`相关输出和进程表；本任务未执行这些命令。collective/P2P基准另行授权。
3. **训练环境**：实际Python/torch/CUDA/runtime/driver/NCCL、compute capability、BF16、attention kernel及版本；UMD13.3不是此清单的替代。
4. **模型与数据**：H3真实checkpoint/config/shard清单、必要精度/buffer/LoRA范围、许可；train/eval prompt集合、encoder缓存是否生成及准确身份。许可不满足时先调整合法reward/模型方案并记录目标变更。
5. **质量目标**：T2VA还是主要视频质量；即使视频目标也需明确audio状态/score演化政策。目标分辨率、帧数/时长、训练时长预算、heldout评价、reward权重与CPS元素均值/模态等权/终步选择。
6. **复现要求**：接受本文显式新profile的K8、float64/epsilon1e-4、无EMA/CPU评分范围，还是要求严格对应某一上游配方；不能含糊混配后称复现。

## 11. 本次实际交付与验证状态

- 已完整读取并汇合三份第二轮Q1—Q10回应；六份角色报告保留。`summary.md`保持第一轮讨论状态，本文给最终设计取舍，不追加轮次或子代理。
- 本轮亲自回查：UniRL full-state loader及25%覆盖门、VisualRL强制decode与media释放限制、上游无优势clamp/当前GRPO clamp、CPS q0采样、H3默认构造/精度/recipe、当前optimizer先于prepare、record的Gaussian约束和execution提交顺序。其余链条承接已完整读取的角色报告与首轮汇总证据，不代造他人实验。
- 目标八份输出是 `round_1/a.md`、`round_1/b.md`、`round_1/c.md`、`round_2/a.md`、`round_2/b.md`、`round_2/c.md`、`summary.md`、`final.md`。本文件写入后将只读回读并列举确认；独立 `validate_reports.py` 检查由后续工作流执行，本Agent仅阅读其结构规则，未运行。
- **完成的是文档审查与设计，不是代码实施、GPU适配或性能/恢复验证。** 所有硬件、目标依赖、真实峰值、SP伴随、DCP/fresh恢复与质量收益缺口均保留；没有修改源码或执行未经授权操作。
