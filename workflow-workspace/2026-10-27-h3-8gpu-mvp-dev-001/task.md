# H3 单节点8卡最小实现：Orchestrator + 3 Coding/Simplifier 配对

## 1. 目标与权限

用户要求实际开发最简单可用版本，而非再写一轮纯分析。目标代码是当前真实framecode仓库的visual_rl及相关配置/测试，不是外层历史inputs快照。起始源码基线d818b8a231b52a9a024fa2f522057acdbb7bf099（已修好七项baseline并清理旧文档）。

允许：Mac本地源码/配置/测试修改、所列CPU测试、本地原子commit与最后整合merge。禁止：push、服务器操作、SSH、下载模型、占用GPU、终止现有进程、修改共享Python环境、改写历史报告或忽略用户未提交变化。真正GPU验证需要另行确认8个GPU UUID/可用窗口、模型/数据/评分资源和训练环境；本次完成仅可称“实现与CPU门禁交付，真实8卡GPU待验收”，不允许冒称跑通真实H3。

角色只写本任务目录内指定的计划/交付/审查记录；外部用户进度文档由主会话维护，不执行旧AGENTS中的跨项目写入或服务器部署。主工作树应保持干净。只在引擎分配的Git worktree/分支工作；不得切到原main直接改源码。原子commit按功能/合同小步切分，拒绝“全部重写后一次大提交”。最终由orchestrator批准，工作流Git引擎负责真实snapshot/merge/integration验证和安全fast-forward；不绕过最终验证手工推main。

## 2. 固定范围：最小闭环，不追求性能或框架大全

- 单节点，目标8×NVIDIA CMP170HX；用户表每卡64GiB，可见10卡但只授权计划8卡，具体集合未定。
- 无NVLink，两NUMA，各5卡，PIX配对(0,1)/(2,3)/(6,7)/(8,9)；只讨论FSDP2同步同模型rollout。默认SP1/TP1，不实现HSDP、多节点、异步rollout、独立vLLM推理后端或4+4卡池。
- 基线任务H3 T2VA LoRA；联合音视频状态、双schedule和score必须正确，不能删音频状态来伪装支持。
- rollout/replay microbatch1；固定小画布、固定group和相同rank工作量；一次完整collection一次更新；有界CPU轨迹和媒体；reward先可用CPU/现有接口，不新造通用服务平台。
- 同一模型完成真实采样→真实评分→advantage→旧策略锚→replay/backward→optimizer→checkpoint→同拓扑fresh恢复/下一步的可调用路径。重模型/评分依赖可lazy-import，但生产路径不能以mock、常量reward、零loss或NotImplemented占位冒充实现。
- 本地CPU可用tiny/fake重量级叶子验证真实控制/数学/状态链；明确这种证据不证明CUDA FSDP2、H3权重加载或质量。
- 旧Flash SMALL40/140、SD3/Wan语义保持，不把H3功能强塞进Flash批准实验，不全面删除guard/artifact、不泛化整个框架。
- CPS代理score与Gaussian密度分开；选定一套明确上游参考及音视频权重，不能把三套语义混用或靠epsilon/删有限性检查掩盖差异。
- 仅完整collection消费并提交tail后的checkpoint；同world/topology恢复，无弹性和mid-collection恢复承诺。
- 保留现有唯一训练入口及controller；允许必要的torchrun入口装配，不另造整套Runner。

## 3. 先读的材料

所有工作流资料路径以仓库根的`workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/`为前缀（下称W）。

- W/reference/architecture-review.md：上一轮设计；是证据与候选，不是必须照抄的完整工程清单，优先最小闭环。
- W/reference/hardware.md、topology.md：用户硬件观测，topology补充优先；高占用历史图不代表卡空闲。
- W/reference/upstream-comparison.md、sources.json：固定上游提交。
- W/reference/{UniRL,verl-omni,Flow-Factory}/：精简只读参考，非安装依赖，部分import目标不在快照。各LICENSE一并保留。不得修改参考文件或将其整个拷入生产代码；优先官方依赖接口，必要移植明确许可证和最小范围。
- 当前仓库AGENTS.md、README.md、pyproject.toml和实际visual_rl源码是实现事实。AGENTS中的旧Flash部署指令不是本次启动/停止服务器授权。

## 4. 七个持久角色及工作约束

### Orchestrator

1. 先真实跑七项baseline，读取现有代码和MVP边界，写W/plan.md并把任务切成小原子项，每项有ID、目标、允许改动路径、输入输出合同、测试/验收与非目标。
2. 在W/assignments/a.md、b.md、c.md明确active/idle、任务ID、基线/前置已合并提交、允许文件、接口锁和验收；不同并行任务不得修改同一文件或未稳定公共接口。
3. 可以按依赖顺序调用任一配对，也可选择parallel-wave。建议先稳定H3状态/score最小合同，再考虑并行模型、分布式、闭环接入；分工不是固定必须三人同时忙。
4. 顺序步骤在同run分支工作区执行；并行时引擎为各coding任务分配独立worktree分支，coding将实际branch/HEAD写入交付。并行分支先在run内临时汇合供review，这不是接受或发布main。对应simplifier可读取汇合后的源码，但只审自己的任务及影响；若有依赖污染需退回。
5. 每次审核三类事实：源代码真的实现、对应simplifier通过、真实CPU命令成功。更新W/merge-ledger.md，记录任务→coding提交/branch→simplifier裁决→集成基线。报告承诺不是测试结果。
6. 需返工则重新派给同coding agent，明确issue ID，不跳过对应simplifier。避免无任务角色制造无意义改动或伪工作量。
7. 不直接编写功能源码或修复别人的实现；集成发现缺口也应分派给对应coding/simplifier。工具级Git冲突/未知外部副作用进入needs_attention时保留工作区，主会话核查，不强制覆盖。
8. 最终再次跑固定七项及H3目标CPU集，核对最小真实调用链、无占位/隐藏mock、无越界diff、旧实验配置不变、所有任务审核完成，再写W/final.md并返回complete。引擎随后在独立candidate重跑integration命令，通过才合入本地main。
9. final必须区分已实现、CPU验证、未执行GPU验证、硬件/资源待确认；附实际源码路径/提交/配置/运行命令和后续GPU验收步骤。

### Coding A/B/C

- 只接受orchestrator在本配对assignment中的当前任务；active=false时idle，不改源码，不冒称跑了测试。
- 在引擎指定worktree/branch修改。源代码原子项尽量只解决一个合同/功能；大任务先向orchestrator建议拆分，不顺手重构无关模块。
- 每次变更后跑同一组七项CPU门禁，H3新增功能对应的目标CPU测试也要跑。失败就修复，不能skip/xfail、删断言、缩样本、替换7项、关闭生产保护使测试变绿。
- 用真实工具输出记录命令、exit code、7个测试结果、所测工作区和源码diff；不能把历史通过当本次通过。
- 写W/coding-<slot>/<task-id>.md，含实际分支/HEAD、修改文件、合同、测试结果、待验证风险、simplifier需关注点。完成一个原子项才交付；引擎在角色结束时形成真实Git snapshot，不手动reset/rebase/force或提交共享main。
- 若资源/依赖缺失导致不能完成，报告blocked和证据，不写伪实现。无权安装升级共享环境。

### Simplifier A/B/C

- 与同字母coding配对，只审核、不直接修源码。比较总task、assignment、实现diff/调用链和真实测试记录。
- 检查过度设计、重复实现/校验、无消费者接口、过多artifact、新旧两套执行路径、不必要base/factory、超范围性能优化。
- 区分能删的机械复杂度和不可删的CPS/旧锚/归一化/分片同步/恢复语义；不能为少代码破坏MVP正确性。
- 顺序review返回approved或changes；changes必须具体到文件/位置/问题/最小修正和验证要求，退给同coding再测再审。
- 并行review把approved/changes/blocked/idle写进自己的报告；返回done只是review步骤结束，绝不代表代码获批。orchestrator据报告安排返工。
- 只写W/simplifier-<slot>/<task-id>.md。需要时可独立重跑允许的CPU命令；不修改tests使其通过。

## 5. 固定测试命令

运行位置必须是实际引擎worktree根目录。使用主checkout已有venv，但通过PYTHONPATH=.保证导入当前worktree，测试自身也会核验导入身份。

七项原CPU baseline（必须原样保留七个函数，不能换一套同数量测试）：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

它们是Flash回归，不足以证明H3。新增H3针对性CPU测试集中在`tests/test_h3_mvp_cpu.py`，作为单独的功能验证，不冒充原七项，也不擅自跑全量768测试：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
```

最终H3 CPU集至少实质覆盖：joint状态/packing及双schedule、CPS与Gaussian边界、旧锚冻结、world级advantage/loss归一化、实际调用序列/不等rank拒绝、checkpoint状态覆盖与fresh恢复、tiny端到端真实控制路径。针对性测试可分原子项逐步添加，但不能只有mock controller/常量输出来替代所测实现。CPU集合不skip/xfail掩盖未实现。真正FSDP/CUDA测试另列待验证，不让CPU伪装它。

## 6. 小步实现建议（由orchestrator重新拆分定责）

- M0：baseline/依赖事实与接口锁；不要重新修已修好的作者数据路径。
- M1：最小H3 joint/CPS数据与数学合同及目标CPU测试。
- M2：可加载的H3 adapter/recipe与lazy外部依赖、精度/LoRA范围、真实rollout/replay入口。
- M3：同节点FSDP2 prepare、rank-local采样、全局统计/梯度缩放和统一阶段计划。
- M4：reward/解码有界资源、collection尾部提交、分布式checkpoint同拓扑fresh恢复，最小torchrun使用说明。
- M5：最终CPU整合、对应simplifier签收、清洁candidate验收和本地主分支合并；明确GPU_PENDING。

不为了并行强拆交叉接口；可以让某组连续idle。没有真实GPU准入条件时不要把M5改称真实8卡成功。

## 7. 完成定义

所有必需原子项都有coding交付和对应simplifier批准，未决必需功能不能隐藏成“后续优化”；实际生产入口可构造H3八卡路径而非只提供文档或伪代码；七项回归与H3目标CPU测试在最终candidate真实通过且未修改受跟踪文件；最终报告准确标记CPU实现交付/GPU未验证。引擎验证并安全发布后才可标workflow completed。若必须依赖本机不可获得的API/资源且没有真实可实现路径，明确blocked，不用stubs换取完成。
