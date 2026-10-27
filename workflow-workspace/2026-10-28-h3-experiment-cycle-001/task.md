# H3 真实实验闭环：Experimenter ↔ Coder

## 用户目标与最新权限

1. 完成新服务器环境配置，在真实H3权重、真实GPU、真实reward下，Flow-GRPO与Flash-GRPO各完成独立5次optimizer更新。报错则根据真实反馈修复、Git提交推送部署、复验，直到两者最小测试均完成。
2. 两者通过后按用户正式计划启动并监视大型实验，完成获准目标，保留真实PID/log/checkpoint/退出状态。正式阶段优先Flash，不将Flow的工程测试自动扩大为正式实验。
3. 主会话启动独立watchdog，监视workflow与owned job。不把无输出或GPU0%直接当卡死，不重放执行状态不明的命令，不停止别人的进程。
4. 用户明确同意本阶段跳过追加CPU回归、优先真实GPU；不得称未跑的CPU测试已通过。必要语法检查不是GPU验收。
5. 用户明确提供并授权SSH：`ssh -i ~/.ssh/id_ed25519_tailscale_ubtroot_192_168_102_193 -o IdentitiesOnly=yes ubtroot@192.168.102.193`。此新指令取代旧Flash文档中只用OpenCLI/禁止SSH的限制，仅限此主机与本任务。

任务目录W=`workflow-workspace/2026-10-28-h3-experiment-cycle-001/`。本任务不复用此前失败workflow，不改写其历史，也不修改旧reports；它从已推送87013b3主线成果继续。

## 硬件和已知环境

- 单节点计划使用8×CMP170HX，实测各64GiB；主机125GiB RAM，无NVLink，双NUMA。
- 优先候选物理卡0/1/2/3/6/7/8/9，对应PIX配对。只在实时查明空闲且没有其他owned任务冲突后分配；不占用额外两卡，不杀vLLM/别人的训练，不操作sudo/docker。
- GPU UUID见W/reference/hardware.md与topology.md。10卡0MiB、约112GiB RAM可用只是上次观测，不是永久预约。
- 服务器源码原目录：`/home/ubtroot/qifan/Efficient_Diffusion_RL_for_Visual_Generation`。
- 已准备独立环境：`/home/ubtroot/qifan/h3-env-20261027-001/venv`，历史记录Python3.11.18、torch2.8.0+cu128、diffusers0.42.1等；必须复核实际版本/API，不将UMD版本或一次NCCL probe当模型可运行。
- 原生缓存FL2VA/Ref2VA为旧MiniMaxH3DiTModel/VideoVAE命名，新Diffusers类不同。先核对真实参数/配置映射；不修改原权重，不盲改_class_name后假称兼容。必要转换写本任务独立artifact目录，由Coder提交可追溯转换脚本再运行。
- 当前代码是H3固定全步CPS路径；H3 Flash单步实现尚未确认。两条算法必须有真实不同的policy-step/credit/replay实现，禁止仅改名字、复用同一全步run或用tiny/mock代替真实H3。

## 角色

### Experimenter（唯一服务器操作owner）

- 负责环境核对、独立环境修复、模型/缓存/数据/HPS准入、部署、运行、观察、证据回收；不直接修改本地或远程算法源码。
- 开始先读W/formal-plan.json、reference、当前源码与使用说明。用真实inspect建立state.md/state.json（阶段、已完成项目、阻塞、当前owned job、最后反馈）。
- 按W/operations.py协议写W/request.json并执行唯一允许操作入口；每次id唯一。先bootstrap和inspect，再publish/deploy；所有source只能通过Git，严禁把编辑后的.py上传到服务器或在服务器hotfix。
- 远程运行根为 `/home/ubtroot/qifan/h3-experiment-cycle-20261028-001`。源码在sources/<SHA>新worktree；运行输出在runs/<唯一job-id>；保留失败目录，不用旧目录重跑。
- 本地publish只推固定实验分支`experiments/h3-cycle-20261028-001`，不force、不抢先改main。部署后核对git/import路径和配置。
- 测试按最小规模逐步进行，先实际加载/rollout/评分可用，再每算法5次更新。可以预检或单步诊断，但最终每算法必须有一个真实成功的5-update run，不能把若干失败/不同配置的片段相加凑5。
- 后台实验使用launch操作，必须声明timeout_seconds和独立job_id。正常训练等待返回waiting，由固定wait步骤延迟后再次检查；不得长时间阻塞一个SSH工具调用。每次观察更新W/state.json的jobs条目（job_id、sha、stage），供watchdog读。
- 有确定的源码问题或缺少必要功能，写W/issues/<id>.md及W/issues/current.md，含真实命令、退出码、栈、SHA、环境、模型/数据路径和最小成功条件，返回repair给Coder。环境问题自己处理，但不装/改系统Python或其他任务的环境。
- 远端stdout/SSH超时不代表命令未执行。先inspect原job/进程/输出，必要时用reconcile记录实际效果；不换job_id盲目重复训练，不重放未知部署/安装。
- 每个5步验收要检查真实GPU使用、模型参数量/权重身份、真实HPS有限反馈、optimizer计数、真实反向/参数变化、step5完整checkpoint和exit0。零信用样本不能删掉；若完全没有学习信号如实报告，不称RL效果通过。
- formal-plan.json里max_optimizer_steps目前未确认：先推进并完成两条5步测试，不因这个正式阶段缺参提前停掉环境与最小测试；二者通过后、正式启动前必须返回needs_input，请用户明确目标。不能将历史40步pilot当“大型实验”的授权数值，也不能把4×8/全局32偷偷换成8×8或不同group统计。
- code变动后对受影响算法复验；最终两条最小测试和正式实验必须对应最终生产代码或可证明生产代码无差异，不把旧SHA测试当新SHA通过。正式resolved_recipe的math中须明确global_prompts/samples_per_prompt/global_batch，分别4/8/32；不能只写含混的group_size让实际分组不可核查。正式model字段也必须对应批准画布/LoRA。
- 完成全部获准实验后写W/evidence.json、final.md，真实证据可由只读验证重新查询，才返回done。正式实验只是launch成功不能当已完成。

### Coder（唯一源码修改owner）

- 读取总需求、Experimenter当前issue和实际源码，只做解决真实问题或补齐缺失算法的最小改动。优先检查已有origin/h3-pytorch-training与origin/h3-readable-integration分支中已用过的H3加载/参数映射，不能仅凭类名判定权重不兼容或重复发明转换。通过W/coder-request.json写action=local_git_read、ref及可选path，再执行operations.py --reference，只读查询Git对象。
- 本地引擎worktree上原子修改，保持当前生产入口，不另造训练Runner。Flow和Flash共享合理组件，但单步语义真实独立，保留音视频状态和正确的旧锚/replay。
- 原权重转换/离线cache产生工具如必要，应提交在scripts/并明确格式、许可、精度和输出；不能下载或覆盖原始权重、不能在自己的角色操作服务器。
- 不为了跑过而fake reward、fake weights、写死SUCCESS、关闭必要数值/分片检查、删失败证据或隐式缩减正式目标。
- 不运行整套CPU回归或SMALL40；可用--syntax检查源码语法，并把需要的真实GPU复验交回Experimenter。
- 写W/coder/<issue-id>.md，含改动、风险、真实验证要求。返回retest后引擎自动snapshot，Experimenter再推固定实验分支、Git部署并复验。
- 若缺少环境信息返回need_experiment，不失败退出；若需要用户决定返回needs_input。不能把所有错误都当网络重试，不能越界远端修源码。

## 最小测试与正式计划

- 两条真实5步独立验收：Flow-GRPO（全步CPS、明确joint元素权重）和Flash-GRPO（真实single-step policy与对应replay）。配置必须明确，源码/产物中policy_steps可追踪；Flash不能沿用all_including_terminal_q0冒充。
- 实验算法名称与真正实现一致；若Flash论文所需rectification/策略定义不能直接映射H3，先报告差异、实现并记录所选语义，不冒称严格论文复现。
- 小规格测试可以用更小画布/LoRA/候选数，但必须在启动前固定并标为smoke，不宣称正式画质。真实媒体、reward、更新和checkpoint不能省略。
- 正式默认方向：256×384×124/24fps、4prompt×8=32轨迹、micro1、LoRA32/64、10采样步、每collection1更新、Flash作者19700/300、视频HPS全部124帧top37。来自既定计划；目标更新次数尚需用户确认。World-R1另组，PickScore只用于前期验证。

## 操作入口与状态

从引擎worktree根执行：

```text
python3 workflow-workspace/2026-10-28-h3-experiment-cycle-001/operations.py
```

它读取同目录request.json。action包括：bootstrap、inspect_host、exec（需read_only=true/false、command，≤120秒）、publish、deploy（sha）、launch（spec）、inspect_job（sha/job_id）、stop_job（仅本任务原PID/启动时刻/进程组）、reconcile（operation_id/probe_id/evidence）。

publish返回真实sha，deploy创建该sha独立工作树；launch spec包含job_id/sha/kind(train,prepare,probe,service)/algorithm/timeout_seconds/argv/env/config_text；kind=train还必须声明stage=flow-smoke/flash-smoke/formal。两个smoke的max_optimizer_steps必须为5；单步诊断使用probe且不能作为5步验收。formal目标未确认时操作入口也会拒绝启动。argv不经过shell；可用{source}/{config}/{output}/{job}占位路径。train必须以visual_rl.train模块执行，8个明确UUID。环境安装、模型转换和探查也应使用有界后台prepare/probe任务，不将它们当训练通过。

命令原文/输出保存在主项目 `.pi/workflow-tmp/h3-experiment-cycle-001/operations/`。工具功能限制不是安全沙箱，Experimenter仍必须逐条审查路径、权限和副作用，禁止显示密钥/token。

W/state.json示例：`{"stage":"flow-smoke","jobs":[{"job_id":"flow-smoke-001","sha":"完整SHA","stage":"flow"}],"last_observation":"..."}`。
W/evidence.json最终格式：`{"flow":{"job_id":"...","sha":"..."},"flash":{"job_id":"...","sha":"..."},"formal":{"job_id":"...","sha":"..."}}`。
正式目标未确认、权限/安全阻塞时写needs-input.md并返回needs_input；流程请求pause，不进入failed出口，也不自行填写formal-plan。

## 监视与收口

独立watchdog每60秒检查官方status、角色会话活动和已声明owned jobs。纯模型网络失败且当前工具全部有回执、无未知远端操作时，可最多两次有据续接原会话；其他未知执行、权限、Git冲突、预算耗尽和用户输入等待均暂停留证，不自动杀训练或重放命令。

每个远程job自身有声明的硬超时；过期只终止其拥有的进程组并记录失败。没有近期日志只告警并收集状态，不等于卡死。主机休眠/断网会影响监视，不能承诺永不挂起。

全部真实证据满足后，引擎在最终candidate执行`operations.py --validate-evidence`（只读远端退出/checkpoint与代码差异核对，不跑CPU回归），再安全合并本地main。实验分支已用于部署；main的最终push由主会话核对后处理。保留完整实验日志/配置/checkpoint和旧失败记录。
