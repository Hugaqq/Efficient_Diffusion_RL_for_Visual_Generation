# Planner round-005 — F0独立验收与正常外部等待

## 实际commit/diff与代码审核

当前HEAD `0eb2a5b6703fc726a063918eaaacdf63672e876f`；实际log确认功能commit `9b0a6b3`（父`684b7b2`），后续`1bbb064/0eb2a5b`为工具快照。HEAD stat只收录40行Simplifier报告，无Simplifier代码commit。

已读取Coder round-004/commit收据、Simplifier round-004，执行生产基线完整授权源码diff。输出超过工具50KB上限，未访问提示中的workspace外/tmp日志；用read补读当前**完整**h3_rollout_profile.py、两步launcher、collect/dual接线及原manifest函数2120起至末尾全部F0回归。原G0/I1/P1不是本轮新改动。提交前时序与7文件stat来自Coder已存收据；Planner未使用非白名单Git命令。

独立检查：
- off不构造profiler/探support/创建profile目录，hook为None；on规范化rank配置先一致，固定step1/row1/index0一次尝试。
- public CPU+CUDA支持/实际DeviceType事件双非零；关闭shape/memory/stack/flops额外采集；CPU合成事件仅在测试叶子，无生产CPU fallback。
- 原cell运算/选择/生成器/forward/result gates/预测释放保持，scope不覆盖row初始H2D。主线程start/stop/事件finalize；正常启停失败门、原model异常主错误与cleanup note/无救援collective/无重放符合计划。
- 停止后才在file_phase的CPU worker序列化/stream-hash/发布，主线程心跳。临时文件所有权、NOREPLACE hard-link、fsync、未知文件保留与失败无伪receipt逻辑保留；128MiB/64KiB是发布界，不是Kineto RAM/临时磁盘硬界。
- 小receipt完整性和trace大小末尾检查没有冒充再次全trace hash；trace SHA已记录，后续独立验收必须流式重验。原两步终态发布前检查命中，原P1/PredictAudit/monitor/图/权重/Adam/媒体/保存条件不变。
- 原七项内真实tiny collect两轮结果/40前向/原gate顺序/record/action/terminal/choice/generator及全局CPU RNG一致，start/stop失败分别停于10/11前向；真实dual/backend接线和失败/产物/未命中/未知文件回归有效。

结论：F0符合范围，无发现的须优先修复正确性缺口；接受Simplifier无必要简化、不造commit的决定。**真实CUPTI兼容/导出资源/NCCL占比仍未经GPU验证**，不能从CPU合成活动得出通过。

## Planner实际门禁与当前状态

再次完整读取 `inputs/run-cpu-gate.sh`，其内容仍是原固定环境和原七项pytest。实际执行：

```text
7 passed, 3 warnings in 44.91s
```

三警告仍为DCP单进程save/load及TypedStorage；原当前worktree导入/Git/config证明与G0/I1/P1保留通过。Coder提交前51.52s、Simplifier独立53.75s的7/7已读；CPU时间变化不是训练speedup。

初始status仅原未跟踪.pi/，保留不读取内部/清理/stage。Planner只写自己的计划/候选表/报告/最小测量卡，未改训练源码或授权文件。写完后实际git diff --check空输出通过；最终status仅这些Planner文件及原.pi/，不宣称整树clean。

## 可见证据缺口检查

重读TASK与MEMORY报告，实际ls inputs仅七份保存摘要/估算/规则/CPU入口，没有逐样本D/H、VAE load组成或CUDA operator trace。另以获准find在可见workflow-workspace搜索 `**/phase-times.csv`、`**/*attempt003*`、`**/*memory*.jsonl`、`**/train.log` 均未找到；不由此声称工具排除的私有档案不存在，也未越权跨workspace取档。

重新看aux_turns owner生命周期与现有HPS onload/offload/score日志能力：旧原文若提供可先离线分解，不必为了填表新增GPU。当前摘要的42.6min/51.318min/13min不足以确定VAE/通信/HPS下一项收益：
- VAE load GPU计时含CPU构造/读取及H2D，用户8s未核边界；新增pair驻留/offload与RNG/失效风险需要据实际权重/余量和临界路径设计。
- P1默认off且未GPU验证，真实D/H/CPU竞争和填充排空未知；不能把串行VAE省192s加到P1估算，也不在未测P1上直接堆跨owner异步。
- F0已就绪，但56.49GB接收仅字节，不等于Gen2实测通信时间；DDP静态超64GiB，D2要1D norm/归约/checkpoint兼容与实际峰证据，不能先换mesh。
- HPS CPU模型已长驻、逐请求搬运确实存在，但占50s的多少未知；T1重复检查也无占比，不为凑循环做微优化或删门。

## 决策/恢复条件

本轮 `waiting_external`，**不是exhausted、不是最终交付/预算耗尽或已达2倍声明**；没有把已确认失败或不明副作用包装成等待。已实施项的当前CPU/代码审查均通过，开放候选尚需实测决策输入。

已写MEASUREMENT-CARD.md：
1. 优先M0补已保存第二轮32样本事件及来源SHA/可用权重header/host/拓扑事实，新增GPU工作=0；用户提供到获准inputs并明确恢复即可复审选择代码项。
2. 若M0不足，仅由用户另批M1全新目录、明确执行者/最终SHA/设备服务归属/资源时窗的一次fresh两更新探测；P1/F0 opt-in但不新增warmup/第三更新/额外HPS评分，不自动执行。该有profile扰动的测量不能冒充最终严格配对速度/逐帧数值验收。

当前没有最终integration或perf发布成功证据；后续候选交付需获准integration核真实门禁，Planner不能越权push/部署。最终GPU逐帧/top37配对（不得有损MP4替代原输入）、最终两步速度/保存恢复仍需具体新授权。工作流等待后只有用户明确恢复才继续，不自动重放工具/恢复Euler-B。

本轮及整个当前角色未执行GPU、torchrun/profile、训练/推理、SSH、实时HPS/HTTP、服务变更或部署。原3db2a05 source/run与Euler-B暂停保持。
