# Coder round 004 — F0 固定单cell观测入口

## 决策与边界

按 planner/PLAN.md，只实现 F0 必要测量前置，不改mesh/FSDP、reward/VAE、科研profile、精度/RNG/更新数学、memory journal或checkpoint。起点 `684b7b2a03adfe30d8bb6a4c90e554dcb27a7a22`；初始status仅原未跟踪.pi/。重读TASK、当前PLAN及恢复标记：RECOVERY-003已resolved，不重新应用P1恢复补丁；当前上游真实P1为7/7并已提交，用户旧5/7提示只适用于历史恢复点。

## 实现

- 新 `visual_rl/runtime/h3_rollout_profile.py`：固定版本/窗口 `optimizer_step=1,row=1,index=0`，每rank仅一次尝试；非目标scope透明。只接收小metadata与gate，不接收/保存latent、prediction、模型或optimizer。目标重入/重复失败，不能沿用旧success状态或重放body。
- public torch.profiler CPU+CUDA，record_shapes/profile_memory/with_stack/with_flops全部False；不增加warmup、forward、backward、同步训练更新、autocast/cast或empty_cache。显式检查public API/活动支持，停止后观察实际CPU/CUDA DeviceType事件均非零才可出receipt，无CPU-only fallback。
- 主训练线程初始化/启动/停止/finalize及事件检查；start/stop失败经原同位置gate交换，自己的错误保留为主异常。body/model/collective失败只本地stop并附cleanup note，不加救援collective、不发布成功或重试body。成功停止后CPU Chrome序列化/stream hash/文件发布通过已有file_phase工作线程，主线程仅原heartbeat/PG等待；不在local-only _phase action中隐藏PG工作。
- `h3_dual_rollout.py`：可选None cell_scope只围住原forward前gate、预测、transition/result gate与原prediction释放；原每行初始H2D在scope外，receipt显式注明不覆盖。所有运算/循环/原异常门不变。
- `h3_dual_stages.py`：可选callable能力仅传递到rollout collect，与anchor/reward/train分开。默认None。
- 两步launcher新增strict-bool默认off的 `--profile-rollout`。bind在首个model调用前直接_exchange规范化flag/固定窗口/设置/预算版本；差异直接失败。随后local-only attach _phase创建能力/安装hook。off不实例化profiler、不探测CUPTI、不创建profile目录。正常最终checkpoint阶段经原_phase检查恰好一次成功window/receipt，未命中不能silent success；原P1选项及fresh两步/graph/weight/monitor/媒体审计保持。

## 产物与安全合同

仅本次输出固定 `rollout-profile/rank-N/update-2-row-1-index-0.{trace,receipt}.json`，拒绝已有rank目录/目标和symlink。检查rank目录所有权；临时文件记录device/inode，只删除确认owned文件，保留未知内容。

trace发布上限128MiB、receipt64KiB；trace用1MiB块流式hash并检查size，未read_bytes。fsync文件/目录，使用no-replace原子hard-link先发布完整trace再receipt，任何可检测导出/receipt/cleanup失败均报错并尝试回滚确认owned发布物；回滚IO失败附note，不声称永久文件系统故障下必能删除成功，不覆盖历史内容。最终local检查小receipt摘要/存在性和trace大小；receipt保存trace SHA供后续独立重验，不宣称任意并发外部写盘下的文件系统锁。

receipt包含schema/version、launcher clean/expected guard沿用的SHA、runtime已admitted rank/UUID、固定/实际窗口与sample/selected index、活动设置/实际CPU与CUDA事件数、instrumented wall、trace bytes/SHA、覆盖与限制。无prompt全文、媒体、activation、weights或tensor。CPU与CUDA时长不能相加；NCCL含原gate，不全归FSDP。窗口wall含profiling/gate开销，不是完整rollout/端到端墙钟。

128MiB是**发布上限**，不是Kineto采集RAM/临时磁盘硬cap。PyTorch必须先导出才能检查；单cell/关闭shape/stack等限制范围但不能保证峰值，获准实验须另预留空间。超限失败关闭，不截断trace、不继续额外更新。

## 实际CPU检查

保持七个原函数/所有原断言，不新增/skip/替换/参数化测试函数。必要F0回归置于原manifest函数；现有两步CPU wrapper仅补profile_rollout=False及真实配置交换所需8份CPU echo，其他门禁不放松。

- 真实两轮tiny dual collect：off factory/support为forbidden仍完成；on仅第二轮第二样本index0启动一次。记录/actions/old_joint/terminals/choices、每样本generator与全局Python/NumPy/Torch CPU RNG一致，40 forward及原gate次序/数目一致（额外profile门单独识别）。
- 实际H3DualIterationStages.rollout调用真实collect并传hook，20次CPU forward、仅一个window；主线程启停一次、停止后才在显式CPU文件线程导出。合成CUDA DeviceType仅为测试叶子，trace带test_only_synthetic_activity_leaf，**不是GPU证据**；生产没有fake/fallback选项。
- start/stop/peer-start/peer-stop失败门、CPU-only事件拒绝、body原model异常+stop错误note且无救援collective、重复/未命中/非主线程拒绝。实际collect中start失败只到目标前10次predict，stop失败到目标11次，之后predict/transition未执行，不重放。
- 实际发布/stream SHA/metadata/设置/覆盖检查；覆盖已有目标、目标/root symlink、trace超限、receipt超限/IO失败、export IO失败与receipt篡改。未知文件完整保留，失败无虚假receipt/trace或正常可清理临时残留。Path.read_bytes设为forbidden时成功发布/完成检查仍通过。
- 实际TwoUpdateBackend.bind：off禁止support探针且不落盘，on正确安装bound recorder.cell及保留PredictAudit/P1选项；peer flag差异在attach前拒绝；enabled未命中阻止parent terminal finalize，off不要求receipt。真实CLI help包含新选项并在Git/GPU入口前退出。原两步wrapper两轮审计、G0/I1/P1回归均通过。

原短入口已确认是固定七项环境/命令，本轮实际运行依次：7 passed, 3 warnings in50.13s；7 passed, 3 warnings in47.71s；最终 **7 passed, 3 warnings in51.52s**。三条仍为原DCP单进程保存/加载及TypedStorage警告。当前worktree导入断言、真实Git/config证明完整保留并通过。`git diff --check`空输出通过。代码/测试范围仅本轮计划五个文件（含新recorder）；本角色报告/commit材料另计。原未跟踪.pi/不读内部/不清理/不stage，不宣称整树clean。

## 成本依据与未验证项

重读保存MEMORY-REPORT与sharding-estimates：第二轮采样42.600min/40个rank-local root前向约63.9s只是含gate/转移/日志的粗均；对127.73min保存轮约33.4%（分母**含优化/校验/保存，不含初始化等**），对前两段93.918min约45.4%（**不含优化/保存**）。每遍allgather接收56.49GB是字节估算，不是Gen2x4实测有效带宽；标准DDP最低常驻64.21GiB已超64GiB，不盲改mesh。没有新的通信/compute/搬运实测分解。

F0本身无直接speedup估算，开启会增加采集/同步/导出开销。CPU仅验证协议/计数/失败/产物，不证明CUDA/CUPTI兼容、真实export时峰内存、NCCL占比、硬件最优或GPU收益。后续测量只能按新授权实验卡在同协议两更新中采固定窗口，不增加warmup或重跑旧任务；须核最终SHA/独立HPS实际绑定/8训练卡/新输出和两更新保存恢复，分别报告window instrumented wall与完整含初始化/优化/校验/保存墙钟。

本轮没有GPU执行、torchrun/profile命令、SSH、服务请求/修改、部署或真实训练；Euler B继续暂停。交Simplifier独立复审后由Planner选择下一项/测量卡；VAE/P2/FSDP/HPS候选仍开放，最终integration/perf发布未做，不宣称工作流完成。原子提交后SHA见coder/round-004-commit.md。
