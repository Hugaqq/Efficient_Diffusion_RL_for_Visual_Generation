# Planner round-004 — P1独立验收与采样profile前置

## P1实际commit/diff审核

HEAD `f4ef71e4247f701338f7359f3c6968213921a1a3`。实际log确认功能commit `9e183bb`，父`b704adc`；后续`7f57c30/f4ef71e`为工作流快照。HEAD stat仅新增46行Simplifier报告，无Simplifier代码commit。

已读Coder round-003/commit收据（含历史5/7恢复失败、核对空diff后一次apply、resolved与最终门禁）、Simplifier round-003。没有重放恢复命令/清理未知改动。提交前时序/10文件stat来自已存收据，Planner未越权运行非白名单Git命令。

实际生产基线累计diff超过50KB工具截断；没有读取提示中的workspace外/tmp日志。改用read补读当前完整h3_reward_pipeline.py、h3_dual_stages.py、h3_memory.py、两步launcher、handoff/lease及当前测试wrapper；累计diff中P1全部新增回归段可见，G0/I1前两轮已审且未见额外变化。

审查结论：
- P1默认False且strict-bool、paired Flash准入；同位置gate比较模式/预算，原aux owner顺序和更新屏障保留。
- Queue1与decode前Semaphore2约束正在decode/排队/消费的总在途；原storage CPU FP32 handoff，原lease退出锁释放，owned单消费者close；不保留整轮媒体/future。
- 消费原score_one/observed saver，FIFO样本表、124/top37/manifest/JPEG/identity/finite门均未改；CUDA工作仍原owner stream，24GiB host reserve、40GiB GPU准入/16GiB margin保持。
- 取消与完成信号独立、业务错误优先、取消等候许可/队列、join后drain关闭、不返回partial rows/重试POST。无法强杀已在途HTTP/ffmpeg的边界已明示，不能将3600s整attempt失败当正常排空。
- CPU日志scope独立thread-local stack，counters=None且不调用CUDA/reset/sync；写入仍原锁/seq/budget/fsync；默认GPU lossless peak-reset路径不变。
- 测试实际驱动tiny decoder/handoff、Event重叠与第三decode阻塞、完整JPEG请求/结果/RNG一致、真实gate mismatch、真实trainer saver失败不更新/不下个owner、交错日志/失败释放和backend/CLI离线接线。CPU叶子不证明远端GPU等价。

未发现需要优先修复的P1正确性缺口，接受Simplifier无必要简化/不造commit的结论。P1代码阶段通过不等于可以默认部署或已实现18min节时。

## Planner实际CPU

已先读获准短入口`inputs/run-cpu-gate.sh`，确认仅set -euo pipefail及原固定PATH/数据/PYTHONPATH/解释器/七项pytest，不加训练/其它测试。实际执行结果：

```text
7 passed, 3 warnings in 51.64s
```

三警告仍为DCP单进程save/load与TypedStorage；原七函数、当前worktree导入与真实Git/config证明保留。Coder最终51.68s、Simplifier独立51.79s的7/7已读，CPU时间波动不作训练speedup依据。

起始status仅原未跟踪.pi/；不访问其内部、不清理、不stage。写完计划报告后实际git diff --check空输出通过；最终status仅Planner自己的PLAN/CANDIDATES修改、round-004新报告及原.pi/，无源码修改，不宣称整树clean。

## 下一项证据/成本

重读TASK、inputs/MEMORY-REPORT与sharding-estimates。重新读真实collect cell循环、h3_forward.py、PredictAudit以及FSDP shard policy、file_phase线程/PG分工：
- rollout42.600min对每rank40前向，粗均63.9s；anchor/current各4次也约64s/次，backward约66s/次。不能据此辨别计算/通信各占几成。
- 存证56.49GB接收/遍历不是PCIe实测延迟；标准DDP64.21GiB静态超卡，D2/reshard改动不能从allocated峰小直接推出安全。
- 42.6min占含优化/校验/保存、不含初始化的127.73min分母约33.4%；占不含优化/保存的前两段93.918min约45.4%。F0没有直接节时目标，只降低选择FSDP/搬运优化的不确定性。
- VAE复用仍开放，但P1开启后旧串行192s节省不宜简单叠加：在D45/H50条件假设下，owner内缓存若只加快第2..4 decode，首decode不变，Dfirst+4H临界路径可能不变。这个反例不是实测或永久排除VAE；须读未来真实加载/解码/评分/等待事件确定。

## 决策：F0单cell opt-in profile入口

当前仍有明确可实现的测量前置，不需要现在waiting_external或FINAL。详见PLAN.md：默认off，固定第二更新/第二本地sample/index0、一cell/rank、只CPU+CUDA public profiler；捕获真实forward/transition/gate窗口、不改变任何采样/更新次数。与所有rank协商配置、start/stop错误门、main线程CUDA与文件worker导出分离、单trace及小receipt/字节发布界、SHA和coverage、异常优先、未命中失败都需必要CPU回归。

该入口仅代码交付；不运行profile/GPU、不切DDP/mesh/精度、不加warmup或重跑旧两步。trace包括gate自身NCCL且可能compute/comm重叠，不能将所有NCCL或CPU enqueue总和当FSDP独占墙钟。

Coder一个功能+回归原子commit，先7/7/diff检查，Simplifier独立审查后再回Planner。P1和F0都默认不开启；后续真实设备隔离、profile及两更新保存恢复/完整墙钟必须另批新实验卡。V1/P2/F1/F2/H1等仍开放/待合理测量，不宣告2倍或候选耗尽。原Euler-B继续暂停；本轮没有GPU/SSH/HTTP/服务/训练/推理/部署。
