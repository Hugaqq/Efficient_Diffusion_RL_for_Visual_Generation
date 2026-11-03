# Simplifier round 004 — F0 独立复审

## 结论与提交

F0 真实实现符合本轮计划，未发现必须修复的正确性缺口或有明确维护收益的必要等价简化。**不修改源码/测试，不创建 Simplifier commit，不写无用途的 COMMIT_MESSAGE**。保留启停错误门、owned文件清理和业务异常优先逻辑，不为减少行数删除安全检查。交 Planner 复审，不宣告工作流完成。

实际 git log 确认功能提交 **9b0a6b3**（父684b7b2）；当前 HEAD **1bbb06490a80a9520b0720e4d211ff68a05fb18d** 为工作流快照。已执行生产基线授权源码累计diff；超过50KB的输出被工具截断，未访问其外部临时日志。以 read 补读当前完整 h3_rollout_profile.py、h3_dual_rollout.py、两步launcher、file_phase、dual stage相关diff及全部F0测试区间，不只依据Coder总结。累计G0/I1/P1不是本轮新增。

## 输入/硬件依据

重读 TASK、planner/PLAN、Coder round-004，以及保存的MEMORY-REPORT阶段表和sharding-estimates静态估算。保存采样42.600min尚无可靠compute/NCCL/传输/host分解，allgather字节不是有效带宽测量，标准DDP静态超64GiB；不按型号猜互联，不由reserved低估驻留风险。本轮是观测前置，不给提速倍数或耗时收益估算。

## 实际源码审查

- 默认off不构造/启动profiler、探测support或创建目录，collect默认cell_scope=None；off路径只多透明nullcontext，不增forward/随机调用。launcher strict-bool及首个模型调用前规范化rank配置交换保留，配置不一致先失败。
- 固定optimizer_step=1,row=1,index=0；每rank一次尝试，重复先清空成功状态并拒绝，不重放。metadata只含有界sample/selected index等标量，不接收/持有模型、activation、latent或prediction。hook仅传入rollout，不进入reward/anchor/train。
- 实际collect中scope包含原forward gate、预测、transition、result gate及prediction删除；每行初始H2D在外，receipt明确标记。运算、selected-step算法、generator顺序、dtype、精度和原异常门未改。
- support/API检查及实际CPU/CUDA事件非零检查均fail closed，无生产CPU fallback。主线程start/stop/事件finalize；正常body后的启停失败在同位置gate传播。模型/collective失败只做本地stop，原异常优先、cleanup附note，无救援collective或body重试。
- 停止后才通过原file_phase执行CPU导出/hash/发布，主线程走原heartbeat；不是在local-only _phase action内隐藏collective。profile对象局部生命周期结束后不跨cell缓存。线程不可强杀，file worker卡死沿原3600s整attempt失败边界，不当正常排空或允许续更。
- 新run固定rank目录拒绝已有目标/symlink；临时文件登记device/inode，stream hash、文件/目录fsync、hard-link NOREPLACE发布完整trace后receipt。失败仅清理身份匹配的owned文件，未知文件不删，cleanup错误传播/附note而不伪造成功。
- 128MiB trace/64KiB receipt是发布界，**不是Kineto RAM或导出临时磁盘硬界**。没有新增媒体队列或改变P1的Queue1/总许可2及owner顺序；没有模型常驻/mesh修改。最终检查恰好一次成功、目录/小receipt完整性及trace大小；receipt保存trace SHA供外部独立核验，不将最终大小检查声称为再次全量hash或文件系统锁。
- launcher原fresh两步、PredictAudit、monitor、权重/图/Adam、媒体及terminal finalize仍在；F0命中检查在parent终态发布前，无额外update/保存。样本身份/RNG/更新数学及checkpoint实现不变。

## 独立实际CPU验证

先读取 inputs/run-cpu-gate.sh 确认仍为原固定环境与七项命令，再实际执行短入口：

**7 passed, 3 warnings in 53.75s**。

三条警告仍为DCP单进程保存/加载和TypedStorage。`git diff --check` 实际空输出通过；当前worktree导入断言、真实Git/config证明和原G0/I1/P1回归包含于通过的门禁。

已实读原manifest函数内F0回归：两轮真实tiny collect仅一次factory/start/stop/export、40次predict与原gate顺序及所有records/terminal/choice/generator/CPU全局RNG一致；真实dual stage传hook；off禁止support/factory探针；start/stop/peer/CPU-only事件/body+cleanup/重复未命中/非主线程拒绝；已有目标/symlink/大小/导出IO/receipt IO失败、owned清理及禁止read_bytes仍成功；真实TwoUpdateBackend.bind安装与配置差异拒绝、未命中不finalize。仅CUDA/CUPTI及transport等不可用观察叶子为合成，CPUtrace明确标识test-only，不是GPU证据。

初始status仅原未跟踪.pi/；本轮仅新增本报告，无stage/commit。保留.pi/，不读取内部、不清理、不提交，不宣称整树clean；最终状态另用实际命令检查。

## 交Planner剩余事项

1. Planner独立审核9b0a6b3及本轮门禁，再选择下一代码单元或最小测量卡；VAE/P2/FSDP/HPS候选仍开放。
2. F0默认关闭，CPU通过不证明真实CUPTI兼容/导出峰内存/真实CUDA活动/NCCL占比或速度收益。开启增加观测开销；window instrumented wall不能冒充完整含初始化/优化/校验/保存墙钟，CPU/CUDA重叠时长不能相加，NCCL也包含gate。
3. 真实profile、P1数值与设备/资源验证、最终同协议两更新保存恢复需新批准实验卡；最终integration/perf发布未执行。无GPU/torchrun/profile执行/SSH/HPS请求/服务修改/训练/推理/部署，Euler B继续暂停。
