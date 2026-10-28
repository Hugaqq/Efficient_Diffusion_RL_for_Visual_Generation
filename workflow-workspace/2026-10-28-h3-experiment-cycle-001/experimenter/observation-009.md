# 第九轮：旧probe真实exit1；修正探查日志后新probe运行

## 范围与身份

当前本地协调HEAD `3121a478fff8045537c4f4169eee75b5fe25dae3`；生产部署SHA仍为 `fa99cf22c11670a67c06cd16508dd8f4ab5664b3`。重读task/formal-plan、reference、state/current、Coder r2、实际loader和操作/owned job实现。未修改生产源码、操作脚本或监视器；未追加CPU回归。正式目标保持null，此阶段不请求确认。

## 旧任务反馈（保留失败）

`exp009-inspect-payload-01`：code0/uncertain=false，原`artifact-payload-verify-001` running=false、exit1，started1793160025.4458957、ended1793160695.551259。日志实际显示：

- 40个产物文件hash全部核验。
- `loaded`：1238 state keys，estimated_weight_bytes66887998080，加载阶段已报告peak RSS67183656960 bytes。
- 27 shards逐tensor比较全部完成，base_tensors_compared638。
- 末尾明确错误：`TypeError: dict() got multiple values for keyword argument 'peak_rss_bytes'`，位于探查argv第35行调用report及第8行dict构造。

这证实上轮记录的Experimenter命令错误，不是生产loader失败。不能因receipt存在把exit1改称成功。

`exp009-payload-receipt-host-01`：code0/uncertain=false；读取旧receipt原文，SHA256 `76a2a03ba5ddc056228d64bbb583023adc4debea105a1f2fc768a2124f5f4e4c`。其写入点在全部检查之后、错误日志调用之前：40 files、638 base、600 LoRA、1238 state keys、66363578880 state bytes、69770481664 peak RSS bytes、661.4172599315643秒、training_updates0。旧child `/proc/976841` 不存在。host MemAvailable120944046080 bytes，磁盘Available453890859008 bytes。没有stop或重放未知操作。

## 新有界复验

确认旧任务退出与资源释放后，`exp009-launch-payload-02`启动全新 `artifact-payload-verify-002`，code0/uncertain=false。保持完整hash/loader/逐tensor比较及LoRA/buffer断言；仅修正末尾日志为 `report('payload_verification_complete', result=result)`，新receipt写合法JSON。生产代码、预算与原始/转换权重不变。

- owner PID985422/start_ticks74828602，created1793161131.9621313；child985423。
- timeout3600秒；空CUDA可见、offline、OMP_NUM_THREADS2。
- 实际入口：独立venv python -u -B -c（完整argv保存在操作回执及remote spec.json）。
- 路径：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/artifact-payload-verify-002/`。
- `exp009-inspect-payload-02` code0/uncertain=false：running=true、exit=null；实际import位于sources/fa99cf2/visual_rl/__init__.py，日志hash推进至transformer2/27；heartbeat1793161132.5355933，log_mtime1793161153.8286917，当前报告peak RSS533082112 bytes（早期hash阶段，不是最终峰值）。

本轮四次远端操作均首次成功，无新增uncertain operation。state.json保留旧失败并追加新job供watchdog观察。

## 后续

返回waiting；下次先inspect原probe002，不重复launch。exit0后核对receipt及最终日志，再推进实时授权八卡FSDP2物化/真实cache/AV/reward与Flow5，随后真实single-step Flash5。H3-EXP-001整体仍未关闭；Flow/Flash均0，资产读取不算训练验证。已知旧日志缺陷无需交Coder修算法；若新复验出现真正生产问题，保留完整栈再交接。
