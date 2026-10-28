# 当前状态 — 第七轮 / native-artifact-conversion / waiting

当前本地协调snapshot为`54e0516185845241b3bff3c20fd3821ef5545a49`；正在运行job与此前发布/部署/import核验的源码仍为`fa99cf22c11670a67c06cd16508dd8f4ab5664b3`，未热更新。未跑CPU回归。

## 已执行

已读Coder H3-EXP-001-r2及完整准备脚本/direct-IO loader。初次SSH与前3次只读重试连接超时，第4次重试成功；遵守用户最多5次重试要求，不改规则程序。实测10卡空闲、host112GiB available/disk499GiB available、无compute processes。随后Git发布部署核验，启动独立转换job。

## 当前owned job

- job_id：native-artifact-convert-001，kind=prepare，SHA同上。
- owner PID952148，start_ticks74454370，created1793157389.6425683；child952149。
- timeout3600秒，CUDA_VISIBLE_DEVICES为空，显式offline，host working limit48GiB。
- 日志：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/native-artifact-convert-001/job.log`。
- artifact目标同job目录`artifact`，进行中独立`artifact.partial`。原始权重和旧不完整目录不写入。
- 最近inspect exp007-inspect-convert-01：首次连接code0，running=true、exit=null；heartbeat1793159443.9584627/elapsed2054.2724s；日志mtime1793159459.7605774。
- transformer638键/27 shards、video703键/5 shards、audio1087键/单文件已转换；全部33个validated_shard事件已出现（脚本完整key/shape/dtype后逐tensor finite检查）。
- 源转换后完整hash已再次完成，正在输出inventory hash。已报告进程peak RSS11,994,386,432 bytes（约11.17GiB），不是最终退出峰值。仍缺最终conversion.json/发布和exit0，不能宣布完整准备通过或用partial加载。

返回waiting，保持job运行。下次必须先inspect同job，禁止重复launch；网络失败按最多5次只读重试，不停止任务。

## 待验收

H3-EXP-001仍open/retest进行中。等exit后核对conversion.json、严格覆盖/源输出hash/实际峰值/文件路径，然后才进入真实完整payload加载probe与GPU FSDP复验。准备完成不是训练更新；Flow0/Flash0，H3-EXP-002真实single-step queued。两个smoke通过后再确认正式步数，不改4×8=32。

本轮只读inspect首次成功；未停止或重启job。最新证据：experimenter/observation-007.md；上轮网络重试证据：experimenter/observation-006.md；启动/部署证据：experimenter/retest-005.md。旧3个job退出及所有网络/模型失败报告保留；无未知mutating操作，无stop/kill或服务器hotfix。
