# 当前状态 — 第六轮 / native-artifact-conversion / waiting

当前本地协调snapshot为`7b808ab349ca7c3525c1ab133f98155824dced4c`；正在运行job与此前发布/部署/import核验的源码仍为`fa99cf22c11670a67c06cd16508dd8f4ab5664b3`，未热更新。未跑CPU回归。

## 已执行

已读Coder H3-EXP-001-r2及完整准备脚本/direct-IO loader。初次SSH与前3次只读重试连接超时，第4次重试成功；遵守用户最多5次重试要求，不改规则程序。实测10卡空闲、host112GiB available/disk499GiB available、无compute processes。随后Git发布部署核验，启动独立转换job。

## 当前owned job

- job_id：native-artifact-convert-001，kind=prepare，SHA同上。
- owner PID952148，start_ticks74454370，created1793157389.6425683；child952149。
- timeout3600秒，CUDA_VISIBLE_DEVICES为空，显式offline，host working limit48GiB。
- 日志：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/native-artifact-convert-001/job.log`。
- artifact目标同job目录`artifact`，进行中独立`artifact.partial`。原始权重和旧不完整目录不写入。
- 最近inspect exp006-inspect-convert-retry-05：running=true、exit=null；heartbeat1793158517.0586038/elapsed1127.3725s；日志mtime1793158505.511192。
- 全部输入payload已记录hash；conversion_start记录working33,715,998,528/available120,528,416,768 bytes。随后真实transformer转换输出535原键→638键、27 shards、61.73GiB。
- peak RSS828,567,552 bytes仅是conversion_start前最近报告值，不能作为转换实际/最终峰值。双VAE、完整输出校验/发布与退出均未验收。

返回waiting，保持job运行。下次必须先inspect同job，禁止重复launch；网络失败按最多5次只读重试，不停止任务。

## 待验收

H3-EXP-001仍open/retest进行中。等exit后核对conversion.json、严格覆盖/源输出hash/实际峰值/文件路径，然后才进入真实完整payload加载probe与GPU FSDP复验。准备完成不是训练更新；Flow0/Flash0，H3-EXP-002真实single-step queued。两个smoke通过后再确认正式步数，不改4×8=32。

本轮初次及前4次只读重试连接超时，第5次成功；未因网络失败停止或重启job。最新证据：experimenter/observation-006.md；启动/部署证据：experimenter/retest-005.md。旧3个job退出及所有网络/模型失败报告保留；无未知mutating操作，无stop/kill或服务器hotfix。
