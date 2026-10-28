# 当前状态 — 第五轮 / native-artifact-conversion / waiting

本地、实验分支与远端新worktree SHA均`fa99cf22c11670a67c06cd16508dd8f4ab5664b3`；实际import核验在该source树、tree干净。未跑CPU回归。

## 已执行

已读Coder H3-EXP-001-r2及完整准备脚本/direct-IO loader。初次SSH与前3次只读重试连接超时，第4次重试成功；遵守用户最多5次重试要求，不改规则程序。实测10卡空闲、host112GiB available/disk499GiB available、无compute processes。随后Git发布部署核验，启动独立转换job。

## 当前owned job

- job_id：native-artifact-convert-001，kind=prepare，SHA同上。
- owner PID952148，start_ticks74454370，created1793157389.6425683；child952149。
- timeout3600秒，CUDA_VISIBLE_DEVICES为空，显式offline，host working limit48GiB。
- 日志：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/native-artifact-convert-001/job.log`。
- artifact目标同job目录`artifact`，进行中独立`artifact.partial`。原始权重和旧不完整目录不写入。
- 最近inspect exp005-inspect-convert-02：running=true、exit=null；heartbeat1793157569.723841/elapsed180.0378s；日志mtime1793157584.2423587。
- 实际日志已hash audio payload和前10个transformer shards；报告peak RSS828,567,552 bytes（仅此时点峰值，非最终转换峰值）。当前尚无转换成功证据。

返回waiting，保持job运行。下次必须先inspect同job，禁止重复launch；网络失败按最多5次只读重试，不停止任务。

## 待验收

H3-EXP-001仍open/retest进行中。等exit后核对conversion.json、严格覆盖/源输出hash/实际峰值/文件路径，然后才进入真实完整payload加载probe与GPU FSDP复验。准备完成不是训练更新；Flow0/Flash0，H3-EXP-002真实single-step queued。两个smoke通过后再确认正式步数，不改4×8=32。

完整本轮证据与请求索引：experimenter/retest-005.md。旧3个job退出及所有网络/模型失败报告保留；无未知mutating操作，无stop/kill或服务器hotfix。
