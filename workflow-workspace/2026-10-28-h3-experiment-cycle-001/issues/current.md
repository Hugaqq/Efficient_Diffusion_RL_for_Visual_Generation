# 当前状态：H3-EXP-001 retest运行中 — waiting，不派新修复

Coder H3-EXP-001-r2已部署至`fa99cf22c11670a67c06cd16508dd8f4ab5664b3`，GitHub/远端Git/import身份一致。Experimenter启动真实独立转换job `native-artifact-convert-001`，PID952148/start_ticks74454370，timeout3600秒。

最近exp006-inspect-convert-retry-05：同owner running=true、exit=null，elapsed1127.3725s；全部输入hash完成，transformer已报告535原键→638键、27 shards/61.73GiB。双VAE及完整产物校验/发布尚未验收。RSS828567552 bytes是conversion_start前最近报告值，不是当前或最终峰值。**尚不能声称转换成功或关闭issue。** 下次先inspect同job，不重复launch、不停止任务。完整spec/日志路径/回执见experimenter/retest-005.md与state.json。

本轮网络首次及前4次重试失败，第5次只读重试恢复；未改操作规则，全部失败回执保留，无unknown mutation。最新证据见experimenter/observation-006.md。后续转换若明确源码错误再附新issue反馈Coder；如果成功则继续独立完整payload加载及真实八卡实验，不以准备/CPU检查算5更新。

H3-EXP-002真实Flash single-step仍queued；Flow/Flash0更新，正式缺步数此阶段不暂停。原H3-EXP-001、materials-r1与全部失败记录保留。
