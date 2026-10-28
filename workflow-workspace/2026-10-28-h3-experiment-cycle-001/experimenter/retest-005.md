# 第五轮：H3-EXP-001-r2真实复验行动记录

当前本地SHA `fa99cf22c11670a67c06cd16508dd8f4ab5664b3`，初始工作区干净。已读完整准备脚本、direct-IO loader、Coder r2交付、task/formal-plan/reference。未跑CPU回归。

## 已执行

- exp005-host-01：只读inspect_host，code255，stdout空，port22连接超时；started1793157050.036181 ended1793157065.049183。
- exp005-publish-fa99cf2-01：code0，实验分支从d5c10c9 fast-forward至fa99cf2；未force或写main。仅本地GitHub发布，不是服务器部署/训练成功。
- exp005-host-retry-01：第一次只读重试，code255，同样连接超时；started1793157106.902774 ended1793157121.9890869。

- exp005-host-retry-02：第二次只读重试code255；started1793157176.0103612 ended1793157191.022806。
- exp005-host-retry-03：第三次只读重试code255；started1793157230.020742 ended1793157245.0354052。
- exp005-host-retry-04：第四次只读重试code0；started1793157274.008482 ended1793157286.284339。10卡0MiB/0%，host112GiB available，disk499GiB available。
- exp005-deploy-fa99cf2-01：code0，Git fetch/FETCH_HEAD与新detach worktree均fa99cf22c11670a67c06cd16508dd8f4ab5664b3。
- exp005-admission-01：code0，实际import位于新SHA source树，git status干净；旧三个job都有exit记录，新job目录不存在；GPU compute-apps为空。
- exp005-launch-convert-01：code0，后台owned job native-artifact-convert-001已启动。不是转换/训练成功证据。
- exp005-inspect-convert-01：code0，running=true、exit=null；heartbeat1793157419.6926997，child952149，elapsed30.0066秒。日志已真实hash audio payload与前两个transformer shards，最大已报告peak_rss_bytes=828567552。最后日志mtime1793157436.2555296。

按用户最多5次只读重试指令，本轮初次加3次重试失败，第4次重试成功；没有重放任何mutating操作，不修改操作/监视程序。全部工具回执uncertain=false。

## 当前owned job / waiting

job native-artifact-convert-001，kind prepare，SHA fa99cf22c11670a67c06cd16508dd8f4ab5664b3；PID952148/start_ticks74454370/created1793157389.6425683，child952149。timeout3600，CUDA_VISIBLE_DEVICES为空、显式offline、host limit48GiB，输出仅该job的artifact（进行中artifact.partial）。

根目录 `/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/native-artifact-convert-001`；完整spec/owner/child/heartbeat及job.log保留，当前exit.json尚不存在。运行已提交scripts/prepare_h3_artifact.py，不编辑原权重或旧不完整目录。

第二次观察exp005-inspect-convert-02（started1793157590.2106628，ended1793157593.239298，code0/uncertain=false）确认同owner仍running=true/exit=null，heartbeat1793157569.723841/elapsed180.0378s，日志mtime1793157584.2423587，输入哈希已推进到transformer第10/13 shard，peak RSS仍828567552 bytes。没有卡死判断或重启动作。

本步骤返回waiting保持job运行。下次先inspect同job，不重复launch；等真实exit0后检查conversion.json覆盖/hash/文件无symlink与实际peak RSS，才决定资产准备通过。当前只是输入hash阶段，不能称权重已转换或加载。若成功再独立真实payload加载probe（80GiB上限且保留实时24GiB余量），再推进GPU/FSDP与真实cache/reward。Flow/Flash仍0；正式缺步数不阻止前置工作。
