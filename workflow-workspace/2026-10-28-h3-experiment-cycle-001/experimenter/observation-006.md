# 第六轮：观察原转换job（不重启）

本地HEAD7b808ab349ca7c3525c1ab133f98155824dced4c；远端job固定源码fa99cf22c11670a67c06cd16508dd8f4ab5664b3。先读取state/task/formal-plan、当前准备输出验证代码和Coder后续payload复验说明。

初次只读inspect exp006-inspect-convert-01：started1793158065.760042 ended1793158080.7717671，code255、stdout空，port22连接超时。
第一次只读重试 exp006-inspect-convert-retry-01：started1793158126.373813 ended1793158141.3886158，code255、stdout空，同样连接超时。

第二次只读重试 exp006-inspect-convert-retry-02：started1793158213.0034618 ended1793158228.02591，code255、stdout空，同样连接超时。
第三次只读重试 exp006-inspect-convert-retry-03：started1793158297.386626 ended1793158312.4006，code255、stdout空，同样连接超时。

第四次只读重试 exp006-inspect-convert-retry-04：started1793158365.8424928 ended1793158380.855755，code255、stdout空，同样连接超时。各次观察起点间隔约61/87/84/68秒，没有高频轮询。

第五次只读重试 exp006-inspect-convert-retry-05：started1793158542.3231711 ended1793158545.063141，code0/uncertain=false，连接恢复。

## 最新真实状态

- 同owner PID952148/start_ticks74454370/created1793157389.6425683，running=true、exit=null；child952149。
- heartbeat1793158517.0586038，elapsed1127.372531414032秒；log_mtime1793158505.511192。
- 已记录所有13个transformer原shard、audio payload、video payload和小配置/index的真实输入SHA。
- 日志conversion_start：working_bytes33715998528、available_bytes120528416768；peak_rss_bytes828567552只是在转换开始前最近报告的进程峰值，不代表转换期间或最终峰值。
- 随后真实官方converter输出：`transformer: 535 original keys -> 638 diffusers keys in 27 shard(s), 61.73 GiB.`
- 尚无artifact_published/最终receipt验收、exit0或训练成功证据。双VAE和完整输出严格校验状态仍待后续观察。

本轮初次加前4次重试失败，第5次重试成功。以上均只读且uncertain=false，没有新mutating操作，不修改任何操作/监视规则，不将观察失败当job失败。3600秒硬超时归原worker所有；本轮没有启动替代任务、stop/kill或CPU回归。

返回waiting保持原job。下次先inspect同job；若exit0再检查conversion.json完整覆盖/源输出hash/文件路径与实际峰值，并继续真实payload加载；明确源码失败则留证交Coder。H3-EXP-001保持open，Flow/Flash仍0，正式目标此阶段不询问。
