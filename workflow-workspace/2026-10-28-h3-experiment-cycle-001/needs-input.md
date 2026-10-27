# 请求步骤边界暂停：固定SSH连接连续失败

请用户/主会话确认本机到`ubtroot@192.168.102.193`（train170hx）的网络及SSH已恢复稳定，再继续本workflow。仍使用已授权固定主机/密钥；无需提供密钥内容，不请求扩大主机权限。

真实回执：
- exp002-discovery-001：code255，连接port22超时，无stdout。
- exp002-host-connectivity-002：code255，仅返回train170hx与GPU0空闲一行，随后`Timeout, server 192.168.102.193 not responding.`。

两次均只读；没有重放、安装、部署、转换、启动/停止训练或未知mutating操作。上次owned probe最后确认exit1，此次未取得新的完整状态。停止继续远端尝试是网络安全边界，不是取消任务或将workflow判failed。

详细证据：experimenter/environment-002-connectivity.md。恢复后先只读核查资源/owned jobs，再补Coder需要的converter完整原文、原生配置、源码hash和既有转换产物信息。H3-EXP-001保持open；Flow/Flash均0更新。

本次不询问正式更新次数；仍按计划先完成两个真实5步测试。
