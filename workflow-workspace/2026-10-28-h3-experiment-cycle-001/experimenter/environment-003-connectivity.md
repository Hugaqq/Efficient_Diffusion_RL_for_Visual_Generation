# 第三轮：主会话确认恢复后，工作流固定入口再次连接超时

本地HEAD `1b70b7b102e3cff48fa87b614238a340dcd71311`。已采用用户Temp最新恢复记录：主会话2026-10-28 09:47+08两次只读SSH成功，原probe确认exit1/running=false、journal无unknown mutating。未把这份时点记录当永久连通保证，未重放exp002请求或重启原probe。

## 本轮独立实际观察

|操作|请求|started / ended（epoch）|真实结果|
|---|---|---|---|
|exp003-host-001|inspect_host，只读|1793152206.028769 / 1793152221.0378659|code255，stdout空，port22 Operation timed out|
|exp003-connectivity-minimal-002|exec read_only=true，command=hostname，timeout_seconds=30|1793152244.8936698 / 1793152259.910014|code255，stdout空，port22 Operation timed out|

两次stderr均为：
```text
ssh: connect to host 192.168.102.193 port 22: Operation timed out
```
均state=finished、uncertain=false（只读），无工具调用待返回。第二次主动缩到最小hostname查询以区分远端材料/显卡命令耗时：仍在SSH连接阶段超时，没有任何hostname或GPU输出。不能将本轮错误归于模型代码、nvidia-smi卡死或认证失败。

实际入口为获准operations.py固定SSH列表：同用户指定密钥和ubtroot@192.168.102.193，IdentitiesOnly/BatchMode/ConnectTimeout15/ServerAliveInterval15/CountMax2。仅阅读脚本核对，没有修改连接配置或使用其他主机、通道、密钥。

## 结论与边界

请求needs_input/再次步骤边界暂停，请主会话核查**workflow实际执行环境**的固定SSH连通性与路由；可对照同operations入口，而非仅凭另一上下文SSH成功恢复。具体根因仍未知；没有证据断言Tailscale、服务器、沙箱或网络配置中的某一个故障。

本轮没有mutating远端操作、新job、publish/deploy、下载、转换、CPU回归或源码修改。未停止任何进程。旧probe最后状态来自主会话09:47恢复核查，当前角色本轮未能刷新；无新增未知mutating操作。

Coder请求的converter全文/原生配置/已安装源hash/既有converted信息仍未取得。H3-EXP-001 open、H3-EXP-002 queued，Flow0/Flash0。此暂停仅因网络，不询问或修改正式更新目标。连接恢复后重新以新id只读inspect/owned job核查，再继续材料收集。
