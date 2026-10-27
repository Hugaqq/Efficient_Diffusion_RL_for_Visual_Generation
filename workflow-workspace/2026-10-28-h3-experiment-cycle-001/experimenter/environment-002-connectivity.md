# Experimenter第二轮：材料收集因SSH连续失败暂停

本地HEAD：8f5b9db5d045668acb348272c80e84050f3d2773；开始status干净。重新读取task、formal-plan、reference、state、current及Coder H3-EXP-001报告，核对当前loader预算与decoder入口。本轮无生产改动、publish/deploy、安装、转换、GPU job或CPU回归。

## 本轮实际回执（固定operations入口）

1. `exp002-discovery-001`，action=exec，read_only=true，timeout_seconds=90。
   目的：在qifan有限深度范围发现既有converted/固定converter，列code目录顶层，读取三种已安装Diffusers源文件SHA256与已知owned-run退出记录；不加载模型、不读tensor payload、不改文件。
   - started=1793131746.488352，ended=1793131761.504967。
   - code=255，stdout为空。
   - stderr：`ssh: connect to host 192.168.102.193 port 22: Operation timed out`。
   - state=finished，uncertain=false（只读操作）。没有取得所请求的环境材料。
2. `exp002-host-connectivity-002`，action=inspect_host，只读独立连通性复核，不重放第一次发现命令。
   - started=1793131780.30632，ended=1793131834.024982。
   - code=255。
   - stdout仅：
     ```text
     train170hx
     0, GPU-4a964c9b-4342-56a1-e7d7-84067d4a83ab, 65536 MiB, 0 MiB, 0 %
     ```
   - stderr：`Timeout, server 192.168.102.193 not responding.`
   - state=finished，uncertain=false（只读操作）。主机一度可连接，但全卡/RAM/磁盘观察不完整，不能宣称资源再次准入。

这是连续网络/SSH观察失败，具体原因未知；可能网络或主机响应问题，但没有证据确定。无sudo、服务调整、kill、换主机或反复重试。不因第二次部分stdout而盲目继续部署。

## 状态与恢复入口

- 请求needs_input/步骤边界pause：用户或主会话确认固定主机的网络/SSH稳定后恢复。
- 不请求正式更新次数；该问题与formal-plan缺参无关。
- 上轮artifact-admission-probe-001最后成功inspect为running=false、exit1。此次未重新核验job状态；无新job，旧失败证据保留。
- 没有新的未知mutating操作；两个失败请求都只读，未自动reconcile或重放。
- Coder请求的完整converter/安装源hash/原生配置/既有转换产物材料仍未取得。H3-EXP-001 open；H3-EXP-002 queued；Flow0/Flash0。
- 恢复后先新id只读inspect_host和owned job核查，再继续有限发现/材料收集。若找到converted根，再按当前代码进行有界meta/header/双VAE配置探查，禁止直接full CPU loader或宣称训练成功。
