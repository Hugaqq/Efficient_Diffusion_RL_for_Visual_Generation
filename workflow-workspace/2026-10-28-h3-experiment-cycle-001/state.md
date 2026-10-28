# 当前状态 — 第四轮环境材料补齐 / repair

本地HEAD a195f13b384d54f5ac231c698f2f1e73645ff715；已部署探查代码d5c10c9f00e44a816e1f0881c7e28ba8accbcdae。本轮未部署新的生产代码。正式计划未修改，Flow0/Flash0。

## 最新进展

- 已采用用户最多5次只读SSH重试授权；本轮所有请求首次成功，未修改操作/监视规则程序。历史网络暂停解除，needs-input保留历史而标明已解除。
- 实际读取三种installed H3源文件，hash均与Coder固定版本一致；无需盲目升级包。
- 官方Diffusers固定aaf57d03通过有界owned Git prepare取得，converter SHA256匹配。完整原文阅读转录保存为experimenter/converter-004-part1/2/3.txt，来源/许可/实际hash见provenance。未执行converter、未转换权重。
- 原生FL2VA小配置/metadata内容及原hash/blob路径已归档experimenter/native-configs-004.json。
- 发现新版layout候选`/home/ubtroot/code/downloads/MiniMax-H3`，但全部transformer14个shard、vae3个shard、audio权重缺失，不能作为ready artifact。
- 真实public H3+Accelerate+PEFT meta成功：638 base、600 LoRA、1238 state，state bytes66,363,578,880、buffer64、全部参数meta。随后headers/decoder拒缺shard，probe exit1。没有调用full CPU loader、optimizer或训练，不把meta检查当GPU验收。

## Owned jobs

远端root `/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/`：
1. artifact-admission-probe-001：旧失败保留，PID765033/start_ticks71836860，exit1，首轮实际八卡BF16环境探查；不是H3训练。
2. pinned-diffusers-source-001：PID933370/start_ticks74226462，created1793155110.5590448，timeout600；started1793155110.6031036、ended1793155116.1823854，exit0/running=false。只Git取回上游源码。
3. converted-meta-admission-001：PID938447/start_ticks74282963，created1793155675.5641956，timeout300；started1793155675.6076143、ended1793155683.9933171，exit1/running=false。真实meta/header准入探查。

三job的spec/owner/exit/job.log均保留；无活跃训练，无新未知mutating操作，未kill或stop。

## 交接与待办

见issues/current.md与不可变issues/H3-EXP-001-materials-r1.md。Coder完整核对转换器/config，最小提交准备工具/必要严格加载预算修复；不丢安全guard、不虚填host预算、不移植单卡runner。收到snapshot后Experimenter Git部署、独立artifact准备与真实模型GPU复验，再推进Flow5、single-step Flash5。正式步数到两个smoke成功后确认。

历史证据：environment-001.md、environment-002-connectivity.md、environment-003-connectivity.md及原H3-EXP-001报告均未覆盖。本轮未跑CPU回归，无模型/reward/训练成功声明。
