# 首轮环境/部署/真实GPU探查证据

所有远端动作均经固定operations.py，未改source或操作脚本；仅Git publish/deploy。

## 操作回执

|operation id|结果|
|---|---|
|exp001-host-inspect-001|code0，train170hx，10卡0MiB，host112GiB available，disk500GiB available|
|exp001-bootstrap-001|code0，任务root owner初始化|
|exp001-environment-inspect-001|code0，版本/路径，旧server checkout f51033af981204f510ad5e4d76ab47e57f26980c干净，无GPU进程|
|exp001-artifact-discovery-001/002|code0，找到下载缓存位置线索；旧环境脚本只读未执行|
|exp001-publish-d5c10c9-001|code0，创建固定实验分支，无force/main写入|
|exp001-deploy-d5c10c9-001|code0，Git fetch/FETCH_HEAD=请求SHA，新detach worktree|
|exp001-model-layout-001/002|code0，真实config/index/文件字节数；布局002工具视图截尾，原文仍由operation journal保存|
|exp001-launch-artifact-probe-001|code0仅表示后台已启动；PID765033，300秒有界|
|exp001-inspect-artifact-probe-001|code0，job已exit1，running=false，真实栈保留|
|exp001-resolve-artifact-links-001|code0，查实HF symlink落点与完整block0/raw根键；部署树仍干净|

所有操作回执uncertain=false，无需reconcile。完整请求/stdout/stderr由operations写在任务指定journal（非本角色另建调度器），远端job spec/log/exit可只读复查。

## 模型与数据

原模型候选`/home/ubtroot/code/downloads/MiniMax-H3-FL2VA/FL2VA`，解析到HF snapshot42ed227ee7df40d41602854ae760620d6eb651fe。13个transformer shard全部存在，535张量、33,122,992,912元素是header计数，不是已验证训练参数/完整payload hash。完整真实forward/loading未通过。

- `/home/ubtroot/qifan/Flash-GRPO/dataset/video/train.txt`：1,513,156 bytes，19700行，sha256 `610e16e6b0e8a30bbb1e75cf87413bf592fe491faa74d918d57d914d865e0016`。
- 同目录test.txt：23,615 bytes，300行，sha256 `96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5`。
- 缓存producer/HPS服务未验收；未使用假的cache、reward或SUCCESS。

## 八卡实际探查

固定物理0/1/2/3/6/7/8/9的完整UUID，逻辑0..7。torch实际UUID与授权表一致（API字符串无GPU-前缀）；capability=[8,0]，native_bf16=true。每卡128×128 BF16 matmul结果128，真实backward finite，peak_allocated17,219,584 bytes。

实际FSDP gradient-divisor API与三种Diffusers H3公开类存在，但本轮未执行NCCL/FSDP、真实模型rollout/replay、AV decode、HPS或optimizer step。不能以这次探查称Flow/Flash完成；实际计数0/0。

## 交接

H3-EXP-001是真实运行阻塞：build_h3_meta `_local_file`因原HF config symlink拒绝；同时实读native fused-QKV/schema需要受审计的兼容路径。参见不可变issue报告。Coder先查两个已知H3分支，再最小本地修复；Experimenter后续Git部署独立复验。formal-plan不修改、不提前暂停。
