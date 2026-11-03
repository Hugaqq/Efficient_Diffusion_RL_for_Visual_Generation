# Simplifier round 006 — H1 独立复审

## 结论与提交

H1 当前实现符合本轮计划；未发现必须修复的正确性缺口或有明确维护收益的必要等价简化。**不改源码/测试，不创建 Simplifier commit**。保留显式状态、释放失败标记、UUID/位置复核、监控清理与业务异常优先逻辑，不为减少行数删除安全门。交 Planner 复审，不宣布工作流完成。

实际 git log 确认原子功能提交 **3f02fdb**，父0190004；当前 **7696c6584c18ffc8f8111432692a54590d202507** 为工作流快照。已执行相对生产基线的累计授权源码diff，输出超过50KB被工具截断；未访问其外部临时日志。补读当前完整 scripts/serve_hpsv3_video.py、原测试内完整H1区间及UUID规范化源码，对照当前PLAN和Coder收据，而非仅采信交接。历史G0/I1/P1/F0不计为本轮新增。

## 输入与成本证据

重读 TASK、当前PLAN、planner/M0-ANALYSIS.md、Coder round-006与收据、原CPU短入口。M0已提供，本角色不重新请求/重跑M0，也不读取工作区外归档或声称独立重hash480条原档。保存的32行归因显示onload+offload共790.792s、占服务评分父段约50.3%，与源码每POST搬运路径相符；这支持H1选择，但不是新代码收益。本报告不新增速度/耗时估算，不将服务父子阶段重复相加。

设备/资源事实沿用保存证据，完整UUID声明不是独占或free证明，不按GPU型号猜互联。标准DDP静态超64GiB的限制不因HPS模型驻留而改变。

## 实际源码审查

### 驻留、身份与资源

- 默认off、strict bool；on要求完整规范非零UUID且CVD恰为该单UUID，off拒绝额外UUID。只读环境，不改LOCAL_RANK/CVD。实际logical0/current/observed UUID复核支持已知CUuuid而拒绝任意stringable。
- CPU初始化/eval/frozen，真实loaded_tensor_sha及processor/package/config/source检查保留。只移动同一模型，不新增第二CPU/GPU模型、dtype cast、batch/compile、请求媒体缓存或CPU推理fallback。
- cold/warm均读真实card free并沿用原min_free门；warm不足或TTL到期先真实offload+unused-cache cleanup，再读真实free，够才cold一次。不以free+权重虚构准入，不重试POST或已经开始的评分。
- _location只有确认转移/位置检查成功后才设为目标；释放失败unknown/poison，ready/health/POST拒绝复用。清晰CPU状态下的临时admission不足保留503语义，区别于未知状态。转移尝试/完成分别计数，warm不伪造零耗时onload/offload。
- health身份纳入声明策略、源码自然新身份；命中/TTL/状态计数不改ready identity。旧manifest差异拒绝覆盖，客户端/validator未放松。未来部署须新实例/endpoint/manifest/资源授权，当前代码flag不代表服务已启用。

### 生命周期、错误与取消

- HTTP仍单线程；service_actions使用同线程0.5s poll，无新idle线程/评分队列。59/60s边界、低headroom卸载与next POST expiry检查存在；active不卸载。TTL受正常loop调度/阻塞影响，不是卡死HTTP/kernel的硬实时释放保证。
- 请求monitor覆盖评分与retain/offload清理，stop/join在finally；监控错误、非法telemetry、<16GiB余量失败都不返回partial scores。晚到monitor错误也不留warm可复用状态。
- 释放/日志失败保留原业务错误并附cleanup notes；日志异常不能阻止一次尚未执行的必要CPU释放尝试，失败.to不盲重放。close幂等且main finally关闭server并尝试backend释放，不能确认物理释放时只报告失败/unready，不假称资源已释放。
- 无新增强杀/请求取消机制；驱动永久阻塞导致monitor超时仍是unsafe/poison，不声称线程已消失。有限可控错误路径CPU测试证明join；真实驱动/服务异常仍需后续实验和运维边界验证。

### 科研协议

- 原Handler JPEG/JSON/sample/prompt/service identity/帧数尺寸检查、逐帧inferencer.reward调用、inference_mode、scalar_fp32、aggregate/top37/wire保持；无帧减少/重排/批处理/精度改变。
- 训练样本身份、P1 Queue1/总许可2、owner顺序、冻结行为策略、dtype/RNG、梯度更新与checkpoint边界没有本轮生产diff。服务复用不缓存分数或媒体，不跨更新混用奖励输入。

## 独立实际CPU门禁

读短入口确认原固定环境与原七项pytest命令后实际执行：

**7 passed, 3 warnings in 51.32s**。

三条仍是DCP单进程保存/加载及TypedStorage。原七函数完整保留，当前worktree导入/Git证明及G0/I1/P1/F0回归一并通过。`git diff --check`实际空输出通过。

已实读H1测试并核对其限制：实际backend构造/CPU loaded hash、_score、离线Handler完整JSON/JPEG及原validator执行；仅重模型/inferencer、CUDA/host/socket/clock为叶子。off双请求每次上下卡，on cold一次后warm复用，124次/请求、全部FP32逐帧及视频字节/top37含ties、输入顺序、混合FP16/FP32参数和CPU全局RNG一致。覆盖cold/warm不足真实回退、idle/TTL/active/重复close、UUID/CVD/current/telemetry/monitor/margin、onload/offload/reward/log多错误、unready拒绝、main finally与旧manifest不覆盖。测试fake socket未监听端口，CUDA位置为合成观察，不冒充GPU严格数值证明。

初始status仅原未跟踪.pi/；本轮只新增本报告，无stage/commit，保留.pi/不读内部/不清理/不提交，不宣称整树clean。最终状态另用命令核验。

## 剩余问题/交Planner

1. Planner独立核验3f02fdb及本轮原七项结果后，继续选择VAE/P2/FSDP等下一独立单元，不再等待已提供的M0。
2. H1仍默认off。真实GPU逐帧bitwise/top37、专卡实际UUID/资源、TTL/回退及端到端收益未验证；需另批专有新服务/端口/manifest实验，不能借原M1无服务变更授权启用。
3. 最终integration/perf发布、最终同协议两更新保存恢复未完成。本角色无GPU/SSH/真实HPS请求/监听端口/服务替换重启/部署/训练/推理，Euler B保持暂停。
