# Coder round 006 — H1 专卡单模型有界驻留

## 范围与输入

按当前 planner/PLAN.md，仅实现 H1：`scripts/serve_hpsv3_video.py` 与原七项测试内部必要回归；无其它训练/协议/客户端/服务配置/依赖变更。起点 `0190004234200308131207a2dbbc9370eb7db24b`，初始status仅原未跟踪.pi/。已重读TASK、当前PLAN和planner/M0-ANALYSIS.md；M0已提供并解除成本等待，不重复请求/执行M0。RECOVERY-003现标resolved且已核对，没有重放历史P1补丁；旧5/7提示不是当前源码状态。

## 实现与不变量

- 新默认off的 `--gpu-resident`，必须另配 `--resident-gpu-uuid` 完整规范非零UUID；strict bool、非法/缺失/额外UUID或不等于单一完整UUID的CUDA_VISIBLE_DEVICES在heavy加载前拒绝。只读验证，不改CUDA_VISIBLE_DEVICES/LOCAL_RANK，不硬编码历史GPU4 UUID。
- 开启后cold/warm与idle/close相关GPU生命周期操作核logical0/current device/实际properties UUID。支持PyTorch已知_CUuuid规范化，拒绝任意stringable对象。实际参数/buffer位置与冻结eval状态复核，不依据“有缓存”猜位置。
- CPU / CUDA-idle / active / unknown-poisoned / closed状态显式记录；只确认成功.to后更新位置，最多同一个model对象。保持原dtype，不复制第二模型、不cast/量化/compile/批处理、不CPU评分fallback、不缓存请求/媒体。
- cold仍真实free>=原min_free（CLI原40..80边界），warm每次也真实free>=同门；低headroom或过期先真实.to(cpu)+原unused-cache cleanup，然后重新读真实free，够才走一次原cold评分，不足原503。没有free+weight虚拟预算，也不重做已开始的124前向/重试POST。
- 原124帧逐帧调用inferencer.reward(prompts=[prompt],image_paths=[image])、inference_mode、scalar_fp32、原Handler JPEG/输入/aggregate/top37/wire字段原样保留。参数/processor/package/config/实际CPU loaded_tensor_sha检查原样保留。
- 成功后原同步/全请求16GiB采样检查和unused-cache释放不删；仅on且实际free仍够、UUID/位置明确时retain。warm没有虚构零秒搬运span，只有真实.to才有onload/offload span；遥测记录cold/hit/fallback/retain/idle/offload、reason/declared+observed UUID/free，另区分转移attempt和确认完成计数。
- HTTPServer.service_actions在原同一串行HTTP线程随0.5s poll做idle housekeeping，60s idle或低headroom卸载；next POST再验expiry。active期间不卸载，无新idle线程。TTL是正常loop期限+调度/poll延迟，不是能强杀卡死HTTP/kernel的硬实时保证。
- close幂等；main异常/KeyboardInterrupt/manifest冲突等均finally关闭server并释放owned resident模型，双cleanup错误附note，不覆盖首个业务异常。不主动shutdown其它服务或发请求。
- 原请求monitor保留到offload/retain清理后，所有onload/reward/offload异常路径stop/join；monitor异常与非法telemetry显式传播、不返回partial scores。未知释放失败不盲目重放.to，poison后health=503/unready、POST=503且不复用。纯cold资源不足且CPU状态明确仍按原临时503语义，不把它当未知模型状态。
- 日志失败仍fail closed；日志坏了也不能阻止一次必要CPU释放尝试，已执行/失败的.to不自动重试。cleanup失败保留主错误+notes；无法确认释放就不宣称释放成功。

## 身份/部署边界

health top-level schema不变，identity_facts新增明确**声明策略**的version/enabled/expected UUID/固定TTL字段，纳入原canonical digest；不是尚未发生的GPU观察或独占证明。实际模型/processor/源码等事实照常生成，ready时identity不随hit/TTL/计数改变；unready时不再向客户端声称ready。

源码/策略自然生成新service identity；旧manifest不同则仍拒绝覆盖，原客户端/validator保持严格匹配。后续H1启用需另批专有GPU4的新实例/端口/manifest/日志归属及新run，M1既有无服务变更授权不能拿来开启H1。本角色未创建/替换真实manifest或操作健康旧服务。

## 原七项CPU证据

七个原函数/全部既有断言保留，不增删/skip/替换/参数化套件。H1回归位于原manifest函数内。实际HpsBackend构造/loaded_tensor_sha/identity/_score与真实Handler JSON/JPEG/校验路径执行；仅HPS重模型/inferencer、CUDA设备/转移/显存/clock、host admission与socket为CPU叶子。tiny模型真实混合FP16/FP32 CPU参数冻结、真实torch标量，CUDA位置是显式观察叶子，不是GPU证据。

- 连续至少两请求：off每次cuda→cpu；on cold一次上卡、warm无第二次上下卡，仍124 reward/请求、输入顺序/参数/dtype/global CPU RNG不变。离线Handler用BytesIO输入真实JSON/JPEG；全部FP32逐帧数组及原validate_score/aggregate的top37 indices（含ties）bitwise相等。
- cold<40零forward；warm<40先卸载/empty-cache，真实free复核够时只做一次cold评分，不够503/无新forward；next POST过期、真实idle hook59/60s、低headroom卸载、active不卸、off idle/close不探CUDA，重复close无重复卸载。
- cold与warm/idle/close UUID错误、current!=0、CVD错、任意stringable UUID、非法/运行中非法telemetry、运行中<16、monitor异常均拒绝/poison且不虚报health/后续POST。已知CUuuid表示可规范化。安全上不能确认的GPU位置不对错误设备强行搬运。
- reward/onload/offload/idle/close失败、业务+cleanup双失败、score scope/empty-cache journal失败均保留主错、清理可确认资源，实际monitor已退出；失败.to(cpu)不盲重试。测试没有不可teardown的永久阻塞线程。
- 真实CPU tensor hash相同、健康digest/validate_health保留，策略导致新identity、计数与TTL不改ready时identity；原客户端拒绝不匹配artifact。main拒绝覆盖旧manifest。
- 真实CLI help/非法UUID组合、错误CVD/低于原CLI headroom预算在heavy/socket前退出；main仅fake backend/server无socket，验证idle hook、KeyboardInterrupt与idle异常、server/backend双close失败finally顺序和主错误note。

固定原七项短入口实际执行四次，依次 **7 passed, 3 warnings in48.48s /50.92s /49.11s /53.69s**；最后一次含逐帧及视频标量的struct.pack FP32字节比较，最终提交前为53.69s。三条仍是原DCP单进程保存/加载及TypedStorage警告。当前worktree导入/Git证明、G0/I1/P1/F0原回归均保持通过。`git diff --check`空输出通过；实际diff stat仅上述两个代码/测试文件。原未跟踪.pi/保留，不读内部/不清理/stage，不宣称整树clean。原子提交后SHA另记round-006-commit.md。

## 实测输入、条件收益与剩余风险

M0选用32行/归因来自上游保存材料，本角色没有读取/重hash工作区外480条原档，也没有新GPU调用。旧onload386.598s+offload404.194s=790.792s（13.180min），约占服务评分父段50.3%；这支持减少重复搬运，但不是新代码收益。

按Planner用旧D/H且无争用的反事实：P1主体33.069min→warm HPS主体27.730min，H1相对P1约省5.339min，而不是简单叠13.180min。旧127.73min保存轮分母**含采样/评分/优化/完整校验/保存、不含初始化等**，同模型代入P1后109.59→104.25min约增量1.05×；不叠I1、不承诺2倍。cold启动、TTL卸载、headroom回退、PCIe/CPU竞争与监测/日志成本均需实测。

CPU仅证明逻辑/输入/标量/状态/错误边界，不证明真实GPU逐帧bitwise等价、可用内存、独占/UUID实际绑定或端到端speedup。监控join若驱动永久阻塞，3s检查会报告unsafe/poison而不是谎称线程已消失；不能强杀该线程/HTTP/kernel，正常有限错误测试已验证无残留线程。idle/close失败也不保证物理释放，只明确拒绝后续复用并报告失败。

本轮未GPU/SSH/真实HPS请求/监听端口/服务修改/部署/训练/推理，Euler B继续paused。交Simplifier独立审核后Planner再审；VAE/P2/FSDP等仍开放，最终integration/perf发布与GPU同协议两更新/保存恢复/严格数值验收均未完成，须另批准实验卡。
