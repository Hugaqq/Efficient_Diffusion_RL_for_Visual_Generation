# Round 007 — V1v：只在一个aux owner内复用视频VAE，默认关闭

## 上轮独立验收与本轮选择

已读H1实际累计diff、完整最新serve_hpsv3_video.py及原测试完整H1区间，确认原子commit `3f02fdb`（父0190004）。当前HEAD `2f4cce22e8b53c2c21919c42905377163d29649d`为工作流快照；Simplifier无代码commit。Planner重新执行原七项：**7 passed, 3 warnings in 51.96s**。H1默认off、身份/free/state/monitor/idle/close与原逐帧数学保持，未发现应先修复的已知缺口；真实GPU/服务效果仍未验证。

本轮唯一单元 **V1v：有界owner-local视频VAE对象复用**。不复用audio VAE，不做跨owner/跨update缓存或HPS/FSDP修改。返回implement。M0已满足收益选择，不再请求同一数据或执行GPU。

## 真实收益与边界

M0视频load32条232.2621479s，其中每rank第一个load合计63.7945410s；若每owner四样本只加载一次，可去24次视频load **168.467607s≈2.808min/轮**（仅decode子段，不含优化/保存，未扣资格/失效/资源管理成本）。audio额外可省不足0.5min，暂不承担第二个缓存模型/weight-norm状态复杂度。

M0反事实P1+H1后D预计成为临界路径，复用视频更可能转为墙钟收益；保持旧调度假设、不叠I1，含优化/完整校验/保存、不含初始化的约104.25min条件分母，上界约104.25→101.44min、增量约1.028×。不是实测/承诺，也不能简单加所有VAE load4.50min；若资源/RNG资格导致fallback，可能没有收益。原8次cold保留，不加预热/额外decode。

headers当前NA不阻碍写保守策略：首次仍按原loader加载，读取**实际模块字节与实际free**才决定是否可保留；不从19.71GiB allocated峰推断pair驻留安全。

## 一个独立单元的允许范围

1. `visual_rl/models/implementations/h3_decode.py`：视频复用scope/借用释放、资源/RNG/状态/失效门、正常/异常退出。可选新增窄helper `visual_rl/models/implementations/h3_decode_reuse.py`，避免把decoder数学复制两份；不是通用缓存框架。
2. `visual_rl/models/implementations/h3.py`：懒scope/关闭owned decoder的桥接；不改predict/encoding/latent数学。
3. `visual_rl/runtime/h3_binding.py`：由真实完整哈希取得私有视频文件凭据并传给scope；decode普通API保持。
4. `visual_rl/composition/preflight/artifacts.py`：**仅增加可选只读file observer/receipt输出能力**，从现有完整哈希的同一遍读取产生凭据；现有返回identity/schema/hash编码及所有强检查不变，无额外大文件扫描。这是缓存安全必需接线，不是另一个完整性优化。
5. `visual_rl/runtime/h3_lifecycle.py`、`visual_rl/algorithms/trainer/h3_dual_stages.py`、`scripts/launch_h3_resolution_memory.py`：默认False的strict-bool `--reuse-video-vae`（或同等明确命名）、真实binding/scope接线、同位置rank配置门；仅dual_flash/paired_4x8开启。
6. `tests/test_flash_collection_runtime.py`：原七函数内部必要回归，保留所有原断言/函数数量，不改其它测试文件。
7. Coder自己的round-007/commit材料。

不改H1/HPS、P1容量/评分协议、F0窗口、FSDP/精度/geometry/科学profile/依赖/transaction/checkpoint语义或授权文件。不合入audio缓存/跨owner流水。若范围必须扩大先报告。

## A. 缓存范围与所有权

- off与原单样本load/use/drop路径一致，不探GPU/新预算、不创建新文件/线程。普通decode_one与lease/handoff不放宽。
- on在**现有aux_turns的owner工作内部**建立scope，覆盖该owner四条sample的串行或P1工作；至多保留**一个video VAE model**，audio每条仍原loader/drop。不常驻8个owner的缓存，不跨collection/optimizer/policy/geometry。
- CUDA加载/使用/丢弃保持原producer工作线程/stream，consumer只处理原CPU媒体；P1先join/drain，scope再销毁模型，最后才owner-end/下一owner/credit/forward。不要通过callback从CPU consumer清缓存。
- 新scope进入/退出、decoder.close/adapter.close有明确防重入/并发/queued-after-close合同。close幂等；活跃decoder/lease/producer不能从foreign thread强行销毁。正常成功退出后cache不持model/weight/媒体引用；失败只保证清除自己引用、poison并禁止后续owner/model，异常traceback可能暂持张量，不能假称物理内存已释放。
- 不改变VAE tiling、未patchify/denorm、24/32通道、audio stereo、clamp、FP32、blocking D2H、finite/shape/statistics/rate/hop检查。不要复制计算实现、预解码/缓存媒体或减少decode次数。

## B. 借用既有完整哈希的真实凭据，禁止重复大扫描

当前 `_hash_artifact` 已为每个文件算content_digest后合入整个model digest，但丢弃per-file结果。不要在每个owner scope/hit再次stream数GiB权重抵消收益。

- 可选observer只接收**本次hasher已完成原身份/文件稳定性检查**的relative path、streamed SHA256、size和独立stat快照(dev/inode/size/mtime_ns/ctime_ns等)。在同一文件读取中收集vae/子树，无caller verified=True、无跳过哈希接口。observer失败仍失败关闭；默认None保持所有老调用/返回identity字节不变。
- Binding仅在本次完整 `verify_h3_content` 成功且原资源/model/source/numerics门都成功后保存私有、独立的video文件凭据，绑定已核model SHA/root；validate_live每个原边界仍完整重验并更新凭据。不要把路径/mtime/observer凭据加进semantic/checkpoint identity或改变原合同。
- scope从实际binding借到凭据，不能由stage/CLI声明一个hash冒充已验证。scope拷贝只读凭据，核root/文件清单/config/class/路径、symlink/index安全及每个文件stat；每次decode/hit和scope退出廉价复核。包括ctime，检测内容被改后人为恢复mtime；metadata只是**失效哨兵**，content SHA和原全内容校验仍是身份权威。
- `_configs()`和原所有path/config/statistics检查每次decode保留；任何文件清单/配置/路径/设备/对象/状态漂移失败关闭，不能静默接受新weights或用旧媒体得分继续更新。无文件系统锁/恶意并发写盘的原子快照保证，不做过度声明。
- 普通构造/restore/关闭cache模式不需要凭据；请求on但没有真实凭据必须明确拒绝，而非无凭据复用。不得减少原checkpoint/init全模型/code/cache核验，也不能用mtime缓存替换这些扫描。

## C. RNG与模型资格（科学语义优先于命中）

- 第一次视频loader调用前后读取Python/NumPy/Torch CPU与**已初始化的当前目标CUDA**随机状态（可复用RankRNGSnapshot或窄helper）。这是只读观察，不能为了检查新初始化其它GPU/RNG，不能save/restore来改变baseline消费。首次仍执行原load。
- 若加载消耗任何所检查全局RNG，**本owner禁用视频复用，走原每sample load/drop，保留实际已消费状态**。不重放load、restore RNG或伪造generator offset；不因不符合缓存资格而停止合法baseline训练。若不能无副作用观测CUDA RNG则不启用复用，不探未授权设备。
- 只有首load RNG-neutral且同一已加载、eval/frozen、目标device/FP32、相同class/config/geometry、无不受控持久tensor状态时才能借用。首次按原顺序load视频再load audio；不新增warmup/预加载。
- 每次借用前/解码后核参数/buffer对象/shape/dtype/device/grad/eval与版本等只读状态，必要的轻量buffer检查可保留。检查失败或模型decode显式改变注册状态时invalidate/drop；不允许后续sample用变过的状态。不要为检查把所有GPU权重D2H重hash（会抵消7s/样本收益）。视频参考实现无audio weight-norm钩子，audio维持原drop；任何不能建立安全资格的状态只disable reuse或失败关闭，不盲缓存任意插件。
- 不让cached model暴露给score/consumer；支持的固定stateless视频decode资格需写清，不宣称可自动证明任意Python模型没有隐藏副作用。CPU测试应故意注入加载RNG消费和注册buffer漂移，证明不会改变原序列/后续复用错误状态。

## D. 内存准入与fallback

- 固定缓存逻辑上限建议16GiB、最多1个video模型。这是上限不是驻留承诺：读取首load实际参数/buffer字节；检查不受控/unregistered tensor状态，GPU实际allocated增量也需计入可能的persistent workspace，不能只用checkpoint文件大小冒充全部占用。资格/容量不够则不保留。
- **原每次decode的40GiB实际card-free准入与16GiB全decode监控保持**。warm入口若保留模型导致free<40，先在无in-flight decode处drop自己的缓存/unused-cache释放，再读真实free，够才原cold path，不够仍拒绝。禁止free+weight虚构准入、下调阈值/FP16/CPU decode fallback。
- 因视频模型会在audio加载时仍在GPU，audio cold load前也要有不弱于原40GiB的真实free条件；不够则先drop idle video缓存、禁用本owner复用，再按原路径load audio。audio从不复用，不允许两个VAE并行forward。
- drop必须真清owned模型引用，不只是empty_cache；unused-cache仍原操作，只在新的scope结束/drop需要时增加有原因的释放，不作显存曲线美容。日志坏了也不能阻止必要清引用，主错误优先、cleanup附note。
- 模型是GPU复用，不保留额外一份CPU权重；如果实现发现额外CPU持久张量须纳入上限/host reserve或拒绝资格。保持P1原host预算/媒体上界和CPU引用生命周期。不要把16GiB逻辑cap或当前allocated当整个设备/RSS硬界。
- 成功scope退出必须在下一FSDP/model阶段前清空模型；释放/monitor异常就poison/中止attempt，不能带缓存继续训练。正常monitor stop/join/日志失败清理如需补齐属于本生命周期必要修复；不能谎称驱动永久阻塞已释放。

## E. 接线/遥测

- backend strict bool默认False，仅paired Flash可启；将真实binding scope工厂传给dual reward，在work中context包住现有串行/P1整体。所有rank同位置协商开关/容量版本，mode不一致先于aux失败；worker无PG。
- TwoUpdateBackend/CLI支持新flag但不运行main；P1/H1/F0可独立开关，原所有fresh两步/监控/图/参数/媒体/保存审计保持。对既有CPU wrapper只补必要默认字段，不放松断言。
- 只真实load记h3_vae_load_begin/end，hit/miss/invalidate/RNG-ineligible/resource-fallback/scope-release独立事件；记录cached bytes/free/资格原因。原h3_vae_decode_begin/exit每sample照旧；新cache scope不要新增peak-reset span来改曲线。
- warm `incremental_peak` 的baseline含cached weights，须明确该口径/记录cached bytes和总allocated，不能拿增量变小宣称总显存下降。原lossless allocator事件与HPS/P1 CPU日志隔离不改。

## 必要CPU回归（仅原七项内部）

1. 原hasher真实小文件：observer on/off返回identity完全相同、原每文件只读/哈希一次，捕获SHA/stat/ctime正确；原symlink/修改/observer抛错拒绝。旧init/checkpoint完整验证仍调用，私有凭据不入contract/semantic/RNG。scope不做额外权重payload read（可设置读取spy），mtime恢复但ctime变化的实际文件改动仍拒绝。
2. 真实tiny视频/audio decoder计算：off四条各load4次；on有资格视频只load1、audio仍4。四条video/audio原张量值/布局/shape/FP32/metadata/terminal/global及sample RNG完全一致，原正常单lease与handoff不变；成功/消费异常后cache引用清空，下一scope重新load，不能跨update复用。
3. loader故意消费Python/NumPy/Torch CPU（以及CUDA观察叶子）RNG：on自动不缓存但不restore/replay，四条结果/调用数/最终RNG等于off；故意注册buffer变化/foreign model/config/device/dtype/grad/eval漂移，不得热复用错误状态。不能只用无随机fake模型宣称解决RNG。
4. 真实free/实际字节上限CUDA叶子：warm<40先清cache后cold再验，仍不足拒绝；audio前不足先evict，无虚拟free；16GiB监测/非法telemetry/monitor/日志/load/decode/cleanup故障无后续owner或update、不留可复用模型。显式允许外部刻意alias/异常traceback限制，不谎称可撤销任意引用。
5. 真实dual reward→复用scope→P1/serial：确保scope在owner worker，P1 consumer已join后才释放、owner-end之前无cache，Queue1/总许可2/原score_once/124-JPEG字节/有限标量一致；任一错误不进credit/update/下一owner，scope exit仍复核文件。实际backend/CLI off/on、rank配置不一致与close生命周期接线需执行，不仅源码字符串断言。
6. 不GPU测试；只替不可用重模型/CUDA/transport/host观察叶子。可构造真正tiny safetensors/header文件，不需真实VAE大权重，不改外部fixtures或依赖。已有G0/I1/P1/F0/H1全部回归和七函数保留。

## 原子commit与权限

Coder提交前实际执行原七项和git diff --check：

```text
bash workflow-workspace/2026-11-03-h3-training-speed-001/inputs/run-cpu-gate.sh
```

必须7/7，功能与必要回归同commit，建议 `perf(h3): reuse verified video VAE within bounded owner turns`；不amend/rebase/squash/换分支。Simplifier另审，必要简化独立commit并先7/7，不造空commit。保留原未跟踪.pi/，只stage所属文件，不改其它角色报告/授权/ignore。

本轮不操作GPU/服务/SSH/真实HPS/训练/推理/部署；Euler-B保持暂停。GPU缓存容量/模型数值/实际命中/收益、H1新服务与最终两更新验收仍待另批实验卡。M0已可用，不重索同一输入，不写FINAL或宣告2倍/硬件最优。更细FSDP/P2机会仍保留，不夹带。
