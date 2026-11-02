# Round 003 — P1：显式可关闭、单owner内的有界解码/HPS双缓冲

## 决策与上轮验收

本轮唯一优化单元 **P1：在一个既有aux owner turn内部重叠解码与远端HPS/观看副本消费**。不改变8个owner顺序，不跨owner排队，不复用VAE模型，不改FSDP/HTTP/reward数学。返回implement。

I1已独立复审：原子commit `a211cbf`，Simplifier无必要修改/无commit。当前HEAD `266a25c43d5b7c76c0caed63da0ca737b35484c1`真实累计diff及最新源码与计划相符；Planner固定原七项 **7 passed, 3 warnings in 48.32s**。旧boundary仍自校验、snapshot每次fresh验证、后置tail/source复核及lifecycle立即消费均通过。没有新发现的I1正确性缺口；不把调用次数减少称为已测节时。

## 真实依据与收益分母

- `h3_dual_stages.py::reward`：当前owner worker对4条样本依次decode→score_one，后者含原HPS校验/JPEG/同步HTTP及可能观看副本导出。
- `H3Coordination.aux_turns`：依次owner0..7；主线程心跳，只有owner工作线程可做VAE；直到work结束才进入owner-end门。可以在work内部加一个**CPU消费线程**，不改变PG序列和CUDA生产者线程/stream。
- 已存事实HPS物理4独立于训练0/1/2/3/6/7/8/9；不是GPU5授权，也不是实时服务绑定证明。默认关闭，并在后续批准实验中独立核查服务实际GPU，角色不得探测/部署/请求HPS。
- 现有decoder一条lease持有锁至消费结束，必须明确移交CPU媒体所有权；直接在lease内提交future会阻止下一decode，直接出with则媒体失效。
- 现有`h3_memory.Journal.stack`为进程共享列表；decode/reward_score/video_save的并发嵌套会错配、跨样本归因并重置另一个阶段的峰。**必须随此功能修复局部并发观测合同**，不能关日志、删reset前落盘或靠全程锁把流水串行化。

保存基线第二轮：42.600+51.318+12.994min，校验16.799、保存4.021；127.73min含优化/校验/保存，不含初始化/其它审计。用户D≈45s（假设加载8+decode37）、H≈50s不是单独实测。

本方案8个四样本owner各自fill/drain，理想辅助主体为`8*(D+H+3*max(D,H))`，而非32样本全局流水`D+H+31*max(D,H)`。若D=45/H=50，主体3040→1960s，条件省18min/轮（实际还受CPU竞争/日志/导出/传输影响）；对已存阶段总量仅作近似：93.918→75.918min约1.24×（前两段，**不含优化/保存**），127.73→109.73约1.16×（**含优化/校验/保存，不含初始化**）。不叠加I1未测收益，不宣称达到51→27min或2倍。跨owner全局流水留下一独立候选。

## 修改边界（同一逻辑单元的必要部分）

允许代码文件：
- 新 `visual_rl/algorithms/trainer/h3_reward_pipeline.py`：窄本地双缓冲辅助函数，不是训练循环/工作流调度器；输入有序样本与decode/score能力，返回原reward rows。
- `visual_rl/models/implementations/h3_decode.py`：显式、单次CPU媒体所有权移交/关闭；不改VAE加载/数学/精度/显存门。
- `visual_rl/algorithms/trainer/h3_dual_stages.py`：默认串行，opt-in仅支持本任务`dual_flash + paired_4x8`；接到现有aux_turns内部，原score_one回调不绕过。
- `visual_rl/runtime/h3_lifecycle.py`：显式strict-bool运行选项及现有预算传递/模式准入，不改科研profile/schema/保存条件/restore。
- `visual_rl/runtime/h3_memory.py`：仅流水CPU消费者的隔离观测上下文；保留默认串行日志格式/peak-reset合同，不做通用重写。
- `scripts/launch_h3_resolution_memory.py`：可选`--reward-pipeline`（默认False）传入既有backend，保留fresh两步所有门禁/审计；此flag不是执行授权。
- `tests/test_flash_collection_runtime.py`：原七函数内部扩展必要覆盖；不改其它测试文件、函数数量、skip或参数化。
- Coder自己的round-003/commit message。

不改HPS服务/客户端、transaction/codec、依赖、任务授权、科学配置或其它角色报告。不扩大到CPS/dual_flow/通用组件缓存。若需要范围外修复先报告。

## 设计合同

### 1. Opt-in、模式一致及设备

- 默认路径严格维持当前串行行为。backend布尔开关应拒绝非bool；开启只接受本任务dual_flash/paired_4x8，launch仍保持`validate_fresh`的480×320、124@24、10步、4×8、LoRA32/64、两步/cadence5。
- 在进入aux_turns前，所有rank通过原coordinator的**同位置**gate协商串行/流水模式及队列/字节预算；配置不一致必须先失败，不能一rank启线程、一rank走不同collective序列。worker不做PG/模型collective。
- CUDA解码始终在已有owner辅助线程及其work_stream；消费者不创建/选择CUDA设备、不做VAE/forward/optimizer，只操作已完成blocking D2H的CPU FP32张量及原远端score_one/导出。原40GiB准入、16GiB安全边界、VAE释放与monitor保持。
- 开启条件是**另行批准实验卡确认独立HPS GPU**；不把CLI声明当观测证明，不新发health/score请求、不改服务identity。这里交付代码，不启动GPU。

### 2. 精确媒体所有权和严格内存界

- 建议`H3AVMedia`增加明确single-use transfer/handoff接口：只接收活跃、已完成的detached CPU媒体，返回一个可close/上下文管理的owned媒体对象，张量storage、dtype、shape、顺序、metadata完全相同，**不clone/reencode/resize/quantize**。原媒体立即失效；旧lease退出释放decoder锁；owned对象关闭后也失效。双重移交/过期访问拒绝，close释放幂等。
- 原decode_one标准单lease语义不放宽；只通过明确handoff把独立CPU存储生命周期移到有界消费者，不延长GPU模型或decoder锁。不可把消费者通过Python保存的任意alias称为可撤销；测试区分合法owner引用与外部故意留引用。
- 一个producer、一个consumer；队列容量固定1，另有**在decode前获取**的总在途许可固定2。许可从开始decode持有到score+观看副本完成或失败关闭；计数包括正在decode、等待队列和正在消费。只有Queue(maxsize=1)不够，它可能容许消费1+排队1+解码1=3条。
- 不保留整轮媒体列表、不建32个future/无界executor。完成rows仅保留小标量；terminal沿用原CPU latent绑定，及时释放局部引用。2条原生AV payload上界约`2 * 229881600 bytes`（约438.5MiB），解码工作区、JPEG/JSON/encoder scratch另算；不能拿这个值冒充整个RSS上界。
- 使用已有预算：单payload仍不超过media_bytes，解码工作区仍受decode_bytes。开启前在owner工作内对`decode_bytes + media_bytes`的保守额外工作集做host可用内存准入（保留现有24GiB reserve惯例，可复用`h3_loading._admit_host_memory`），不改budget含义/配置。CPU测试只替换telemetry叶子；不足须显式失败或在工作开始前明确选择串行，不能半途吞错/丢样本。

### 3. 顺序、数学、更新屏障

- packet不可变地绑定ordinal/sample_id/group/member/prompt/selected index；producer按r.samples顺序，consumer严格FIFO。只调用原score_one一次/样本，原有限Python float门、124帧/top37校验、manifest/JPEG/identity和observed_score/saver保持。
- 原始张量storage移交，消费者导出必须在owned媒体活跃期间；不能用保存MP4回读评分或先关lease。
- 同一owner只发一个HPS请求；不改变服务请求顺序/微批/精度。下一owner必须等本owner所有packet评分/导出完且consumer退出后才开始；credit/optimize必须等所有owner完成。没有冻结策略跨更新混用。
- 不新增随机数调用/改变生成器次序。加载/解码仍由原producer串行执行；CPU消费者不得采样或改变全局RNG。保留r.behavior_id与局部read_behavior_id在工作前后的一致性保护；不得从worker调用带PG的_version。

### 4. 背压、错误、取消、排空

- producer入队/许可等待、consumer取队使用可取消轮询或等价有界机制，不能依赖满队列永久put sentinel。完成/错误信号与队列解耦。
- 任一decode/transfer/score/非finite/导出/日志/清理失败：记录原错误，停止新生产/新POST，唤醒队列/许可等待；关闭未消费owned媒体；producer/consumer退出并join，排空后再向原aux_turns传播，不能返回partial rows。
- 不重试POST，不把失败改成默认reward，不继续credit/update；多重cleanup错误附note而非覆盖首个业务错误。错误退出不进入新owner或模型工作；不新增finally collective。
- 线程不能强杀。取消时对已在进行的HTTP/ffmpeg操作只能等待原超时/结束；不并发关闭仍在用的共享client，不伪称已取消请求。保留aux的3600s整attempt deadline：不可停的外部/kernel卡死是poison/整attempt退出，不可将其当正常排空或允许新更新。正常/可控失败路径必须join真实worker，不留后台consumer。报告这一硬边界；CPU测试用可释放的阻塞叶子验证取消过程。

### 5. 并发观测不能破坏lossless日志

建议只给CPU消费者一个从reward父上下文捕获的隔离观测scope（thread-local独立stack/context），decode继续现有GPU span/reset路径。CPU消费者中的reward_score及既有exporter.video_save嵌套必须走此隔离scope：
- 正确记录本sample/thread/父reward上下文、begin/end/错误/壁钟；不push/pop共享GPU stack。
- **不重置CUDA峰、不调用CUDA synchronize**；如记录allocator快照须明确它是共享进程观测而非CPU评分专属分配。可以用独立cpu_span_begin/end事件，不伪造独占阶段峰。
- journal所有写仍经原锁、序号、byte bound、flush/fsync；默认GPU reset前持久化/high-water不变。任何日志失败仍fail closed。
- 此scope只用于新consumer，不悄悄关闭日志/吞异常，不改原baseline单线程schema/事件语义。避免大范围contextvars重构；不持journal锁跨整个HTTP/导出以抵消流水收益。

## 必要CPU验收（全放原七函数内部，实际驱动而非字符串断言）

1. handoff：原媒体/过期lease失效，owned storage指针/内容/metadata相同；关闭/重复移交/consumer异常清理；原普通单lease重入保护仍在。使用真实tiny CPU decoder或其已有fake重模型叶子，验证下一decode可在前一owned消费中运行。
2. helper＋实际dual reward接线：4条样本顺序不变、只评分一次、float一致；Event/Barrier控制证明第二decode与第一score重叠，不用sleep耗时比宣称speedup。只替重模型/transport，不把整个reward/helper mock掉。
3. 严格容量：阻塞消费者时最多2个包括在decode的许可；第三decode未启动，队列max1，释放后完成。用weakref/活跃计数确认完成/失败packet不再被合法owner/队列持有；不存在整轮视频列表。
4. 数学/身份：同一组原始CPU张量分别串行/流水传给真实H3HPSReward＋原JPEG/validate_score（本地fake transport），相同sample/prompt/124帧顺序/top37/标量结果；至少断言payload/JPEG相等，不只fake计数。观测回调在lease活跃时能导出，serial/pipeline RNG状态与terminal不变。
5. decoder第2条失败、consumer第1条失败/非finite、saver失败、transfer失败、取消时队列满/producer等许可等情形：无deadlock、无新POST/下一owner/update、所有worker已退出、每个已交接owned媒体关闭，原错误传播。线程超时测试用有界Event并确保测试teardown不留线程。
6. 双线程跨序退出的日志：阻塞score时decode span可以完整进出，score中嵌套video_save；无stack mismatch/样本污染，CPU span不reset/sync，原durable-before-reset/high-water/byte-budget/错误测试继续通过。
7. opt-in默认off、模式不符/非bool/跨rank配置不一致门禁、host预算不足在启动前拒绝；两步launcher原fresh/graph/weight/monitor/媒体审计仍在（只CPU叶子验证，不执行main训练）。原I1/G0测试与七个函数完整保留。

## 原子提交与未验证项

提交前仅例行原七项＋diff检查：

```text
PATH=/home/user/projects/h3-visual-rl/.scratch/h3-media-tools:/usr/local/bin:/usr/bin:/bin VISUALRL_FLASH_AUTHOR_DATA=/home/user/projects/h3-visual-rl/.scratch/flash-grpo/source/dataset/video PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/user/projects/h3-visual-rl/.scratch/h3-cpu-venv/bin/python -m pytest -q tests/test_flash_collection_runtime.py
```

必须7/7、当前worktree导入断言保持、`git diff --check`通过；功能及必要回归同一个可回退commit，建议 `perf(h3): pipeline owner-local decode and reward with bounded media ownership`。Simplifier独立复审，有必要等价简化另原子commit并先7/7；无必要不造commit。保留未跟踪.pi/，只显式stage所属文件。

本轮CPU不证明GPU性能/数值等价/真实独立设备绑定。后续实验卡须明确授权新的输出、最终SHA、物理训练8卡＋HPS4、保持原模型/数据/种子/精度/几何/步数/评分/更新，先核资源和服务身份，验证媒体字节/逐帧与top37排序容差、样本/梯度/两更新/保存恢复；收集decode/score/export/queue等待/峰内存和完整含优化保存墙钟。**此处仅计划，禁止执行或重跑旧两步，Euler-B保持暂停。** VAE复用、跨owner流水和FSDP候选仍开放，当前不写FINAL。
