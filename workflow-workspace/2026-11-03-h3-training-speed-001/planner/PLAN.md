# Round 006 — H1：默认关闭、专卡UUID约束的有界HPS权重驻留

## 恢复/决策

主会话已明确恢复并直接提供M0；M0成本输入已足以选择本轮，**不再等待同一M0，也不申请先跑GPU才写代码**。本轮唯一单元H1：对原HPS服务代码增加opt-in单模型复用，带原显存准入、低余量offload回退、闲置60s卸载和退出清理。返回implement。

只写代码/CPU验证，不启动/请求/替换/重启现有服务、不SSH/训练/推理/部署；M1与任何GPU执行仍未授权，Euler-B仍paused。

## 已有结果复核，不重复已发生操作

HEAD `e5cfeb446bb20d1222a1e8df09516f4682572eee`实际log/HEAD stat确认：相对上一已审`0eb2a5b`只是Planner四份报告快照，无新源码commit，初始status仅原.pi/。G0/I1/P1/F0代码/commit审查和Planner上轮实际原七项 **7 passed, 3 warnings in 44.91s**有效，本步骤不重复跑该门禁/恢复补丁。Coder本轮提交前仍必须对自己新代码实跑7/7。

## 真实成本/排序/收益边界

选用32行原值存于M0-ROWS.csv，完整归因/来源指纹与原摘录交叉核对见M0-ANALYSIS.md。原归档路径只作索引，未读取工作区外；没有伪称独立重hash整个480条归档。

实测每样本：D=45.613s，score_full=50.405s；服务hps_score=49.112s，其中onload12.081s+offload12.631s=24.712s，**占服务父段约50.3%**。32条搬运共790.792s=13.180min；剩余服务24.400s不是纯GPU kernel，不能父子相加。VAE两load共270.054s=4.501min；owner内保留一次后可去24次reload的旧serial上界约197.290s=3.288min。

现读真实 `scripts/serve_hpsv3_video.py`：CPU模型已初始化一次，**每POST却完整model.to(cuda)→124次原reward→model.to(cpu)**。因此单模型有界GPU复用是直接可实施、显著冗余消除，不依赖缺失的VAE headers/完整拓扑/CUDA trace。

不能把13.18min全叠到P1：以M0各owner顺序和旧D/H不变的反事实理想调度，P1主体约33.069min；warm H'=H-on-off后decode转为瓶颈，主体约27.730min，**H1相对P1约省5.339min/轮**。旧完整保存轮127.73min分母（含优化/完整校验/保存，不含初始化等），仅此模型推算P1后约109.59→104.25min，增量约1.05×。这是条件估算，未实现/未实测；不叠I1、不能承诺2倍。首个cold load、最终offload/TTL、实际争用/回退/监控开销须实测。

## 严格修改边界与原子单元

代码只允许：
1. `scripts/serve_hpsv3_video.py`：可选驻留状态机、CPU健康身份策略、原请求路径复用/回退、同线程idle housekeeping/close及main资源finally。可在文件内用小helper组织，不造通用服务框架/新调度器。
2. `tests/test_flash_collection_runtime.py`：原七函数内部必要回归，所有既有断言/函数数量保留；不得新增测试函数、skip/替换/参数化套件或编辑其它测试文件。
3. Coder自己的round-006/commit材料。

不改原HPS客户端/wire/aggregation/processor/model/权重/精度/帧batch、P1/F0/训练算法/FSDP、依赖、科研配置/旧manifest、TASK/AGENTS/其它角色报告。若必须扩大代码范围，先报告具体原因，不混做VAE缓存或HPS批处理。

建议commit：`perf(hps): reuse dedicated GPU weights with guarded idle residency`。必要功能/回归同原子commit，先7/7/diff检查；Simplifier必要等价简化另commit并先7/7，无必要不造commit。

## 不变量

- 原124帧全部逐帧、frame microbatch1、原inferencer.reward/processor/prompts/images顺序、原dtype/autocast/inference_mode、scalar_fp32和top37/排序/HTTP字段/输入校验不变。不假设模型所有权重都FP32，**不新增dtype cast/量化/compile/改batch**。
- 构造仍CPU加载/eval/requires_grad(False)，原loaded_tensor_sha及processor/package/config/weight身份检查不跳过；不开启时原每POST上卡/下卡/40GiB准入/16GiB监测/unused-cache cleanup行为保持。
- 只保留同一个冻结model对象，最多一份GPU模型，无复制CPU+GPU双模型、无跨设备转移/CPU推理fallback；不保存images/媒体/请求队列。HPS HTTP仍单线程串行，不并发访问同一inferencer。
- 当前这是服务**代码**优化，不是服务操作。任何实际启用须未来独立批准专有GPU4、新实例/端口/manifest/身份/日志归属；现有服务/原3db2a05 run及旧证据不动。

## 建议策略与状态合同

### A. 明确opt-in、设备与健康身份

- CLI默认off，例如`--gpu-resident`；on必须配`--resident-gpu-uuid`完整规范UUID，off时额外UUID参数应拒绝，非bool/非法UUID/不完整声明先拒绝。idle TTL本轮固定60s，别引入任意schedule或调参框架。
- on要求显式CUDA_VISIBLE_DEVICES恰为该单一完整UUID；**只读验证，不改环境/LOCAL_RANK**。在每次warm reuse/cold onload及idle/close相关GPU生命周期操作前核实际logical0/current device/完整observed UUID与声明一致；不按名称/物理序号记忆猜。可复用已有严格UUID规范化方法，支持PyTorch已知_CUuuid，不能接受任意stringable替代。
- M0里GPU4 UUID为`GPU-0e1cd165-126d-e889-2b87-240d8faddddb`，只能作保存证据；代码不能把此历史值当实时独占/可用内存证明，仍需future实验核准。不要自动占GPU5/训练卡。
- 保留原health top-level schema；在identity_facts加入**声明为运行策略**的版本/启用/expected UUID/TTL字段并纳入原canonical digest，不能称其为未发生的实际GPU观察。模型hash/processor等事实照常实际生成，source SHA自动随代码改变；实际观察只记遥测。
- health/manifest identity一旦就绪保持稳定，不随命中/TTL/state计数改变；源码/策略变更自然形成**新身份**，不得硬编码旧identity/改旧manifest/放松健康比对掩盖差异。新artifact由未来获准实例按原真实代码生成，只用于新run。
- 模型卸载/UUID观察/监测失败造成状态不明时要fail closed并报告unready/拒绝后续POST；不能健康端点仍宣称可安全复用。不要吞异常后自动重试同一次POST。

### B. 每请求原准入与可解释的fallback

可用CPU / GPU-idle / scoring / poisoned-closed小状态，标明模型实际位置，只有确认成功转移后才更新状态。

1. cold：原实际free>=min_free_gib（原CLI40..80边界不改）才model.to(cuda)，dtype不变；按原路径评分。
2. warm：先核UUID、实际位置、TTL与**实际card free>=原min_free_gib**。不得以free+cached_weight构造“虚拟free”绕过40GiB门，原16GiB全请求采样余量保持。
3. warm余量不足/TTL过期：先对**自己的**model安全.to(cpu)+原unused-cache释放，重新观测真实free；足够则本次走原cold路径，不足按原503/异常结束。只回退模型驻留策略，**不重试POST/不重做已开始的124前向/不降CPU计算**。
4. 成功评分后同步/原余量检查照旧；仅on且实际free仍满足原min_free、状态明确时保留GPU权重。否则原offload cleanup；任何评分/monitor/margin/identity异常都不留warm可复用状态，可靠offload后传播；cleanup失败附note/poison，不能盖住首个业务错误。
5. 复用本轮不删原unused-cache释放；warm时它只释放unused，不能谎称已return-to-CPU。fallback/TTL真正offload所需cleanup明确原因/计数，不为曲线新增无谓empty_cache。

### C. 有界idle与退出：不新增服务线程

- 利用同一HTTPServer主线程的service_actions/serve_forever poll（如0.5s）做idle housekeeping，或等价明确的单线程周期hook。默认off的idle不探CUDA；on仅已驻留且无active request时检查真实UUID/free和monotonic idle age。
- 满60s idle或free低于原min_free即主动卸载自己的模型；下一POST入口再检查expiry，避免漏tick后继续无限复用。active scoring期间绝不TTL卸载，不并发close inferencer。
- TTL是正常服务loop中的检查期限＋poll/调度延迟，**不是能强杀卡死HTTP/kernel的硬实时保证**；不得为这条保证开启无界后台线程。意外阻塞/卸载失败记录失败/poison，不能伪称完成释放。
- 新close幂等；shutdown/KeyboardInterrupt/server异常通过main try/finally关闭server并释放owned resident模型。初始化仍不提前上卡；不影响其它服务/进程。close与idle都不发评分请求。
- 原请求monitor stop/join必须在所有业务/转移异常下执行。若需为本状态机修复“offload异常导致原monitor未停”的局部路径，可同单元处理，但不扩为其它优化；monitor异常不能静默丢失。多错误保留原business异常＋cleanup notes，无法确认卸载就不再ready。

### D. 遥测/安全证据

在原journal/JSON机制记录hit/miss/cold/fallback/TTL/offload/retain及reason/声明与observed UUID/free、实际onload/offload次数；只对真实.to动作记原onload/offload span，warm跳过应有明确hit事件，不能把虚构零秒搬运当实测。idle offload是独立idle阶段，不错绑成某个sample的评分时长。原日志失败仍fail closed，source/manifest/health完整门不删。

## 必要原七项CPU回归（不能发真实服务请求）

- 使用真实HpsBackend `_score`/handler/策略路径，只有HPS重模型/inferencer与CUDA/显存/UUID/clock/HTTPServer socket为CPU叶子；`object.__new__`配tiny模型也可，但不能把整个评分或驻留逻辑mock掉。
- 同一CPU tiny冻结模型/原torch scalars，连续至少两请求：default仍每次cuda→cpu；on cold一次上卡、warm无第二次上/下卡，124调用/请求、参数/输入顺序/模型dtype/RNG不变。比较全部逐帧FP32数组/原aggregate与top37 indices **bitwise相同**，含ties；用原wire validator/真实Handler离线BytesIO输入的JSON/JPEG，不能只断言计数。预留真实GPU配对未验证，不拿CPU结果代替。
- cold不足40拒绝且零forward；warm不足40先卸载/empty-cache，真实free复核够时只执行一次cold评分，不够时503/零forward；运行中余量<16/无效telemetry/monitor异常/错误UUID/错误CVD均失败关闭。覆盖状态不明不复用，health/POST不虚报ready。
- fake monotonic＋实际server idle hook：59s不卸，60s后卸，低headroom先卸，active不卸，off不探CUDA；过期在next POST重验、shutdown/重复close无重复卸载，无idle线程残留。
- 原业务reward失败、onload/offload/idle/close失败、多重业务+cleanup失败、journal失败：传播主错误、monitor确实stop/join，不返回partial scores、不自动POST重试；后续不能用poisoned GPU状态。不要构造无法teardown的永久阻塞线程。
- manifest/health：真实CPU loaded_tensor_sha与原digest/validate_health保留；策略改变形成新identity，未匹配artifact被原客户端拒绝；状态/hit计数不改变ready时identity，不覆盖旧manifest。不得修改protocol validator实现来让测试通过。
- 实际CLI --help/非法组合在heavy模型/GPU/socket之前退出；main用fake server/backend叶子验证finally close/idle hook接线，绝不监听真实端口。G0/I1/P1/F0原回归与当前worktree导入证明全部保留。

## 提交门禁与风险/后续

Coder新代码每个commit前实跑原七项：

```text
bash workflow-workspace/2026-11-03-h3-training-speed-001/inputs/run-cpu-gate.sh
```

必须7/7＋git diff --check；功能及必要回归一起提交，只显式stage所属文件，不处理原.pi/，不amend/rebase/squash。记录真实SHA、测试和未测收益。Simplifier随后独立审核状态/身份/生命周期/科学等价；Planner再复审。

GPU风险仍是实际显存/host/CUPTI/服务数值、残留缓存、TTL/争用收益；本轮不运行M1。之前MEASUREMENT-CARD的M1无服务变更授权，**不能拿它开启H1**；后续如需H1实验必须另批新专有服务/manifest/endpoint及资源，不自动替换健康旧HPS或改旧run。原生同输入GPU逐帧/排序先要求完全一致，任何非零差异先调查，不自动放宽容差/换reward。代码/CPU通过不称已达speedup。

M0已解除同一成本等待；VAE复用在H1后重新成为高优先下一候选，P2/FSDP等仍开放，本轮只H1，不写FINAL/不判耗尽。
