# Planner round-007 — H1独立验收；选择video-only VAE有界复用

## 真实代码/commit复审

HEAD `2f4cce22e8b53c2c21919c42905377163d29649d`；log确认原子功能commit `3f02fdb`，父`0190004`。后续`7696c65/2f4cce2`是工作流快照；HEAD stat仅53行Simplifier报告，Simplifier未造代码commit。

读取Coder round-006/commit收据与Simplifier round-006，实际执行生产基线授权源码累计diff；超过50KB截断后不读外部/tmp文件，通过read补读当前完整serve_hpsv3_video.py及tests原manifest函数2405起至末尾的H1全部回归。历史G0/I1/P1/F0不误算为H1新增。提交前时序/四文件stat依已存收据，未越权执行非白名单Git命令。

H1复核：
- strict bool/default off、规范非零单UUID/CVD与实际logical0/current/observed UUID，未知设备/位置不复用；无环境改写或按型号猜余量。
- cold/warm均真实free准入；warm低headroom/TTL先卸自己的模型、释放unused后重新读真实free，未虚构free+weight，未CPU推理fallback/POST重试。
- 同一个冻结eval模型，原dtype/124逐帧reward/inference_mode/FP32标量/top37/Handler JSON/JPEG/identity验证保留。健康facts含声明策略、自然新service identity，ready时状态计数不改身份，旧manifest拒绝覆盖。
- monitor覆盖retain/offload，stop/join及late monitor失败传播；业务错误优先、清理notes、未知释放失败unready不盲重放。无法确认实际释放不是正常成功，真实驱动永久阻塞仍需未来执行者终止owned进程，不假称硬实时清理。
- 同一HTTP线程idle60s/poll，active不卸、入口再验TTL，main finally关闭server/backend，default off无idle CUDA观察。
- CPU测试实际构造/loaded hash/_score/离线Handler和原wire validator，合成CUDA位置叶子但非整路径mock；off/warm完整FP32逐帧/视频字节、top37含ties、mixed dtype/输入/RNG一致，覆盖UUID/free/TTL/monitor/日志/转移/双清理等错误、unready及main finally。

未发现需先修复的H1已知正确性缺口；接受Simplifier无必要简化的决定。CPU不证明真实GPU等价/资源/TTL收益，H1仍默认off，实际新服务/manifest/endpoint未授权。

## Planner实跑门禁

固定短入口实际：**7 passed, 3 warnings in 51.96s**。三警告仍为DCP单进程save/load及TypedStorage，原七函数/当前worktree导入/Git证明和G0/I1/P1/F0全部保持。Coder提交前53.69s、Simplifier51.32s的7/7已读，非Planner本轮新测时间；所有CPU时间都不作训练speedup证据。

初始status仅原.pi/；不读内部/清理/stage。Planner没有写训练源码。

## 下一项真实源码与成本分析

重读M0-ANALYSIS与当前decoder/adapter/binding/dual reward/hash实现：
- video load总232.2621479s，8个首load63.7945410s，因此owner内删24次video reload上界168.467607s≈2.808min（decode子段、不含优化/保存、未扣资格/资源管理）。audio额外仅约0.480min且有weight-norm状态，选择video-only避免pair复杂度。
- H1后decode预计变成critical path；使用旧调度反事实而非新GPU数据，含优化/校验/保存、不含初始化等的约104.25min条件分母，最多减至101.44min、增量约1.028×。首cold/争用/回退会削弱收益，不承诺。
- 原decoder每sample视频/audio分别load/drop，标准lease/handoff只能持CPU媒体；真正复用须scope保留模型且在下一owner/model前drop，不能改成全run常驻。
- 缓存最重要的科学边界是loader可能消费RNG。不能靠“固定权重应该无随机”猜；首load若改变全局流，本owner不复用且不restore/replay，保持原load次数/实际状态。支持固定stateless视频模型，注册状态变化/资格不足拒绝热复用。
- 缓存最重要的收益边界是**不要为cache每scope重复hash大权重**。实读 `_hash_file_record` 已在原完整哈希中stream content_digest，后合入global digest再丢弃per-file值；可选只读observer同遍输出视频凭据可供失效，保留原全部强门/返回identity，私有stat/ctime不入semantic/checkpoint合同。
- headers目前NA不阻塞保守首load实际字节/free资格判断；保留原每decode40GiB/16GiB，并在保留视频时为audio cold加载再守真实free。资源不够先drop自有cache再cold，不能降低阈值/换dtype或CPU decode。

## 本轮唯一计划V1v

PLAN.md规定默认关闭、单owner/最多单视频模型/逻辑16GiB且真实free门优先、无跨更新缓存；串行或P1同一work内scope，consumer join后释放。hash observer/真实binding凭据/RNG资格/预算与生命周期/离线回归都是此单元必需组成，不混audio缓存、P2或通信变更。

允许窄文件范围已列明；必要测试须同功能原子commit，先原七项7/7及diff检查，Simplifier再独立复审。已有init/checkpoint全model/code/cache校验不减少；observer不增加大权重读取、不提供verified绕过。原P1媒体容量、H1服务、F0窗口与所有科学数学保持。

M0已可用，仍不等待同一输入；GPU/SSH/HPS/服务/训练/推理/部署从未执行，Euler-B暂停。P2/FSDP等仍开放，最终integration/perf发布及最终GPU两更新/保存恢复/严格等价验收待另批，当前不写FINAL/不称2倍。

## 报告检查

Planner本轮仅更新PLAN/CANDIDATES和round-007，保留原.pi/。写完后实际git diff --check空输出通过；最终status仅上述Planner文件和原.pi/，无源码修改，不宣称整树clean。另在候选表补记：D2约32.91GiB静态常驻即与原40GiB VAE free准入冲突，不能以改mesh/删门解决。返回implement。
