# Round 002 — 同一checkpoint安全边界复用一次完整live校验结果

## 决策与前置结论

唯一单元 **I1：一次新鲜校验同时产出boundary与live bindings**。返回implement；不和VAE复用/流水/FSDP/profile混做。

G0已复审：Coder原子commit `e324aaf`，Simplifier无必要修改且无commit。Planner在HEAD `78fb5c544442df51369b727b00f7677bc38daba3`实际检查累计源码diff：仅原测试函数局部ROOT夹具，生产零diff；固定原七项 **7 passed, 3 warnings in 47.76s**。无正确性遗留阻塞。

## 实际代码与收益依据

- `visual_rl/runtime/h3_binding.py::checkpoint_boundary`：检查同一stages/prepared，读取已提交tail状态，调用完整`validate_live()`；随后检查Adam/scheduler、捕获RNG、构造progress/proof/data并返回四元组。
- `visual_rl/runtime/h3_lifecycle.py::run`：同一个`checkpoint_validation` span、同一个`_phase(...,'checkpoint-boundary',...)`中，紧接再次`bound.binding.validate_live()`拿live，之后立即交原`H3CheckpointTransaction.checkpoint`。
- 两次之间没有模型forward/optimizer/scheduler/data更新，只有只读边界构造。`validate_live`每次又完整stream模型/code、`preflight_h3_profile`全缓存finite/身份、source/model/numerics等；不能改成“信任初始化hash”。
- `inputs/baseline-MEMORY-REPORT.md`第二轮整个validation1007.942s≈16.8min，**没有单次validate耗时**。若两个相同全扫描各占约一半，可省约8.4min/保存轮：127.73→119.33min条件约1.07×（分母含采样/评分/优化/校验/保存、不含初始化等）。旧两步每步诊断保存；正式cadence5不能说每update省8.4min，稳态仅假设摊1.68min/update，111.08min含优化/摊销校验保存分母约1.5%。实际收益可能更低，未有GPU/IO实测，不宣称实现速度目标。
- 选在VAE前：无需额外CPU/GPU权重驻留、RNG loader或新offload假设，当前重复调用可直接确证；VAE/P1仍是开放高优先后续项，不是已排除。

## 修改范围

仅此单元必要文件：
1. `visual_rl/runtime/h3_binding.py`：提供一次调用产生 `(boundary, live)` 的窄API（建议`checkpoint_snapshot(stages, trainer)`；boundary仍是既有progress/proof/data/rng四元组）。
2. `visual_rl/runtime/h3_lifecycle.py`：仅正常run保存边界切到该API，原_phase/all-rank错误交换、保存条件、memory spans、原transaction调用保留。
3. `tests/test_flash_collection_runtime.py`：扩展原七个函数内部的必要覆盖，不新增/替换/skip/参数化函数。允许复用现有H3 CPU fixtures，只替换原本CUDA/重模型/HTTP叶子；不编辑其它测试文件。
4. Coder自己的round-002报告/commit message。

不得修改transaction/codec/schema/recipe/dependencies/任务授权；不做跨边界模型或embedding缓存。若确需超出这三个代码/测试文件才能保持正确，先报告而非扩大本轮优化。

## API/校验合同

- 所有现有`checkpoint_boundary(stages,trainer)`调用继续获得原四元组，仍是自校验、失败关闭的API；**不能**把它改成不校验的低层接口。可委托新snapshot API并丢弃live；不留两套复制实现。
- 新API必须自己完整执行`validate_live()`一次并返回**该次真实返回值**，不能取`_live_ids()`/`self.content`冒充已验证结果，不能接受caller给的live/verified/token。
- 每次调用/每个保存边界都重新完整验证；无`self.last_live`等持久cache、无跨更新receipt、无可选跳过校验flag。
- 建议先做合法tail/foreign binding检查，再完整验证得到局部live，最后做原有Adam/scheduler/RNG/progress/proof/data构建，并把两者一同返回。这是保留旧内层校验顺序的最窄重构；不需要“先算boundary再在lifecycle第二遍验证”。
- 在长校验前后对tail/source边界做**廉价只读一致性复核**，确认source state/step/behavior/next batch不变，拒绝校验期间推进或新reservation；不是再跑全内容扫描。必要时轻量检查完整对象所有权仍一致。不得把外部可变嵌套引用直接当历史快照比较（用独立JSON/deepcopy快照）。这里的结果只在同步run里立即使用，无外部异步复用。
- 保留原foreign stages/prepared、tail/清空grad、Adam complete counter、scheduler、selection provenance、RNG identity/proof ready等断言及所有full `validate_live`子检查；不修改其含义、模型数据hash、finite扫描、HPS manifest检查。
- 原验证本身不提供文件系统锁/原子全盘snapshot。新实现同样不能声称能阻止任意并发写盘；完整hash前后检查保留，后续边界仍全重验，transaction文件/参数证明仍保留。任何已检测内容漂移必须抛错、不能重试吞掉。

## 必要CPU回归（放入原函数）

### 真实binding层
- 用现有CPU binding/source/stages fixture完成真实小迭代，调用新API；`validate_live`用**委托真实实现**的spy计数：每次snapshot恰好一次；旧boundary API每次也恰好一次，返回格式、progress/proof/data/RNG与不变状态时旧路径相等。
- 连续两个合法更新边界各自再验（计数递增），不能跨更新复用live；同一边界重复调用也重新验，非memoization。验证只读：Python/NumPy/Torch RNG（CPU测试可观测部分）及模型/Adam/source不变，不probe `.step()`。
- 非法pre-tail、foreign binding、残留grad、counter/scheduler不符仍失败关闭。
- 真改模型/VAE文件、cache index或HPS manifest中至少一个受保护artifact：下一次snapshot必须被真实检查拒绝；不要仅靠mock raise证明强检查保留。沿用已读fixture的drift模式可避免无关巨大模型。
- 故障注入：在真实校验成功后、返回前修改source cursor/behavior或打开reservation，要求廉价复核拒绝且不产生保存receipt；禁止把此测试做成自动修复/回滚生产状态。

### 实际lifecycle分支
- 在原七项内用CPU fake trainer/transaction叶子驱动真实`H3LifecycleBackend.run`保存分支（可在一个tiny两步序列验证每个边界），确认run只调用新snapshot一次/边界，且把该次live精确交给原transaction接口，**没有额外validate_live调用**。
- 验证snapshot抛错后transaction未被调用；保留_phase错误传播。不要只靠字符串搜索/源码assert证明已接通。
- 保留restore-prepared/restore-bound/final-file完整性路径不变；不将新snapshot扩用于restore门禁。

这些是同一优化的功能回归，必须与实现同commit。CPU fake计数证明调用/失败边界，不证明每次扫描节省几分钟或NCCL正确性。

## 固定协议/安全不变量

480×320、124帧24fps、10步first5选1、4×8、LoRA32/64、lr1e-4、精度、reward124/top37、优势/Flash/缩放/逐样本反向释放、随机流与checkpoint进度/保存恢复数学均不变。无新线程/驻留/queue，无GPU或服务操作；原Euler-B保持暂停。正式cadence5及首个/terminal/pause特例不变。

## 验收与原子commit

每个代码commit前实际运行：

```text
PATH=/home/user/projects/h3-visual-rl/.scratch/h3-media-tools:/usr/local/bin:/usr/bin:/bin VISUALRL_FLASH_AUTHOR_DATA=/home/user/projects/h3-visual-rl/.scratch/flash-grpo/source/dataset/video PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/user/projects/h3-visual-rl/.scratch/h3-cpu-venv/bin/python -m pytest -q tests/test_flash_collection_runtime.py
```

必须7/7，当前worktree导入断言保留；`git diff --check`通过，真实diff仅本单元。记录实际输出、diff、SHA与GPU未验证。保留未跟踪`.pi/`，不访问/清理/提交，不以ignore隐藏，不宣称clean。

Coder建议原子标题 `perf(h3): share fresh live validation at checkpoint boundaries`；功能+必要测试一起，先测后commit，不amend/rebase/squash。Simplifier独立复审；必要等价简化另commit并先7/7，不造空commit。Planner下一轮再核真实diff/数学/测试，并继续VAE/流水等候选；本轮不是结束或GPU性能证明。
