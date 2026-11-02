# Planner round-002 — G0独立验收与I1单元选择

## 上轮真实复审（不是只信交接）

当前HEAD `78fb5c544442df51369b727b00f7677bc38daba3`。`git log -12 --oneline`确认：Coder `e324aaf fix(tests): isolate launcher output fixtures in managed worktrees`，父`a71ea2f`；其后`b8f4ade`与`78fb5c5`为工作流快照，不是Simplifier代码commit。`git show --stat --oneline HEAD`显示最新快照只收录Simplifier的29行报告。

实读Coder round-001/commit收据、Simplifier round-001；实际执行完整生产基线授权源码diff。diff仅原 `test_approved140_candidate_compiles_typed_cadence` 内：
- 真实ROOT子输出拒绝且不创建；模拟root与子目录亦拒绝；
- owned生命周期局部ROOT context，保留fresh/retry/owner/latest/orphan/terminal原断言；
- 两个run-002 prepare_output分别局部context，真实Git/config证明不被patch污染；
- context退出均检查真实ROOT恢复。无visual_rl/scripts改动，无测试函数增删。

接受G0为独立必要前置，无已知正确性缺口。Simplifier无需抽helper或造空commit的决定合理。按当前可见Git log、累计diff和自身测试验证代码及commit存在；Coder暂存统计/提交前时序的详细证据来自其已存报告，Planner未越权运行未允许的git show <SHA>命令。

## Planner实际门禁

固定原七项命令在当前worktree实跑：

```text
....... [100%]
7 passed, 3 warnings in 47.76s
```

warnings为原有DCP单进程save/load与TypedStorage；没有skip。当前checkout导入断言与真实Git/config证明均随原函数执行通过。Coder报告提交前7/7/46.53s，Simplifier独立7/7/45.90s；CPU结果不证明GPU性能。

本轮起始status仅`?? .pi/`；保留不访问/清理/提交，不宣称clean。计划/报告写完后实际`git diff --check`空输出通过；最终status仅Planner的PLAN/CANDIDATES修改、新round-002报告及原`?? .pi/`，无源码修改。

## 本轮重新读取与成本审核

已再读TASK、保存baseline-MEMORY-REPORT/sharding-estimates/bootstrap CPU证据、候选表及最新 `h3_decode.py`、loader入口和真实decoder CPU fixture；VAE当前逐视频/音频各load/drop，fixture明确原先视频对象在audio加载前释放，复用需新增驻留/失效/RNG合同，不能以19.71GiB allocated峰推导安全。没有GPU/HTTP/SSH测量，8s加载尚未拆分为CPU构造/磁盘/H2D。

重新实读 `h3_binding.py` 保存边界及 `h3_lifecycle.py::run`：同一个同步保存分支连续两次完整validate_live，之间仅只读counter/proof/RNG构建。继续读transaction bindings/progress/data验证、RNG capture实现及现有真实CPU binding/bound-checkpoint/controller entry fixtures，确认可以用原七项内部扩展覆盖：新鲜一次验证产出两个结果、旧API兼容、lifecycle接通、跨边界重验和失败不保存。

## 002决策：I1

仅做同一安全边界的完整校验结果复用，使用新窄snapshot API返回boundary与本次真实live；旧boundary接口仍独立自校验，禁止跨边界cache/verified绕过。保留full hash/finite/model/source/HPS等覆盖与transaction/restore/最终文件安全门；补充长验证期间source/tail不变的廉价独立快照检查。方案见PLAN.md。

理由：这是真实可证的重复工作，不需额外GPU/CPU权重驻留或改变加载RNG；比立即缓存VAE的正确性/预算假设少。V1与P1并未消失，继续作为后续高优先项。

收益仅假设：整保存轮validation16.8min；若两次扫描大致等时、且主导该段，省约8.4min/保存轮；含优化/校验/保存的127.73min分母条件约1.07×，不是已经达到。正式cadence5需摊销（约1.68min/update；稳态111.08min含优化/摊销校验保存分母约1.5%）；旧两步每步诊断保存不可冒充正式cadence。实际节时下轮代码也不能仅凭CPU计数宣称测得。

## 交接与未完成

返回implement。Coder仅h3_binding.py、h3_lifecycle.py及原七项内部必要回归，功能/测试同原子commit；Simplifier随后独立审查。无GPU运行授权、未启动训练/推理/服务，Euler-B继续暂停。

G0完成≠代码交付完成。VAE复用/有界流水/FSDP测量和其它候选仍开放；本轮不写FINAL、不声称2倍。最终integration、专用perf分支发布及另行授权的两更新GPU验收尚未执行。
