# Round 001 — 恢复工作流worktree内的原七项CPU门禁

## 决策

本轮唯一单元为 **G0：测试夹具路径隔离修复**，不是性能优化。返回 `implement`。首次审计后本拟优先VAE复用，但实际固定门禁失败；先修复可确定的普通测试问题，不等待外部、不夹带训练优化。

起点HEAD `86a628cb58ffe558fc91da9daf9fcff2946398a9`。相对生产基线的`visual_rl scripts tests/test_flash_collection_runtime.py` diff为空；尚无Coder/Simplifier实现可审核。

## 真实故障与收益依据

Planner实际执行原固定命令：`1 failed, 6 passed, 3 warnings in 46.53s`。失败函数 `test_approved140_candidate_compiles_typed_cadence`，line1321第一次 `launch.prepare_output(tmp_path / 'owned', ...)`；`tmp_path`由工作流置于当前源码树`.pi/workflow-tmp/pytest-of-user/...`。生产 `scripts/launch_flash140.py:233` 正确抛出 `RuntimeError: runtime outputs must stay outside the clean source checkout`。

这不是训练运行失败，也不是硬件阻塞。`inputs/bootstrap-cpu7.txt`在此前环境7 passed/46.85s，不证明当前managed worktree门禁通过。恢复门禁无训练speedup，但为所有后续原子commit必要前置。

## 修改边界（只改一个文件）

允许：`tests/test_flash_collection_runtime.py` **现有** `test_approved140_candidate_compiles_typed_cadence`内部路径夹具与对应回归。Coder另写自己的round-001报告/commit message。

禁止：改生产launcher/ROOT计算/输出准入规则；改pytest全局配置/依赖/AGENTS/.gitignore/.pi；移动临时输出到worktree外；改其它测试数量、skip/xfail/参数化；合入VAE、流水线或checkpoint优化。

建议实现：
1. 使用当前`tmp_path`内一个明确的模拟源码根，和它的**兄弟**输出目录。用`monkeypatch.context()`仅在调用真实`launch.prepare_output`的夹具区间临时替换 `launch.ROOT`。这模拟真实“源码与输出分离”部署，不模拟/替换`prepare_output`自身。
2. 原函数中有两段输出准入调用：首次`owned`生命周期，及末段`original/run-002`修复owner生命周期；两段都必须隔离，不只修首个异常。所有创建仍在当前授权worktree的测试临时目录。
3. `launch.preset_config`与 `repair.git_read/prove_launcher_upgrade`的真实Git/config证明必须使用**实际checkout ROOT**，不能因为长作用域patch而转到伪源码根。保存真实root并限定patch区间；不要伪造真实Git证明或放松历史升级allowlist。
4. 增强同一原函数内的拒绝覆盖：模拟root本身和其子路径均被真实`prepare_output`拒绝，拒绝后无owner/输出副作用；兄弟输出可正常fresh/resume/retry；保留既有owner mismatch/latest/orphan/terminal全部断言。可另验证实际root内部路径在未patch时也拒绝，且不创建目录。
5. 保持函数后续其它断言不变；若运行暴露同源夹具问题，限于本文件内同一逻辑单元修复并记录。若需要扩至生产改动，先报告，不擅自扩项。

## 不变量

- 仍七个原测试函数，仍完整运行固定命令，不跳过/替换任何原功能断言。
- 生产源码、科学协议、数据/RNG/梯度/精度/更新/保存恢复、路径/owner安全门全不变。
- 原Euler-B暂停；不训练/推理、不SSH、不请求HPS、不服务变更。
- `.pi/`在Planner门禁后变为untracked，是命令产生的工作流临时资料；**保留、不读取内部/不清理/不提交、不以.gitignore掩盖**。只用授权显式`git add`提交所属文件。报告树状态，不宣称clean。

## 测试及验收

唯一例行CPU门禁（每个代码commit前必须实际全通过）：

```text
PATH=/home/user/projects/h3-visual-rl/.scratch/h3-media-tools:/usr/local/bin:/usr/bin:/bin VISUALRL_FLASH_AUTHOR_DATA=/home/user/projects/h3-visual-rl/.scratch/flash-grpo/source/dataset/video PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/user/projects/h3-visual-rl/.scratch/h3-cpu-venv/bin/python -m pytest -q tests/test_flash_collection_runtime.py
```

同时`git diff --check`；检查真实diff仅本测试逻辑及自己报告，不扩大staging。导入当前checkout的原有断言保持并执行。记录完整结果、commit SHA与剩余未验证项。

不重跑GPU或其它套件；不是CPU性能benchmark。原门禁后遇到普通错误自行诊断，不包装为waiting_external。

## 风险/失败场景

- 整个函数patch ROOT会破坏后续真实Git/历史config证明，必须局部作用域。
- 只修第一次prepare_output会在末段run-002再次失败。
- 把生产保护改成允许测试/`.pi`路径、捕获异常后pass、改tmpdir到外部、skip均不接受。
- 测试结果可能生成工具内部未跟踪目录；不处理不属于本轮writePaths的副作用。

## 预期原子commit与交接

Coder一个原子commit，建议标题 `fix(tests): isolate launcher output fixtures in managed worktrees`；必要回归与修复同commit，先7/7及diff检查，不amend/rebase/squash。Simplifier独立复审；只有必要且等价的简化才另commit，同样先7/7，不造空commit。

下轮Planner必须读实际SHA/diff/test证据，确认G0没有放松安全门，再选V1（有界VAE复用）或P1/I1等单一独立项。当前没有已实现speedup；无需FINAL或宣告耗尽。
