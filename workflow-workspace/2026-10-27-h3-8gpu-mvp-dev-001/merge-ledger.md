# Merge ledger

## 当前集成事实

- 实际起点：`076635a61263ffd5e7c5be4cc64d77ef99242c51`；本轮实际HEAD / 集成基线：`b7f938506f16034ac5d2896ddfa44c16eb4af83d`。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`
- 初次及本轮接收时status均干净。M1a/M1b/M2a1/M2a2均已读实际代码、交付及配对approved；本轮核对完整loader和五项新增测试。无changes、无失败门禁待返工。
- 引擎历史由 `git log -12 --oneline` 核对：`f8e4d0e`（首轮协调）→ `7b29f4b`（M1a coding）→ `8a6763c`（M1a审查）→ `42a11a8`（第二轮协调）→ `725f48f`（M1b coding）→ `8ed954f`（M1b审查）→ `f8f83f7`（第三轮协调）→ `edd5c87`（M2a1 coding）→ `63fe34c`（M2a1审查）→ `282dd4a`（第四轮协调）→ `919b6ab`（M2a2 coding）→ `b7f9385`（M2a2审查，当前HEAD）。角色阶段由交付/审查HEAD和顺序结果支持；本轮新增授权的累计范围diff另见下。
- Orchestrator 本轮仍只改协调文件；不自行提交。当前原子项功能及 CPU 审核接受不等于最终 candidate 或用户 main 已发布。

| 任务 | Coding branch / snapshot | Simplifier | 测试/审查证据 | 集成/批准状态 |
|---|---|---|---|---|
| M0 baseline | 上述实际分支/起点 | 不适用，协调核对 | orchestrator 真实执行固定七项，exit0，7 passed in12.90s；git diff --check exit0 | baseline 已核对，不是 H3 实现 |
| H3-M1a-joint-schedule | 同上述分支；前置 f8e4d0e7f126cbd654ebe43c157902d506a0d37b；coding snapshot 7b29f4bf8d990794d4b9bc826a82bd390861d530 | approved；记录 snapshot 8a6763cf3defcb07415dfeda501204e630b42f5a | coding Flash7/11.98s、H3六项/0.65s；simplifier独立Flash7/12.19s、H3六项/0.58s；orchestrator本轮Flash7/12.73s、H3六项/0.64s，全exit0 | 功能/配对/CPU接受，前置合同而非完整MVP；历史范围审计限制见下 |
| H3-M1b-cps | 同上述分支；前置42a11a8c3fc6473aa52238473e20bc002e66db4c；coding snapshot725f48f4d67c67d90c60cc3536f4bc40f12802f2 | approved；记录snapshot8ed954f2562cf9d1f5432a7bb84ada181cdae6a9 | coding Flash7/12.12s、H3十二项/1.13s；simplifier Flash7/12.35s、H3十二项/1.03s；orchestrator Flash7/12.78s、H3十二项/0.88s，全exit0 | 功能/配对/CPU接受；非旧锚、非生产闭环 |
| H3-M2a1-forward | 同上述分支；前置f8f83f7064f121a00526241048e68ceda83eb4b1；coding snapshot edd5c874f10ffa917a71bd9f2fa9085247075a6e | approved；记录snapshot63fe34c99f96a115ef3cb743965aa4338e3653d6 | coding Flash7/12.78s、H3十七项/1.41s；simplifier Flash7/13.10s、H3十七项/1.20s；orchestrator Flash7/12.27s、H3十七项/0.91s，全exit0 | 功能/配对/CPU接受；tiny叶子桥接，不是权重/完整adapter |
| H3-M2a2-loader | 同上述分支；前置282dd4ad05bbabb8bdb8ddf301e11b057d6b0846；coding snapshot919b6abc7920f2dbcea764907152523882a09a59 | approved；审查snapshot b7f938506f16034ac5d2896ddfa44c16eb4af83d | coding Flash7/12.12s、H3二十二项/3.12s；simplifier Flash7/12.95s、H3二十二项/3.55s；orchestrator Flash7/13.05s、H3通配符22/3.36s，全exit0 | 功能/配对/CPU接受；非真实H3/FSDP证据 |
| H3-M4a1-anchor | W1-A；依赖现CPS；共同基线b7f938506f16034ac5d2896ddfa44c16eb4af83d；实际分支/SHA待引擎 | simplifier-a待审 | 未实现/测试 | active parallel，未批准 |
| H3-M2a3a-text-cache | W1-B；依赖现forward；共同基线同上；实际分支/SHA待引擎 | simplifier-b待审 | 未实现/测试 | active parallel，未批准 |
| H3-M3a-prepare | W1-C；依赖已批准loader；共同基线同上；实际分支/SHA待引擎 | simplifier-c待审 | 未实现/测试 | active parallel，未批准 |

## 首轮 baseline 命令（历史记录）

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

工具实际输出：`....... [100%]` / `7 passed in 12.90s`。H3 测试文件尚不存在，未执行 H3 命令。没有 full suite、GPU、服务器、部署、下载、环境升级或 push。

## 接收规则

每个新条目读取实际源码 diff、coding 命令结果、对应 simplifier verdict 后再定批准。引擎 run 内暂存合并不代表批准。changes 回原 coding 并重测重审；不得用历史 baseline 通过填充新源码测试证据。最终所有必须项批准后 orchestrator 亲跑两集、写 final；引擎独立 candidate 验证后才能安全发布本地 main。

## M1a接收检查与历史证据限制

- Orchestrator 已读 `coding-a/H3-M1a-joint-schedule.md`、`simplifier-a/H3-M1a-joint-schedule.md`、`models/h3_joint.py`、`algorithms/dynamics/h3_schedule.py` 和完整 `tests/test_h3_mvp_cpu.py`。实际实现与已批准窄合同相符，无新增返工项。
- 在当前 HEAD 亲自原样跑 plan 中两条固定命令：Flash `7 passed in 12.73s`、H3 `6 passed in 0.64s`，均 exit0。当前原七命名函数存在，grep 无 skip/xfail；此事实不能替代逐断言历史比较。
- Coding 交付记录的最终 status/diff 声明是仅新增指定文件、既有跟踪文件未改；simplifier与本轮 status 均干净。仅据这些和实际源码审核接收 M1a 功能。
- 当时白名单不含历史diff，未以clean工作区冒称审计。**本轮已授权并执行d818b8a→HEAD累计范围及Flash完整diff，下节关闭原七项不变证据缺口**；不声称每个中间snapshot均做过patch比较，最终candidate仍需重查。

## M1b接收历史

- 读回 `h3_cps.py`、完整H3测试及 `coding-a/H3-M1b-cps.md` / `simplifier-a/H3-M1b-cps.md`。CPS实际消费M1a，唯一stats公式、FP32元素均值、detached FP16固定action、精确双schedule、末步q0/RNG与Gaussian边界符合本项；无新返工issue。
- 当前HEAD亲跑两条固定命令：Flash `7 passed in 12.78s`、H3 `12 passed in 0.88s`，均exit0。不是最终candidate测试，不证明GPU。
- 本轮交付报告记录既有diff仅H3测试214 insertions/1 deletion（import扩展），新CPS与交付文件在status；原六项断言保留。Orchestrator已读当前全部12项，历史比较限制仍未关闭。
- 下一项 B/H3-M2a1-forward：T2VA几何与attention packing、双t、真实签名单次callable、两流符号转换；现joint/schedule/CPS只读，B独占目标测试。A/C idle，不启并行。
- 读取上游bundle/vendor初始化与transformer签名、packing真实几何；首次猜测Flow-Factory `adapter.py`读到ENOENT，已用find定位并读取实际 `adapters.py`/`dependency.py`，非资源阻塞。Flow-Factory依赖文件提供Diffusers H3真实模块路径（要求>=0.40.0），只作后续API选型线索，不证明本机可导入。
- 当时find未返回VENDOR_COMMIT.txt，记录曾误写为快照缺文件。本轮直接read已成功，修正见下；不编造其来源或宣称这份历史文档证明当前依赖可用。

## M2a1接收历史

- 读 `coding-b/H3-M2a1-forward.md`、`simplifier-b/H3-M2a1-forward.md`、实际 `h3_geometry.py` / `h3_forward.py` 与五项新增测试。真实T2VA族几何、FP64 rotary、attention布局、双显式sigma、单callable和两流各一次符号转换与前项合同相符，无返工issue。
- 亲跑两条固定命令：Flash `7 passed in 12.27s`，H3 `17 passed in 0.91s`，均exit0。Coding记录H3测试仅追加163行、原断言无删除；这一历史范围声明来自交付报告，白名单缺历史diff的限制仍未关闭。
- 下一项B/H3-M2a2-loader，锁定真实lazy公开API、本地config/safetensors、参数meta/构造buffer保留、主blocks六target LoRA/FP32 master与精度岛、完整base FQN覆盖和显式CPU初态。只初始化owner读全CPU state；不宣称流式常量内存，不创建PG/FSDP/optimizer。
- 本轮查读实际 `common_diffusers.py::apply_peft_lora`（旧helper为suffix/gaussian）、`ParameterStateManager`（立即绑定Parameter身份）及上游真实transformer精度岛/构造buffer。故B不修改旧helper、不提前绑定参数；后续C须在最终分片物化后构造optimizer。
- A/C idle，B独占目标测试；无并行派工。未做环境升级/服务器/GPU/下载/push。真实H3可选依赖、模型资源与8UUID仍未验证。

## 本轮M2a2接收与W1并行

- 已读完整 `h3_loading.py`、五项loader测试及coding-b/simplifier-b的M2a2记录。实际meta/PEFT限定范围、FP32岛/非持久buffer、owner CPU全覆盖与预算实现符合本项；无返工issue。C必须恢复to_empty后的rope并最终绑定optimizer，不据bool owner声明推定分布式已实现。
- 独立真实门禁：Flash固定7项 `7 passed in 13.05s`；新通配符H3 `22 passed in 3.36s`，均exit0。目标尚只有原文件，W1后将扩展三个feature模块；最终必须通配符，不跑全量。
- 应用用户最新指令，实际派parallel W1：A冻结锚、B离线text cache、C FSDP2 prepare；全部active，每组一个原子项。文件/接口独立、同波无相互import，详见decisions/wave-01.md与assignments。原tests/test_h3_mvp_cpu.py和既有合同只读，不再以共享测试阻碍并行。

## 新授权历史审计（本轮实跑）

1. `git diff --name-status d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD` exit0：净变更为新增H3六源码模块、H3目标测试和workflow-workspace资料/报告；没有旧源码、旧实验配置、原Flash测试的M/D记录。
2. `git diff --stat d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD` exit0：97 files /23248 insertions；绝大部分是相对该更早baseline的只读参考/任务资料，**不将这些计为功能实现或本轮coding成果**。
3. `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` exit0，无输出：原七项文件含全部断言在当前HEAD与该baseline逐字相同。此前该证据缺口在当前HEAD关闭，最终再次核验。
4. 该累计diff不是每个中间snapshot patch，也不能据资料的A标记断言其全由本run引入。未改task/reference/workflow；目前允许命令不作额外未经授权历史遍历。

## 参考材料修正

本轮直接读 `reference/UniRL/unirl/models/minimax_h3/vendor/VENDOR_COMMIT.txt` 成功，记录源Diffusers PR14355 commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`，vendored 2026-08-03。前轮仅find无结果就推定缺失不成立，本轮修正；不推断文件何时出现。该文档的draft/导入验证为历史事实，且明确没有numerical parity；不能覆盖当前公开依赖准入与GPU_PENDING。无移植/参考修改。

状态：MVP 未完成；GPU_PENDING。
