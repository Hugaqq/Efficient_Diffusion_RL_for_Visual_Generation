# H3-M4a4-object-integration — Simplifier A / W4

verdict=approved

返工清单：无。A 本项满足三真实对象同链 CPU 验收要求，可交 orchestrator 关闭此前 W3 组合证据缺口；不代表真实 FSDP2/GPU、生产 RunController 装配或完整 MVP 完成，不代替 B/C 本波审核。

## 身份与范围

- 当前 HEAD：`fb085fb35fb239cdd4c151d39decbd7fae268c9d`。
- 审核分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-daf8a51f-8100-46ae-8769-51eb2030e7e8`；开始 status 干净。
- 已读取当前 task/plan/assignment A、Coding A 交付、完整 `tests/test_h3_mvp_cpu_integration.py`，核对当前 stages、coordination、HPS acquire/score/close 与复用 HTTP helper 的实际代码。
- 本角色仅写审查记录，未修改生产源码、测试或其他报告。A 本项交付限新增测试及交付记录，无需新增生产框架。

## 实现与证据核对

### 真正组合，而非高层替身

`_setup`/`_run` 为八个模拟 rank 分别构造并运行真实 H3IterationStages→GRPOTrainer/BaseTrainer、H3HPSReward、H3Coordination；评分绑定真实 `owner.scorer.score_one`，协调 factory 返回真实 `H3Coordination`。没有使用旧 `_CoordinationLeaf`，没有用 lambda reward 或本地 clip_grad_norm_ 替代生产对象。stage 观察 wrapper 仍调用原 stage 并检查同一 identity。

真实 pool/CPS/anchor/world credit/objective、prepared guard、AdamW/LambdaLR、decoder lease、原 HPS factory/client/JPEG/validators 在同链执行。8 个 worker 都运行完整数据和阶段工作，不是复制 rank0 receipt。独立 sample/group、条件绑定和两轮 behavior/data/next-step 连续推进均被断言。

### 替身严格定位与有界性

- `_MemoryWorld` 只代替 WORLD/CUDA 观察及通信叶子，按 rank 操作序号和 kind 匹配，20秒等待/90秒 future 超时；失配或非协作异常使等待者失败，测试不会把错误当成功。
- `_ToyGather` 是 toy 模型内的 weight gather/平均 reduce-scatter autograd 叶子，不执行阶段控制或优化器操作。每格真实 backward 穿过它。
- `_dtensor_parts` 仅给本 rank 真实参数/梯度提供逻辑 shard 观察；生产 `_trainables`、`_local_gradient` 和 norm 数学未替换。不是实际 DTensor/FSDP2/AC/NCCL 证据。
- H3/双 VAE 仅重量级模型叶子 tiny；HPS helper `_Transport` 实际解码收到的124张 JPEG 后生成变化分数，原 wrapper/校验器真实执行，无 socket。

这些是本项需要的测试仿真，不进入生产代码；无第二训练 runner 或通用调度器。测试不更改共享线程数/环境。

### 四个 cases 的实质断言

- 正常与零信用两例均执行8rank×2次完整 iteration，K2/N2含末格q0。共同事件 identity/sequence 一致；完整 rollout→aux0..7→完整 anchor→逐格 current/backward→norm→optimizer→tail 的顺序及各 rank no_grad/grad、样本条件、每格调用均验证。
- aux work 实际峰值并发1、顺序严格0..7，每 sample 一次 POST/all124/top37，媒体不进入 receipt gather；client/response 恰当关闭，媒体弱引用释放，池仅在 tail 后 released。
- 信用参考来自实际 JPEG-derived frame scores，手算top37、完整world std/组均值；norm/clip参考来自归约前各rank backward vectors，而非照抄 norm 输入标量。正常例双模态参数有限变化；零信用由相同真实媒体得到，仍真实 backward、norm0、Adam step2、scheduler/data tail，不靠生产零loss占位。
- rank3 HPS sample_id 协议错误由原 validator 拒绝；rank3 current第二格 NaN 在 backward 前 gate 共同拒绝。八rank全部得到真实 H3CoordinationError，禁止step/tail/norm，旧reservation abort一次、next-step0、Adam仍空、poison不可重试，清理/重试不新增 collective。raw model collective失败仍明确依赖上层 teardown，未虚称可恢复。

## 当前汇合代码独立 CPU 门禁

原样执行以下允许命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

- Flash exit0：**7 passed in 12.54s**。
- H3通配符 exit0：**108 passed, 9 warnings in 6.94s**（此前93+A4+B4+C7）。四个组合cases实际执行，不以A分支97项历史结果替代。
- 9条 warnings 来自本波 C checkpoint tests 的明确单进程 DCP save/load 和损坏fixture覆盖提示；未过滤。本报告不据此批准 C codec。
- 目标文件无 skip/xfail 匹配，未跑全量。
- `git diff` 无输出；`git diff --check` exit0。
- 原 Flash 完整文件相对 `d818b8a231b52a9a024fa2f522057acdbb7bf099` 的获准历史 diff 无输出、exit0，七项及全部断言保持。

## 结论边界

此前“真实 stages+HPSReward+H3Coordination 未在同链执行”的 **CPU 组合证据缺口已由本项实测补足**，由 orchestrator 正式登记接收。本测试仍使用显式数据 fixture 和模型/通信叶子，不是默认 runtime/controller/recipe 装配或完整 checkpoint 发布/fresh恢复证据。后续必须完成这些必需生产环节，并做最终 candidate 两集门禁。

没有真实HTTP/PG/GPU、服务器、下载、环境升级或push。video-only HPS 不证明音频质量；真实权重、HPS124资源和CUDA/FSDP2/NCCL仍未验证。**GPU_PENDING，MVP 未完成。**
