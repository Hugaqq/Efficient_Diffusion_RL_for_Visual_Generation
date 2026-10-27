# H3-M4c1-runtime-binding-r1 — A-BIND-01

implementation_ready=true

本次仅完成原配对返工，交 simplifier-a 复审；不自行关闭 issue 或批准 binding。原交付和 changes 报告保留。GPU_PENDING。

## 实际身份 / 范围

- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-25da9cef-5a0f-40d1-bc85-63c30c9a504d`。
- 开始/结束源码基线 HEAD：`cc6e19f7d35a34d443fe7942a3c33e00ea7bb631`；开始 status 干净。assignment integration_parent 为 d7c75b7，不将其冒称当前 HEAD。
- 已重读 task.md、plan.md、assignments/a.md、原 simplifier-a/H3-M4c1-runtime-binding.md、实际 binding/prepare/prepared handle/codec/stages 与测试。
- 仅修改 `visual_rl/runtime/h3_binding.py`、`tests/test_h3_mvp_cpu_binding.py`，新增本报告。未修改公开接口、fixture 接口、原七项、兄弟角色文件或其他生产 owner。
- 未自行提交/切分支；引擎后续形成 snapshot。

## 最小实现 / issue 对照

统一 `_observe_model` 调用 `_observe_scheduler`，初始构造、每次受控 scope/forward、validate_live、post-tail checkpoint 复用同一校验：

1. scheduler 精确为 LambdaLR，且 `scheduler.optimizer is prepared.optimizer`。即使两个 optimizer 使用相同参数集合也拒绝错配。
2. 初次观察保存实际 scheduler/optimizer 引用，后续两对象实例均必须稳定。相同类型、相同 state、正确 optimizer 的新 scheduler 仍拒绝。
3. 只读观察现 prepare/codec 的 constant-LambdaLR 表示：固定 optimizer lr/initial_lr、scheduler base_lrs/_last_lr 均为 1e-4；last_epoch 为非负整数，_step_count=last_epoch+1；lr_lambdas 序列化状态为 `[None]`（不接受 codec 不支持的有状态 callable）。原 optimizer 全超参数 fingerprint 检查保留。
4. 冷 Adam 必须对应 epoch0；warm Adam 必须覆盖所有最终 optimizer 参数，各 scalar finite step 与 scheduler epoch 一致且为正。完整 checkpoint 仍由原边界额外比较 trainer/data/tail step，不在两阶段恢复尚未恢复 data 时增加错误的 source 计数依赖。
5. 校验不调用 LR callable、scheduler.step 或 optimizer.step；这是可观察常数状态及对象链验证，不是任意 Python callable 未来行为的形式化证明。不新增 scheduler 注册器或替换 prepare 创建逻辑。

## 实际测试

新增两项独立 binding cases：

- 初始外部 optimizer 的 scheduler 拒绝（外部 optimizer 恰使用同一参数集合）；base/current lr 和两个计数漂移拒绝；合法 cold bind/live 通过。校验期间两个 step 方法以失败 spy 禁止调用，模型调用数不变，两个 Adam 均仍为空。
- 合法一轮真实 stage/tail 后，分别注入 scheduler.optimizer 引用漂移、同状态同类型 scheduler 替换、base/current lr、last_epoch/_step_count、optimizer initial/current lr、单参数 Adam step 漂移。每种都在 live、post-tail checkpoint、scope 拒绝；恢复原值后完整边界逐值相等。另以真实 source 预留下一请求，逐项执行 binding.predict，确认在重模型 forward 前拒绝；模型调用数不变、step spy 零调用。

原合法 stage 测试扩展为连续两轮：真实 BaseTrainer/pool/CPS/anchor/Adam/scheduler/data tail 执行，验证 trainer/progress/proof/Adam/scheduler 都为 step2、cursor16、_step_count3、固定 lr1e-4；每轮 optimize 返回但 tail 未完成仍拒绝 checkpoint。原 cold/active/poison/未清 grad 等断言全部保留，active 测试使用实际下一步 reserve(2)。

测试沿用已明确的重模型/VAE、HTTP、CUDA/FSDP/collective 叶子替身；不替换 binding、scheduler、Adam 或 stage 控制。没有真实网络/PG/GPU 操作，也没有新 publisher 或 controller 实现。

## 最后源码编辑后的真实门禁

工作区根目录原样运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3 exit0：**137 passed, 18 warnings in 14.18s**，当前基线135+本项2；不包含W7尚未汇合的B/C新增项。DCP no-dist/损坏fixture覆盖 warnings 原样保留。
- Flash exit0：**7 passed in 12.53s**。原七项未改，无 skip/xfail、全量替代或 warning 过滤。本轮无失败门禁。
- `git diff --check` exit0。
- `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` 无输出，exit0；当前工作树 diff 也未修改该文件。
- 已读实际 diff：仅上述两文件功能/测试变化，所有旧断言保留。

## 待办与边界

原 simplifier-a 复审并明确 approved/changes；orchestrator 登记 snapshot，结合W7 B/C裁决和汇合两集门禁接收。本轮不批准兄弟任务、不声称默认唯一RunController闭环/整run恢复已完成。未执行GPU/服务器/HTTP/PG初始化/下载/环境升级/push；CPU不证明实际H3权重、FSDP2/NCCL、多卡恢复或音视频质量。GPU_PENDING。
