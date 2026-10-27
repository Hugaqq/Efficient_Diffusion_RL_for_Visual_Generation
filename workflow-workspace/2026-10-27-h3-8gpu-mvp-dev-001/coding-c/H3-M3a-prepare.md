# H3-M3a-prepare — Coding C

## 身份、范围与判定

- `implementation_ready: true`，仅指当前 prepare 原子项，交 simplifier-c 审核，**不是 MVP 完成或 GPU 通过**。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-3bc7567a-9042-4d85-a3c3-9f22a1647954`。
- 开始/结束 HEAD：`44918392c5903cf54c9d8af389317814b60d3e04`；引擎分配的隔离 worktree，初始干净。integration_parent b7f9385 与 loader 919b6ab 在实际 git log 中；未自行提交、切分支或合并。
- 仅新增：`visual_rl/models/lifecycle/h3_fsdp.py`、`tests/test_h3_mvp_cpu_prepare.py`、本报告。未改其他配对文件、共享 H3 测试、Flash 七项、旧 runtime/配置、loader 或 prepared 基础类。
- 已读 task/plan/assignment C、AGENTS、loader/forward/prepared/components 与原 tiny helper；无既有 simplifier-c 反馈目录。重读 UniRL Apache-2.0 LICENSE 全文及 fsdp/wrap.py、state.py；独立组合公开 PyTorch API，未 vendor/import reference。

## 实际入口与下游合同

`prepare_h3_fsdp(meta, *, process_group, device, rank, local_rank, local_world_size, host_weight_limit_bytes, init_seed, sp=1, tp=1, lr=1e-4, betas=(.9,.999), eps=1e-12, weight_decay=0., constant_scheduler=False) -> H3PreparedModel`

1. 调用者负责预先初始化并选择 device。本入口只接受实际已初始化 NCCL WORLD、world_size8、实际 rank 与参数相符、local_rank==rank、local_world_size8、SP1/TP1、显式 cuda:local_rank 且为当前 CUDA device。不选 UUID、不初始化或销毁 PG，不做资源授权推断。
2. 重 API lazy 导入并检查必需签名；不支持时在权重读取/包装/重 collective 前拒绝。要求全参数 meta、loader 精度/shape/state 一致、trainable 恰是主 blocks 内 LoRA、内部 gradient_checkpointing 关闭、真实非持久 `rope.inv_freq`。
3. 先 all_gather_object 一个固定大小 receipt（rank、hostname SHA、模型/初始化 SHA），要求八条逐 rank 一致。摘要包含 config、完整 state shape/dtype、base 映射、blocks/targets/LoRA、实际 LoRA rank/alpha/scaling、rope 值、seed、预算、optimizer/scheduler 设置；不 gather 权重。节点或元数据不一致早拒。
4. 实际 rank0 才调用真实 `load_h3_cpu_state`，完整 base+LoRA CPU 输入；其他 rank 不读权重。一次 prepare attempt 标记不可复用，部分失败由上层结束整个 attempt，不能拿已部分包装模型重试。
5. `DeviceMesh.from_group(WORLD, 'cuda', mesh_dim_names=('dp_shard',))` 复用单 WORLD；每个 loader block 先真实 `checkpoint_wrapper(NO_REENTRANT)`，再对 wrapper `fully_shard`。FSDP 位于 AC 外，`reshard_after_forward=True`、param BF16、reduce FP32、cast_forward_inputs=False。每个 block 检查 FSDPModule/DTensor，root/refiner 不 shard。没有 CPU/伪 DTensor 生产 fallback。
6. 检查 canonical state FQN 不变后才 `model.to_empty(device)`；然后**每个 rank** 调用真实 DCP `set_model_state_dict(full_state_dict=True, broadcast_from_rank0=True, strict=True)`，rank0 完整 state、其他 rank 空 dict。未用 assign=True 作生产分片 setter。
7. 显式恢复 `rope.inv_freq`，核对 dtype/shape/device/逐值相等；其各 rank 初值由先前摘要准入保证一致。持久 buffer 随严格完整 state 加载。检查物化后的全部 state FQN/shape/device/dtype 和精确 LoRA trainable 范围。CheckpointWrapper 对 state_dict 隐藏的 `_checkpoint_wrapped_module` 只对安装的 block 路径作精确 named_parameters 映射，不放宽 strict。
8. 最后 model.eval()（不禁止 grad），才构造真实 PreparedModelBundle、AdamW，复用 `validate_optimizer_parameter_subset` 检查 optimizer 恰持有最终 trainable 参数对象。可选 LambdaLR 是常数1；无预先 ParameterStateManager、无 Accelerator.prepare 或第二次分布式包装。

`H3PreparedModel` 是既有 `PreparedComponentHandle` 的窄子类：可直接作为 `h3_t2va_forward` 的 callable，提供既有 optimizer/scheduler/prepared_root/close，并暴露最终 trainable_named_parameters（实际 bundle/wrapper FQN）。复用原 forward guard，raw module `__call__` 旁路被拒；close 幂等，只释放本地 guard/引用，不 collective/关 PG。`accumulate` backend 仅提供同步 backward 的上下文能力，不冒充完整 Accelerator，不做 no_sync、loss 缩放、step bookkeeping；world loss/global norm 是后续任务。

## CPU 真实证据与替身边界

新增10项，复用原测试模块只读 tiny fixture（原文件未改）：

- 实际本 worktree 导入身份；真实已安装 FSDP2/DCP/AC 必需签名准入。
- 对实际 FSDPParam/FSDPParamGroup dtype 方法作无 PG 的 CPU probe：BF16 frozen + FP32 trainable 在指定 BF16 forward/FP32 reduce policy 下被当前安装版本的 dtype 验证接受。仅用普通 CPU Parameter 设置 dtype 元数据，未构造 DTensor、未运行 FSDP forward，**不证明 CUDA mixed-dtype 行为**。
- 真实 Accelerate meta/PEFT/safetensors loader、真实 nonreentrant checkpoint_wrapper、真实 to_empty、严格 CPU assign=True setter 叶子替身、最终真实 AdamW/LambdaLR、prepared guard 和实际 H3 bridge 双模态 forward/backward/optimizer step。
- 事件顺序：cuda准入→metadata→rank0 read（peer 无 read）→单 mesh→AC/shard 每块→物化/setter→buffer恢复/最终校验→optimizer。setter 明确破坏 rope 为 NaN，生产恢复成功才允许 optimizer。最终 optimizer 参数身份与物化后 requires_grad 参数集合完全相等、不含旧 meta 对象。
- 双模态各自 backward 均产生真实 LoRA 梯度，base 全冻结；Adam 改变最终参数，scheduler lr 保持不变。
- rank1 模拟使用空 state 进入同 setter，fixture 预先提供模拟 rank0 payload；对照最终所有 tensor 和 rope。这是单进程控制测试，**不是八 rank 实测**。
- raw forward/二次 prepare、非法 WORLD/rank/device/SP/TP、跨节点/元数据不一致、stray trainable、缺失/损坏 buffer、内部 AC、物化模型、非法超参、缺失权重 state、unsupported API、setter 失败/最终 precision 错误均拒绝。失败不创建 optimizer 或暗加 cleanup collective。
- 单独调用实际生产叶子函数，拦截其 PyTorch 调用验证精确 mesh/FSDP/DCP 参数；no-op fully_shard 被真实生产后置检查拒绝。

替身明确限于 CUDA device 准入、WORLD 观测/metadata exchange、mesh、FSDP shard、DCP collective setter，以及重 H3 class。未 mock 整个 prepare、loader、AC、optimizer、guard 或 bridge。当前环境实际 dtype probe 最初 fixture 缺 sharded_param 导致一次 `22 passed, 1 failed`；改用真实 FSDPParam dtype 方法和 CPU 参数，随后全部通过，没有 skip/xfail，原22项和原七项断言未改。

## 最后源码编辑后的两组门禁

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# exit 0: 7 passed in 12.32s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# exit 0: 32 passed in 2.74s (本隔离分支原22项 + C新增10项)
```

未跑全量。`git diff --check` exit0；`git diff` 无已有受跟踪文件改动，新增文件尚未暂存，不以空 diff 冒充新增文件检查。已 read 复读实际源码；status 仅上述新增文件及报告目录。

`git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` exit0/无输出，结合当前无已有文件改动，原七项文件相对该历史基线不变。

## 待审与未完成

请 simplifier-c 审公开 API 顺序/AC 外层分片、canonical FQN、单 owner/全rank setter、buffer恢复、最终 optimizer 身份和 guard 复用。changes 回 C 修复并重跑两集；引擎 snapshot SHA 待生成，orchestrator 登记与汇合后全通配符复测。

当前仅提供 prepare 原子入口，尚未接 controller/launcher；world 计划/数学/global norm、评分/完整 tail/checkpoint/fresh 恢复不在本项。真实 H3 权重、CUDA FSDP2/DTensor 物化、NCCL 广播与多卡 forward/backward/AC 行为均待授权环境验证。CPU dtype/API probe 不替代这些验证。未执行 GPU、PG 初始化、多进程网络、服务器、下载、升级、push。**MVP 未完成，GPU_PENDING。**
