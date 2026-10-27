# H3-M4b1-state-codec — Coding C / W4

## 判定 / 实际身份

- `implementation_ready: true`，仅本原子项交 simplifier-c；不是完整checkpoint/整run恢复完成。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-5dfce464-611a-49d3-b174-e07b3c2246c5`。
- 起始/结束 HEAD：`477325141c0911aeb3cdd8a85937a6843532be87`，初始干净。assignment integration_parent 为7cf28dced4c8193ccf9321d62b5b2e479bd904f7；未自行提交/merge/切分支。
- 仅新增 `visual_rl/artifacts/checkpoint/h3_state.py`、`tests/test_h3_mvp_cpu_checkpoint_state.py` 与本报告。旧manager/protocol/transaction/state/RNG、prepare/coordination、旧tests及A/B文件只读；不import同波新adapter/组合测试。
- 已读最新task/plan/assignment C/wave-04、配对W3 approved反馈、现checkpoint safe-point/progress/manager/state，以及现prepared/FQN/健康receipt合同。独立组合公开PyTorch API，无vendor移植。

## 实际给现manager的接口

```python
identity = H3StateIdentity(
    topology_id=..., model_id=..., semantics_id=..., group_geometry_id=...,
    world_size=8,
)
# 所有rank调用；safe_point是各rank真实proof，progress与identity是共同expected绑定。
receipt = save_h3_state(
    prepared, shared_staging, process_group=WORLD,
    identity=identity, safe_point=safe_point, progress=progress,
)
# 所有rank在新prepare模型+真正空Adam上调用；caller先验证完整checkpoint/树hash/信任。
receipt = load_h3_state(
    fresh_prepared, checkpoint_root, process_group=WORLD,
    identity=identity, progress=saved_progress,
)
```

四个 identity 都是显式SHA256；不能以占位身份代替实际模型/LoRA/拓扑/语义绑定，具体绑定由后续runtime/manager实现。生产只接实际已初始化NCCL WORLD8，真实H3PreparedModel及最终单组AdamW对象集合。当前prepared的固定scope为FP32 trainable master、单组非AMSGrad/非capturable/非fused/非differentiable AdamW，scheduler为None或既有constant LambdaLR。

仅写caller目录内的新 `h3-state/`：
- `h3-state/dcp/`：真实公开DCP保存的完整model（冻结base、持久buffer、LoRA）/Adam/非持久buffer；DTensor仍为sharded state，`full_state_dict=False,cpu_offload=False,ignore_frozen_params=False,strict=True`。生产不full_tensor、不每rank全权重CPU/GPU gather。
- `h3-state/manifest.json`：bounded schema/expected identity/progress_id/step、完整tensor FQN shape/dtype、trainable/组FQN顺序与超参、scheduler精确状态和rope身份。manifest/.metadata上限16MiB，健康汇合只发路径和摘要，不发权重。
- receipt只有relative_path/schema/manifest_id/progress_id/step，供外层现manager树hash及发布使用。**没有complete/latest/终态发布、没有第二manager。raw staging不是可恢复run checkpoint。**

rank0仅负责小manifest写入并fsync；所有rank在writer外依次进入真正state getter、DCP save/load和setter。生产FileSystemWriter明确`overwrite=False,sync_files=True`，既有codec目录/已发布output拒绝，不重用半写目录。

## save强门与纯度

- 使用原 `CheckpointSafePoint.assert_ready(CheckpointProgress)`，再核对actual rank/world、group_geometry、committed/global/next step一致且step>0；open reservation/reward/dynamics、GA位置、非同步、poison等均拒。所有模型梯度必须None，不把零grad当已清。
- proof必须由caller在BaseTrainer整轮tail返回/next-step推进之后提供。本codec验证已有proof的字段和一致性，不从一次optimizer.step推造tail安全证明。
- **调用任何DCP optimizer helper之前**验证当前每个最终trainable的Adam state恰为step/exp_avg/exp_avg_sq，step为CPU FP32标量、finite正数且等于已提交progress，moments全局shape/dtype匹配、local值finite且平方moment非负；cold/partial/stale对象/缺字段/未清grad先共同拒绝。
- 之后所有rank调用公开`get_model_state_dict`及仅对已完整warm Adam调用`get_optimizer_state_dict`，复核完整canonical FQN/shape/dtype/group覆盖，再共同进入同步DCP save。root与主block checkpoint-wrapper FQN精确映射，不放宽strict。
- 独立保存真实FP32 `rope.inv_freq`，小值hash进入共同manifest，加载时核验；不遗漏to_empty之后需恢复的非持久buffer。scheduler None-schema与constant-schema精确区分，计数/lr必须符合committed step。
- 真实测试比较save前后全部model/buffers/Adam moments-step-groups/scheduler/grad及RankRNGSnapshot一致，且step spy零次。实现不恢复/重设全局RNG来掩盖副作用。

## fresh空Adam的实际API发现与安全路径

本轮首先对真实公开`set_optimizer_state_dict`做了空Adam probe，step替换为禁止spy。实际失败栈：

`set_optimizer_state_dict -> _load_optim_state_dict -> _init_optim_state -> optim.step(closure=None)`。

这是实证的API副作用，不是GPU结果；最初门禁为93 passed/1 failed。没有安装升级或把这个helper默默用于fresh恢复。改用以下公开、安全的组合，最终测试全通过：

1. fresh load检查`optimizer.state`真正为空，**不调用fresh get_optimizer_state_dict或set_optimizer_state_dict**。
2. 在独立目标dict构造完整canonical FQN destinations：step空CPU标量、`empty_like(parameter)`两moment（生产保留DTensor shard）。这些只是DCP接收容器，不装入optimizer、不作零梯度step、不预造假warm state。
3. 读取本地DCP metadata，以公开`DefaultSavePlanner`的local schema核对完整keys/global shape/dtype；检查缺失/截断/逃逸storage路径后通过共同健康门。此处不进行state collective。
4. 所有rank实际`get_model_state_dict`取得目标model视图，`dcp.load`从磁盘填充model/moments/buffer；验证所有已读取非零训练step、完整moments/FQN/groups、finite以及rope hash。
5. 共同验证成功后使用公开`set_model_state_dict(strict=True)`和公开 **`AdamW.load_state_dict(loaded_fqn_state)`**。PyTorch optimizer setter通过saved-ID（这里是已核验FQN字符串）→当前Parameter的zip映射装入状态，无`_init_optim_state`/dummy step；组FQN顺序在读前严格比对，不能靠位置错绑。恢复buffer及scheduler后再核验最终对象/状态/schema并共同确认。

因此不是用热optimizer恢复冒充fresh，也不是调用危险setter后“回滚”权重。CPU中从新的真实tiny loader/PEFT模型、新空Adam、新scheduler实际恢复；故意扰动fresh冻结base/持久buffer并使用不同初始lr，恢复后全部状态与不中断对象一致；固定下一步真实backward/Adam/scheduler结果逐值一致。CUDA DTensor通过公开optimizer setter的行为仍须实际GPU验收，CPU证据不替代它。

## 兼容/失败/信任边界

- 现有有界健康receipt用于save预检/capture/写完，以及load预检/已读验证/应用完成；共同路径/manifest摘要、拓扑/模型/语义/progress不一致共同拒绝。本地metadata/shape/覆盖/路径问题在潜在state collective前检查。
- getter/DCP/model setter内部死亡或collective故障直接交上层整attempt退出；不在finally补barrier，不回滚或发布假complete。load可能已修改目标model，失败后必须丢弃该fresh attempt，不能继续训练。
- DCP `.metadata`是pickle格式。只读取外层manager已验证信任、完整marker与树hash的artifact；本codec检查schema和storage覆盖/长度，不重复实现全树hash/发布协议，不能当不可信checkpoint沙箱。
- 不恢复data cursor/behavior/全局RNG。外层完整恢复门必须最后调用既有RankRNGSnapshot恢复；这里save/load均不抢先restore RNG。

## 七项独立CPU测试 / 替身

- 真tiny H3 meta loader、真实PEFT/Accelerate/safetensors、真实nonreentrant checkpoint wrapper、PreparedModelBundle guard、真实AdamW/LambdaLR两步训练。
- 真磁盘DCP `.metadata`/`.distcp` save/load及公开state APIs；CPU leaf显式`no_dist=True`包装，没有mock整个codec/save/load成内存dict。
- 仅替WORLD rank/健康transport、CUDA/DTensor-local观测，及本来就需替换的重H3 class；CPU参数不冒称DTensor。proof是明确caller测试fixture，不冒称真实trainer tail。
- save纯度与step禁止spy；新模型/真正空Adam恢复（也以模拟rank1运行真实文件load/getter/setter），非零moments/step2、完整base/持久和非持久buffer/组lr/scheduler逐值对照；下一步step3与不中断对照一致。
- cold/部分state/错误momentshape/step/非有限/stale对象/未清grad/unsafe proof提前拒绝，state helper入口计数0且未创建codec目录。
- None scheduler roundtrip与warm-load/重复save拒绝；wrong topology/model/semantics/schema/FQN/shape/scheduler/counters/groups、重复JSON key、缺/损坏metadata、截断storage拒绝。
- 真DCP重写出缺moment或wrong-shape损坏fixture，load在读取tensor前按真实metadata拒绝；公开optimizer FQN setter单独证明fresh不step、不改权重。
- 独立叶子测试核验生产DCP进程组、overwrite=False/sync_files=True及strict/nonfull/不丢frozen选项；生产sharded观察器真实拒普通Tensor。

## 最后源码编辑后的门禁

均exit0；无skip/xfail、无全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.28s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 100 passed, 9 warnings in 4.39s（基线93+C新增7，不含W4兄弟新增）
```

9个warning是公开DCP的显式单进程模式提示和损坏fixture故意重写tmp_path文件提示，未隐藏/过滤；生产writer不覆盖。已read复读最终实现，status仅上述两新文件及本报告，git diff --check exit0。新增未暂存文件不计入tracked diff，未把空diff当内容审查。原Flash完整文件相对d818b8a231b52a9a024fa2f522057acdbb7bf099历史diff无输出，当前该文件未改。

## 待审 / 必需后续

Simplifier C重点审cold纯度、标准optimizer FQN setter替代危险DCP setter的依据、fresh物化和真实DCP文件、完整state及buffer、全rank调用位置与raw staging边界；changes回C修复两集重测。引擎snapshot待生成，orchestrator登记并在W4汇合后重跑完整两集。

后续仍必须把本codec接原AtomicCheckpointManager共享staging/单writer发布，绑定真实data/资源/双schedule/cache/topology/模型语义身份，完整tail safe point与rank RNG/data、同拓扑fresh整run恢复/下一步；本项不是完整checkpoint交付。无GPU、真实PG、网络、服务器、下载、升级、push或手工Git提交。**MVP未完成，GPU_PENDING。**
