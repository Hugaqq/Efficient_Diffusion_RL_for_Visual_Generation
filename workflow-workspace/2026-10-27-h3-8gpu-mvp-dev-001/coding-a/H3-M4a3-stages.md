# H3-M4a3-stages — Coding A / W3

## 判定与身份

- `implementation_ready: true`：本分支六阶段接线实现及指定CPU门禁完成，交simplifier-a；不是W3审核/汇合批准或RunController完成。
- 实际worktree：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-264dad1a-fc45-4b29-83f5-fa37bb012962`。
- 实际branch：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-264dad1a-fc45-4b29-83f5-fa37bb012962`。
- 开始HEAD/基线：`282539fa94c9ca2ec7fda8602e2cc2980ada6d62`，初始干净。assignment integration parent为`e436c192778a6bbf24e2bee7ebef777332138f9a`；实际基线含W3协调更新。
- 已读task/plan/assignment A/wave-03、对应pool approved审查、实际BaseTrainer/StageValue接口、world/credit、CPS/pool/anchor及prepared/decoder接口。无前置changes待返工。
- 引擎snapshot SHA pending；无暂存、手工commit/merge/push、切分支或访问其他worktree。

## 允许范围内的实际改动

1. 新 `visual_rl/algorithms/trainer/h3_stages.py`。
2. 新 `tests/test_h3_mvp_cpu_stages.py`。
3. `visual_rl/algorithms/trainer/h3_pool.py`、`visual_rl/algorithms/optimization/h3_anchor.py` 只增加可选逐格status_hook及对应安全异常边界；已有单进程调用默认语义不变。
4. 本交付记录。

未修改原grpo.py/stages.py/runtime/controller、CPS数学、objective、已有测试及断言、Flash配置或B/C本波文件。不import同波B/C新模块。独立控制接线，无上游代码复制/新增依赖/第二runner。

## 下游绑定接口

```python
stages = H3IterationStages(
    data=data_port, prepared=prepared_h3,
    predict=predict, decode=decode, score_one=score_one,
    coordination_factory=coordination_factory,
    process_group=world_pg, rank=rank, device=device,
    read_behavior_id=read_behavior_id,
)
trainer = stages.trainer()  # 恰一次；返回现GRPOTrainer，不自写run_iteration
trainer.prepare_run(PrepareRunContext(...))
result = trainer.run_iteration(step)
```

必需端口：

- `data.reserve(step) -> H3IterationRequest`；`data.commit(request)`真实提交本轮data/collection并把behavior更新为request.next_behavior_id；`data.abort(request)`放弃失败reservation而非回滚权重。没有默认dataset或no-op实现。
- request携带原IterationIdentity、有序sample三元组、collection/behavior/next_behavior、精确dynamics、geometry_id、按sample绑定的prompts/CPU detached finite `[1,L,D]` conditions、CPU FP16初态/显式双RNG及三类预算。geometry_id及条件模型身份由后续模型/数据绑定层如实提供；本层不import具体H3模型类。
- `predict(prepared, *, sample_id, x, index, condition, dynamics) -> joint_noise_ward_prediction`；必须使用给定唯一prepared callable。condition是CPU输入，未来adapter负责其device搬运；CPU测试使用真实h3_t2va_forward。
- `decode(terminal_cpu_fp16) -> contextmanager(media)`，未来模型域用已批准decoder/geometry闭包装配。
- `score_one(*, sample_id, prompt, media) -> finite Python float`，必需注入，与wave-03一致。
- `coordination_factory(plan)`返回具备`gate(event, *, local_error=False)`、`aux_turns(work)`、`clip_grad_norm(prepared, *, max_norm)`的对象，与wave-03一致；无默认no-op或CPU分布式fallback。

## 真实六阶段与tail

- **Prelude/build**：reserve→本地请求/身份/初态/条件/预算/RNG/prepared校验→真实admit_h3_world。普通本地预检异常转换为local_error进入共同receipt拒绝，不先调用模型。schedule identity从layout/双FP32 grid/eta/CPS family/quantization实际值生成。取得plan后才构造coordinator，再以小digest gate核对step/source/phase/next_behavior共同语义。所有stage保留请求的同一IterationIdentity对象。
- **Rollout**：prepared root EVAL；调用已批准pool.collect，完整sample-major/N步含q0。version前后和逐格forward/result gate同序；无decoder调用或假媒体。
- **Reward**：仅完整rollout后调用aux_turns；每个sample取真实terminal，进入单条decode lease、score_one、退出后即删除媒体引用。只产出真实小reward五元组，不聚集整批媒体。
- **Advantage**：真实gather_h3_credit执行world表/身份/完整性与全world归一化数学。
- **Credit**：独立stage把本rank信用扩展到完整N格，产生真实PolicyLossInputs，algorithm_weight1、clip .005、beta0；finite检查完成后共同gate。
- **Optimize**：先一次现H3ReplayAnchor.capture，固定有序闭包把pool sample与index绑定正确conditions；完整成功/共同gate后才current replay。每格重建临时设备record、prepared prediction、现CPS.evaluate、现ClippedSurrogateObjective，乘且只乘plan.loss_scale(n_slot=1, reducer_divisor=prepared.reducer_divisor)，立即backward并删除该格图/预测。prepared必须暴露divisor8及真实AdamW；无no_sync、无再除world/GA/N。全部格完毕/version确认后唯一clip_grad_norm(max_norm=1)，一次optimizer.step、如有scheduler则一次step、zero_grad。
- **既有BaseTrainer tail**：optimize返回后，由原commit_iteration调用data.commit、核对新behavior、共同tail完成gate，随后pool.release及本binding进度更新。next_optimizer_step仅由原BaseTrainer维护。返回结果为轻量H3UpdateResult（IDs、格数、scaled_loss、norm），无轨迹/图。
- build开始置poison，只有完整成功tail解除。preflight/评分/old/current异常不能step/tail；部分Adam或data-tail失败不尝试权重回滚，也不允许原step再次执行。stage失败由BaseTrainer调用abort；tail失败原BaseTrainer不再调用abort，本binding仍poison。失败池仅在close清理，不伪装成功tail release。

## Hook与异常分类

pool.collect、anchor.capture新可选`status_hook(event, *, local_error=False)`：每格`cell/{local_ordinal}/forward`与`cell/{local_ordinal}/result`。本stage绑定后加`rollout/`或`anchor/`前缀；events不含各rank不同sample ID。

raw model forward/AC不在数值捕获try内；其异常不补发恢复collective。模型正常返回后的CPS sample/evaluate、量化/finite错误在result gate报告local_error。current score/objective/scaled-loss错误在`current/cell/{ordinal}/backward`门前共同拒绝。raw backward异常同样不在finally补gate，要求上层attempt teardown。

其他事件包括prelude/binding/digest、rollout/prepare/version/complete、anchor/version/complete、credit/ready、current/prepare/complete、optimizer/complete、tail/ready/complete。aux内部事件由coordinator拥有。C需按实际调用序列处理这些有界字符串；本项未推定raw collective失败可恢复。

## 真实CPU测试

新增14个实际pytest cases：两个正常路径（两次更新/零信用）与12个故障参数case（preflight、identity、condition、score、rollout、anchor、current、current版本漂移、raw forward、raw backward、部分step、部分tail）。已有65项及断言不改。

- 真实BaseTrainer.run_iteration、StageValue、pool/CPS/anchor、world admission/credit数学、既有objective、真实guarded H3PreparedModel/PreparedModelBundle和AdamW/LambdaLR执行；真实H3 forward bridge与双VAE decoder控制。
- 仅H3/VAE重量级模型叶子tiny、评分transport用测试media-dependent score、WORLD/NCCL/coordination/norm叶子显式CPU替身。未mock整个trainer、stage、pool、anchor、world算法或目标函数。
- 实测每轮sample-major三组rollout/anchor/current各6次forward，全N含末格；前两组no_grad，current每格正常backward；精确forward/result/backward gate序列，评分发生在全rollout之后；原identity对象保持。
- 两次更新双模态有限非零梯度/参数变化、Adam state step=2、data position/behavior/next-step每轮一次；pool在data tail时仍ready，只在完整tail后released。weakref确认预测不跨格保图、媒体不跨lease保留。
- 同reward专门case产生零信用，但仍6次真实backward、norm0、真实Adam/scheduler/data commit；没有zero-loss生产占位。
- 错误预检真实local_error receipt、模型零调用；评分/old/current finite错误无step/tail；raw forward/backward错误后没有额外gate；部分step/tail只发生一次，不就地再更新、不假回滚；close释放失败池。

## 最后源码编辑后的门禁

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3通配符exit0：**79 passed in 4.55s**（本基线65+本项14；不包含未汇合W3 B/C新增tests）。
- Flash固定原七项exit0：**7 passed in 12.97s**。
- 无skip/xfail，无全量测试。前两次开发中通配符也全部通过77/78项；最终以上述79项为准。
- git diff已读：仅pool/anchor逐格hook扩展，35 insertions/11 deletions。新stages和feature测试尚未暂存，未包含在tracked diff/stat中。
- git diff --check exit0；只覆盖tracked diff，不冒称untracked源码也被该命令检查。新实现已读回并真实import/测试。
- 获准历史比较原tests/test_flash_collection_runtime.py相对d818b8a231b52a9a024fa2f522057acdbb7bf099完整diff为空，exit0；本轮未改它或其他旧测试。

## 待审核与明确边界

交simplifier-a审真实六stage复用、每格固定锚/单一归一化、图释放、安全gate与完整tail/poison语义。changes回A两集重测。

W3汇合后仍须验证与B真实HPS scorer/C真实coordinator结构和事件budget兼容；本分支CPU叶子不替代该集成验证。后续仍必须接ModelAdapter/recipe/唯一RunController、真实data port与模型条件/资源身份、checkpoint safe-point及fresh恢复/下一步。此项只提供可装配的真实六阶段对象；未执行HTTP、CUDA/FSDP2/NCCL、真实模型/VAE/HPS权重或8卡资源准入，不宣称音频质量。

无服务器/GPU/网络访问部署/下载/环境升级/push。**GPU_PENDING，MVP未完成。**
