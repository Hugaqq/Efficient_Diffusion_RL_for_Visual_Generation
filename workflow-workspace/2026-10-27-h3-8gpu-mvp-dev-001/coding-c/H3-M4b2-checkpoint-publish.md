# H3-M4b2-checkpoint-publish — Coding C / W5

## 判定、身份与续跑核对

- `implementation_ready: true`：仅本原子项交 simplifier-c；不是完整默认RunController装配或GPU验收。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-7b9f8550-68d5-442f-b7a4-f95d64914a4e`。
- 本attempt起点：`d2a6b8563f5bcc2d5f95bf9e24458dc25d257b91`，开始干净。续跑当前HEAD：`5da7331f7ddf460bf793e9de032ae14857385134`；实际git log显示它是紧接d2a6b85的工作流snapshot。未自行commit/merge/切分支。
- 网络中断发生在两组门禁和diff检查均已返回成功之后，没有待返回bash。按用户续跑指令，已重新read当前源码/测试、核对clean status及snapshot、历史七项diff；**没有重放源码写入或已成功测试**。本次续跑只补本报告及角色交付。
- 已读本轮task/plan/assignment C/wave-05与本配对W4 approved反馈；旧codec/protocol/state/RNG及A/B本波文件保持只读。

## 文件与最小manager扩展

- 精确修改 `visual_rl/artifacts/checkpoint/manager.py`。
- 新增 `visual_rl/artifacts/checkpoint/h3_transaction.py`。
- 新增 `tests/test_h3_mvp_cpu_checkpoint_publish.py` 六项；原108项及Flash七项未改。

现 `AtomicCheckpointManager` 增加 `create_staging(step) -> CheckpointStaging` 和 `commit_existing(staging, contract, *, progress, fault_injector=None)`。receipt是本manager的进程内一次性能力：校验对象归属、root/step/真实目录、device/inode、无symlink/发布元数据、final不存在；复制/外来/已消费receipt、未知目录、替换目录及重复step拒绝，不接管或删除未知状态。

原 `commit(writer)` 和新入口调用同一 `_publish_staging`；该方法沿用原tree hash、contract/progress/complete、rename、latest和故障注入逻辑，没有复制第二套发布器。旧writer异常清理、rename前失败清理、rename后失败保留合法complete语义保持。新入口不接模型callback，是纯文件发布。修改前的tracked diff为manager 71 insertions/1 deletion，主要为能力入口与抽取原发布体；新文件不计入该stat。

## H3事务公共接口

```python
transaction = H3CheckpointTransaction(manager, process_group=WORLD)
committed = transaction.checkpoint(
    prepared,
    contract=contract, progress=progress_template, safe_point=rank_safe_point,
    identity=h3_state_identity, bindings=bindings, data_state=rank_data,
)
pending = fresh_transaction.restore_prepared(
    fresh_prepared, committed_path,
    expected_contract=current_expected_contract,
    identity=current_state_identity, bindings=expected_bindings,
)
restored_progress = fresh_transaction.restore_bound(
    validate_live=validate_actual_live_bindings,
    restore_data=actual_source.load_state_dict,
    read_data=read_actual_source_envelope,
)
```

所有rank调用这三个方法；manager只由rank0创建共同staging/发布。返回实际 `CommittedCheckpoint`，所有rank收到相同path/contract/progress/tree身份。

`bindings` 精确字段：`model, topology, semantics, data, cache, score, dynamics, resources, code`，值均显式SHA256。model/topology/semantics与H3StateIdentity匹配，code与现CheckpointContract一致；完整绑定表写入H3 manifest，恢复逐项比对。实际身份采集由未来runtime负责，不能用假hash/假G3 receipt充数。

### 冻结data envelope（future runtime需从真实source构造）

```python
{
    "schema_version": 1,
    "rank": rank, "world_size": 8,
    "content_id": source_content_sha256,
    "source_id": progress.next_source_id,
    "next_optimizer_step": progress.next_optimizer_step,
    "next_global_batch_id": progress.next_prompt_batch_id,
    "cursor": actual_global_source_cursor,
    "behavior_id": actual_behavior_id,
    "open_reservations": 0,
    "payload": actual_source.state_dict(),
}
```

精确key/schema、rank/world/next-step、source/cursor和progress对应关系、content binding与共同behavior/next batch均核验。只接受有限JSON值，深度<=16、编码<=64KiB；不接受Tensor/Generator/model/轨迹。envelope是拥有副本的快照，不保live对象；必须同时满足真实 `CheckpointSafePoint.assert_ready` 与既有codec的强门，不能以`open_reservations=0`代替tail proof。C不制造真实cursor，也不import本波A数据实现；外层必须在source允许的完整边界调用state_dict。

## 保存流程 / 全局与rank身份

1. 校验现CheckpointContract/Progress/SafePoint、step>0、rank/world/geometry/transform及JSON data边界；复用有界receipt共同拒绝本地预检错误。proof必须来自BaseTrainer完整tail/data/behavior/pool/next-step推进之后，不从optimizer.step推造。
2. 每rank用既有 `RankRNGSnapshot.capture_current(rank)` 捕获私有RNG，仅汇合有序rank→rng identity小表。将progress模板的rng_state_id替换成 `SHA256({schema_version:1, rank_rng_ids:[[0,id0],...,[7,id7]]})`，**不是要求八个私有hash相同，也不是删除RNG身份**。返回的CommittedCheckpoint.progress包含该真正全局摘要。
3. rank0以manager创建staging并广播basename/step/binding/progress；所有rank在自己的声明root下严格解析同一真实路径。
4. **所有rank在manager writer外直接调用真实 `save_h3_state`**。cold/partial/未清grad等依旧由已批准codec在helper前共同拒绝；不修改或绕过codec，不dummy step。
5. 每rank写自己的 `h3-ranks/rank-N.pt`，仅含现RNG codec payload、JSON data/behavior、safe point和rank/world/identity/progress/codec摘要；没有把DTensor权重写普通torch.save。附件<=1MiB，独占创建、flush/fsync，返回小hash descriptor。
6. 校验恰8份附件、文件hash/大小、私有RNG/data/proof身份、共同cursor/behavior与全局RNG表。rank0写小 `h3_transaction.json` 后，调用原manager的纯文件 `commit_existing`，全树覆盖DCP及附件。只有成功才返回commit receipt。

任何失败poison本attempt；DCP/模型collective内部故障交上层超时退出，无finally补barrier、无伪rollback。发布前失败可能保留本attempt非resumable staging供检查；不把它当恢复源。rename后latest写失败保留可由原manager inspect/latest扫描识别的合法complete，不倒删目录。

## 两阶段恢复

- `restore_prepared`：先实际 `manager.inspect_complete(expected_contract=...)` 重验完整tree/contract/progress，再核验H3绑定、codec manifest及8份私有附件。全rank实际调用 `load_h3_state` 恢复新模型/真正空Adam/scheduler/buffer；返回待处理的本rank data/RNG与checkpoint receipt，**不恢复data或RNG**。
- `restore_bound`：必需 `validate_live()` 在本次调用中重新检查实际模型/资源/data/cache/score/双schedule/topology，返回新的 `(CheckpointContract,H3StateIdentity,bindings)`；无noop默认。再次实际inspect完整文件树及所有身份，并和prepared阶段receipt一致，不能用第一次检查缓存跳过。
- 调用 `restore_data(saved_payload)`，再用必需 `read_data()` 读取实际恢复后的完整envelope，校验cursor/behavior及全部payload；共同成功门之后，**最后**调用现 `RankRNGSnapshot.restore_current()`。其后没有IO/model helper/collective。
- 两个恢复方法各只可一次，部分data/model恢复失败poison，不原地重载或声称回滚。真实绑定回调必须由后续runtime实现；类型接口不能自行证明调用者真的重新读取资源。

## 真实CPU证据 / 替身边界

新增六项实际测试：

- 真manager目录/tree hash/complete/latest和inspect；未知/复制/外来/已消费receipt、替换目录、symlink、重复step拒绝。旧commit故障注入仍执行原发布逻辑。
- 八个进程内worker各有真实tiny loader/PEFT/AC/guarded prepared/Adam两步训练；所有rank实际执行state getters/save入口/load/model setter。spy确认这些都不在manager publication中。保存模型/Adam/scheduler/RNG不变。
- 实际磁盘DCP `.metadata/.distcp` 与原codec、8附件及全局RNG摘要；八私有RNG不同仍产生一个全局progress。完整目录可被原manager.inspect_complete读取。
- 新模型、真正空Adam禁止step spy下恢复，八rank状态逐值对照；另有单进程真实RNG恢复case，故意在data回调消耗Python/NumPy/Torch RNG，验证调用顺序live-refresh→data→read-data→RNG-last、下一随机数与保存状态对照、下一固定Adam/scheduler更新与不中断对象逐值一致。
- 附件/DCP失败不发布；**实际latest.json写IO失败**发生在rename后，合法complete保留且原manager.latest_complete仍能发现。失败后重试拒绝，无补救collective。
- bound阶段实时cache binding变化、两门间DCP篡改、缺rank文件、manager重新hash发布的错误私有rank/topology/data fixture、midcollection/data错误及部分data恢复失败均拒绝/poison，RNG不提前恢复。

替身限于重H3类、WORLD/CUDA/DTensor观测和通信：八线程内存传输模拟PG；DCP通信包装让八份相同CPU replica全部进入codec/getter/save，再以真正no-dist DCP写共享文件，所有rank读取真实文件。私有进程RNG以独立Random/RandomState/Generator构造快照；八线程末端RNG恢复观测单独替换避免同进程global竞争，另一个case确实执行原restore_current。data对象和live绑定为明确外部接口fixture，不冒充尚未汇合A/B实际data/profile或默认RunController/G3；没有高层mock整个事务、manager或codec。

初轮H3有两项fixture失败：从线程模拟转入主线程单进程恢复时，API spy未设置TLS rank。已补真实主线程rank0观测，未删除断言；随后全部通过。

## 最后编辑后的已执行门禁（续跑未重放）

中断前以下命令均已实际返回exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 14.59s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 114 passed, 16 warnings in 11.81s（基线108+C新增6；非W5兄弟汇合结果）
```

无skip/xfail或全量。16warnings是明确no-dist DCP及原损坏fixture重写提示，未过滤。已保存的最终diff --check为exit0；续跑未再改源码/测试，仅补本报告。恢复快照5da7331实际包含当前实现，接续时status干净；原Flash完整历史diff相对d818b8a再次无输出。未自行提交此snapshot，未以clean状态代替此前实际测试结果。

## 待审 / 未完成

交simplifier-c审manager唯一发布体、所有rank state IO位置、RNG全局/私有层次、两门刷新及RNG最后；changes回C修复重测。orchestrator记录恢复snapshot与交付报告snapshot，W5汇合仍需完整通配符/Flash及真实data/profile/publisher组合验收。

下一步仍须现唯一RunController完整装配，提供真实运行contract/G3/live身份、data envelope及完整tail proof，把恢复的progress交回controller继续分支；本原子项不是整run完成。真实CUDA DTensor/DCP/NCCL、多卡保存恢复、H3/HPS资源和质量未验；继承manager现有文件hash内存行为，不能据tiny文件测试承诺真实大权重IO/RSS预算。无GPU、真实PG/网络、服务器、下载、升级、push或手工Git提交。**MVP未完成，GPU_PENDING。**
