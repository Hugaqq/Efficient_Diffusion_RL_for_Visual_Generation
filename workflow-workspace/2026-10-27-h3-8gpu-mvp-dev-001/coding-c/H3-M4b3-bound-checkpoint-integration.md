# H3-M4b3-bound-checkpoint-integration — Coding C / W7

## 身份与范围

- `implementation_ready: true`，仅当前合法对象组合测试交 simplifier-c。**不批准 A-BIND-01，不宣称默认RunController或GPU完成。**
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-8b40db49-6c81-41a6-859c-cd6bf45266a5`。
- 开始/结束 HEAD：`cc6e19f7d35a34d443fe7942a3c33e00ea7bb631`，初始干净；未手工提交、切分支或合并。
- 仅新增 `tests/test_h3_mvp_cpu_bound_checkpoint.py` 和本报告。所有生产文件、已有tests、同波A/B文件只读；不依赖A本波新符号。
- 已读最新task/plan/assignment C、C-PUBLISH-01 approved报告、实际binding/source/stages/transaction及BaseTrainer、RunController阶段顺序。

## 实际组合与断言

新增两个case，内部覆盖正常继续、fresh恢复、实际cache漂移及checkpoint漂移：

1. 使用真实完整tiny transformer文件（全base safetensors）、profile/cache/模型与code内容hash；真实adapter load、meta/PEFT/AC/prepare/Adam/合法constant scheduler、真实已acquire HPS wrapper。
2. 八个rank各自构造真实H3PromptSource和H3RuntimeBinding，再构造真实H3IterationStages/BaseTrainer。没有用旧 `_contract/_identity/_data` fixtures替代绑定输出，没有mock binding/source/checkpoint逻辑。CheckpointContract、H3StateIdentity、live绑定均来自实际binding；progress/proof/data来自完整tail后的 `binding.checkpoint_boundary(stages,trainer)`。
3. 真正执行第0轮全六stage及tail；冷边界拒绝且codec调用为0。optimizer post-hook观察到“更新已执行但tail未完成”时boundary仍拒绝。完成后所有rank step1/cursor8；gradient norm正数、Adam含非零moments。
4. 所有rank调用真实transaction.checkpoint，真实磁盘DCP、rank附件、原manager manifest/tree/complete/latest及inspect执行。模型/Adam getter/save/load/model setter计数覆盖0..7，spy禁止它们出现在单writer发布体内。私有RNG使用独立进程状态观察叶子，八份实际CPU RNG字节不同，有序全局rng_state_id一致。
5. 连续对象实际执行第1轮，保存其真实request与完整结果。另建全新adapter/prepare/空Adam/source/binding/transaction/trainer，先restore_prepared：恢复model/moments/scheduler，source仍step0、全局RNG不变、无dummy step、fresh路径未调用optimizer getter。
6. restore_bound重新调用真实binding.validate_live，真实source.load_state_dict，再从source公开state/属性生成实际envelope回读。**没有在fresh trainer上调用checkpoint_boundary来伪装已完成tail，也没有修改trainer/source私有计数。**随后以公开PrepareRunContext(start_optimizer_step=恢复进度)准备新trainer并真正运行第1轮。
7. 连续/恢复路径逐值比较request身份/样本/prompt/condition/FP16初态/两个显式generator初始state，完整IterationResult（含loss/norm）、全部参数/Adam/scheduler/data/behavior与next-step2；新真实tail再次产生合法checkpoint边界。
8. 串行rank0 case在事务内部真正执行原RankRNGSnapshot.restore_current；data回调真实load后故意消耗Python/NumPy/Torch随机数，观察顺序为live→data→read-data→RNG-last，验证恢复后的完整RNG身份与下一随机数一致。
9. 两门之间分别实际修改cache/index.json或DCP文件字节，第二门通过真实live检查或manager全树hash拒绝；所有模拟rank失败/poison，data仍未恢复、RNG不变、没有额外codec调用。随后还原测试文件用于清理，不修改生产防护。

初轮为135 passed/1 failed：最后一项boundary检查误在主线程使用了peer rank观察。已将该检查放入对应worker，保留原断言；修复的是测试设备/rank叶子使用方式，未发现需要越界修生产代码的组合缺陷。

## CPU替身精确范围

- 复用已存在只读artifact/重模型/VAE/HTTP/prepare叶子helper；本文件自行装配八rank真实对象。真实HPS factory/client/JPEG/响应校验继续执行，分数来自实际提交JPEG像素，不是常量reward。
- `_World`仅为有限等待的进程内通信叶子，不创建PG/socket或训练runner。CPU使用完整模型副本：backward hook模拟SUM/8梯度同步；真实norm通过DTensor观察叶子只读取各rank独占的Shard(0)梯度片段，实际clip后通信叶子汇集片段，使各CPU副本以相同完整梯度执行真实Adam。
- 这不是物理DTensor/FSDP验证。参数原始形状保持，norm观察使用实际global shape与本rank非padding slice；没有把重复全梯度计八遍或替换高层norm。
- DCP通信叶子先断言八个CPU副本完整state逐值一致，再执行真正no-dist磁盘save；所有rank实际读文件。并非把IO变成内存dict，也不把副本称真实分片。
- 八线程共享进程global RNG，所以并行最后恢复仅模拟各私有进程观察；另有串行case在实际事务末端执行真实restore_current。没有多进程网络、GPU、全局线程数修改或环境升级。

## 两恢复门/唯一controller接线的实际约束

实际controller顺序仍为 prepare→restore_prepared→transform→bind→restore_bound→prepare_run（controller.py约279–336），本测试没有更改或运行该controller。

当前transaction第一门需要full expected_contract/identity/live表。测试证明：**实际prepare结束、source已构造但未reserve、HPS已真实acquire**后，即可构造一次真正H3RuntimeBinding得到这些数据；不要求source先恢复或trainer先完成tail。随后原对象可以穿过模型恢复和source恢复，再正常运行下一轮。

该binding构造同时安装prepared/numerics能力，并非纯声明对象。因此下一默认生命周期需要显式安排这些真实对象在第一门前就绪，并在后续bind阶段复用同一binding（不要二次构造/重新包装最终参数）；transform不能破坏已冻结参数身份。如果上层要求把所有绑定安装严格推迟到BIND阶段，则需对应owner提供合法prepared-contract分离入口，不能用fixture/G3假receipt补洞。本项给出对象组合证据，不替orchestrator擅改controller阶段。

第二门的read_data可直接使用 `source.state_dict()`、`source.source_id`、`source.next_global_batch_id`、`source.read_behavior_id()`及实际binding.validate_live()['data']形成事务既有envelope；本文件 `_envelope` 展示精确适配，不需要新trainer的tail proof。第一门不得提前source.load，第二门最后恢复RNG，prepare_run使用返回progress的next_optimizer_step。

## 最后测试编辑后的门禁

原样执行，均exit0；没有skip/xfail、全量或warning过滤：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.27s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 137 passed, 22 warnings in 21.93s（基线135 + 本项2，不含W7兄弟未汇合新增）
```

22warnings为明确no-dist DCP保存/读取及既有损坏fixture重写提示。git diff --check exit0；原Flash完整历史diff相对d818b8a231b52a9a024fa2f522057acdbb7bf099无输出。新增文件尚未暂存，不以tracked diff为空代替内容证据；生产源码与旧tests无修改。

## 待审/未完成

交simplifier-c审实际合同/来源/完整tail、fresh强门及下一轮断言、叶子替身和生命周期说明；changes回本coding。引擎snapshot待登记；A-BIND-01仍需原A复审，W7汇合后必须重跑完整目标。当前合法scheduler组合成功不能批准其所有权负例。

下一项仍须默认composition/唯一RunController与runtime session装配，并做完整入口验收。当前未运行真实H3权重、CUDA/FSDP2/NCCL、多卡DCP、HPS服务或音视频质量验证。没有服务器、网络部署、下载、push或手工Git提交。**MVP未完成，GPU_PENDING。**
