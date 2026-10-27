# H3-M4a2-rollout-pool — Coding A / W2

## 交付状态 / 身份

- `implementation_ready: true`，仅本原子项源码与本分支CPU门禁完成，交simplifier-a；不代表W2汇合批准、controller闭环或main发布。
- 实际worktree：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-b3628d4a-b90b-4ad2-8eae-d93f7a30b47f`。
- 实际branch：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-b3628d4a-b90b-4ad2-8eae-d93f7a30b47f`。
- 开始HEAD/基线：`8e674e8894c2af61188f7523933cff585a6812d6`，初始干净。assignment integration parent `47546f94e7490dafcc3fdbc2eb57279f0839899d`；实际基线包含W2协调更新。
- 已读当前task/plan/assignment A/wave-02、前置anchor的approved报告，以及实际CPS/anchor/forward/geometry。无前置changes需修复。
- 引擎snapshot SHA：pending；未暂存、切分支、手工commit/merge/push。

## 仅新增

1. `visual_rl/algorithms/trainer/h3_pool.py`
2. `tests/test_h3_mvp_cpu_pool.py`
3. 本报告

没有改既有源码、旧测试及断言、B/C同波文件；不import兄弟同波新模块，不复制CPS数学或PPO，不新增runner/依赖。task/reference/AGENTS/workflow未修改。

## 固定接口

```python
pool = H3RolloutPool(
    collection_id=cid, behavior_id=bid,
    samples=((sample_id, group_id, member_index), ...), group_size=K,
    dynamics=cps, max_samples=..., max_cells=..., max_trajectory_bytes=...,
)
pool.collect(
    initial_states,  # mapping sample_id -> owned-at-capture CPU detached FP16 [1,D]
    generators,     # mapping sample_id -> (video_generator, audio_generator)
    collection_id=cid, behavior_id=bid, device=device,
    predict=predict_sample,  # (sample_id, x_on_device, index) -> noise-ward joint
    read_behavior_id=read_policy_version,
)
cells = pool.records(collection_id=cid, behavior_id=bid)
samples = pool.samples(collection_id=cid, behavior_id=bid)
terminal = pool.terminal(sample_id, collection_id=cid, behavior_id=bid)
# Future trainer owner calls only after complete optimizer/data tail:
pool.release()
```

- 构造时核验显式有序sample三元组、唯一sample、各组唯一member且完整覆盖0..K-1、K>=2；group_id是上层给定prompt occurrence，不由文本生成。绑定一个collection/behavior及既有精确CPS合同。
- 构造时检查max_samples/max_cells/max_trajectory_bytes。预估 retained tensor payload 为 `samples*N*(4*D+4) + samples*2*D`：每格FP16 x/action、FP32 rollout diagnostic及每样本独立FP16 terminal。不计Python metadata、caller-owned输入/读取副本或模型临时workspace，不冒称整个进程RSS界限。
- collect在任何predict前验证全部初态的IDs/shape/CPU/B1/FP16/detached/finite，验证全部generator IDs、pair、目标device及对象独立性；视频/音频及不同sample不共享Generator对象。CUDA目标必须显式device index。初态由caller提供，不生成零初态、不自动seed/补样；预检后一次拥有所有初态CPU副本，按sample消耗，复用预算中的terminal槽位。
- transient `collecting` 状态防止重入；一次成功则ready，任何失败则failed，无局部重试。release幂等、释放轨迹/终态引用并使所有读取拒绝；collect中不能release。new/failed/released都不能读出部分池。
- no_grad中固定sample-major/index0..N-1顺序，predict一次、真实H3CPSDynamics.sample一次。callback收到当前x的副本，CPS action成为下一x；逐步通过CPS public接口构造拥有CPU记录，立即释放设备record/prediction，设备上不累计轨迹/媒体/计算图。末步q0沿用现CPS的双noise draw语义，全部N格保存。
- 前后read_behavior_id必须匹配，漂移拒绝整池；只有完整成功才能公开记录、元数据和终态。caller负责EVAL、固定参数及每次policy mutation更新version；本项不声称检测未申报的修改或实现全rank门。
- records返回有序tuple及既有冻结H3CPSRecord；其公开tensor属性返回副本，禁止修改内部下划线buffer。terminal返回CPU FP16 clone，数值恰等最后stored action。所有reader核对collection/behavior，terminal核对sample ID。pool在生成记录时绑定sample来源，不接受外部伪造的partial记录。
- release只释放pool自身持有的引用；外部已借出的records必须由读者释放。未调用optimizer/data tail，不能把本地release测试称为tail提交。

## 独立六项真实测试

- 导入路径属于当前worktree；合法小几何32×32×124（仅CPU fixture，非生产画布更改），tiny trainable H3叶子通过真实h3_t2va_forward和CPS生成4 samples×3 transitions，固定顺序、no_grad、双模态连续、FP16/FP32/CPU/detached、实际payload bytes和末态一致。
- 同seed复现、不同seed变化；固定初态与audio RNG仅改变video RNG时音频不变，证明模态流隔离。逐步与直接真实CPS reference action/score一致，generator最终state也一致，包括末步draw。采样不污染全局RNG。
- 实际pool输出进入现H3ReplayAnchor，明确有序闭包逐格核对sample顺序、index、x/action，再提供正确sample condition预测；修改当前模型后old不变/current变化；真实ClippedSurrogateObjective/PolicyLossInputs beta0 backward，两模态有限非零梯度。未mock池/锚/CPS/目标函数。
- 样本/格数/bytes超界、重复或不完整组、非法member/ID、最后一个sample的非法初态、缺少/多余IDs、缺失/重复/错误generator均在第一个predict前拒绝。非法初态包括shape、dtype、B2、meta、带图、NaN/Inf。
- 中途非有限预测在已有完整样本后失败仍没有partial可读池；前后version或identity冲突拒绝，失败不可重试；release幂等。
- callback篡改caller原始初态不影响后续样本（已拥有快照）；修改记录公开副本/终态读取不污染池；reader identity错误拒绝。weakref验证临时callback输入不跨步积累、pool不保callback/model、release后原记录可释放。

## 最后源码编辑后的真实门禁

在上述隔离worktree根目录执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3通配符 exit0：**50 passed in 3.13s**（本基线44+本项6；不包含尚未汇合的B/C W2测试）。
- Flash原七项 exit0：**7 passed in 12.07s**。
- 无skip/xfail，无全量测试，无既有断言变化。
- `git diff`无输出，新增文件尚未暂存；`git diff --check` exit0，仅检查tracked diff，不冒称覆盖untracked文件。新增源码已读回并真实import/测试。
- `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` exit0、无输出；原七项完整历史文件含断言不变，本轮也未改。

## Simplifier / 后续

请审核有界payload、原子ready发布、身份/样本顺序、双模态完整链、失败不可重试、release所有权及无重复CPS/PPO。接口供未来RolloutStage/RewardStage消费，但本项未改stage/controller，不做reward/credit/anchor捕获（只在测试中串接）或更新loop，不依赖W2 decoder/world新模块。

等待simplifier-a审查；changes回A重测两集。W2汇合仍须orchestrator实跑完整通配符与原七项。本轮没有GPU/CUDA实际搬运、服务器、网络部署、模型下载、环境升级或push。真实权重/音视频质量/FSDP2/NCCL/全rank闭环未验证。**GPU_PENDING，MVP未完成。**
