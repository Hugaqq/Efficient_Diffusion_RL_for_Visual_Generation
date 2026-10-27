# H3-M4c2a-runtime-session — W7 Coding B

## 状态与身份

- `implementation_ready: true`，仅本原子项实现/CPU门禁完成，交simplifier-b；不批准A binding或C组合测试，不代表默认入口完成。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-cda11d5b-7bd1-430d-afb4-dab60628f6e5`。
- 开始/结束HEAD：`cc6e19f7d35a34d443fe7942a3c33e00ea7bb631`；初始干净。未切分支、手工commit/merge/push；snapshot待引擎生成。
- 已读当前task/plan/assignment B/wave-07、最近本配对approved反馈及实际profile/facts、WORLD receipt、prepare准入合同。不依赖A-BIND-01新符号或C本波测试。

## 改动与接口

仅新增`visual_rl/runtime/h3_session.py`、`tests/test_h3_mvp_cpu_session.py`及本记录。旧composition/controller/lifecycle/train/profile/binding/prepare、全部原135项和Flash七项未改。

```python
session = H3RuntimeSession.create(profile, facts)
# 只读：process_group, rank, local_rank, local_world_size, device, gpu_uuids
# 下一默认create_runtime阶段将这些实际值传给既有prepare/binding。
session.close()  # 幂等，仅自己WORLD，无barrier
```

无可传verified=True的旁路；直接构造拒绝。模块import无CUDA/PG动作，torch/distributed及现receipt在create/close处lazy使用。无模型/optimizer/HTTP/checkpoint/训练循环责任，无Accelerator、新mesh、子组、RPC或CPU生产fallback。

## 实际准入顺序

1. 要求真实H3Profile/H3LocalFacts。核验launch binding ID与semantic payload摘要，并核对配置模型/固定数学/seed/max steps/expected身份/数据parse与hash/cache/model-config事实内部一致；不是只因都是hash字符串就认为相同。session不重新做权重materialization或把facts的未验权重状态升格为ready。
2. torchrun环境必须显式且规范：WORLD_SIZE=LOCAL_WORLD_SIZE=8、GROUP_WORLD_SIZE=1、GROUP_RANK=0、RANK=LOCAL_RANK且0..7，MASTER_ADDR非空且有界、MASTER_PORT为有效端口。缺失/非法/多节点均拒绝，不默认rank0或物理0..7。
3. CUDA_VISIBLE_DEVICES必须恰为profile的八个有序完整UUID；无数字index/截短UUID/MIG/多余十卡fallback。真实CUDA device_count必须8，并读取每个logical device的`get_device_properties(index).uuid`逐一对照；重复/错序/缺能力拒绝，绝不用name/总显存推断身份。
4. dist/NCCL可用且不存在外来已初始化PG；CUDA可用。设置显式`cuda:local_rank`，回读current_device，并调用公开`is_bf16_supported(including_emulation=False)`核验选中卡native BF16。缺该API能力/无支持明确拒绝，不借emulation或版本号放行。
5. 只调用一次公开`init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=180))`。180s为本窄session显式固定有限timeout；不复用无关HPS timeout、不自动重试。保存成功初始化返回后的WORLD。
6. 实际回读backend/world/rank；无效通信拓扑不能进入固定8rank收据。有效WORLD后复核current device和UUID映射，复用`h3_world._exchange('session/admission',...)`的有界JSON bytes收据：同host hash、rank顺序、profile source/declaration/binding/semantic IDs、有序UUID和local ranks必须一致。没有gather模型或媒体。

现公共CUDA属性uuid支持完整带GPU-前缀或裸UUID字符串及stdlib UUID对象，统一为canonical GPU-UUID；缺属性或部分值明确报UUID观察能力不可用。CPU测试不调用真实CUDA，因此**没有证明当前Mac或目标服务器的该属性/驱动能力可用**；目标环境必须实际走此分支准入，不能补假UUID或安装升级绕过。

## 失败与所有权

- 进程内creation lock避免两个本owner同时初始化；已经有PG则在CUDA设备选择之前拒绝，不销毁或重配它。
- 仅init成功返回后的WORLD归本owner。后续结构/metadata/transport错误释放该已取得的PG；raw init失败不产生成功owner，也不把未知外部PG认作自己所有。
- 有效WORLD上的可协作local错误发送失败收据，保留原异常链；raw transport失败不在finally追加collective/barrier或retry。PG建立前的本地失败无法共同通信，其他进程需依有限timeout/外层torchrun停止整个attempt；未承诺死rank故障可恢复。
- close只在保存的PG仍是当前WORLD且已初始化时，调用`destroy_process_group(saved_pg)`；显式传原对象。外部替换WORLD则不误销毁新PG。close先失效owner引用，重复close不再销毁；cleanup失败不覆盖create的原异常，附note后传播。
- 不reset CUDA context、不改环境/RNG/线程/NUMA；选中的CUDA device保留，不声称释放全部CUDA内存。无模型/scorer close、杀进程或删除输出。

## 七项CPU证据

测试使用真实tmp profile/facts与实际session控制；仅CUDA availability/count/properties/selection/BF16及dist PG/通信API叶子替身，无真实GPU、PG、网络或服务调用：

1. 顺序模拟rank0..7合法观察，检查公开API参数/调用顺序、一次init/一次收据、只读属性、PG独占与幂等close、环境和Torch RNG未变。
2. profile/facts绑定及自洽摘要但数据事实错误、环境缺失/不规范/uneven/多节点/端口错误，在CUDA/PG前拒绝。
3. visible UUID缺失/数字/错序/重复/MIG及实际设备十卡/错序/重复/缺uuid/短uuid拒绝；设备name+64GiB不能替代UUID。
4. CUDA/dist/NCCL/BF16能力缺失或旧BF16 query API、外来PG拒绝，无生产CPU fallback。
5. 同host/rank/UUID/facts/local-world收据分歧、peer失败、超收据大小拒绝；初始化后设备漂移真实走失败receipt并保存cause。
6. 创建后backend/world/rank错误、rendezvous/transport timeout不重试、不补rescue collective；拥有PG销毁一次，cleanup二次错误不掩原错误。
7. WORLD被外部替换不销毁新owner；明确teardown错误时close仍幂等。

失败spy同时禁止重模型/VAE/HPS health/HTTP/optimizer构造；barrier/new_group均失败spy。收据替身只模拟通信返回，不mock整个create、profile校验或UUID映射验证。这里不是实际八进程NCCL测试。

## 最终门禁

最后源码/测试编辑后原样执行，均exit0，无skip/xfail、无全量或warning过滤：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 11.90s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 142 passed, 18 warnings in 13.50s（基线135+本项7，不含W7兄弟未汇合新增）
```

18条既有checkpoint no-dist/损坏fixture warnings原样保留。`git diff --check` exit0；原Flash完整文件相对d818b8a获准历史diff为空。当前仅三个授权新文件，不以空tracked diff代替新文件审核。

## 后续边界

交simplifier-b；changes回B修复并重测。Orchestrator汇合后再次检查A/B/C各自裁决与完整目标。

本session只有请求UUID与实际CUDA映射一致性检查，**不是用户授权、独占/空闲窗口、GPU或主机资源余量证明**。未读取历史硬件图推断资源ready；真实8UUID/窗口、BF16/UUID/NCCL API、H3/HPS/权重与性能仍须另行授权验证。

session/binding/恢复组合稳定后，下一任务将本owner接现唯一RunController/default create_runtime并按原生命周期关闭；本项未修改默认入口或实现完整整run恢复。**MVP未完成，GPU_PENDING。**
