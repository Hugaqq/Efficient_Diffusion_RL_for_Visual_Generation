# MiniMax-H3 分布式 RL：VeRL-omni / UniRL / Flow-Factory 源码对照

核对日期：2026-10-27。只读上游源码及配置，没有安装模型、执行训练、操作服务器或修改 framecode。下面是当前配方/实现的事实，不是本次复测结果。此前对 H3 可训练性的疑问已通过三个仓库真实 MiniMax-H3 接入得到澄清。

## 1. 固定版本与结论

| 项目 | 本次上游提交 | 本地只读 checkout |
|---|---|---|
| verl-project/verl-omni | `ffaa9e959faba3a24d6aa282278b5f084b6457ba` | `/tmp/h3-distributed-review/verl-omni` |
| Tencent-Hunyuan/UniRL | `8cfbac17765004afb68fef08a4e011a4c8daf555` | `/tmp/h3-distributed-review/UniRL-current` |
| X-GenGroup/Flow-Factory | `3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1` | `/tmp/h3-distributed-review/Flow-Factory` |

旧桌面参考副本并非这些最新版本。本次未更新用户已有仓库。临时目录可能被系统清理，以提交固定的 GitHub 链接为长期引用。

- VeRL-omni：**训练/推理双引擎**。Actor 用 Diffusers + FSDP2，rollout 用 vLLM-Omni TP/SP；默认 H3 V1 是同卡分阶段同步执行，不是独立卡池异步训练。
- UniRL：**trainside 单模型路径**。FSDP2/HSDP 训练模型直接在 no_grad 下生成；H3 README 明确 trainside only，不把其他模型的 vLLM/SGLang 支持算到 H3 上。
- Flow-Factory：**同一 adapter + Accelerate**。DDP/ZeRO-2/FSDP2 承载采样与训练；H3 配方没有专门的 vLLM TP/SP 推理引擎。
- 三者比较的主路线是 T2VA（文本到视频+音频）的 LoRA Flow-GRPO/GRPO；另有 NFT、FL2VA、Ref2VA 配方，但不能混成同一个 benchmark。

## 2. VeRL-omni

### 实现链

`examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_v1.sh`
→ `verl_omni.trainer.main_diffusion_v1`
→ V1 sync trainer / TransferQueue / ReplayBuffer
→ H3 AgentLoop
→ vLLM-Omni H3 rollout adapter
→ reward / advantage
→ Diffusers H3 actor replay + FSDP2 backward
→ 将最新权重/LoRA 同步回 rollout engine。

证据：
- `verl_omni/trainer/diffusion/v1/trainer_sync.py:15-67`：训练与rollout colocated；采样结束sleep replicas，更新后同步权重。异步请求服务器不等于异步策略更新。
- `verl_omni/pipelines/minimax_h3_flow_grpo/vllm_omni_rollout_adapter.py:15-100`：继承 vLLM-Omni 原生 H3 pipeline；CPS joint video/audio转移和replay元数据；`supports_request_batch=False`，不是单个推理副本同时把n个视频塞进大batch。
- 同目录 `diffusers_training_adapter.py`、`weight_sync.py`：训练端replay与LoRA命名映射。
- `verl_omni/workers/rollout/vllm_rollout/vllm_omni_diffusion_strategy.py:240-312`：TP/Ulysses/Ring/encoder/VAE参数转交底层引擎并核验分配拓扑。

### T2VA V1 默认配方

| 配置 | 默认 |
|---|---|
| GPU | 1节点×8 |
| rollout DiT TP | 2 |
| text encoder TP | 2 |
| Ulysses / Ring | 1 / 1 |
| rollout workers | 8 / 2 = 4；四个双卡推理组 |
| prompts | `data.train_batch_size=32` |
| 每prompt候选 | `rollout.n=8`；名义每轮256个音视频样本 |
| 训练 | FSDP2；Actor SP=1 |
| PPO minibatch | 配置16，trainer乘n后为128条轨迹；不是全局16条视频 |
| 每GPU微批 | 1 |
| LoRA | rank64 / alpha128，attention Q/K/V/out和FFN投影 |
| optimizer | lr3e-4 / weight_decay1e-4；KL关闭 |
| 精度/内存 | BF16、activation checkpointing；FSDP reshard=true；默认无参数/optimizer CPU offload |
| train canvas | H256×W384，requested frames=121，24fps，10步，guidance1 |
| eval canvas | H512×W768，requested frames=121，40步 |
| 随机策略步 | CPS，noise0.8，从[0,8)选择连续3个transition |
| reward | CLAP + ImageBind(audio_video)，权重各1 |

batch展开证据：`trainer/diffusion/v1/trainer_base.py:1356-1357,1476-1483`。不能把prompt batch、trajectory batch和forward microbatch混同。

### 并行配置边界

- 推理组大小 = `TP × Ulysses × Ring`；外部副本数量约为总卡数/组大小。更大TP/SP通常降低单样本显存压力，但减少独立副本数，吞吐未必提高。
- Actor SP独立于rollout TP。`ACTOR_SP=2`时8卡上Actor DP=4、每个样本序列分两卡。要求SP整除attention heads（released H3为56）；packing padding/mask处理在训练adapter中。
- 纯Ulysses示例为 TP1/Ulysses4/Ring1，text encoder TP4，VAE patch parallel4、tile模式。README明确这一路拓扑的当前GPU acceptance仍pending，不能当已验收推荐值。
- H3不允许Ulysses>1与Ring>1同时开启；encoder TP只能为1或整个DiT组且整除8；VAE patch parallel只能为1或整个DiT组，mode=tile，CFG parallel=1。
- `run_minimax_h3_t2va_lora.sh` 已标 deprecated v0。它有ROLLOUT_USP/RING和VAE环境变量便捷映射；V1脚本只保留TP-only环境默认，其他拓扑需显式Hydra字段并同步worker数，不能只设置这些环境变量。
- README有16 NPU上的`fsdp_size=8`/CPU-offload示例，不能不加区分当成已验证16 GPU配置。H3 dedicated `separate_async` recipe并未提供。
- `pyproject.toml` 固定 vLLM-Omni commit `12e9280010afc41ce5c6a1f46900fc329cf08377`；不是任意安装最新版均相同。

来源：
- [V1 T2VA launcher](https://github.com/verl-project/verl-omni/blob/ffaa9e959faba3a24d6aa282278b5f084b6457ba/examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_v1.sh)
- [H3说明：SP/默认值/支持边界](https://github.com/verl-project/verl-omni/blob/ffaa9e959faba3a24d6aa282278b5f084b6457ba/examples/flowgrpo_trainer/minimax_h3/README.md#rollout-vae-and-sequence-parallelism)
- [sync trainer](https://github.com/verl-project/verl-omni/blob/ffaa9e959faba3a24d6aa282278b5f084b6457ba/verl_omni/trainer/diffusion/v1/trainer_sync.py)

## 3. UniRL

### 实现链

Hydra recipe → Ray分布式角色/DP dispatch → 同一train pipeline的TrainsideRolloutEngine → prompt分组reward/advantage → TrainStack微批累积 → FSDP2/HSDP更新。

- `unirl/rollout/engine/trainside/engine.py:18-98`：直接保存pipeline引用；`DP_SCATTER`；设eval、no_grad；按forward_batch_size切分；finally恢复模式。
- `unirl/distributed/group/dispatch.py:84-106`：按DP rank派发；不是把FSDP shard数当推理TP。
- `unirl/trainer/diffusion.py:45-103,129-161,710`：按root prompt tree派发，候选fork；prompt batch必须满足DP几何，train端按生成样本数分片。
- `unirl/train/backend/fsdp/wrap.py:51-174,271-324`：FSDP2 fully_shard、FP32 reduce、master参数、root_wrap；hybrid mesh=(dp_replicate,dp_shard)。
- `unirl/algorithms/flowgrpo.py:90-177`：old_logp_source=replay时在更新前额外no_grad replay建立旧策略锚，再做有梯度replay；不要误当纯重复一致性检查删除。

### 默认8卡T2VA配方

| 配置 | 默认 |
|---|---|
| GPU | 8 |
| prompts / 候选 | batch_size8 × samples_per_prompt16 = 128音视频样本/rollout |
| 推理 | TrainsideRolloutEngine；每forward1个；H3配方无TP/SP设置 |
| 更新微批 | 1；默认num_updates_per_batch=1 |
| 分布式 | FSDP2 full；reshard_after_forward=true；cpu_offload=false |
| 精度 | param BF16，trainable master FP32，reduce FP32；trajectory FP16，logprob FP32 |
| 包裹范围 | MiniMaxH3TransformerBlock；root_wrap=false；LoRA module_prefix=transformer_blocks |
| 内存 | activation checkpointing；meta-init transformer；辅助组件CPU；可选离线text embedding |
| LoRA / optimizer | rank32/alpha64，lr1e-4，eps1e-12，weight_decay0 |
| rollout | 256×384，124帧，10步，guidance1，CPS eta0.7 |
| policy steps | 未指定sde_indices，默认这10步均训练 |
| old logprob | replay，不直接依赖rollout logprob |
| reward | CLAP + ImageBind(mode=all)，GPU，scorer batch2 |

`algorithm.timestep_chunk_size`当前实现可逐块backward，但这个H3 YAML未显式启用。其注释指出高分辨率全10步replay的图占用风险；不能把代码存在的优化说成当前默认已启用。

### 2节点×8卡HSDP验证配置

```yaml
num_devices: 16
batch_size: 16
num_rollouts: 1
backend:
  fsdp_cfg:
    fsdp_mode: hybrid
    hsdp_shard_size: 8
    expandable_segments: false
sampling:
  height: 768
  width: 768
  samples_per_prompt: 4
  num_inference_steps: 2
  sde_indices: [0]
```

- mesh为2个副本组×每组8个参数分片；节点内参数all-gather/reduce-scatter，副本间梯度同步。
- 这是参数分片和数据并行，不是把一段视频token分成8份的sequence parallel。未配置SP时，两个8-rank分片组不意味着只有2条数据并行样本通道。
- 全局64候选；reward转CPU；text encoder按需搬GPU编码后回CPU。
- **这是num_rollouts=1、2次采样转移/1个训练转移的验证配方，不是768×768长期高质量训练完成的证明。**
- 默认YAML关于历史reward增长也明确说来自另一套配置；ImageBind/audio-only指标可能掩盖黑视频或静音退化，不能单凭reward判定质量。

来源：
- [8卡T2VA](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml)
- [2×8 HSDP验证](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/examples/diffusion/minimax_h3/minimax_h3_t2va_trainside_hsdp_2x8_validation.yaml)
- [Trainside engine](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/rollout/engine/trainside/engine.py)
- [FSDP wrapper](https://github.com/Tencent-Hunyuan/UniRL/blob/8cfbac17765004afb68fef08a4e011a4c8daf555/unirl/train/backend/fsdp/wrap.py)

## 4. Flow-Factory

### 实现链

`ff-train CONFIG` → Accelerate多进程 → BaseTrainer adapter.rollout()/sample_batch → H3 joint音视频CPS转移 → 记录选中训练transition和logprob → reward/advantage → GRPO逐microbatch、逐timestep replay/backward → optimizer。

- `src/flow_factory/trainers/abc.py:3208-3254`：同adapter的no_grad采样循环。
- `src/flow_factory/trainers/rl/grpo.py:76-189`：selected transition采样、逐timestep backward、sync_gradients时提交optimizer。
- `src/flow_factory/models/minimax_h3/workflow.py:300-405`：构建音视频schedule、调用adapter.forward、保存联合/分模态logprob、decode。
- `src/flow_factory/hparams/training_args/_base.py:475-511`：每rank sample batches = prompts×group/(world_size×microbatch)，auto累积由期望更新数派生。
- `hparams/args.py:1158-1159`：累积计数还乘训练timestep数，不能只看样本微批次数。
- H3 adapter B=1用于预处理/rollout/eval；默认配置更新也B1。没有找到H3专属TP/Ulysses/Ring配置，不能把项目支持FSDP2误称推理TP支持。

### 两份不同定位的配置

| 参数 | default.yaml | quality_720p_fsdp2.yaml |
|---|---|---|
| 后端 | Accelerate + ZeRO-2 | Accelerate + FSDP2 |
| 拓扑 | 8进程 | 2节点×8，共16进程 |
| rollout/训练模型 | 同adapter | 同adapter |
| 画布 | 512×768，明确unvalidated starting geometry | 768×1344 |
| 帧数/fps | 124/24 | 124/24 |
| 采样步数 | 24 | 24 |
| 每卡microbatch | 1 | 1 |
| prompts/group | 48/16，即768样本每采样epoch | 同左 |
| 目标更新次数/epoch | 2 | 2 |
| policy transition | CPS noise0.8，从候选[0..9]选3步 | 同左 |
| LoRA | rank64/alpha128 | 同左 |
| optimizer | lr3e-4，weight_decay1e-4，eps1e-8 | 同左 |
| PPO clip / KL | 1e-4 / beta0 | 同左 |
| EMA | decay0.9，interval4 | decay0（禁用） |
| 轨迹存储 | CPU offload | CPU offload |
| reward | CLAP + ImageBind(audio_video)，GPU | 仅CLAP，CPU，batch1 |
| activation checkpoint | true | true |

以16卡配方为例：768样本/16=48个本地B1样本；2次更新→名义每次全局384视频，样本累积24；3个训练transition→执行层累积计数还要乘3（72），这是不同维度。

`config/accelerate_configs/fsdp2.yaml`指定FSDP2、TRANSFORMER_BASED_WRAP、reshard_after_forward=true、offload_params=false、activation_checkpointing=true、FULL_STATE_DICT。高分辨率YAML覆盖进程/节点数16/2；不能只读base文件中的8/1。

上游说明的验收边界：低分辨率debug完成rollout/reward/replay/backward/checkpoint/resume；native-resolution quality路径完成初始化/checkpoint/decode/CLAP evaluation。其36-cell矩阵采用debug几何。**不能把文件开头“validated on16 GPUs”扩张成768×1344长跑RL质量已经验证。**

来源：
- [默认ZeRO2配置](https://github.com/X-GenGroup/Flow-Factory/blob/3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1/examples/grpo/lora/minimax_h3_t2va/default.yaml)
- [高分辨率FSDP2配置](https://github.com/X-GenGroup/Flow-Factory/blob/3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1/examples/grpo/lora/minimax_h3_t2va/quality_720p_fsdp2.yaml)
- [FSDP2 base](https://github.com/X-GenGroup/Flow-Factory/blob/3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1/config/accelerate_configs/fsdp2.yaml)
- [GRPO loop](https://github.com/X-GenGroup/Flow-Factory/blob/3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1/src/flow_factory/trainers/rl/grpo.py)
- [H3实现及验证边界](https://github.com/X-GenGroup/Flow-Factory/blob/3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1/.agents/knowledge/topics/minimax_h3.md)

## 5. 对 VisualRL 的架构启示（建议，不是已实现）

1. 优先借鉴UniRL/Flow-Factory的同步同模型路径，打通H3联合音视频状态、FSDP2 LoRA和安全恢复；减少初期双引擎映射/同步成本。是否选FSDP还是HSDP取决于节点互联与显存，不凭GPU数量一刀切。
2. 采样吞吐确为瓶颈时，再增加可替换rollout backend，参考VeRL-omni的TP副本、encoder TP及VAE并行；不先复制其所有TransferQueue/actor系统。
3. 提高global batch优先靠样本分配和梯度累积；这三条H3配方核心执行均以微批1为主，不是每卡堆大video batch。
4. FSDP主要减少参数/梯度/optimizer的存储压力，不直接切分长音视频序列的activation。高分辨率需求需要独立评估SP、timestep chunking、activation checkpoint、VAE decode和轨迹CPU内存。
5. H3特有精度岛、audio/video双schedule、old-policy anchor是算法/数值条件，不应当成不必要保护全部删掉。
6. 质量目标需明确是T2V还是T2VA。CLAP只约束音频文本，ImageBind(audio_video)也不能替代视频视觉质量/动态质量评分。三者当前reward组合不同，不可直接横比reward数值。
7. VeRL脚本requested frames121与UniRL/Flow-Factory显式要求124存在后端约定差异。本次未提取底层vLLM帧对齐实现，未验证最终输出帧数；迁移时必须核对真实geometry，不静默互换。
8. 未取得同卡型、分辨率、步数、reward和batch条件的吞吐实验，不能给三者速度排名或承诺VisualRL移植后的提速倍数。

建议下一步先确定GPU型号/数量、单机NVLink与跨节点网络、H3任务类型和目标画布，再设计第一版拓扑及统一性能测量基线。
