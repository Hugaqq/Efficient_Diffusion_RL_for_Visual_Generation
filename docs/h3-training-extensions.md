# H3 可选训练扩展

这些能力不代表新的实验已获准执行；beta、EMA 参数由实验卡决定。

## 配置与身份

在既有 YAML 中配置：

```yaml
overrides:
  training:
    # 其他原有设置照旧；非零 KL 仅支持 dual_flow / dual_flash。
    kl_beta: 0.0
    ema:
      enabled: false
      decay: 0.9
      update_interval: 8
      counting: committed_optimizer_steps.v1
    advantage_std_domain: batch  # 可选 group
```

HPS 仍要求 `launch.hps.origin`，可另给 `origins: [origin1, origin2, ...]`。
必须是 1–8 个不同的 canonical loopback HTTP origin，第一项等于 `origin`。
单项等于旧 origin 时不会新增身份字段或更换客户端。

- 默认字段不进入旧 payload；保留旧 semantic id、numerics、checkpoint schema。
- 任一非默认科学设置进入 semantic id；即使 EMA 关闭，改变 decay/interval 也记录身份。
- 多 origin 的有序列表和 `hps.scoring-identity.v1` scoring identity 进入运行身份。规范化复制 facts 后，只移除 `gpu_residency_policy.expected_gpu_uuid`，连同完整视频契约、协议及输出设备/精度哈希；residency 其余字段、queue、模型/processor/源码/依赖等全部保留。池内 scoring identity 必须完全相同，各端点仍自证、固定并使用自己的完整 service identity 发请求和验分。单 origin 的原身份、请求和 artifact 精确匹配规则不变。
- **源码身份仍严格检查。** 新源码不可能具有旧源码 hash；本扩展不提供 ignore-code、跨科学配置 resume 或 step-20 warm-start/import。读取旧状态格式不等于自动准许升级旧 run。跨版本恢复仍须独立审核的代码升级路径；不扩张历史任务专用白名单。

## KL 与 EMA

KL 对每条保存的训练转移先做一次无梯度、关闭 LoRA 的双流前向，然后恢复 adapter 再建立当前策略图。视频和音频各用自己的转移 std，元素数加权；KL 不做 TGR。beta=0 不创建参考 capability、不执行参考前向。

每个 optimize 的 cell 循环只捕获一次 RNG、buffer 和 requires_grad 标志；参考前向仍逐 cell 切换 adapter、no_grad 并恢复。H3 EVAL 优化路径无随机算子，参考 RNG 恢复复用同一更新快照；不适用于引入随机 policy forward 的新协议。buffer 在循环退出时只比较一次，变化则恢复并拒绝，检查完成前不执行 optimizer.step；异常同样恢复 adapter/flags/RNG。参考路径的全模型验证移到更新边界，不在每个 cell 重复遍历。

EMA 保存每个 rank 的 local `Shard(0)`，CPU FP32，不聚合全模型。第 k 次成功提交后计数；仅在 `k % update_interval == 0` 时执行 `shadow.lerp_(live, 1-min(k/(9+k), decay))` 的等值更新。k 是已提交优化步数，不是微批次数。训练 rollout 永远使用 live；EMA 默认不改变训练。

默认 decay=0.9、interval=8 的依据是官方 `bd6051f:scripts/train_wan2_1_flash_1node.py` 第 521–523 行显式构造 `EMAModuleWrapper(..., decay=0.9, update_step_interval=8, ...)`；`flow_grpo/ema.py` 类默认的 0.9999 不是该脚本的实际参数。本实现按已提交优化步计数，不声称复制官方微批循环的逐次调用时序。

启用时 checkpoint 的 `ema_state_schema` 为 `h3.lora-ema.local-shards.v1`，新增 `h3-ema/rank-N.pt`，由既有完整状态树哈希及原子发布管理。恢复严格检查 schema、配置、rank、分片 shape/offset/dtype、有限值、提交及更新计数。关闭时不产生这些文件。

评估/导出侧可在所有 rank 的已提交边界同步使用 `with binding.ema_weights(): ...`；内部 `H3LoRAEMA.evaluation_weights()` 在 `finally` 中恢复 live shards，禁止有梯度、嵌套 swap、swap 中训练和保存。调用者必须确保没有在途 forward/backward；不要在此作用域启动训练 rollout。

## 导出与监控

在项目环境中运行（CPU）：

```bash
python -m scripts.convert_h3_lora_to_comfyui export --checkpoint CHECKPOINT --output EMA_FILE --weights ema
python -m scripts.convert_h3_lora_to_comfyui convert --checkpoint CHECKPOINT --output COMFY_FILE --weights ema
```

`--weights live` 为 checkpoint 导出的默认值；EMA 缺失时明确拒绝，不回退 live。默认 live 不写入 `weights` 元数据，保持 7c85751 的张量与完整 safetensors header 键集合及值；只有 EMA 写入 `weights=ema`，读取缺失标识解释为 live。转换已有 safetensors 时从元数据继承 live/EMA 标识；显式选择冲突会拒绝。EMA 导出先校验 complete/tree，再按 rank/offset 重建完整 LoRA。

独立解释器重复执行 7c85751 的 export/convert 已观察到文件 SHA 不同：原始 header 的 `__metadata__` 键序不同，而解析后的完整 header 与 tensor payload 字节一致。因此回归门禁比较 header 内容与张量，不要求文件 SHA；不修改基线的序列化行为。

非默认扩展开启时，既有 `h3-last-metrics.json`、终端 `metrics.jsonl` 及每步 `h3_training_metrics` 事件增加 `kl_video/kl_audio/kl/policy_loss/kl_loss/lora_norm`；EMA 开启还增加 `ema_live_distance/ema_committed_steps/ema_updates`。`lora_norm` 和距离是全 rank LoRA A/B 参数的 L2，不是 ΔW 范数。`kl_video/kl_audio/kl` 是未乘 beta、scale 的原始 KL，先按本 rank 的 cell 取平均，再按 rank 平均（当前 plan 强制各 rank 等长）。`policy_loss` 与 `kl_loss` 是真正进入 loss 的两分量：每 cell 都包含 scale，KL 分量还包含 beta；rank 内求和、rank 间平均，与总 `loss` 同口径，二者之和在浮点舍入误差内等于总 loss。beta=0 的非默认扩展记录 `policy_loss=loss, kl_loss=0`；全默认仍无新字段。通用日志契约要求附加数值为 float，EMA 文件中的计数仍是严格整数。默认指标格式不变。

## 多 HPS 与限制

P2 每个 rank 有独立请求线程，因此调度不能仅依赖进程内计数。H3 使用 run-owned `hps-dispatch.json` 和短 `flock` 临界区共享在途计数，选择最少请求的实例、平局轮转。每次请求使用独立 Session；队列满只在同一 origin 内沿用原重试规则。任一实例失败使整个池报错，绝不剔除实例或跨 origin 重试；新尝试由生命周期在 output lock 下初始化新 generation。

保留原 P2 timeout/queue 准入，不因多实例而放宽。UUID 可不同，但任何其他 identity facts 字段不同都拒绝。文件 dispatch state 仅正常退出回写/flush；崩溃不回收 lease、不修复在途计数，不据此跨实例重试。

## 既有测试改动与 checkpoint 缺口披露

`7d58251` 修改了既有 CPU 测试；独立审查在 `7c85751` 与 `7858c33` 重跑相同 checkpoint 测试组均为 11 failed / 9 passed，不能把修正后的全绿当作新实现补齐了旧缺口。具体降低覆盖的改动如下：

| 既有测试 | 在 7c85751 已不成立的断言/前提 | 7d58251 改成了什么 |
| --- | --- | --- |
| `test_h3_mvp_cpu_checkpoint_state.py::test_h3_codec_real_disk_save_purity_fresh_adam_restore_and_next_step` | 对 fresh 的冻结 `context_embedder.weight` 人为加 123 后，restore 再 `_equal(_snapshot(fresh), before)`（内部 tensor equal）失败：冻结参数未被恢复 | 删除这一次冻结参数扰动，保留 persistent buffer 加 17、恢复后完整 snapshot/Adam/下一步相等断言；**不再测试修复损坏的冻结基座** |
| `test_h3_mvp_cpu_checkpoint_publish.py::test_h3_allrank_publish_private_rng_full_tree_and_prepared_restore` | 要求八 rank 都调用 `model-set`，adapter.v2 已不调用全模型 setter | 只要求八 rank `load`，另明确断言没有 `model-set` |
| `test_h3_mvp_cpu_bound_checkpoint.py::test_bound_source_tail_publish_fresh_restore_and_real_next_iteration` | 要求八 rank 都调用 `set-model`，同样不成立 | 删除 `set-model` 调用计数要求，保留八 rank `load` 及 snapshot/下一次真实更新相等 |

其余既有 fixture 修改不是新学习能力的证明：prepare/binding/checkpoint 的 file heartbeat、broadcast、aux turns、CPU decoder 补齐 CPU transport；all_reduce fixture 从无维度 sum 改为 sum(0)；发布 fsync 的线程断言从 rank0 TLS 改为 `h3-checkpoint-publisher`；假 contract 的 EMA/reference schema 改为 `none`；两个 dual-Flash 测试设 num_steps=10，符合已经存在的 first-five 非终点约束。基线部分测试因此在上述语义断言之前就失败。

**冻结基座不由 checkpoint 恢复、依赖 `verify_h3_content` 的模型 artifact 内容校验，是 7c85751 已有的 adapter.v2 缺口，不是本分支引入。** 本轮不修改生产恢复逻辑去修这个缺口；后续应独立核查 LoRA 子集 restore 与冻结基座运行时内容保证的边界。

## 默认路径独立提交回归

`tests/test_h3_default_subprocess_parity.py` 从 `git archive 7c85751` 和运行时的 `git archive HEAD` 提取两份完整源码，各用一个 `python -I -B` 子进程执行默认 dual-Flash 两次真实 CPU AdamW 更新。比较的是 **HEAD，不是未提交生产修改**。每侧运行自己的 profile、binding/predict、HPS client、lifecycle、stages、checkpoint transaction/codec；逐模块验证 `visual_rl` 导入来源属于该侧 archive。没有向旧实现注入新版生产方法。旧的局部方法对照现明确命名为 stage-math 测试，不再作为完整提交证明。

两侧使用同一份固定 CPU 测试 fixture 模块：小模型/decoder、HTTP 响应、CUDA/FSDP 观察、线程内八 rank 通信、独立 rank RNG 观察与 DCP no-dist 传输叶。仅测试输入 safetensors 的 header 键序做规范化，输入文件内容哈希必须完全一致；生产导出和 checkpoint 序列化不改动。HTTP 叶根据实际请求 JPEG 像素给确定分数，并非真实 HPS 模型。

逐项比较默认 semantic payload/id、训练配置、每步 rollout 的状态/action/old_joint/终态、reward/advantage、loss/grad norm、梯度、每步参数/Adam、最终 scheduler/buffer、两步完整 checkpoint 逻辑内容、两步指标字段及值、32 个 HPS POST 请求字节。张量比较 dtype/shape 和原始字节。checkpoint 每侧先独立 `inspect_complete` 校验真实磁盘树，再比较所有 JSON、八 rank 状态及从真实 DCP 读取的全部状态；不是只比较写入函数的参数。

必然变化字段显式单列于测试输出 `separate-fields.json`：真实源码 hash/file_count/byte_count，以及由源码派生的 resources、recipe、algorithm、runtime-bound、checkpoint-contract 身份；派生哈希由测试用标准 JSON/SHA256 独立重算。物理文件差异仅分离 DCP `.metadata`（保存 UUID 等）、`h3_transaction.json` 的 SHA 与 state-tree 身份，仍比较文件集合/大小、完整逻辑内容并校验各自 tree。其余身份（含 semantic、progress、RNG/data、manifest）不忽略。完整提交 SHA 另存 `source-revisions.json`。

无法在这个 CPU 小模型中比较的层：CUDA/WORLD 准入、真实 FSDP 分片/通信、file-phase heartbeat/aux-turn 调度和进程全局 RNG 恢复。fixture 对这些层使用相同替身（RNG checkpoint 为固定的逐 rank 观察值，不是八线程共享的全局 RNG）；实际 rollout 仍使用生产代码创建的独立 generator。DCP 使用真实磁盘格式，但替换分布式传输为校验八份副本一致后的 no-dist 保存。替代证据是实际两步参数/Adam、rollout/action、落盘状态与请求字节对照，不把被替换的层计入完整实现回归。

这证明限定 CPU 叶下的默认回归，不证明真实 H3/FSDP2/NCCL、GPU 数值/性能或真实 HPS。八进程 gloo EMA 测试另行覆盖真实 DTensor 分片、空 shard、保存/恢复及异常换回，不替代完整训练的 GPU 验收，也不将这里的八线程通信冒称真实分布式训练。

## 启动器短测与 EMA 磁盘预算

正式入口默认仍是 200 更新；显式 `--target-updates N`（1..199）要求 profile 的 `launch.max_optimizer_steps=N`，其他正式几何/数据/优化器/cadence 约束不变，不与 `--max-optimizer-steps` 混用。N 已属于原有 semantic payload 和 formal-run identity：不同 N 恢复同一 run 必须拒绝。因此 2+1 测试 fresh/resume 均设 N=3，step-2 用 PAUSE，在 step-3 正常完成（终点不会抛 SafeBoundaryPause）。不能 fresh 用 N=2、resume 改 N=3。

短测磁盘按剩余每个更新都可能 PAUSE 保存预算，覆盖 step-1、step-2 暂停和 step-3 终点；视频仍按原 retention。监控按剩余更新数检查寿命/字节，不改每步秒数、磁盘 1.2 倍余量或运行时安全阈值。目标在既有 launch identity/argv/log 中记录；未传新参数时 parser 字段、身份、EMA 关闭预算与 7c85751 相同。

EMA 预算只读实际 transformer safetensors header 的六个 LoRA target，推导 A/B shape 与 WORLD8 Shard(0) 的每 rank 元素数。每个 checkpoint 增量为各 rank `numel*4 + tensor_count*8192 + 65536` 总和（与 load/export 文件上界一致），inode 增量为八文件加一个目录，不再翻倍整个 checkpoint。真实八进程 CPU 保存的文件总字节由测试确认不超过预算；CPU 数据不代表 GPU 验收。

## GPU 验收清单（这里只列出，不执行）

- `kl_beta>0` 首个参考前向必须成功；确认真实 H3 forward 没有懒写 rope/cache/counter buffer。变化时应恢复并拒绝，而非放宽保护。
- 首步零 LoRA 的逐流/合并 KL=0；扰动后有限且有梯度；检查 scaled `policy_loss+kl_loss≈loss`，KL 不乘 TGR。
- 八 rank 在 acquire 时读到同一 `hps-dispatch` generation；读失败立即拒绝，不等到评分。
- bind 时 EMA 构造成功；首个 EMA 更新的每 rank shadow 与手算 lerp 逐位相等（含空分片）。跨更新边界保存/恢复及下一步与连续运行一致。
- 默认全关真实同输入对照 7c85751；EMA swap 异常恢复；双 HPS 同 JPEG 的帧分数/top37/总分一致及实例故障拒绝；记录 KL/EMA 增量耗时和内存峰值。

新增 CPU 测试确实使用 8 个进程、gloo、`init_device_mesh('cpu',(8,))` 和 `distribute_tensor(Shard(0))`，未替换 `local_parameter/_local`，覆盖行数 2/3/9/10、空分片、offset、restore、swap 与首步 lerp。checkpoint 默认 CUDA 校验不变；EMA 显式按参数 device 复用同一 WORLD8/Shard(0)/process_group 校验。CPU 测试仍不证明真实 FSDP2/NCCL、bf16 数值或 GPU 性能。
