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
- 多 origin 的有序列表也进入 semantic id；每个实例都必须通过完整 manifest/identity 校验。
- **源码身份仍严格检查。** 新源码不可能具有旧源码 hash；本扩展不提供 ignore-code、跨科学配置 resume 或 step-20 warm-start/import。读取旧状态格式不等于自动准许升级旧 run。跨版本恢复仍须独立审核的代码升级路径；不扩张历史任务专用白名单。

## KL 与 EMA

KL 对每条保存的训练转移先做一次无梯度、关闭 LoRA 的双流前向，然后恢复 adapter 再建立当前策略图。视频和音频各用自己的转移 std，元素数加权；KL 不做 TGR。beta=0 不创建参考 capability、不执行参考前向。

EMA 保存每个 rank 的 local `Shard(0)`，CPU FP32，不聚合全模型。第 k 次成功提交后计数；仅在 `k % update_interval == 0` 时执行 `shadow.lerp_(live, 1-min(k/(9+k), decay))` 的等值更新。k 是已提交优化步数，不是微批次数。训练 rollout 永远使用 live；EMA 默认不改变训练。

启用时 checkpoint 的 `ema_state_schema` 为 `h3.lora-ema.local-shards.v1`，新增 `h3-ema/rank-N.pt`，由既有完整状态树哈希及原子发布管理。恢复严格检查 schema、配置、rank、分片 shape/offset/dtype、有限值、提交及更新计数。关闭时不产生这些文件。

评估/导出侧可在所有 rank 的已提交边界同步使用 `with binding.ema_weights(): ...`；内部 `H3LoRAEMA.evaluation_weights()` 在 `finally` 中恢复 live shards，禁止有梯度、嵌套 swap、swap 中训练和保存。调用者必须确保没有在途 forward/backward；不要在此作用域启动训练 rollout。

## 导出与监控

在项目环境中运行（CPU）：

```bash
python -m scripts.convert_h3_lora_to_comfyui export --checkpoint CHECKPOINT --output EMA_FILE --weights ema
python -m scripts.convert_h3_lora_to_comfyui convert --checkpoint CHECKPOINT --output COMFY_FILE --weights ema
```

`--weights live` 为 checkpoint 导出的默认值；EMA 缺失时明确拒绝，不回退 live。转换已有 safetensors 时从元数据继承 live/EMA 标识；显式选择冲突会拒绝。EMA 导出先校验 complete/tree，再按 rank/offset 重建完整 LoRA。

非默认扩展开启时，既有 `h3-last-metrics.json`、终端 `metrics.jsonl` 及每步 `h3_training_metrics` 事件增加 `kl_video/kl_audio/kl/policy_loss/kl_loss/lora_norm`；EMA 开启还增加 `ema_live_distance/ema_committed_steps/ema_updates`。`lora_norm` 和距离是全 rank LoRA A/B 参数的 L2，不是 ΔW 范数。`kl_video/kl_audio/kl` 是未乘 beta、scale 的原始 KL，先按本 rank 的 cell 取平均，再按 rank 平均（当前 plan 强制各 rank 等长）。`policy_loss` 与 `kl_loss` 是真正进入 loss 的两分量：每 cell 都包含 scale，KL 分量还包含 beta；rank 内求和、rank 间平均，与总 `loss` 同口径，二者之和在浮点舍入误差内等于总 loss。beta=0 的非默认扩展记录 `policy_loss=loss, kl_loss=0`；全默认仍无新字段。通用日志契约要求附加数值为 float，EMA 文件中的计数仍是严格整数。默认指标格式不变。

## 多 HPS 与限制

P2 每个 rank 有独立请求线程，因此调度不能仅依赖进程内计数。H3 使用 run-owned `hps-dispatch.json` 和短 `flock` 临界区共享在途计数，选择最少请求的实例、平局轮转。每次请求使用独立 Session；队列满只在同一 origin 内沿用原重试规则。任一实例失败使整个池报错，绝不剔除实例或跨 origin 重试；新尝试由生命周期在 output lock 下初始化新 generation。

保留原 P2 timeout/queue 准入，不因多实例而放宽。当前服务的 resident identity 含设备 UUID，不同设备的 resident 实例因而可能无法通过“身份全部相同”要求；本任务不删改该校验。须选择确实同身份的服务配置，或另行审核服务协议变更。

CPU 测试不证明真实 FSDP2/NCCL、bf16 数值或性能；部署前仍需八卡同输入回归、保存/恢复、EMA swap、双 HPS 分数及内存/时间验收。
