# Wan post-merge validation — `2adfbfd`

日期：2026-07-15

受测产品提交：`2adfbfd`（`fix: preserve Wan SDE latent precision`）

## 结论

`2adfbfd` 修复了 Wan 重算路径中的精度边界：transformer 仍接收 BF16 输入，而 Flash
SDE 的 current/next latent 保留 rollout 产生的原始 FP32。该修复通过本地回归、真实
Wan W7b reference parity，以及 World-R1/Flash-GRPO 两组 bounded 两步训练与恢复复验。

目前证据足以说明这条受测路径能够正确跑通，并在两步范围内精确恢复；它不构成质量、
速度、显存效率或长程稳定性的结论。

## 产品修复与本地验证

旧实现把 latent 转成 transformer 的 BF16 dtype 后，再 `.float()` 交给 SDE；这只能把
已经舍入的数值扩回 FP32。`2adfbfd` 将二者拆开：

- `model_input_j` 只用于 BF16 transformer 前向；
- `sample_j` 与 `next_sample_j` 保持源 latent 精度，并以 FP32 进入 SDE；
- 新回归使用非 BF16-exact 数值，同时检查 transformer 的 BF16 契约和 SDE 的 FP32
  bitwise 保真。

本地验证：

- 非分布式：`883 passed, 2 skipped, 5 deselected`；
- Gloo：`5 passed, 885 deselected`。

远端执行使用只包含 `visual_rl/`、`scripts/`、`pyproject.toml` 与 `train.py` 的最小源码
归档；没有上传 exercise、checkpoint、权重或本机配置。执行身份如下：

| 项目 | SHA256 |
| --- | --- |
| 最小源码归档 | `8c693b44c318aa7e13ee4b925a95bf3534f9e16196017f00c3707238f7ce4856` |
| W7b heterogeneous parity harness | `b91fb4c918c0a9984a4645b43e4416eb7c49948d7fdbfa2b247f5a62822a1adc` |
| W5 attempt 2 resume harness | `909f0b10a10f41bc2757377b58a1ab25b6de7b3361aa6378d33899e2ec805e2e` |
| reward_general probe | `e0d877212864ebd7237f200cf9bb87b5ed55b32d5d9b89727a6a63ca9a64d0ad` |

## W7b：Flash heterogeneous parity

远端结果 `valid=true`。以下门禁全部通过：

- sample 顺序与 tensor 精确一致；
- selected indices、按 selected index 派生的 grouped seed、scheduler metadata 精确一致；
- Flash reference contract 与动态 SDE coefficient 精确一致；
- 恰好保留一个 transition；
- 重算 logprob 与 loss 精确一致；
- 480 个 LoRA gradient tensor 全部 bitwise exact；
- 参数在验证前后不变。

关键身份与哈希：

| 项目 | SHA256 |
| --- | --- |
| adapter 源码 | `87e3b3ea3a0c90855abcba203b4639e9275d353de16c5df90bae90e00af1c967` |
| `w7b_attempt1/result.json` | `68c34aaae935c6d64afb2424ccd311340f1be7a142828b4b4d968d908a469153` |
| `w7b_attempt1/wan_w7b_2adfbfd_attempt1.log` | `fc4ed93963ebcfe050a854875868cf63ab40086cc8dae746203e20eb31ee885e` |

## W5 attempt 1：harness fail closed

第一次 World-R1 与 Flash-GRPO 尝试都在训练前失败，错误为：

```text
deterministic runtime must be configured before CUDA is initialized
```

原因是实验 harness 先调用 `torch.cuda.set_device/empty_cache/reset_peak_memory_stats`，
之后才构造 `ExperimentRunner`；而 deterministic runtime 必须在首次 CUDA 初始化前设置。
这是 harness 顺序错误，不是模型训练失败。失败证据被保留且没有覆盖。

修复方式是将 `ExperimentRunner(config)` 移到所有 `torch.cuda.*` 调用之前，并加入动态
monkeypatch 回归，实际验证 runner 事件先于第一项 CUDA 事件；相关 harness 测试结果为
`3 passed`。

## W5 attempt 2：六段有效，恢复精确

修正 harness 后，两个 backend 各运行 continuous-2、split-1、resume-to-2：

| Backend | Segment | 完成步数 | status valid | marker valid | audit valid | 结果 |
| --- | --- | ---: | --- | --- | --- | --- |
| World-R1 | continuous | 2/2 | true | true | true | valid |
| World-R1 | split | 1/1 | true | true | true | valid |
| World-R1 | resumed | 2/2 | true | true | true | valid |
| Flash-GRPO | continuous | 2/2 | true | true | true | valid |
| Flash-GRPO | split | 1/1 | true | true | true | valid |
| Flash-GRPO | resumed | 2/2 | true | true | true | valid |

两份 comparison 的 11 个 exact gate 均为 true：

1. `all_final_runs_secure`
2. `all_segments_valid`
3. `final_adapter_exact`
4. `final_checkpoint_metadata_semantic_exact`
5. `final_training_state_exact`
6. `manifest_semantic_exact`
7. `metrics_rows_2_1_1`
8. `resume_loaded_step_one`
9. `split_final_hash_equals_resume_initial`
10. `step_one_metrics_exact`
11. `step_zero_metrics_exact`

最终 adapter 与证据哈希：

| Backend | continuous/resumed adapter SHA256 | comparison SHA256 | log SHA256 |
| --- | --- | --- | --- |
| World-R1 | `334e0ff0811f42cc2394b3a3f4a3c50aff6a8729f9ce31c4fafef16bf12a1a75` | `1506ac35afc3c8a429c442db4048564acac2c78f323a54d5544e7347c6d6be31` | `a77842ceef8514b006c27ad81c65e9416ff4000470f065da628dd88f35739aec` |
| Flash-GRPO | `9ee9afc5a03e75816552d92c98f6b3c3c8f7109021a38b13b135cc62cee793e6` | `3ca5fd55c4d475d4d228b4f3dfca8c8e5d4a03f876fc67d2de0dd12a8b80c953` | `ca5dfa01f50cc331311914a648ffe4ad233273ee435e7403434762a2efc17a0d` |

checkpoint tree 的归档字节不作为相等门禁；每棵树先独立通过 SHA256/marker 校验，再比较
反序列化后的训练状态与语义字段。纯运行时间、吞吐和排队延迟属于 observational metrics，
不参与 deterministic equality。

## reward_general attempt 1：loopback reference 一致性

post-merge reward probe 在 `2adfbfd` 上为 `valid=true`。固定输入下的三条路径输出完全
一致：

| 路径 | scores |
| --- | --- |
| direct reference | `[0.260009765625, 0.1943359375]` |
| reference HTTP | `[0.260009765625, 0.1943359375]` |
| VisualRL | `[0.260009765625, 0.1943359375]` |

`max_abs_http_vs_direct=0.0`，`max_abs_infra_vs_direct=0.0`。坏 pickle 与坏图像请求都返回
HTTP 500 并带 traceback，且 `silent_fallback_detected=false`，因此没有把协议或解码错误
静默替换成伪 reward。采证结束后 harness 终止 server，`server_returncode=-15`。

峰值 CUDA allocated/reserved 分别为 `7,874,462,208` / `8,250,195,968` bytes。证据哈希：

| 文件 | SHA256 |
| --- | --- |
| `reward_general_attempt1/result.json` | `6c7ccdb9ce8f0af04660bd722f09f4d331db71adc7d67b46cdb8d084bd16328e` |
| `reward_general_attempt1/server.log` | `82af7c01776ea301fc64d5d4611b579a199f7cf536281d8e468a2e39a482ff13` |
| `reward_general_attempt1/reward_general_2adfbfd_attempt1.log` | `73fa49d0d048684f200551ed6c5fcf535d6265b2f6b048dc0ce77b663450a39a` |

该探针显式限定 `network_scope=loopback_only`、`protocol_mode=reference_v1`、
`wire_format=legacy_pickle` 和 exact trusted host。它只验证这条受信任本机兼容协议的数值
一致性与错误处理，不表示 legacy pickle 可安全扩展到远端，也不构成 reward 效果或性能结论。

## 证据完整性

证据根目录（相对于本文件所在目录）：

```text
evidence/remote/wan/2adfbfd/
```

该目录的 `SHA256SUMS` 列出 101 个保留文件；本地执行完整校验为 101/101 `OK`。证据包括
W5 attempt 1 的失败摘要与日志、attempt 2 六段受控产物及两份 comparison、W7b 结果与
日志，以及 reward_general attempt 1 的结果与日志；没有用历史 W5/W7 结果替代
post-merge rerun。

## 结论边界与后续验证

本轮只证明：

- 真实 Wan2.1 路径在限定配置下可完成两步更新；
- 梯度门禁、artifact/status/marker/checkpoint 安全门禁通过；
- continuous 与 split/resume 在两步范围内的训练语义和最终 adapter 精确一致；
- W7b 的 Flash reference parity 与 FP32 SDE latent 契约成立；
- loopback legacy reward_general 协议与 direct reference 数值一致，并对坏请求 fail closed。

本轮尚未证明：

- 训练后生成质量优于基线；
- 长时间或多节点训练稳定；
- 吞吐、显存或收敛效率提升；
- 不同数据规模、seed、reward 组合下仍保持相同结论。

下一阶段应使用多 seed、足够长的训练预算和固定评估集，分别报告质量、速度、峰值显存
与恢复稳定性，不能把本次 bounded 两步结果外推为成熟度或性能结论。
