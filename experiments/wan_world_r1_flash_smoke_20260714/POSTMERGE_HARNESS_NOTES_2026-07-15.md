# Wan / World-R1 / Flash post-merge harness notes (2026-07-15)

本文件只说明合并后 harness 的验证口径，不改写 `W1_W2_RESULTS.md`、
`W3_W4_RESULTS.md`、`W5_RESULTS.md`、`W6_W7_RESULTS.md` 中保存的历史结果。

## 证据边界

旧 W3/W4/W5 证据来自 checkpoint v2、旧 manifest/status 投影和旧 Flash
rectification 口径，不能直接作为当前合并代码的 checkpoint v4、manifest v2、
authoritative commit marker、动态 Flash coefficient 或 artifact audit 证据。旧 W7/W7b
结果仍可用于说明当时版本的 bitwise parity，但不能替代 post-merge rerun。

post-merge 复验现已完成，受测产品提交为 `2adfbfd`（`fix: preserve Wan SDE
latent precision`）。该修复把 transformer 输入单独转换为 BF16，同时让 Flash SDE 的
current/next latent 保持原始 FP32，避免重算时先经 BF16 往返而丢失精度；回归测试使用
无法由 BF16 精确表示的 FP32 数值锁定这一契约。

本地验证结果为：非分布式测试 `883 passed, 2 skipped, 5 deselected`，Gloo 测试
`5 passed, 885 deselected`。远端证据统一保存在
`../postmerge_validation_20260715/evidence/remote/wan/2adfbfd/`；其
`SHA256SUMS` 含 101 个文件，复核结果为 101/101 通过。完整结论与证据索引见
`../postmerge_validation_20260715/WAN_RESULTS_2adfbfd.md`。

## W5 post-merge 门禁

`remote_wan_resume_run.py` 只接受：

- `completed_steps == authoritative_completed_steps == --max-steps`；
- `marker_valid` 与 `ready_for_aggregation` 同时为真；
- `audit_run_artifacts()` 返回 `valid=true`；
- 预期数量的 metrics、有限且非零梯度和最终 checkpoint 同时存在。

`compare_wan_resume.py` 不再直接反序列化不可信 pickle。它对 continuous、split、
resumed 三段分别验证 commit marker 与 checkpoint tree SHA256，并通过
`read_and_validate_training_state(..., trusted_root=run_root,
use_checkpoint_implementation_identity=True)` 读取安全 checkpoint。最终比较要求
checkpoint v4、manifest v2、status/audit 全部有效；纯运行时间和吞吐字段不参与
deterministic equality，训练语义字段仍须精确相等。

第一次远端 W5 尝试在进入训练前按预期 fail closed：harness 先调用了
`torch.cuda.*`，随后 `ExperimentRunner` 才尝试配置 deterministic runtime，因而得到
`deterministic runtime must be configured before CUDA is initialized`。失败目录和日志
被保留，没有覆盖。修正后 `ExperimentRunner(config)` 位于所有 `torch.cuda.*` 调用
之前，并增加动态 monkeypatch 回归；该 harness 测试集为 `3 passed`。

第二次尝试中，World-R1 与 Flash-GRPO 各自的 continuous-2、split-1、resume-to-2
共六段均 `valid=true`，status、authoritative marker、artifact audit 与 checkpoint
门禁均通过。两份 comparison 的 11 个 exact gate 全部通过；continuous 与 resumed
的最终 adapter SHA256 分别为：

- World-R1：`334e0ff0811f42cc2394b3a3f4a3c50aff6a8729f9ce31c4fafef16bf12a1a75`；
- Flash-GRPO：`9ee9afc5a03e75816552d92c98f6b3c3c8f7109021a38b13b135cc62cee793e6`。

comparison 与运行日志 SHA256：

- World-R1 comparison：`1506ac35afc3c8a429c442db4048564acac2c78f323a54d5544e7347c6d6be31`；
- World-R1 log：`a77842ceef8514b006c27ad81c65e9416ff4000470f065da628dd88f35739aec`；
- Flash-GRPO comparison：`3ca5fd55c4d475d4d228b4f3dfca8c8e5d4a03f876fc67d2de0dd12a8b80c953`；
- Flash-GRPO log：`ca5dfa01f50cc331311914a648ffe4ad233273ee435e7403434762a2efc17a0d`。

推荐在全新输出目录运行三段，再比较（路径仅为示例）：

```bash
python experiments/wan_world_r1_flash_smoke_20260714/remote_wan_resume_run.py \
  --config experiments/wan_world_r1_flash_smoke_20260714/world_r1_real_wan.yaml \
  --output /tmp/wan-w5-postmerge/continuous --max-steps 2

python experiments/wan_world_r1_flash_smoke_20260714/remote_wan_resume_run.py \
  --config experiments/wan_world_r1_flash_smoke_20260714/world_r1_real_wan.yaml \
  --output /tmp/wan-w5-postmerge/split --max-steps 1

python experiments/wan_world_r1_flash_smoke_20260714/remote_wan_resume_run.py \
  --config experiments/wan_world_r1_flash_smoke_20260714/world_r1_real_wan.yaml \
  --output /tmp/wan-w5-postmerge/resumed --max-steps 2 \
  --resume-from /tmp/wan-w5-postmerge/split

python experiments/wan_world_r1_flash_smoke_20260714/compare_wan_resume.py \
  --continuous /tmp/wan-w5-postmerge/continuous \
  --split /tmp/wan-w5-postmerge/split \
  --resumed /tmp/wan-w5-postmerge/resumed \
  --label world-r1-postmerge-v4 \
  --output /tmp/wan-w5-postmerge/comparison.json
```

## W7b post-merge 门禁

`remote_wan_flash_heterogeneous_parity.py` 现在显式使用 `wan_backend=flash`、
`train_cfg=false` 与 `objective_version=reference_v1`。每个 selected index 的 scalar
reference 使用与生产 grouped path 相同的稳定派生 seed、forked RNG，以及从同一次
完整 batch prompt encoding 中切出的对应行；group 顺序不应影响结果。这样 W7b 隔离验证
的是 heterogeneous selected-index 分组与恢复顺序，而不是 T5 在 BF16 下未承诺的
scalar-vs-batch bitwise invariance。后者作为显式、非 gating 的
`prompt_batch_composition_diagnostic` 保留。rectification 不再读取冻结表，而是比较
rollout/recompute 返回的动态 SDE coefficient。结果还必须记录真实 scheduler timestep
metadata，并要求恰好 480 个 LoRA gradient tensor 全部 bitwise parity。

```bash
python experiments/wan_world_r1_flash_smoke_20260714/remote_wan_flash_heterogeneous_parity.py \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --world-r1-root /path/to/World-R1-main \
  --flash-root /path/to/Flash-GRPO-main \
  --source-root "$PWD" \
  --output /tmp/wan-w7b-postmerge
```

`--world-r1-root` 暂时保留为旧命令行兼容参数；W7b 的 active reference root 是
`--flash-root`。

在 `2adfbfd` 上的远端 W7b 结果为 `valid=true`：sample 顺序与 tensor、selected
indices、grouped seed、scheduler metadata、Flash reference contract、单 transition
保留、重算 logprob、loss、参数不变门禁全部通过，且 480 个 LoRA gradient tensor
逐个 bitwise exact。受测 adapter 源码 SHA256 为
`87e3b3ea3a0c90855abcba203b4639e9275d353de16c5df90bae90e00af1c967`；结果 JSON 与日志
SHA256 分别为
`68c34aaae935c6d64afb2424ccd311340f1be7a142828b4b4d968d908a469153` 和
`fc4ed93963ebcfe050a854875868cf63ab40086cc8dae746203e20eb31ee885e`。

## 结论边界

上述证据支持的是：真实 Wan2.1 模型上的 bounded 两步训练可跑通、梯度有限且非零，
并且 continuous 与 split/resume 在已声明的语义字段上精确一致；W7b 还支持修复后的
Flash SDE latent 精度与 reference parity。它不支持图像/视频质量提升、长程训练稳定性、
吞吐提升或显存效率提升等结论；这些仍需更大规模、更多 seed 和对照实验。

## reward_general post-merge 门禁

`remote_world_r1_general_reward_probe.py` 明确限制为 `127.0.0.1` 或 `localhost`，
并显式声明 `protocol_mode=reference_v1`、`wire_format=legacy_pickle`、
`allow_unsafe_pickle=true` 及 exact `trusted_hosts`。该 opt-in 只用于复现固定的本机
reference server 协议，不允许扩展到非 loopback 地址。

`2adfbfd` 上的远端 attempt 1 为 `valid=true`。direct reference、reference HTTP 与
VisualRL 三条路径均返回 `[0.260009765625, 0.1943359375]`，HTTP/VisualRL 相对 direct
的 `max_abs` 都是 `0.0`；坏 pickle 与坏图像均得到 HTTP 500 且包含 traceback，
`silent_fallback_detected=false`。server 在采证后由 harness 终止，
`server_returncode=-15`。峰值 CUDA allocated/reserved 分别为 `7,874,462,208` 和
`8,250,195,968` bytes。

结果、server log、运行 log 的 SHA256 分别为：

- `6c7ccdb9ce8f0af04660bd722f09f4d331db71adc7d67b46cdb8d084bd16328e`；
- `82af7c01776ea301fc64d5d4611b579a199f7cf536281d8e468a2e39a482ff13`；
- `73fa49d0d048684f200551ed6c5fcf535d6265b2f6b048dc0ce77b663450a39a`。

该结果只证明固定 HPSv2 checkpoint、loopback legacy pickle 协议下的数值一致性和
错误 fail-closed 行为，不证明 reward 的效果质量、吞吐或资源效率。
