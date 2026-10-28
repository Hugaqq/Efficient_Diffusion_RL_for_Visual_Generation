# 第七轮：原转换job输出校验进展 / waiting

本地协调HEAD54e0516185845241b3bff3c20fd3821ef5545a49；运行job固定生产SHA fa99cf22c11670a67c06cd16508dd8f4ab5664b3。开始直接inspect同job，未重启或部署。

## 实际回执

exp007-inspect-convert-01：started1793159463.4540539 ended1793159466.165688，code0/uncertain=false，首次连接成功。

- job native-artifact-convert-001；原PID952148/start_ticks74454370/created1793157389.6425683，child952149。
- running=true，exit=null，success=null，checkpoint_complete=null。
- heartbeat1793159443.9584627，elapsed2054.272390127182秒；log_mtime1793159459.7605774。声明硬超时仍3600秒。
- 完整spec与日志根：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/native-artifact-convert-001`。

## 真实日志新进展

```text
transformer: 535 original keys -> 638 diffusers keys in 27 shard(s), 61.73 GiB.
video_vae: 560 original keys -> 703 diffusers keys in 5 shard(s), 9.70 GiB
  (latents_mean/latents_std: 24/24 channels; tiling 256px / 64px min overlap).
audio_vae: 1087 keys carried over unchanged, 0.56 GiB
  (40 latents/s, latents_mean/latents_std: 32/32 channels).
scheduler: shift=12.0 (video), audio_scheduler: shift=3.0 (audio).
```

- transformer全部27、vae全部5、audio单文件共33个`validated_shard`事件已观察到；实际脚本先检查公开模型完整keys/shape/dtype，再逐tensor分块检查finite后才打印该事件。说明这些准备检查已走通，但不是实际模型forward或GPU训练。
- `torch.nn.utils.weight_norm` deprecation FutureWarning保留，没有过滤或误判为训练失败。
- 转换后的原始输入完整第二遍hash已出现在日志；随后已进入输出inventory hash（audio及两个transformer输出shard已记录）。结合本轮读取的prepare源码顺序，进入此阶段意味着源前后inventory比较没有抛错。
- 最新已报告进程peak_rss_bytes=11,994,386,432（约11.17GiB），来自转换后validated_shard及hash事件，比此前仅conversion_start的828,567,552 bytes更完整，但尚非进程退出时最终峰值。
- 输出示例：audio文件605,429,340 bytes，sha256 `52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb`；transformer第一输出2,428,837,440 bytes，sha256 `87f828147fca546e5463d4261cf67edd6fec0558d1725fdbad0959c7bc5cc0f4`。

## 验收边界 / 下一步

尚未观察artifact_published、最终conversion.json或exit0，不能宣布整个准备成功、关闭H3-EXP-001或使用partial启动加载。返回waiting保持原任务。下次先inspect同SHA/job，退出后再独立读取receipt/覆盖/hash/无symlink与资源证据，再进行真实完整payload direct-IO加载及真实GPU链路。

本轮仅一次远端只读观察，重读当前输出验证源码、task/formal-plan及Coder复验说明；没有source改动、新launch、安装、转换重跑、stop/kill或CPU回归。Flow/Flash仍0更新，正式缺步数此阶段不暂停。保留所有旧失败证据。
