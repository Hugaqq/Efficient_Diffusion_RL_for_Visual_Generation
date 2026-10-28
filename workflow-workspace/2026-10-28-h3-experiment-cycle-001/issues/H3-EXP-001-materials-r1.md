# H3-EXP-001 补充r1 — 材料已取得，真实meta通过但新版布局无权重

状态：repair回Coder；原H3-EXP-001保持open，旧失败报告不覆盖。未做算法修复、转换或训练。

## 本轮连接与范围

采用用户最新“只读短暂超时最多重试5次，不改规则程序”指令。本轮所有实际operations回执首次即code0/uncertain=false，未消耗重试，无unknown mutation。正式步数未询问/未填。

本地HEAD `a195f13b384d54f5ac231c698f2f1e73645ff715`；本轮探查明确使用已部署`d5c10c9f00e44a816e1f0881c7e28ba8accbcdae`（import由job实际打印），不是宣称本地新HEAD已被GPU验证。未publish/deploy新生产代码。当前production checkout status仍干净。

## Coder所需材料逐项答复

### 1. 已安装源码身份

实际三类源文件hash全部匹配Coder固定值：
- transformer：`92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef`，33045 bytes。
- video：`58e0ed47fdee8227d738eec5bb1e7a47acc21c65ee3de01aa3795c97d679d015`，42128 bytes。
- audio：`b56c5bb95a7dc99ace03df192544615c89ec01ac3b134a7f1dd7348b03731d57`，30959 bytes。

路径与来源见`experimenter/converter-004-provenance.md`。无需因版本差异盲目升级环境；未安装任何新包。

### 2. 完整官方转换器

经有界owned prepare job用Git取得固定上游SHA `aaf57d03a3edfbd9de457d2872417a92f149a91d`，clean。目录：
`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/pinned-diffusers-source-001/diffusers`。

完整converter为47157 bytes，SHA256 `86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c`已实算匹配。`LICENSE`为Apache2.0，hash和上游链接见provenance。

本地完整可读转录：`experimenter/converter-004-part1.txt`、`part2.txt`、`part3.txt`按顺序阅读（字符边界故可能切在词中）。这是证据文件，不是生产vendor模块；未import/执行converter，连dry_run也未执行。固定远端原文件hash已验证，本地转录尚未独立算拼接hash，不声称本地逐字hash通过。

### 3. 原生小配置

`experimenter/native-configs-004.json`保存全部FL2VA model_index、transformer配置、video wrapper/source、audio wrapper/config.yaml/metadata的实际解析内容及原文件bytes/SHA/HF blob落点。JSON容器格式重排，hash指原文件而非证据容器。

特别注意：audio wrapper output_channel2，yaml audio_channel1，metadata中的attn_proj/bigvgan/rates/latent宽度俱全；请根据真实官方实现核验channel-major stereo语义，不凭字段名字补参数。

### 4. 现成新版布局候选的真实状态

发现`/home/ubtroot/code/downloads/MiniMax-H3`（非FL2VA symlink根）：
- transformer/config.json：新类，完整33B配置，sha256 `74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae`。
- transformer index：638键，sha256 `ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e`；所列14个shard全部缺失。
- vae/config.json：新类，sha256 `78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745`；index703键，sha256 `15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca`，3个shard全部缺失。
- audio_vae目录只有config.json（sha256 `9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6`），没有权重文件。
- 三组件配置/index非symlink，但**不是可复用的converted artifact**。没有覆盖它，未下载这些大权重。

发现范围为qifan目录有限深度（跳过venv/python/.git/.pi/sources等）及已知code/downloads模型根，不声称搜遍全服务器。exp004-final-material-discovery-01的nvidia-smi compute-apps列表为空，但这不是文件无人使用的全证明；不改旧目录。

## 实际meta/header/双VAE准入probe

job=`converted-meta-admission-001`，kind=probe，timeout300秒，CUDA_VISIBLE_DEVICES为空（此阶段不占GPU）；PID938447/start_ticks74282963/created1793155675.5641956，child938448。

远端日志：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/converted-meta-admission-001/job.log`；完整argv/env/exit在同目录spec.json/exit.json。started1793155675.6076143、ended1793155683.9933171，**exit1、running=false**，无SUCCESS/checkpoint。

关键真实调用：
```python
root = Path('/home/ubtroot/code/downloads/MiniMax-H3')
meta = build_h3_meta(root, rank=4, alpha=8)
_weight_headers(root / 'transformer')
H3CPUDecoder(root, max_media_bytes=268435456, max_decode_bytes=1073741824)._configs()
```
未mock重模型类，调用已安装真实公开H3类+Accelerate+PEFT；只构造meta，不物化完整权重、不建optimizer、不执行forward。

实际meta成功输出：
```json
{"state_keys":1238,"base_keys":638,"lora_keys":600,"state_bytes":66363578880,"buffer_bytes":64,"all_parameters_meta":true}
```

实际失败：
```text
<string>:13 -> h3_loading.py:227 _weight_headers
 -> _local_file h3_loading.py:57
ValueError: artifact file missing or path escape: diffusion_pytorch_model-00001-of-00014.safetensors
<string>:17 -> h3_decode.py:195 _configs
ValueError: VAE shard missing or path escape
ERROR_STAGES ['headers', 'vae-configs']
```

因payload缺失，不能给完整真实header shape/dtype覆盖对照；只有declared index和实际meta键数均638，未据此声称键逐个相同。真实双VAE加载、GPU forward/backward、HPS、optimizer仍未完成。

## 资源推导（不是实际full loader新失败）

已测meta目标state66,363,578,880 bytes。现loader预算`2*source_bytes + 2*destination + 2*buffers`，仅目标项就132,727,157,888 bytes（含两份64byte buffer），已高于首轮实测112GiB available；再加源项更高。没有虚填host预算，没有调用full loader或冒称复现OOM。请Coder根据真实目标字节数设计/复用有界严格加载路径，实际峰值仍交GPU环境测量。

## 下一步最小请求

完整converter与配置现在都在允许目录可读，请Coder直接完成最小受审计准备工具（优先复用两个已读分支）或给出无需源码修改的合规准备命令。不要再次只凭类名判断；不要移植单卡runner或造第二入口。转换只能在本任务独立artifact目录，来源/输出身份、完整key/shape/dtype与重建buffer核验不可省略。需要改预算/加载实现时保持实际资源约束，禁止提高声明值越过真实内存。

交付后Experimenter Git发布部署→实际转换/严格加载→真实八卡AV/reward/Flow5，随后真实single-step Flash5。H3-EXP-002仍queued。未运行CPU回归。

## 操作索引

exp004-discovery-01、exp004-native-configs-01、exp004-upstream-source-prepare-01、exp004-inspect-source-01、exp004-converter-part1/2/3-01、exp004-final-material-discovery-01、exp004-converted-layout-01、exp004-launch-meta-01、exp004-inspect-meta-01均有回执。源码准备job exit0；meta job exit1；其他操作code0不等于训练成功。所有失败目录与过去网络失败证据保留。
