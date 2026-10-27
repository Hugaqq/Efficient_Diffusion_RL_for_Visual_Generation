# H3-EXP-001 — 原生H3 artifact尚不能进入生产loader（真实服务器反馈）

状态：open / repair；Experimenter 2026-10-28首次环境阶段。源码 SHA `d5c10c9f00e44a816e1f0881c7e28ba8accbcdae`。

## 实际运行与失败

- 实验分支 `experiments/h3-cycle-20261028-001` 已push；远端Git fetch/FETCH_HEAD与独立worktree均为上述SHA。运行中实际import为该worktree的`visual_rl/__init__.py`，探查后Git status仍干净。
- job `artifact-admission-probe-001`，kind=probe（不是训练），timeout=300秒。
- owner PID765033，start_ticks=71836860，created=1793131214.535802；child PID765034。
- exit.json：code=1，started=1793131214.577547，ended=1793131229.123795；inspect确认running=false，无SUCCESS/checkpoint。
- 远端完整argv/env：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/artifact-admission-probe-001/spec.json`；完整日志同目录`job.log`。所有失败文件保留。
- 关键复现调用（在已部署worktree、PYTHONPATH指向该树、独立venv中）：
  ```python
  from visual_rl.models.implementations.h3_loading import build_h3_meta
  build_h3_meta('/home/ubtroot/code/downloads/MiniMax-H3-FL2VA/FL2VA', rank=4, alpha=8)
  ```
- 真实栈：
  ```text
  File "<string>", line 27, in <module>
  File ".../visual_rl/models/implementations/h3_loading.py", line 107, in build_h3_meta
    config_path = _local_file(directory, "config.json")
  File ".../visual_rl/models/implementations/h3_loading.py", line 57, in _local_file
    raise ValueError(f"artifact file missing or path escape: {name}")
  ValueError: artifact file missing or path escape: config.json
  ```

## 已定位事实，不混淆为已加载失败

1. **本次直接抛错是symlink containment，不是类名检查。** 原缓存实际位于`/home/ubtroot/code/downloads/hf-cache-ref2va/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA`。config/index/shards为指向同HF缓存`blobs/`的符号链接，文件确实存在。config落点为`blobs/94cc6ca1254d9d04b3f7cc06a8c9fc05d0ded065`，不在transformer组件子树。
2. **解引用复制不是全部解决方案。** 实读原配置`MiniMaxH3DiTModel`，hidden5376/layers50、latents_dim24/audio_latents_dim32、token_refiner_num_layers2/ffn_hidden_size14336等旧字段。当前loader要求新类、in_channels/audio_in_channels等新schema和`diffusion_pytorch_model*.safetensors`命名。
3. 实际safetensors header读得535个张量、33,122,992,912元素、13 shards合计66,280,485,936 bytes，无缺shard。不是payload完整hash/finite核验，也不是实际加载成功。
4. 旧权重确有结构映射需求，不仅名字不同：`blocks.0.attn.qkv_proj.weight`=[21504,5376] BF16（融合QKV）；`blocks.0.attn.out_proj.weight`=[5376,7168]；`blocks.0.mlp.fc1.weight`=[28672,5376]；`blocks.0.adaln_proj.linear.weight`=[96768,2688]。现loader LoRA targets为`transformer_blocks.*.attn.to_q/to_k/to_v/to_out.0`和`ff.net.0.proj/ff.net.2`。应根据真实旧/新实现验证拆分/顺序，不只换_class_name。
5. 原视频VAE在`video_vae/source/model.safetensors`（10,415,548,320 bytes），wrapper=`MiniMaxH3VideoVAE`、source=`AutoencoderKLLegacy`；原audio=`MiniMaxH3AudioVAE`（605,429,308 bytes）。两个真实公开Diffusers类可import，但没有加载实际VAE或验证等价。

## 环境与GPU证据

独立解释器`/home/ubtroot/qifan/h3-env-20261027-001/venv/bin/python`：Python3.11.18、torch2.8.0+cu128、diffusers0.42.1、transformers5.21.0、accelerate1.15.0、peft0.21.2、safetensors0.8.0；flash-attn未安装（尚不据此认定必须安装）。公开FSDPModule.set_gradient_divide_factor存在。

所选物理0/1/2/3/6/7/8/9，8个UUID与reference一致。每卡真实执行128×128 BF16 matmul/backward，finite=true、capability8.0、native_bf16=true、peak_allocated17,219,584 bytes。每卡探查前free67,743,645,696 bytes。**这只是CUDA能力探查，非H3 forward、NCCL、reward或optimizer更新证据。** 初始主机约112GiB available、磁盘500GiB available。

## 交Coder的最小请求

- 优先用获准`local_git_read`查看`origin/h3-pytorch-training`和`origin/h3-readable-integration`已有真实H3加载/映射，避免重复发明转换。
- 给出可Git部署的最小原生artifact适配/准备路径（必要时`scripts/`工具）。输出只写本任务独立artifact目录；不改原权重，不放宽路径保护来掩盖问题，不假填身份，不strict=False丢键。
- 先完整核对transformer与双VAE配置、参数FQN、融合拆分/形状/dtype/buffer和真实forward语义；若用转换，应有逐文件来源/hash、完整覆盖与确定性报告、显式RAM/磁盘预算和可复查命令。33B full状态需考虑112GiB host预算，避免多份全模型同时常驻。
- 已有原缓存symlink的物化属于环境准备，可由Experimenter运行已提交工具；**无需为了这个错误删除loader安全guard**。若已有兼容分支可复用，明确最小移植和许可。
- 不运行额外CPU回归；Coder不操作远端。交付后Experimenter再次Git发布部署、原生artifact准备、真实模型/GPU复验，再推进Flow5。

## 最小成功条件

新artifact/入口明确源权重身份，完整严格映射，无伪造参数；当前真实公开类完成meta/schema及严格真实加载，并能开始真实AV forward/rollout/decode。最终仍须真实HPS、5次optimizer、完整step5、exit0；本issue关闭不等于smoke完成。

## 已知后续缺口（非本次栈，勿冒称已复现）

- 当前H3 profile/stages仅full-N CPS；Flash single-step尚未实现，另见H3-EXP-002。先处理模型准入，不以改标签完成Flash。
- `scripts/`目前无H3离线文本cache producer；reader要求layer50/raw/no-special/5120及完整provenance。真实encoder在原artifact text_encoder（14 shards合计66,714,912,872 bytes）。后续需真实产出，不能用零embedding；尚未找到符合本schema的cache，不声称全服务器不存在。
- HPS124服务尚未准入，本轮未访问或修改任何服务。
- 正式global4×8=32与现每rank完整group WORLD8（8×K）需在正式前明确支持，不能静默改batch；正式目标步数在两个smoke通过后再请求用户。

## 操作证据索引

operations入口的不可重放记录：`exp001-host-inspect-001`、`exp001-environment-inspect-001`、`exp001-publish-d5c10c9-001`、`exp001-deploy-d5c10c9-001`、`exp001-model-layout-001/002`、`exp001-launch-artifact-probe-001`、`exp001-inspect-artifact-probe-001`、`exp001-resolve-artifact-links-001`。全部返回JSON code0/uncertain=false（launch/inspect的code0不等于job成功；job exit=1）。原文由operations保存，使用远端只读inspect复查，不重放launch。
