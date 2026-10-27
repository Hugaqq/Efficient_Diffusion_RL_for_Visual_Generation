# H3-M2a1-forward — Coding B 交付

## 范围与身份

- 当前 assignment active=true；无 simplifier-b 当前返工记录（查找无结果）。只完成本 forward 原子项，交 simplifier-b。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。
- 开始 HEAD：`f8f83f7064f121a00526241048e68ceda83eb4b1`；初始工作区干净。未切分支、手工提交或 merge；引擎 snapshot SHA 待生成。
- 唯一总需求、plan、assignment B、当前 joint/CPS、ModelAdapter 与 prepared guard 边界均已读取。

## 实际改动

1. 新 `visual_rl/models/implementations/h3_geometry.py`：冻结 T2VA 输入几何，拒绝自动修帧/放大画布及非 guidance1/24fps/T2VA；固定族常数映射到已有 H3JointLayout。256×384×124 → (24,37,16,24)、3552×96 video、414×32 audio，joint宽354240。
2. 同文件 `attention_inputs`：新建拥有的 CPU FP64 positions，再转目标 device；[text|audio|video]、tags 1/2/0、无条件行与padding；真实非匀速 video clock、channel-major stereo、endpoint=False aspect-normalized空间坐标。无公共可变计划缓存，避免额外完整性/缓存框架。
3. 新 `visual_rl/models/implementations/h3_forward.py`：`h3_t2va_forward(prepared_forward, *, x, geometry, text_embeddings, video_sigma, audio_sigma)`；两显式 detached FP32 scalar → 1-sigma → sorted unique/row mapping；只调用调用者 callable 一次。输入 FP16 B1 原样传模型，文本已trim且非空；验证输出 shape/device/floating/finite。双输出各取负一次，通过原 H3JointLayout pack 保留梯度。不 import algorithms、不 wrap/unwrap、不改参数dtype/training、不创建 autocast/调度/优化器。
4. `tests/test_h3_mvp_cpu.py`：保留十二项原测试及断言，仅追加五项及两个导入身份断言。目标覆盖真实几何、手算位置/周期交界/末位置、完整索引覆盖、一次 callable、双 t 和相同sigma折叠、精度/符号/双梯度、no_grad复用、真实 bridge→CPS sample→固定action replay→score/backward、输入调用前拒绝和输出拒绝。

## 来源与许可

读取 UniRL 固定 `8cfbac17765004afb68fef08a4e011a4c8daf555` 的 LICENSE 全文（Apache-2.0，Tencent）、`minimax_h3/config.py`、`packing.py`、`vendor/packing.py`（MiniMax/HuggingFace Apache-2.0 文件头），以及 `diffusion.py` 的真实双流forward/符号转换和 `vendor/transformer_minimax_h3.py` 实际签名/投影精度岛。

- geometry family及双布局语义来自 config/packing；rotary时空算法来自 vendor/packing。
- 参数名称和 projection 自行 dtype 对齐来自 transformer forward；data-ward→noise-ward来自 diffusion.predict_noise。
- 本项独立实现数学/窄接口，未复制 vendor 函数或引入 reference import；因此未新增移植许可文件或修改 THIRD_PARTY_NOTICES。源码头保留语义来源/固定SHA/许可说明。不编造缺失 VENDOR_COMMIT，不宣称安装了 Diffusers H3。

## 真实 CPU 门禁

最后源码编辑之后运行，均 exit 0，无 skip/xfail、无全量测试：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.78s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
# 17 passed in 1.41s
```

`git diff --check` exit0；`git diff` 仅已有 H3测试文件增加163行，无原断言删除；新源码为未跟踪文件，不计入该 diff stat。status只出现两个指定新源码、H3测试及本交付目录。原Flash七项、旧配置/runtime、已批准joint/schedule/CPS未修改。

## 审查重点与未完成边界

请 Simplifier B 核查独立几何算法、双布局、真实签名/单入口、精度和符号、两流梯度与许可范围。若 changes 回 B 修复后重跑两集。

tiny测试只替重 transformer 叶子（明确trainable双参数，非恒定预测），没有真实H3权重、controller、prepared生命周期或FSDP2执行证明。文本是否真正trim是调用者合同；本函数无mask，不猜padding。每次新建位置计划以保持最小实现，未作缓存/性能承诺。

后续仍必须实现 loader/meta/LoRA、cache/AV decode、profile/ModelAdapter/controller、world/评分/旧锚/tail checkpoint/fresh恢复。本项不是完整MVP。无服务器/GPU/下载/环境升级/网络部署/push操作。**GPU_PENDING**。
