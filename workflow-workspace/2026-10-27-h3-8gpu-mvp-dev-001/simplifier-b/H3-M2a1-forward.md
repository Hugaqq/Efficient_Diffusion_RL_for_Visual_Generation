# H3-M2a1-forward — Simplifier B

## 裁决：approved（仅本原子项）

审核 HEAD `edd5c874f10ffa917a71bd9f2fa9085247075a6e`，分支 `pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`；接收时工作区干净。已读 task、plan、assignment B、Coding B 交付、实际新增源码、完整 H3 测试、joint/CPS 和 ModelAdapter prepared 边界。未修改源码或测试。

## 实现与边界

- `visual_rl/models/implementations/h3_geometry.py::H3T2VAGeometry`：严格整数画布/帧数、32倍数、面积/比例/时长、24fps、T2VA/guidance1 限制；256×384×124 得到正确 video 3552×96 与 audio 414×32。复用已批准 joint 协议，不重复 flatten。
- 同文件 `attention_inputs`：attention 顺序 [text|audio|video]，与 trajectory 分离；FP64 共用时钟、5/3×(1,4,4,4,4) 视频周期、channel-major stereo、endpoint=False 空间网格及模态标签与上游实际算法一致。索引覆盖与首尾/周期交界位置有独立手算断言，非全零占位。
- `visual_rl/models/implementations/h3_forward.py::h3_t2va_forward`：FP16 B1 输入、双显式 FP32 sigma、text/video 与 audio 分别 t=1-sigma，unique timestep/row mapping 正确；只走传入 callable 一次。参数名/形状与参考真实 transformer forward 相符；不 unwrap、不另 prepare、不改变参数 dtype/模式，不增加 root autocast。输出两模态各取负一次并保留梯度，shape/device/floating/finite 拒绝明确。
- `tests/test_h3_mvp_cpu.py` 新五项实际覆盖几何、位置、单 callable、独立双t及相同t折叠、no_grad/grad 复用、负例，以及真实 bridge→CPS sample→固定 action replay→score/backward；仅重 transformer 叶子被 tiny trainable 模型替代。
- 已核对 UniRL LICENSE 全文、vendor/packing.py Apache-2.0 文件头与实际算法、config 常数、packing 的 row_timestep_plan、diffusion 的符号转换、transformer 的真实 forward 与精度对齐。当前为窄数学/接口独立实现，无 reference import 或整文件 vendor，未见需扩大移植范围之处。

## 简化审查

未发现本项必须退回的错误、重复协议或过度抽象。两个窄文件分别负责几何与单次 forward，没有第二 runner、通用缓存/工厂或伪 adapter。每次新建 position 计划虽非性能优化，但符合当前最小实现约束，不要求额外缓存/hash。text 是否已经 trim 是调用者明确合同，后续 cache/adapter 必须履行，不能把本函数当作 padding 验证器。

本项暂只有目标测试消费者，assignment 已明确将 loader/aux/ModelAdapter/controller 分开；不以此前置桥接批准代替生产闭环验收。

## 独立 CPU 证据

以下为本轮实际执行，均 exit 0，无 skip/xfail、未跑全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.10s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
# 17 passed in 1.20s
```

原七个命名测试仍存在且全部通过。`git diff` 无输出、`git diff --check` exit0（写本记录前）；仅证明当前工作区状态，不证明历史 snapshot 范围。历史比较命令未授权，原七项断言及允许路径的历史不变性仍由 orchestrator/引擎提供 diff 完成最终核验，不能用 clean 冒充已审历史。

## 后续与限制

Orchestrator 记录 coding snapshot 与本项批准；继续依赖加载/meta/LoRA、严格 cache/真实 AV decode、prepared/FSDP2、world 数学、真实评分与冻结旧锚、完整 tail checkpoint/fresh 恢复、唯一 controller 整合并逐项复审。此次无服务器/GPU/网络部署/模型下载/环境升级/push。CPU 不证明实际 H3 权重、CUDA/FSDP2/NCCL 或视频质量。MVP 未完成；**GPU_PENDING**。
