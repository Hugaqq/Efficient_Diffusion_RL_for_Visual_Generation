# H3-M2a2-loader — Simplifier B

## 裁决：approved（仅本原子项）

审核 HEAD `919b6abc7920f2dbcea764907152523882a09a59`，分支 `pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`；接收时工作区干净。已重新读取 task、plan、assignment B、Coding B 交付、完整新 loader 和五项新增测试，并核对实际旧 LoRA helper、ParameterStateManager 身份绑定、参考 bundle/transformer/recipe 与公开依赖路径。未改源码或测试。

## 实现及边界核验

- `visual_rl/models/implementations/h3_loading.py:94`，`build_h3_meta`：仅接受现存本地目录，固定本地config，拒路径逃逸/重复JSON键；默认lazy解析真实公开H3符号，无tiny/reference fallback。真实 `load_config(local_files_only=True)` 与 `from_config` 有生产调用路径。族输入/输出投影、主blocks和六线性target有检查。
- 同函数：真实Accelerate参数meta构造，buffer不随参数置meta；构造前未加载完整base权重。精度转换仅作用于参数，真实rope buffer没有BF16往返；base BF16、指定岛FP32、LoRA FP32有后置核验。全部参数返回时仍meta，内部AC关闭。
- 同函数：真实PEFT原位注入全FQN目标，并对期望trainable集合精确核对；refiner/root/base冻结。完整base state仅做已知 `.base_layer` 映射，注入后state与base+LoRA集合精确一致。没有提前创建optimizer或绑定ParameterStateManager。
- `h3_loading.py:213`，`_weight_headers`：单文件/分片index互斥；本地路径、重复JSON key、跨shard重复/错放/未声明/缺失key及真实safetensors header有检查；没有pickle或部分加载fallback。
- `h3_loading.py:258`，`load_h3_cpu_state`：owner前置拒绝，完整base键/shape/dtype核验；显式预算在tensor payload读取前执行。预算包括源、目标、LoRA和buffer的保守驻留余量，明确不是RSS保证或流式常量内存。完整CPU状态与本地身份返回给后续prepare，不擅自物化模型到GPU。
- 同函数：独立CPU generator构建FP32 Kaiming-A/zero-B，既非全零A+B，也非旧helper gaussian；全局CPU RNG不变有实际测试。非持久 `rope.inv_freq` 的真实副本作为单独输入返回，未被当作允许丢失的base权重。
- `tests/test_h3_mvp_cpu.py:629–856`：五项新测试执行真实Torch meta、Accelerate、PEFT、ConfigMixin与safetensors；仅重H3类解析叶子替换。实际单文件/两shard、完整state、种子复现、owner/预算提前拒绝、格式/键/shape/dtype/路径/config/target负例均覆盖。tiny CPU显式物化后分别video/audio backward进入LoRA，base无梯度；不把此CPU setter称为FSDP。

## 简化与许可

未发现必须返工的过度框架、重复旧执行路径或不必要抽象。两个入口与三个窄结果类型提供后续prepare确需的FQN、规格、buffer和完整CPU输入；无需为缩短代码删掉覆盖、精度或预算检查。旧helper的root包装、gaussian初始化和不限定主blocks的接口不适合直接复用，本项未改其旧语义。

对照已读UniRL Apache-2.0许可、实际transformer构造/FP32声明、bundle与recipe，及Flow-Factory带许可头的dependency路径：当前使用公开依赖并独立实现窄加载边界，未复制vendor/bundle或导入reference。未将上游每rank完整加载或通用框架搬入项目。

## 本轮独立 CPU 门禁

以下均实际执行、exit0；无skip/xfail、未跑全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
# 22 passed in 3.55s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.95s
```

原七个命名测试仍存在且全部通过。`git diff` 无输出、`git diff --check` exit0（写审查记录前）。当前命令权限不含历史snapshot diff，因此允许路径/原七项断言的历史不变性仍待orchestrator/引擎最终核验，不以clean工作区替代历史审计。

## 后续必须履行的接口与限制

1. C必须保证仅初始化owner读取完整权重；全rank按同一初始化协议消费并广播。当前bool准入不是分布式rank检查，assignment明确把该职责留给C。
2. 若prepare使用 `to_empty`，必须恢复返回的真实 `rope.inv_freq`，不得保留未初始化buffer；最终参数身份绑定与optimizer必须在分片/物化完成之后。
3. 当前loader只有目标测试消费者，完整controller尚未接入。可选resolver测试允许“符号存在或明确ImportError”两分支，不据此宣称本机真实H3类已构造或权重已加载。
4. 后续仍需真实H3依赖/权重兼容验证、FSDP2/world、aux/profile/controller、评分/冻结旧锚及tail/checkpoint/fresh恢复；本批准不代替这些任务。

本轮无服务器/GPU/下载/环境升级/网络部署/push。**MVP未完成，GPU_PENDING**；CPU通过不证明真实H3、CUDA/FSDP2/NCCL或质量。
