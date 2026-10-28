# H3 单节点八卡入口（GPU 待验收）

H3 的固定 `h3_t2va_mvp_v1` profile 现在通过原 `python -m visual_rl.train CONFIG`
和同一个 `RunController` 执行。不是新的 Runner，也不表示已有真实八卡训练结果。
已有六条 SD3/Wan 配方保持原编译和运行路径。

## 支持范围

- 单节点 WORLD8，block FSDP2，SP1/TP1；同模型同步 rollout 与 replay。
- T2VA joint 音视频、CPS 元素加权代理 score，固定量化 action，更新前冻结 replay 锚。
- 每 rank 一个完整 prompt group，microbatch1、时间步逐格 backward、每 collection 一次更新。
- Video-only HPSv3：124帧全部评分、top37均值；不宣称音频质量奖励。
- 完整 collection 边界 checkpoint；相同代码/模型/语义/拓扑下 fresh 恢复。
- 纯本地原始模型与文本缓存；不下载、不在线补算缺失 embedding、不转换旧权重。

## 环境与资源

使用 Python 3.10/3.11 的独立训练环境。安装训练依赖并确保 Diffusers 包含
`MiniMaxH3Transformer3DModel`、`AutoencoderKLMiniMaxH3` 和音频 VAE 的公开实现：

```bash
python -m pip install -e '.[train]' 'diffusers>=0.40.0'
```

PyTorch 必须支持 `prepare_h3_fsdp` 所校验的 FSDP2/DCP API，包括
`FSDPModule.set_gradient_divide_factor`；版本号、模型名称和驱动显示不代替实际能力检查。
不把旧 `MiniMaxH3DiTModel` / `VideoVAE` artifact 直接当作新版权重；须先核对布局与参数映射。

需要准备：

1. 模型根目录下真实 transformer、video_vae、audio_vae 及其完整配置/权重。
2. `visual_rl.h3-text-cache.v1` 的离线文本缓存，覆盖 train 和 eval；格式见
   `visual_rl/models/implementations/h3_text_cache.py` 文件头。原始 UniRL cache 不会被静默接收。
   可用 `scripts/prepare_h3_text_cache.py --raw-root RAW_FL2VA --model CONVERTED_MODEL --train TRAIN --eval EVAL --output NEW_CACHE --max-tokens 512 --max-prompts 32` 在独立准备阶段生成：显式选择单个获准GPU UUID，使用本地Qwen3-VL原始第50层、CPU权重offload、逐prompt真实前向。超长文本拒绝而不截断，完整输入哈希和实际输出记录在provenance.json；不会下载或修改原权重。此准备程序的语法检查不代表已生成GPU缓存。
   `data/h3_dual_smoke/` 是明确独立的8训练/2评估prompt smoke子集，来源与选择索引已记录；不替换正式19700/300数据。
3. 精确 train/eval prompt 文件及真实内容身份。
4. 对应画布的124帧 HPS manifest 和可用服务。不得复用81帧 Flash manifest。
5. 已获准且空闲的8张卡，按完整UUID固定逻辑顺序；不默认选择物理0—7。

## 配置与启动

以 `configs/v2/h3_t2va_mvp.yaml.example` 为结构模板，填写真实路径、预算、UUID和内容身份，
另存为完整 YAML。示例占位内容会被编译器拒绝，不能原样运行；不通过删除校验启动。
字段的唯一实现定义为 `visual_rl/composition/config/h3_profile.py`。

```bash
# EXACTLY the same ordered UUID list as launch.execution.gpu_uuids:
export CUDA_VISIBLE_DEVICES='GPU-...,GPU-...,GPU-...,GPU-...,GPU-...,GPU-...,GPU-...,GPU-...'
torchrun --standalone --nnodes=1 --nproc-per-node=8 --module visual_rl.train /absolute/h3.yaml
```

启动后先检查配置、资源和GPU映射，只有rank0读取初始化权重；初始化读取失败时先通知全部rank，
再拒绝进入mesh/参数集合通信。输出目录按run独占锁保护；新运行拒绝覆盖已有目录。
每个rank都参与FSDP训练与分片checkpoint，只有rank0发布最终文件。

恢复：保持原语义、目标步数、UUID顺序和output_dir，将 `launch.resume_from` 改为该run的
最新完整 `checkpoints/step-N`。这不是任意步骤/弹性拓扑恢复。终态checkpoint已保存但SUCCESS尚未写出时，
可恢复并完成终态发布，不重复更新；已有SUCCESS的运行拒绝再次训练。

## 输出与验证边界

`SUCCESS` 使用明确的 `visual_rl_h3_run_success` kind；结果还包含真实checkpoint、
resolved_recipe.json、run_manifest.json、metrics.jsonl。旧单进程 `inspect_run/audit_run`
不应被假定支持这个H3终态格式；checkpoint由现有AtomicCheckpointManager和H3事务读取路径验证。
默认不持久保存每条视频，媒体用于评分后释放。

CPU验收覆盖正式默认factory/controller/CLI、两轮连续更新、durable checkpoint后中断、
新实例恢复下一轮以及终态checkpoint恢复。测试只以明确叶子替代重模型、VAE、HTTP与CUDA/通信；
实际模型与NCCL仍须GPU验收，不能将CPU结果称为八卡训练成功。

先在获准资源上完成低画布的一次真实更新与恢复，再扩大batch/画布。不要停止未知进程或直接运行完整长实验。
