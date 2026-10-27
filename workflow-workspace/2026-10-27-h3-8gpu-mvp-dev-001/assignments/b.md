# Assignment B

- active: true
- status: assigned
- task_id: H3-M2a1-forward
- pair: coding-b → simplifier-b
- execution: sequential; A/C idle
- source_dependency_commits: M1a `7b29f4bf8d990794d4b9bc826a82bd390861d530`、M1b `725f48f4d67c67d90c60cc3536f4bc40f12802f2`，均 simplifier-a approved
- integration_parent: `8ed954f2562cf9d1f5432a7bb84ada181cdae6a9`
- 开始先核对引擎实际 branch/HEAD；不自行切分支/commit/merge。当前没有 changes 或失败门禁返工。

## 唯一原子功能

**把已批准 generated joint state 接到 H3 T2VA 单次 transformer forward。** 实现真实几何→attention packing→双 timestep→单次 callable→符号转换→joint prediction；不得只写未使用类型。完整 adapter 太大，已拆分：本项只做 forward，后续分别做本地 artifact/meta/LoRA 加载与缓存/解码，再接现 ModelAdapter/controller。不要提交带 NotImplemented 抽象方法的假 H3Adapter，也不要在本项创建第二训练 loop。

## 允许写入

- 新 `visual_rl/models/implementations/h3_geometry.py`
- 新 `visual_rl/models/implementations/h3_forward.py`
- `tests/test_h3_mvp_cpu.py`：保留现12项及断言，只追加本项目标和导入身份。
- 若最小几何代码移植需要许可：`THIRD_PARTY_NOTICES.md` 仅追加本项来源，及新 `visual_rl/models/implementations/H3_LICENSE.txt`（Apache-2.0全文与对应归属；若只独立实现不需新增许可文件则说明）。若角色工具范围不含这些路径，报告请求，不绕过。
- `workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/coding-b/H3-M2a1-forward.md`

只读：已批准 `models/h3_joint.py`、`algorithms/dynamics/{h3_schedule,h3_cps}.py`、旧 ModelAdapter/Dynamics/runtime、原七项及旧配置。不得改 task/reference/AGENTS/.gitignore/workflow。需要额外路径先报告。

## 输入/输出与数值合同

1. 最小 `H3T2VAGeometry`（可采用同等清楚名称），只支持纯文本生成 AV，不支持 keyframe/ref2va/SP/padding。生产微批严格 B1，guidance1，不引入 negative 分支。family 常数：spatial compression16，video channels24，patch(1,2,2)，audio feature32，stereo2，24fps；pixel frames 必须 `17n+5`，latent frames `5n+2`；audio latents/channel 为 `round(frames/24*40)`。画布正整数、32倍数、宽高比1/4..4、面积<=768*1344，时长5..15s；不自动修帧数或放大画布。固定资格例256×384×124得到 latent37×16×24、video rows3552×96、audio rows414×32，消费既有 `H3JointLayout`，不要再写一个joint flatten协议。
2. 模型 attention 序列为 **[text | audio | video]**，不同于 trajectory 的 [video-flat | audio-flat]。按上游真实 T2VA 算法生成 position_ids(float64)、token_tags、video/audio/text_indices；无条件行。text 长度来自已trim有效 embedding `[1,L,Dtext]`，不把padding token当text。用真实 AV 共用 rotary 时钟：video spans `(5/3)*(1,4,4,4,4)`循环，audio每latent1、channel-major stereo；text长度作媒体时间原点；空间网格endpoint=False、aspect normalization/sqrt(latent area)、scale32；不要用全零position或等距video时间占位。
3. forward桥接接收**调用者提供的单一 prepared/guarded callable**、packed FP16 x、geometry/已构建布局、text embeddings 与两个显式FP32 sigma scalar。模型域不得 import algorithms 或按recipe ID分支；schedule/index解析留算法/runtime，桥接只接video_sigma/audio_sigma。t=1-sigma，text/video行用video t，audio行用audio t；构造unique timestep和row indices，不能平均sigma/误把index当t。
4. 按真实 `MiniMaxH3Transformer3DModel.forward` 参数名一次调用 callable：`hidden_states`, `audio_hidden_states`, `encoder_hidden_states`, `timestep`, `timestep_indices`, `token_tags`, `position_ids`, `video_indices`, `audio_indices`, `text_indices`, `return_dict=False`。不访问 `.module`/raw原模型或自行wrap/prepare，不改变模型参数dtype/training模式、不包整个root autocast。状态在原FP16传入，FP32岛由真实模型自己的projection对齐；不要对整个模型 `.half()`。
5. 返回的 `(video_velocity,audio_velocity)` 是 **data-ward**，对两者各取负恰一次，pack成noise-ward joint prediction交现CPS；保持计算图，检查返回shape/device/浮点与finite。不可删audio，不在forward里运行CPS或调optimizer。调用者后续可在no_grad和grad上下文复用完全同一路径。
6. 输入geometry/state/text shape、空text、B>1、非法sigma/nonfinite或输出shape错误都清楚拒绝；shape/计划错误尽量在model调用前检查。简单数据拥有/复用即可，不造tokenizer缓存框架/通用多模态工厂/每步hash。

## 真实来源与许可证

先读本任务task/plan、已批准joint/CPS源码及当前 `models/interface.py` / prepared调用边界。上游锁UniRL `8cfbac17765004afb68fef08a4e011a4c8daf555`：

- `reference/UniRL/LICENSE` 全文；`unirl/models/minimax_h3/vendor/packing.py` 文件头 MiniMax/HuggingFace Apache-2.0，读实际T2VA几何/position/row timestep。
- `unirl/models/minimax_h3/{config,packing,diffusion}.py`，尤其geometry常数、`row_timestep_plan`与预测符号转换。
- `vendor/transformer_minimax_h3.py` 实际forward签名与FP32 islands，不靠假想API。

不整文件vendor packing（其中有不需要的图像/条件分支）；仅独立实现或合规最小移植T2VA必要部分，移植保版权、许可、来源SHA及修改说明，交付逐项列来源。

依赖事实仅作为后续选型线索：本快照UniRL vendor __init__称来自未合并PR14355，但被引用VENDOR_COMMIT.txt未收进reference；不能编造该提交。Flow-Factory固定快照 `dependency.py` 给真实Diffusers H3模块路径并要求>=0.40.0，这不证明本机已安装可用。本项不安装/升级/下载、不 import reference，也不靠不存在的Diffusers动态接口装作已经加载。forward callable与该已读真实签名兼容；重模型加载在下一原子项另审。

## CPU 验收

- 真实geometry默认例核对上述行数/元素数、拒121帧/非法画布/fps或任务；验证text/audio/video offsets、modality tags与完整无重叠索引覆盖。
- 手算若干首尾/周期交界position，验证stereo channel-major和端点不包含的空间grid；不能只比较实现自己产出的另一个结果。
- 用明确tiny trainable transformer叶子（非mock bridge）记录kwargs和调用次数，桥接只call一次；双sigma的t和row mapping精确、三模态位置与shape对应，data-ward→noise-ward符号只翻一次，两模态参数都能反传；不依赖权重/网络/GPU。
- 与**真实现H3CPSDynamics**串接：bridge预测→sample→相同bridge固定action replay→score/backward，证明不是无消费者独立helper。测试标明只替重transformer叶子，不宣称完整controller或GPU。
- malformed输入在调用前拒绝；不允许假position、恒定预测或预存score取代所测生产实现。扩展导入身份测试以确认当前worktree。
- 最后源码编辑后，原样执行 plan.md 两条CPU命令；原七项仍7/7，目标集全过无skip/xfail，不跑全量。写明命令/exit/数量/时间与实际branch/HEAD、改动路径和diff；`git diff --check`。

## 非目标/审查

本项不负责依赖安装、模型/数据/文本缓存下载或权重加载、LoRA/meta/FSDP2、VAE解码、reward/advantage/旧锚、checkpoint、registry/controller。后续必须补齐，不能用本项fake-leaf通过冒称完成。Simplifier B只审本项，检查最小边界、实际H3几何与签名、单prepared入口、精度/符号/两流梯度及许可；changes回Coding B修复两集重测。
