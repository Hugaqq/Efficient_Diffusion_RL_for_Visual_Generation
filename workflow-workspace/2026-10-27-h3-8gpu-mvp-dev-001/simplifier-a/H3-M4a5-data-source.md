# H3-M4a5-data-source — Simplifier A / W5

verdict=approved

本项返工清单：无。仅批准 A 的本地数据预留/提交与同拓扑边界恢复，不批准 B/C 本波实现，不代表默认 RunController 或整 run 恢复完成。

## 身份与材料

- 审核 HEAD：`230696472b735966d3ac17102120a444f4ff8697`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-5cbffe4d-c73d-4516-8d23-53d5b99580f5`；开始工作区干净。
- 已读当前 task/plan/assignment A/wave-05、对应 Coding A 报告、完整 h3_source.py 与六项 feature 测试，并核对实际 H3Adapter/H3Conditioning/初态端口及 cache 覆盖/漂移检查。
- 本角色未修改源码、测试或其他报告。

## 实现、边界与简化核对

1. `visual_rl/data/h3_source.py::_file/_prompts`：真实有界读取原字节并核对显式 SHA256；严格 UTF-8、LF/CRLF，只移除行终止符，保留首尾空白；空记录/BOM/孤立 CR/坏编码拒绝。train至少8条、eval非空；路径不进入内容身份，无假数据或在线 fallback。
2. `H3PromptSource.__init__/_check_inputs`：实际调用 adapter.verify_coverage 并 encode 取得真实 cache 身份；校验5120文本宽度及 geometry/layout。身份绑定原文件/解析、采样/流版本、精确双 schedule/eta、seed、WORLD8/rank placement、K/N、模型配置/cache/初态版本与预算。模型权重和实际硬件资源身份没有被假称由本源验证。reserve/commit/state/restore 重查文件及缓存漂移。
3. `reserve/_stream`：global occurrence=`step*8+rank`、原序循环、局部完整K组，重复文本不合并 occurrence。实际 adapter.encode/prepare_latents 产生条件和初态；initial/rollout×video/audio 使用独立 SHA-derived 63bit seed 命名空间和显式 generators，无全局 Python/NumPy/Torch RNG 修改。reserve不推进进度，失败 poison，不重抽补样。
4. `commit/abort/conditioning_for`：仅 exact active request 可提交；commit推进一次后清除请求/typed条件引用，abort清除并 poison。公开 embedding 副本和真实 typed conditioning 分离；无历史媒体/轨迹或模型图保留。错误 token/重复操作拒绝，不伪装提交成功。
5. `state_dict/load_state_dict`：只允许清洁 collection 边界导出，JSON拥有副本；fresh-only恢复精确核对整个 binding/schema/rank/device及step/cursor/behavior/batch的一致性，错误恢复 poison，无elastic/热恢复/mid-collection承诺。共同 next_global_batch_id 绑定下一组8个occurrence，不是本rank独有摘要。没有抢先恢复全局RNG。
6. 源码仅一个窄数据owner和必要hash/解析函数；复用现adapter与H3IterationRequest，不造通用dataset、采样runner或checkpoint manager。预算只覆盖明确轨迹payload，不冒称embedding/caller副本/模型workspace或整个进程RSS上限。

## 六项目标测试的实质证据

`tests/test_h3_mvp_cpu_source.py` 使用真实tmp文件和safetensors cache、真实registry/session/H3Adapter（重模型叶子tiny）：

- 八rank不同sample/完整组、相同文本不同occurrence、精确空白/中文、共同身份、两次commit和循环次序；种子与双流初态有独立手算参考，三类全局RNG保持。
- 重复reserve、错误/重复commit、abort/poison、mid-collection导出拒绝；mutation隔离和weakref验证已提交请求/条件释放。
- 两次commit后真实JSON往返→fresh同rank对象→下一request的ID/prompt/embeddings/初态/显式RNG状态逐值一致；同内容移动路径不改身份。
- schema/topology/stream/cache/schedule/seed/预算/进度与hot restore负例，文件/cache漂移、coverage/shape/预算等负例明确拒绝。
- 真实新source注入现H3IterationStages/BaseTrainer，reserve/commit未mock；真实adapter typed predict、prepared guard/Adam及双VAE控制执行一轮，tail推进cursor8/step1并release池，边界恢复后下一request一致。

最后一项明确使用旧stage测试中的协调/评分CPU叶子，不冒充此次新数据对象已与真实HPS/八rank协调/publisher整合。既有W4三对象组合测试仍在通配符中执行。

## 当前汇合树独立门禁

原样执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

- Flash exit0：**7 passed in 14.59s**。
- H3通配符 exit0：**126 passed, 16 warnings in 12.77s**，含本项6例及当前汇合目标，不以A分支114项历史结果替代。
- warnings来自checkpoint feature的明确单进程DCP save/load与损坏fixture覆盖提示，未过滤；不由本报告据此批准C实现。无skip/xfail匹配，未跑全量。
- 原Flash完整文件相对 `d818b8a231b52a9a024fa2f522057acdbb7bf099` 的获准历史diff无输出、exit0，原七项及断言不变。
- `git diff` 无输出；`git diff --check` exit0。

## 后续边界

当前生产源码搜索只有source定义，尚无默认runtime装配。下一步必须把真实source/profile/adapter/stages/checkpoint接唯一RunController，实际测试完整发布和两阶段fresh整run恢复/下一步对照。硬件拓扑、模型/评分资源身份由上层真实准入，不可用本source的字符串身份替代；全局RNG仍由外层最后恢复。

无GPU/服务器/HTTP/PG初始化/下载/环境升级/push。CPU不证明CUDA generator、真实H3权重/FSDP2/NCCL/HPS124资源或音视频质量。**GPU_PENDING，MVP未完成。**
