# H3-M2a3a-text-cache — W1 Coding B

## 状态与真实工作区

- `implementation_ready: true`：仅当前原子项源码与本分支CPU门禁完成，待simplifier-b审核；不代表W1汇合批准或MVP完成。
- 本轮实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-e8cc81cc-68a4-48b2-ba77-835297ad76f1`。
- 实际起始/结束基线HEAD：`44918392c5903cf54c9d8af389317814b60d3e04`。初始工作区干净；没有使用旧main worktree路径，没有切分支/手工commit/merge/push。引擎snapshot待生成。
- 已重读task、plan、assignment B、W1边界、既有配对approved反馈、当前forward与本地身份helper；当前无changes返工项。

## 唯一改动

- 新 `visual_rl/models/implementations/h3_text_cache.py`：严格本地reader，coverage与单条有界读取。
- 新 `tests/test_h3_mvp_cpu_text_cache.py`：6项独立CPU测试。
- 本交付记录。

未改已有22项H3测试、原Flash七项、loader/forward/geometry或旧代码；不引用或修改A/C本波新文件。本分支尚无A/C新测试，局部通配符通过不能代替W1汇合后完整复测。

## API/下游合同

```python
cache = H3TextCache(
    local_directory,
    expected_identity=explicit_artifact_recipe_identity,
    max_tokens=positive_int,
    max_embedding_bytes=positive_int,
)
cache.verify_coverage(train_prompts, eval_prompts)  # 未来模型phase前必须调用
text = cache.get(exact_prompt)                    # CPU detached [1,L,5120]
```

- 不strip/normalize；全UTF8 prompt SHA256为key。重复训练occurrence复用相同prompt，不进入key。
- 构造验证全部已声明shard的header/hash；coverage检查train/eval每个精确字符串并重新验证用到的shard。get每次复核index和该shard内容，clone单条后finite检查；无cache miss在线fallback。
- 存储为精确有效`[L,5120]`，get仅加batch维；FP32/BF16/FP16原dtype保留，没有自动cast/猜padding/补零/截断。合法全零行或有限全零tensor不等于占位，原样接受。测试明确验证零行没有被裁掉。
- `identity_sha256`是固定index原始字节SHA256，绑定identity声明、完整prompt映射和全部shard内容hash，可供未来checkpoint绑定。`embedding_identity_sha256`是规范化identity JSON的SHA256。reader不保留调用者expected dict引用。
- 不保留shard句柄或tensor缓存；同一时刻只打开一个safetensors handle，读取后关闭。仅单tensor materialize/clone，其他tensor仅get_slice header。完整file hash按1MiB块流式读取，非全shard tensor加载。
- `max_embedding_bytes`限制一条tensor payload，不是总RSS；临时clone及文件哈希chunk会有额外有界内存。index/header分别限制16MiB。不是通用缓存或资源平台。

## 精确schema（也在模块docstring）

`index.json`仅允许四字段：

```text
schema_version = "visual_rl.h3-text-cache.v1"
identity = {
  model_sha256, encoder_sha256, tokenizer_sha256, processor_sha256,
  template_sha256, truncation_sha256,   # 各为完整小写64hex摘要
  target_layer: 50,
  normalized: false,
  add_special_tokens: false,
  feature_dim: 5120,
  dtype: "float32" | "bfloat16" | "float16"
}
shards = { "local_basename.safetensors": full_file_sha256 }
entries = { full_SHA256(exact_prompt_UTF8): { shard, tensor, tokens } }
```

每个(shard,tensor)映射唯一；shard包含且仅包含其index声明的tensor，shape为[tokens,5120]，dtype严格等于identity。所有身份由未来artifact/recipe显式提供，reader不会替调用者填身份常量。

每个safetensors的metadata恰为：

```text
{"h3_text_identity_sha256": SHA256(identity canonical JSON)}
```

canonical JSON为sorted keys、separators=(',', ':')、默认ensure_ascii=True的UTF8字节。完整file SHA256另由index绑定，包含header/metadata/payload。没有每条重复大模型身份。schema v1只接受basename，不接受子路径、绝对路径、`..`或任何symlink。

**原UniRL raw 1.0 index不兼容并明确拒绝**：其16hex key与原始fingerprint不满足完整身份与hash要求。保留index+sharded safetensors及[L,5120]布局；本项不提供writer/转换脚本，不声称任意上游cache可直接加载。

## 完整性/边界

- JSON含重复key（包括嵌套index/实际safetensors header）拒绝；重复映射、未声明/未引用shard、tensor key覆盖不符、缺文件、形状/dtype不符均拒绝。
- token/字节上限及header形状/dtype先于payload/hash扫描及get_tensor。无效payload不能借先读tensor绕过预算。
- 每次实际读取使用同一已验证本地path；完整SHA256在读取前后核对，辅以inode/size/mtime/ctime检查检测读期间变化。mtime不是身份来源，测试中恢复原mtime且保持文件长度的payload篡改仍被hash拒绝。index修改也使既有reader拒绝继续读取。
- 文件应在使用期间保持不可变；这里没有跨进程写锁、缓存writer或安全对抗恶意并发文件替换的通用文件系统事务承诺。
- finite检查在get交付前执行；构造/coverage不提前物化所有embeddings，不能把它们称作所有payload的数值检查。

## 来源/许可

重读UniRL固定`8cfbac17765004afb68fef08a4e011a4c8daf555`的LICENSE全文（Apache-2.0）、offline_text_embed.py及text_embed.py。后者真实逻辑是Qwen layer50、移除最终norm、text-only tokenizer add_special_tokens=False。当前reader只核验这些声明和身份；**不从数值证明生成过程真实、权重正确或文本语义正确**。

独立实现严格reader，无复制上游writer/缓存handle平台、无reference import、无encoder/tokenizer加载、无模型或缓存下载。现core/filesystem_identity仅编码身份而不负责文件I/O；本项使用明确原始SHA256与最小内容检查，没有另造通用manifest系统。

## CPU证据

6项新测试使用真实tmp_path JSON及safetensors，不mock reader/hash/safetensors：
- 两shard/三prompt、train与heldout全覆盖、精确空白差异、三dtype、固定5120维、合法零行/全零值、detached/拥有副本、缺失train/eval拒绝。
- get结果实际进入已批准h3_t2va_forward；只重transformer替tiny叶子，单次调用/符号/图与backward验证。
- 模型/encoder/tokenizer/processor/template/truncation身份、层/norm/special/feature/dtype不符、raw UniRL schema、重复JSON/mapping/metadata、keys/shape/dtype/nonfinite负例。
- token/byte/header准入优先于坏payload/hash；路径/子目录/symlink逃逸、缺shard、同长度同mtime篡改及index漂移拒绝。

最后源码编辑后，实际执行新版两条门禁，均exit0，无skip/xfail/全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 11.85s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 28 passed in 3.94s（本分支既有22 + 本项6）
```

`git diff`无输出（只有新未跟踪文件）；`git diff --check` exit0。status仅三个授权新文件；未把空tracked diff冒充新文件历史范围审计。

## 交付后

Simplifier B只审本项；changes回B修复再跑两集。引擎snapshot/汇合不等于批准。Orchestrator必须在W1汇合后读三组裁决并再跑完整通配符与原七项。

VAE decode、真实reward、world/controller接线、完整tail/checkpoint/fresh恢复仍未完成，text cache不是完整aux或MVP。没有GPU/服务器/部署/下载/环境升级/push操作。**GPU_PENDING**，CPU不证明真实encoder产物/权重或视频质量。
