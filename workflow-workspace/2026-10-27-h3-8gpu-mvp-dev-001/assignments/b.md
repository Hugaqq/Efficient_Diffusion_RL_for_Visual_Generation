# Assignment B — W1

- active: true
- task_id: H3-M2a3a-text-cache
- pair: coding-b → simplifier-b
- execution: parallel W1；见decisions/wave-01.md
- integration_parent: `b7f938506f16034ac5d2896ddfa44c16eb4af83d`
- source_dependencies: forward `edd5c874f10ffa917a71bd9f2fa9085247075a6e`、loader `919b6abc7920f2dbcea764907152523882a09a59` 已approved。当前无changes；原loader不再修改。

## 唯一目标/允许文件

实现**严格本地离线H3 text embedding读取与训练/评估全覆盖准入**，供既有forward的trimmed `[1,L,5120]`输入。仅写新 `visual_rl/models/implementations/h3_text_cache.py`、新 `tests/test_h3_mvp_cpu_text_cache.py` 及本目录 `coding-b/H3-M2a3a-text-cache.md`。既有loader/forward/geometry/旧测试只读；不修改或import A/C本波新文件，无共享文件写入。

## 输入输出锁

- 窄 `H3TextCache` 类或等效reader，只接受本地目录与显式expected身份、max_tokens、max_embedding_bytes。没有Qwen/tokenizer加载、在线miss fallback、自动生成缓存或下载。missing train/eval prompt应在模型phase前拒绝。
- 公开 `verify_coverage(train_prompts, eval_prompts)` 与 `get(prompt)`；coverage检查两集合每个精确字符串，不strip/normalize，不把组occurrence身份混进cache key。get返回拥有副本的CPU detached `[1,L,5120]`、保留显式存储dtype、无padding（存储本来为精确有效L）；不得猜零token为padding或返回占位全零向量。
- 来源读 `reference/UniRL/LICENSE`、`offline_text_embed.py` 和 `text_embed.py`：H3层50、未加最终norm、feature5120、text-only无额外special tokens。这里只验证cache声明/身份，不冒称从embedding数值证明生成过程正确。
- 定义一个最小显式版本化本地index+真实safetensors schema并写在模块docstring/交付中；优先沿上游layout，但必须补足模型/encoder权重、tokenizer/processor、模板/截断、层50/unnormalized、dtype等身份校验。expected身份由未来recipe/artifact绑定提供，不能自行填常量通过。若与上游raw index不兼容，应明确拒绝，不能声称任意UniRL cache直接兼容。无需本项实现writer或转换脚本。
- 精确prompt SHA256键及shard/tensor键和完整file SHA256绑定；本地路径逃逸/重复JSON键/重复映射/缺失shard/hash漂移/metadata错配明确拒绝。避免无界打开shard handle或加载整个shard；每次只读一条，读取后关闭。header形状/dtype/token/bytes上限应先于payload访问，finite检查后才交付。合法零值不等于占位，可接受真实有限tensor，但不得生产常量embedding。
- 不按每条记录重复存全模型身份/大manifest，不做通用resource cache平台。身份由reader拥有固定摘要，后续checkpoint可绑定；实际读文件需核验和所声明的hash一致，不能hash另一个路径。文件修改不能继续静默使用旧身份；使用现core/filesystem_identity工具或最小明确检查，不靠mtime单独替代内容。
- feature_dim生产固定5120；tiny测试L可小但维度不需要缩成假H3。FP32/BF16/FP16支持范围明确并与expected dtype严格一致，不隐式cast。上层adapter将自行搬设备，本项不碰GPU。

## 实测验收

用tmp_path真实JSON/safetensors，不mock reader/hash/safetensors：两shard/多prompt、train和heldout覆盖、精确空白差异key、合法L和维度/dtype、输出detached拥有副本。缺train/eval项、错误层/norm/encoder/tokenizer/template身份、重复key、路径和symlink逃逸、hash/tensor篡改、非有限/错误shape/dtype、token/byte上限在payload读前拒绝。测试get结果实际进入既有h3_t2va_forward（tiny transformer叶子）而非只有reader自身assert；不需要C prepare或A锚。

仅新feature测试文件；原22项不动。最后源码编辑后运行plan最新版两条命令：原Flash七项和H3通配符 `tests/test_h3_mvp_cpu*.py`，无skip/xfail/全量。交付记录实际branch/HEAD、命令/exit/数量/时间、source diff/check与schema/下游接口。引擎提交隔离分支，不手动commit/merge/push。

## 非目标

不做text生成、VAE decode/reward服务、loader修订、profile/registry/controller、FSDP/PG/checkpoint。VAE作为后续独立项，不将其偷省为未来优化。Simplifier B检查身份/层语义、严格覆盖和有界真实读取、可调用forward消费者与不必要平台化；changes回B修复重测。CPU不证明真实encoder产物/权重兼容或质量，GPU_PENDING。
