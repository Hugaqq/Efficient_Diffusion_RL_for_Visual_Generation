# 当前交接：repair — H3-EXP-001，材料已齐

请先读`issues/H3-EXP-001-materials-r1.md`。网络本轮正常，所有新请求首次code0，不再等待网络输入。

Coder先前need_experiment所需材料已取得：
- 已安装三个H3类源码hash与固定版本全部一致。
- 官方固定converter已通过600秒有界Git prepare取得、HEAD/clean/hash核验；完整阅读转录为`experimenter/converter-004-part1.txt`、part2、part3，来源/许可见converter-004-provenance.md。没有执行converter。
- FL2VA原生小配置及audio metadata完整内容、原bytes/hash/blob路径见`experimenter/native-configs-004.json`。
- 找到新版布局目录`/home/ubtroot/code/downloads/MiniMax-H3`，但transformer14/14、vae3/3 shards及audio payload缺失，不可复用为完整artifact。
- 真实公开类+PEFT meta成功：1238 state keys（638 base+600 LoRA），66,363,578,880目标bytes，buffer64 bytes，全部参数meta。
- 新probe `converted-meta-admission-001` exit1/running=false：_weight_headers拒缺第一shard；decoder _configs拒缺VAE shard。完整命令/栈/时刻在不可变r1报告，没调用full CPU loader。

请完成最小准备调用/严格映射与实际资源受限加载方案，再交Experimenter Git部署真实复验。不要再次重复索要已提供相同材料，不盲改类名或放宽guard，不让Experimenter写算法/转换脚本。原H3-EXP-001仍open；H3-EXP-002 single-step后续queued。Flow/Flash仍0更新，正式步数此阶段不询问。旧网络与失败报告全部保留。
