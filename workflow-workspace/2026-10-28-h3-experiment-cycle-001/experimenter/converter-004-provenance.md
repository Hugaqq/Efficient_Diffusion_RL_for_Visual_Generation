# 固定官方转换器完整阅读材料（非生产实现）

这些`.txt`仅为Coder复核用的完整源码转录，不import、不执行、不作为服务器source部署。Experimenter没有修改算法或转换器。文件按字符范围切分，边界可能在单词中间；按part1、part2、part3顺序连接即为阅读顺序，无额外分隔符。

- 上游：Hugging Face diffusers，https://github.com/huggingface/diffusers
- 固定Git SHA：`aaf57d03a3edfbd9de457d2872417a92f149a91d`。
- 远端独立目录：`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/pinned-diffusers-source-001/diffusers`。
- 实际通过Git fetch该SHA、checkout detach，HEAD核验相同，git status --porcelain为空。未改installed venv。
- `scripts/convert_minimax_h3_to_diffusers.py`：47157原始bytes，实际SHA256=`86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c`，与Coder要求一致。
- 许可证：上述checkout的`LICENSE`，Apache License Version2.0 January2004；SHA256=`f9e2070c247517b1ddf65f7b11b393484a18da91a958fd18a97bd0f241c3125c`；上游全文 https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/LICENSE 。Coder若引入生产第三方实现仍须保留所需许可/归属。
- 本地阅读文件：`converter-004-part1.txt`=[0:16000]字符；part2=[16000:32000]；part3=[32000:]。通过工具read stdout转录；原文件SHA在远端核验，本地转录尚未独立计算拼接hash，不以转录代替执行时的固定源码hash验证。
- 对应原文操作：exp004-converter-part1-01、exp004-converter-part2-01、exp004-converter-part3-01，均code0/uncertain=false，分段未触发operations末22000字符截尾。
- 下载job：pinned-diffusers-source-001，PID933370/start_ticks74226462/created1793155110.5590448，600秒有界；child933371；exit0，started1793155110.6031036 ended1793155116.1823854；running=false。kind=prepare，CUDA_VISIBLE_DEVICES为空。
- 操作exp004-upstream-source-prepare-01和exp004-inspect-source-01，完整spec/owner/exit/job.log保留。

只取得并读取源码，**没有import或执行converter（包括dry_run），没有转换任何权重，没有产生训练SUCCESS或checkpoint**。文件中tiny/test配置只是上游原文的一部分，本实验不使用它冒充真实训练。

## 已安装公开实现身份

exp004-discovery-01实际从`/home/ubtroot/qifan/h3-env-20261027-001/venv/lib/python3.11/site-packages/diffusers/`读取：

|相对路径|bytes|SHA256|
|---|---:|---|
|models/transformers/transformer_minimax_h3.py|33045|92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef|
|models/autoencoders/autoencoder_kl_minimax_h3.py|42128|58e0ed47fdee8227d738eec5bb1e7a47acc21c65ee3de01aa3795c97d679d015|
|models/autoencoders/autoencoder_kl_minimax_h3_audio.py|30959|b56c5bb95a7dc99ace03df192544615c89ec01ac3b134a7f1dd7348b03731d57|

三者均与Coder提供的固定来源hash完全一致；不能仅凭此称模型或转换等价通过。
