# H3工程验收使用的作者原始视频prompt

来源：Shredded-Pork/Flash-GRPO `dataset/video/{train,test}.txt`，固定参考commit bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d。原始字节保留，不清洗/截取为smoke子集。

- train.txt：19700条、1513156 bytes、SHA256 610e16e6b0e8a30bbb1e75cf87413bf592fe491faa74d918d57d914d865e0016
- test.txt：300条、23615 bytes、SHA256 96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5

train用于训练来源，test只评估；完整缓存必须覆盖两者。一次更新访问4条prompt不等于把数据集截断为4条。19700条训练与300条评估的文本集合必须无重叠，启动前真实校验。
