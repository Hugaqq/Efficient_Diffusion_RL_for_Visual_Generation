# Assignment B — W3

- active: true
- task_id: H3-M2c-hps-reward
- pair: coding-b → simplifier-b
- execution: parallel W3；decisions/wave-03.md
- integration_parent: `e436c192778a6bbf24e2bee7ebef777332138f9a`
- dependencies: 既有Hpsv3VideoClient/flash_hpsv3协议；W2 decoder approved，代码汇合ced9a871df0ee6bfe8b560e90da03bc12d4dda0e。无changes。

## 唯一功能/文件

把**现有真实HPSv3视频协议**接到H3单条AV lease，作为显式 `h3_hpsv3_video_only_v1` reward。只写新 `visual_rl/algorithms/rewards/h3_hps.py`、新 `tests/test_h3_mvp_cpu_hps.py`、W/coding-b/H3-M2c-hps-reward.md。旧client/protocol/resource factory/Flash配置/decoder/tests只读，不import A/C本波新模块。禁止实际HTTP/GPU/服务操作，只实现本地源码并用transport叶子测试。

## 决策与接口锁

先完整读wave-03 reward决策：单一HPS权重1，所有124帧走现协议top30%均值（现VideoContract是int(frames*.3)，124帧top37）；不是CLAP/ImageBind复现，不评分audio语义/AV对齐。**音频仍必须来自真实联合解码，不删除音频状态/CPS**，不声明音频质量。不改旧Flash任何frame/manifest或启动已有服务。

- 结构接口 `score_one(*, sample_id:str, prompt:str, media) -> float`，供A现stage调用。media有video/audio/metadata，与既有H3AVMedia结构一致；无需模型implementation import或继承第二套Reward服务框架。
- 生产默认使用现 `Hpsv3VideoClient` / `jpeg_frames` / `VideoContract` / health与score validator，不能另写HTTP、topK、JPEG或不校验响应的快捷client。若包装已按现ResourceFactory取得的client，要明确constructor要求与所有权；同时提供可由未来runtime构造的真实本地artifact准入路径，不只测试工厂。
- 本地HPS manifest + 显式expected service identity + loopback origin +明确geometry/timeout/response上限；必须校验本地artifact、对应VideoContract与live health一致再允许score。能复用现Hpsv3VideoResourceFactory就复用，不为便利削去identity；如果需要少量窄装配，仍调用原protocol校验。import/配置声明本身不联网，真正acquire才health。生产client失败不降级假reward。
- 严格CPU detached video[T,3,H,W]与lease metadata匹配、音频[L,2] finite和sample rate/实际长度一致；frame contract124、H/W采用实际声明，超现协议128上限明确拒绝。不要把81帧历史服务当124帧ready。video原dtype/范围符合现jpeg编码，帧完整不抽样；音频只校验/保持，不发送并冒称它被HPS评分。
- 调用原client.score_video(sample_id,prompt,frames_base64)，返回真实有限Python float；sample identity/selected indices/frame scores与service identity仍由原验证器严格校验，不信任远端单scalar。无自动POST重试/并发批媒体、无常量reward、无shadow/mock fallback。
- 固定reward profile/version/service/contract identity以小不可变元数据暴露给未来plan/checkpoint；不增加每帧大artifact或在对象长期保留media/JPEG/响应。生命周期close幂等、借用或拥有client责任明确；A在decode lease内调用该端口，退出后无媒体引用。

## 真实CPU验证

真实原Hpsv3VideoClient、JPEG编码、protocol health/score校验、wrapper执行；只HTTP transport leaf替身，不发真实请求。构造本地真实manifest和与媒体内容相关的有限frame score响应，所有帧与sample/prompt发送、124/top37结果和现protocol独立参考一致；至少两不同视频得到不同假服务叶子score，不能mock score_one或校验器。

错误service/health contract/sample ID、缺帧/非有限/非法selected indices或topK总分、超响应/请求大小、非2xx/redirect、timeout/no retry、过期lease/非法audio元数据、close后调用拒绝。测试不同合法几何与不兼容81帧契约早拒，不修改原七项或旧protocol。明确tiny/transport结果不证明服务资源或质量。

最后源码编辑后原样跑plan Flash七项+H3通配符，无skip/xfail/全量；写明branch/HEAD、命令/exit/数量/时间、diff/check及生产构造契约。Simplifier B重点审真正复用现已验证client、不丢identity/全帧规则及显式video-only限制；changes回B重测。

## 非目标

不部署新服务、不联系历史HPS、不装CLAP/ImageBind/下载模型，不改profile registry/controller或同波stage/coordinator。完整生产recipe绑定下一项，当前不能称新HPS服务ready。GPU_PENDING。
