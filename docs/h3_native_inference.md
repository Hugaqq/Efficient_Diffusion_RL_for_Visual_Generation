# H3原生联合音视频：阶段C交付 / D真实验证

## 状态先读

A已经实现并推送 `9d5c1e4afcdbefa5ef61c0754d3ce3bb844b3c75`。B服务器于2026-10-24 17:01:49 UTC发现新root PID939604、`/var/tmp/python`、同特征外连及GPU100%，**没有部署或运行RL5**。这是外部安全阻塞，不是算法失败；不能将A的CPU结果称为五次GPU通过。C按TASK继续本地实施，D仍须安全复查。

本次C代码在同一`h3-readable-integration`分支交付。最终测试/40位VisualRL部署SHA见任务 `outputs/visualrl-h3-integration/local-c-result.md`（文件不能自含自身commit SHA）。不要以分支名/latest代替精确SHA。若服务器B尚未运行，可在最新测试C SHA上fresh运行原RL5；旧run不要热续到新code identity。

## 用户review：两层，而不是另起HTTP流程

```text
python -m visual_rl.infer_h3 CONFIG
  Model: H3InferenceAdapter.load(local assets) / generate(H3Request) / close
    → pinned vLLM-Omni DiffusionEngine, uni executor（GPU worker就在本进程）
    → MiniMaxH3Pipeline：Qwen3-VL hidden50 + native tokenizer/processor
    → packed联合video/audio状态 + 两路sigma/RES solver + condition anchors
    → native video-VAE + audio-VAE → 两路真实tensor
  Algorithm: run_h3_rollout
    → 串行一条/三条原prompt请求 → 校验AV → 导出MP4/float WAV
    → 全部decode/实测metadata/哈希 → 每样本receipt → 完整关闭后SUCCESS
```

模型`generate`是明确的原生推理能力，不是已统一到RL单步`predict`、概率、轨迹或训练ABI。本路径**不支持H3 RL训练**，不构造reward、LoRA、optimizer、ParameterStateManager或训练checkpoint；既有Wan训练/数值/恢复路径不改。没有在VisualRL新建engine registry、服务stage调度器或第二个训练loop。上游库内部必要执行器/请求控制不复制进项目。

`uni`仍会初始化上游one-rank distributed/device/env全局状态，故运行在独立H3 CLI进程/venv，不嵌入正在运行的Wan trainer；这不是外部生成服务。唯一额外子进程为FFmpeg编解码，不执行模型生成。

## 支持范围与原生数值边界

- 第一版仅原始`MiniMaxAI/MiniMax-H3`的完整本地`FL2VA/`分区；T2VA无图，FL2VA明确first/optional last本地图，B=1，原prompt和seed不改。三场景使用T2VA请求、FL2VA权重。
- 不是ComfyUI单文件pruned DiT/NVFP4 AWQ encoder，不是FastH3/Turbo/VDN，不带LoRA。发现量化config、缺shard/processor/tokenizer、非FL2VA或固定DMD base_schedule立即失败；不下载/转换/HTTP fallback。
- 参考请求1344×768、124帧、24fps、20 `res_multistep`。上游拥有packed video/audio、条件prefix、两路shift12/3与RES历史；VisualRL不复写其数学，不套Wan CFG。
- 连续base边界u从1到0，video sigma=12u/(1+11u)，audio=3u/(1+2u)。原生模型的FP32 patch projection、BF16 condition projection、FP32 velocity/solver和VAE各自精度由固定上游实现保留，不对整个pipeline粗暴`.to(bfloat16)`。
- 不绕过上游reference anchor/tokenizer/processor。FL2VA传PIL副本，rollout校验输入图片在生成前后SHA未变，并保存condition image hashes。
- 原生帧边界17n+5，不静默round。audio latent为40Hz，decode×800、32kHz stereo；124帧对应207个音频latent、165600个原生audio samples（约5.175s），视频124/24约5.1667s。保留这一原生时钟差，不补静音/截掉音频假对齐。
- `--diagnostic`显式改成384×224×107，原prompt/seed/20RES保持，所有receipt/成功标记注明diagnostic；**不计入正式三场景展示**。

## 固定后端、依赖与资产

### 后端源码pin与独立环境

后端必须clean **vllm-project/vllm-omni@0f79bd8b0e5ff9379a9a38e0c585db8f8335bf7c**。`verify_backend`核对Git完整SHA/dirty及实际`vllm_omni.__file__`。不能仅相信package version，也不能用本项目文本审查快照当完整可安装仓库。

本地已经读过该pin的native pipeline、encoder、VAE加载、双solver、uni executor、engine/config/request及Flow-Factory直接joint实现；未安装/执行上游模型。vLLM-Omni的依赖由该pin的`setup.py`读取`requirements/cuda.txt`及递归`-r`文件，不是在pyproject里静态全列。本地审查文本副本未包含requirements目录，**不编造已测试的torch/vllm/CUDA精确版本或可安装性**。部署应fetch完整固定Git后读取这些固定版本依赖文件，再仅在新环境安装、`pip check`并保存完整freeze/driver信息；若依赖/driver不兼容即返回真实阻塞，不改Wan环境。源码pin决定所用API，实际freeze和模型文件SHA随run保存。

以下是隔离路径和执行模板，仅server经授权WebCLI、复查安全之后操作；local没有执行：

```bash
export BACKEND=/mnt/data/qifan/src/vllm-omni-0f79bd8
export H3ENV=/mnt/data/qifan/envs/visualrl-h3-native
export H3PY="$H3ENV/bin/python"
export HF_HOME=/mnt/data/qifan/models/h3-native/hub-cache
export PIP_CACHE_DIR=/mnt/data/qifan/cache/pip-h3
export TMPDIR=/mnt/data/qifan/tmp/h3
mkdir -p /mnt/data/qifan/src /mnt/data/qifan/envs "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
# 仅当BACKEND/H3ENV不存在时首次建立；存在则检查，不能覆盖/自动重装。
git clone --no-checkout https://github.com/vllm-project/vllm-omni.git "$BACKEND"
git -C "$BACKEND" checkout --detach 0f79bd8b0e5ff9379a9a38e0c585db8f8335bf7c
git -C "$BACKEND" status --short --branch
# 阅读该pin的requirements/cuda.txt及所有-r引用；确认不改变共享环境。
python3.12 -m venv "$H3ENV"
export VLLM_OMNI_TARGET_DEVICE=cuda
"$H3PY" -m pip install -r "$BACKEND/requirements/cuda.txt"
"$H3PY" -m pip install --no-deps -e "$BACKEND"
"$H3PY" -m pip install imageio-ffmpeg
"$H3PY" -m pip check
"$H3PY" -m pip freeze --all > /mnt/data/qifan/runs/visualrl-readable-h3-20261024/h3-dependencies.freeze
```

FFmpeg需要支持libx264、AAC、PCM float、showinfo/ashowinfo和`-fps_mode passthrough`（建议>=6）。优先既有系统FFmpeg，否则imageio-ffmpeg。每份媒体receipt记录实际binary路径/version；本地真实FFmpeg导出/全decode测试不证明服务器binary版本相同。不要安装NVFP4/Blackwell专属backend来尝试在A100上蒙混通过。

### 原始模型必须独立下载到数据盘，不能拿现有Comfy文件顶替

`MODEL_REVISION`必须是官方HF仓库确定的40位commit（不是main）。本轮未在线查询一个新HF revision，因此不伪造模型SHA。server下载前记录官方revision和下载清单，容量不够则阻塞。`inspect_assets`的`declared_revision`如实名名：这是调用者来源声明，不能单独认证远端；实际所有本地配置/tokenizer/processor/代码/weights文件均独立流式hash记录。

```bash
export MODEL_REVISION='<确认并记录的官方HF40位commit>'
# 下载阶段不要设offline=1；此处只下载文件，不执行模型Python。
"$H3PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download('MiniMaxAI/MiniMax-H3', revision=os.environ['MODEL_REVISION'],
    cache_dir=os.environ['HF_HOME'], allow_patterns=['FL2VA/**'])
print('MODEL_ROOT=' + path)
PY
# 把上面真实返回的snapshot根目录设为MODEL_ROOT；其下必须有FL2VA/。
export MODEL_ROOT='<上面真实返回的本地snapshot根目录>'
```

模型VAE可能通过`auto_map`执行权重目录中的本地Python。preflight仅列出/hash，不加载模型；**server须逐份阅读清单中的模型.py及其相关文件**，确认来源/内容可执行，再把整个代码清单digest作为`--reviewed-model-code-sha256`显式批准。digest缺失/不符则不会load engine。不能不看代码就复制digest；这个参数记录显式审阅，不是自动安全证明。模型与后端文件应保持只读、不可在运行时热改。所有HF/Transformers生成时强制offline，下载与执行分开。

### 容量风险（不是实测承诺）

原完整分区约135GiB（TASK记录），旧机RAM约167GiB、最近available142GiB、系统盘只剩28G。全部依赖/cache/tmp/权重放`/mnt/data`，**不要在系统盘再下载135GiB或升级既有Wan venv**。

CLI保守准入：GPU free>=68GiB；host available>=全部safetensors字节+24GiB；disk>=64GiB。执行期复用原15s心跳和16GiB GPU/24GiB host/64GiB disk门。上游encoder保留50层约51.5GB BF16，加DiT/VAEs必须由原生组件owner分阶段module-offload；pin_memory=False避免额外大规模pinned副本。

**按最近旧机available值，完整原权重很可能在host准入即被阻塞。** 预算不是实测上界，loader临时副本/页缓存/AV workspace可更高，单A100可行性未知。若容量不足，报告确切数值，需安全的更大内存设备或另行证明受支持的权重格式；不删门禁、强行读取Comfy pruned/NVFP4、不编造已生成。D不能因为C有fake测试通过就算完成。

## D：精确Git部署与一条→三条

先按TASK、`server-rl5.md`复查主机和新root异常；新PID939604的证据不能泛化成处置未来新PID授权。若仍同特征活跃，**GPU阻塞**，只落证据，不启动模型/停止服务。安全恢复后，只通过授权origin和hostname的WebCLI操作；main保活idle-only，server确认前台完成，不重放未知启动。

按 `docs/readable_h3_handoff.md` 的任务checkout Git fetch/stash规则部署**local-c-result精确AGREED_SHA**，不改其他旧worktree/SMALL40/服务。核对clean和真实imports：

```bash
export AGREED_SHA='<local-c-result测试通过的40位VisualRL SHA>'
export SOURCE=/home/ubuntu/qifan/visualrl-readable-h3-20261024/source
# 首次worktree或已有任务checkout保留dirty/stash的步骤见A交接。
git -C "$SOURCE" fetch origin h3-readable-integration
git -C "$SOURCE" switch --detach "$AGREED_SHA"
cd "$SOURCE"
git rev-parse HEAD
git status --short --branch
export PYTHONPATH="$SOURCE:$BACKEND"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
"$H3PY" -c 'import visual_rl; from visual_rl.models import h3; import vllm_omni; print(visual_rl.__file__, h3.__file__, vllm_omni.__file__, sep="\n")'
```

明确新OUT，不存在或没有重复活跃作业才启动：

```bash
export H3BASE=/mnt/data/qifan/runs/visualrl-readable-h3-20261024/h3-native
export ONE="$H3BASE/one-001"
export THREE="$H3BASE/three-001"
mkdir -p "$H3BASE"
"$H3PY" -u -m visual_rl.infer_h3 configs/h3/one-scene.json \
  --expected-sha "$AGREED_SHA" --backend-root "$BACKEND" \
  --model-root "$MODEL_ROOT" --model-revision "$MODEL_REVISION" --output "$ONE" --preflight
```

RC非0保留错误返回，不自动放松资源预算。RC0后读取`$ONE.preflight.json`的`assets.model_python_files`，逐文件read审阅；审阅接受才手工设 `REVIEWED_CODE_SHA` 为其中实际`model_code_review_sha256`。

```bash
# 唯一server确认安全/准入/代码审阅后只提交一次，不等同于本地已运行：
LOG="$H3BASE/one-console-$(date +%Y%m%d-%H%M%S).log"
nohup "$H3PY" -u -m visual_rl.infer_h3 configs/h3/one-scene.json \
  --expected-sha "$AGREED_SHA" --backend-root "$BACKEND" \
  --model-root "$MODEL_ROOT" --model-revision "$MODEL_REVISION" --output "$ONE" \
  --reviewed-model-code-sha256 "$REVIEWED_CODE_SHA" >"$LOG" 2>&1 < /dev/null &
PID=$!; printf 'PID=%s LOG=%s OUT=%s\n' "$PID" "$LOG" "$ONE"
ps -p "$PID" -o pid,etime,stat,args
readlink /proc/"$PID"/cwd
```

这次须等真实原生视频+音频、完整decode和自然退出，不能仅见PID就放行三条。必要容量诊断另用新OUT加`--diagnostic`，不可将小规格视频计入正式三条。

一条成功后再复查安全与资源，以同参数、**`configs/h3/three-scenes.json`、`--output "$THREE"`、独立LOG**运行三条；仍是B=1串行，不同时跑两组。不能改scene原prompt/seed或自动扩80条/640条；也不能重用ONE作为THREE目录。

## 结果、验收与Mac下载

每样本：`<OUT>/<sample_id>/{video.mp4,audio.wav,receipt.json}`。
每attempt：`<OUT>.launch/{owner.json,assets.json,load.json,resources.json,result.json}`。
失败保留`.partial`、FAILED/result；不覆盖、自动retry或把旧文件当生成。

- receipt包含原request/seed/actual rates/layout/时长、模型实际assets SHA与声明revision、后端pin、RES/双shift、诊断标记、native stage timings/peak、MP4/WAV字节数和SHA。
- MP4逐帧和audio全解码；实际showinfo/ashowinfo确认宽高/fps/stereo32k，检查帧/样本数及finite/nonzero audio。浮点WAV完全decode并与原生waveform字节一致，不造静音；AAC允许最多2048 samples编码padding且实记观察值。
- `ROLLOUT_COMPLETE.json`仅表示媒体rollout阶段完成；必须`<OUT>.launch/result.json.status=native_av_generated_and_decoded`及`<OUT>/SUCCESS.json`同时存在、diagnostic=false，且native close及最终资源sample成功，才是该run通过。这些均不证明RL5通过。
- 最终三份request须1344×768×124/20RES，fps24、WAV2×165600/32k；seed42009/42012/42013、原prompt完全匹配。复核文件真实SHA，不能用旧Comfy文件或只inspect已有结果冒充生成。

server应从**真实三条新receipt**生成下载清单（实际remote路径/bytes/SHA），通过已有授权OpenCLI分块读取机制拉取到：

`~/Desktop/ZIP Lab/Diffusion RL Infra/artifacts/visualrl-h3-native/`

建议本地名`<sample_id>-seed<seed>.mp4`，同时下载WAV、receipt、launch metadata。已有`outputs/h3-scene-inference-20261024/download_videos.py`硬编码旧ComfyUI路径/哈希，**禁止原样运行来冒充本次下载**；主会话/获准部署角色只复用其64KiB（必要时32KiB）read-only base64分块逻辑，改用本次新receipt manifest与本目标目录，遵守同一bridge lock及idle prompt检查。local编码角色不操作browser或越权写该artifacts目录。

Mac每文件先`.partial`，全部字节到齐后SHA一致才rename、写`.sha256`，再完整decode核对双轨；保存native-download-manifest。不仅发远端路径或飞书旧预览。若server下载权限不足，由主会话处理；未完成这一步不能报告总任务完成。

## CPU门禁和证据范围

C交付前在实际candidate运行原`tests/test_flash_collection_runtime.py`恰好7项，加独立`tests/test_h3_native_inference.py`及A专项/compile/diff检查。H3单测使用显式fake backend/assets，不下载/导入/执行上游模型代码；真实FFmpeg对小型fake视频/音频编码并全decode，不能当H3模型/GPU成功。

最终实际候选门禁：原7项`7 passed in13.95s`；H3专项`8 passed in1.86s`；A专项`10 passed in3.32s`；compileall及diff检查无错误。初轮H3专项5通过/3失败暴露本机FFmpeg移除`-vsync`，改为`-fps_mode passthrough`后全通过；加actual showinfo metadata时发现必须读取输入帧率而非输出未定义0/0，修正后全通过；没有删断言/skip。最终提交SHA及Git状态记录见local-c-result。GPU、原始权重兼容性、独立依赖安装、性能/RAM/HBM及正式三段下载仍待D真实证据。

### 固定源码依据

- vLLM-Omni pin如上，`diffusion/models/minimax_h3/pipeline_minimax_h3.py`：native FL2VA权重解析、tokenizer/processor、no_grad joint diffuse/decode、postprocess uint8 BFHWC+stereo audio。
- 同目录`denoise_loop.py`、`sampling.py`、`time_request.py`：两路sigma/solver、condition anchors、17n+5及40Hz时钟。`encoder.py`为Qwen3-VL layer50；`vae.py`有auto_map trust gate。
- `diffusion/{data.py,diffusion_engine.py,request.py}`及`executor/uniproc_executor.py`：本进程单GPU执行与明确关闭；uni超时不强制打断卡死模型，故资源/进度异常应如实保留并返回，不自动重放。
- Flow-Factory@3dd5ca4dff396c1db9b59c16dceaaca7836c6ed1 `models/minimax_h3/{workflow,blocks,denoise,dependency}.py`作为双状态/模型边界反例参照；没有复制其厚BaseAdapter，也没有把它的SDE训练loop当本次RES纯推理实现。
