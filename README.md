# 多模态决策模型 · Multimodal Decision Models

**将视觉语言模型改造成面向候选选择与动作建议的多模态决策模型。**

我们延续 Qwen3-VL 多模态决策模型的改造方法，为 **Gemma-3n-E4B / E2B、MiniCPM-V-4.5、InternVL3.5-8B / 14B** 实现统一决策通路：融合文字、RGB 图像、带时间戳的视频帧和测量状态，直接输出可供程序使用的决策结果。

## 我们做了什么

1. **将多模态理解接到决策输出。** 保留各基座原生视觉编码与语言融合通路，读取候选标签的下一词元 logits，输出稳定选项 ID、候选概率及推理记录。
2. **实现四类统一决策。** 支持候选选择 `choice`、真假判断 `noul`、等级评分 `score` 和四轴速度动作建议 `velocity4`，无需解析自由文本回答。
3. **实现多模态状态与动作协议。** 将按时间排列的视觉观测与可测量状态接入同一输入协议；四轴动作覆盖 2401 种组合，配有独立的时效、刹车距离和联合路径安全检查函数。
4. **适配五个模型，共用一个决策台。** 提供可切换模型的网页、统一 API、按需加载和单模型驻留管理，保留已有 Qwen 服务的接入方式。
5. **提供可复现的运行与验证代码。** 项目包含独立环境、模型下载、启停入口、测试及脱敏验证记录。新增三个模型均完成真实 GPU 接口验证，并与官方生成第一步的候选 logits 对照。

```text
文字 + RGB 图像 / 视频帧 + 测量状态
                 ↓
       基座模型原生多模态编码与融合
                 ↓
       候选 logits → 候选内概率分布
                 ↓
   选择 / 判断 / 评分 / 四轴动作建议
```

这里的“决策模型改造”指多模态推理与结构化决策通路的实现。本版本使用原基座权重，未进行决策微调；候选概率未经校准，四轴输出为动作建议。验证结果与能力边界在下文列出。

## 模型基座

| model_id | 官方基座 | 适配方式 |
|---|---|---|
| `gemma-e4b` | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) | 原生 Gemma3n processor / conditional-generation forward |
| `gemma-e2b` | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) | 原生 Gemma3n processor / conditional-generation forward |
| `minicpm-v45` | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | 原生图像切片、视觉重采样、语言模型 embedding 融合 |
| `internvl35-8b` | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 原生动态切片、视觉特征与 IMG_CONTEXT token 融合 |
| `internvl35-14b` | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) | 同上 |

决策协议沿用 Qwen3-VL 决策台的 `choice`、`noul`、`score` 和 `velocity4` 接口。可通过 `QWEN_DECISION_URL` 接入已有的兼容 Qwen 服务。

## 快速开始

Linux、Python 3.12、NVIDIA GPU；实测环境为 PyTorch 2.8.0、Transformers 4.57.1、BF16。模型只按需加载，一个后端同时驻留一个模型。预留显存门槛分别为 E2B 16 GiB、E4B 20 GiB、MiniCPM/InternVL 8B 22 GiB、InternVL 14B 34 GiB；长输入可能需要更多。

```bash
git clone https://github.com/maolinqi/multimodal-decision-models.git
cd multimodal-decision-models
./setup.sh
# Gemma 需要先在 Hugging Face 接受模型条款并登录。
.venv/bin/hf auth login
.venv/bin/python scripts/download_models.py gemma-e2b gemma-e4b minicpm-v45 internvl35-8b internvl35-14b
CUDA_VISIBLE_DEVICES=0 ./start_all.sh
```

浏览器打开 http://127.0.0.1:8456，在同一个页面切换模型。视频抽帧需要系统已有 `ffmpeg`、`ffprobe`。启动脚本不会重复安装依赖；停止使用 `./stop_all.sh`。已下载模型可通过 `MODEL_ROOT=/path/to/models` 指向共享模型目录，下载与启动时使用同一个变量。

安装后的 gateway 需要仓库中的 `web/`，因此建议保留 Git checkout 并使用上述 editable 安装。服务默认绑定本机，不包含公开隧道或生产身份验证。

## API

```bash
curl http://127.0.0.1:8457/health
curl -H 'Content-Type: application/json'   --data-binary @examples/choice.json http://127.0.0.1:8457/decide
```

图像通过 `images` 数组传入，每项包含 `base64`（无 data URI 前缀）、`camera_id` 和 `timestamp_ms`。视频以按时间排列的 RGB 抽帧输入，不接收音轨。最多 8 帧，总输入上限 8192 tokens，超限报错而不是截断。InternVL 使用最多四个动态 tile 加缩略图，MiniCPM 使用最多四个切片；这是一种有限计算预算配置。

`choice` 使用 2–26 个稳定 ID 映射到 A–Z；`noul` 返回 true/false；`score` 返回等级概率与期望分数。`velocity4` 分别选择 vx、vy、vz、yaw-rate 的七档，共 2401 种组合。每个分量一次语言模型前向，共四次，独立分量分布不是联合安全概率。动作默认 `executable=false`，需要外部实时遥测与联合路径检查。见 [接口说明](docs/architecture.md)。

## 验证与限制

```bash
.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/validate_model.py minicpm-v45 --out evidence/minicpm-v45.json
```

验证脚本检查图像进入视觉通路、候选概率归一化、四种接口与官方原生生成第一步的候选 logits 一致性。小型合成图像探针不等于现实泛化评测。最新实际结果见 [验证记录](docs/validation.md)。Gemma 既有验证中 E4B 的 `2+3` 探针曾选错，保留这一限制。

没有附带训练后的 LoRA，没有将接口改造描述为学习能力提升。对输入状态的约束和安全门不替代真实系统安全验证。

## 项目结构

```text
src/multimodal_decision/  模型适配、协议、安全门、API 与网页网关
web/                     统一前端
scripts/                 模型下载与真实 GPU 验证
examples/                请求示例
tests/                   CPU 协议与安全边界测试
docs/                    架构、验证和贡献说明
evidence/                脱敏后的实际验证记录
setup.sh / start_all.sh / stop_all.sh
```

## 许可与贡献

本仓库自有接口代码采用 Apache-2.0；模型权重与运行时加载的官方自定义代码仍遵循各自上游条款，见 [模型许可说明](docs/model-licenses.md)。仓库不含模型权重、账户凭据或部署信息。欢迎提交 issue 与 PR，见 [CONTRIBUTING.md](CONTRIBUTING.md)。
