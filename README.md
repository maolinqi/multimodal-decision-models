# Multimodal Decision Models｜多模态决策模型

**公开占位仓库。当前仅发布项目说明；模型适配代码与可复现验证将陆续上传。**

本项目将多模态模型改造成候选决策接口：输入问题、测量状态、RGB 图像或按时间排序的视频抽帧，输出候选集合内的相对概率与选择结果。计划统一支持 `choice`、`noul`、`score` 和四轴动作建议 `velocity4`。

## 模型基座

| 模型 | 精确基座仓库 | 当前进度 |
|---|---|---|
| Gemma-3n-E4B | [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) | 已在现有部署中完成决策接口及原生首步 logits 一致性验证；代码待上传 |
| Gemma-3n-E2B | [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) | 已在现有部署中完成决策接口及原生首步 logits 一致性验证；代码待上传 |
| MiniCPM-V-4.5 | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | 决策接口适配与验证待完成 |
| InternVL3.5-8B | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 决策接口适配与验证待完成 |
| InternVL3.5-14B | [OpenGVLab/InternVL3_5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) | 决策接口适配与验证待完成 |

其中 **Gemma 决策模型的基座明确为 Gemma-3n-E4B-it／Gemma-3n-E2B-it（指令版本）**，不是 Gemma 3，也不是从零训练的新模型。

## 改造方式与边界

- 沿用既有 Qwen 多模态决策接口的思路：读取最后位置的候选 token logits，在候选集合内做 Softmax；不解析长篇自由文本。
- Gemma 适配保留每次前向的新临时缓存，以正确执行其层间 KV 共享。
- 当前 Gemma 改造属于推理接口适配，**未对基座权重进行训练或 LoRA 微调**。
- 候选概率是相对偏好，未经正确率校准。接口验证不等于决策准确率或泛化能力证明。
- 四轴输出只提供动作建议；没有验证真实飞行或导航收益。
- 少量合成探针仍存在错误：Gemma E2B 有位置判断错误，E4B 有算术候选选择错误。后续发布会保留这些结果。

## 计划发布内容

1. 各模型独立的视觉输入与候选 logits 适配器。
2. 统一 API 与可切换模型的前端。
3. 项目独立运行环境及初始化、启动、停止入口。
4. 真实模型接口验证记录与明确的能力限制。

仓库不发布模型权重、账号凭据或私人服务器配置。模型权重需从对应发布者获取，并遵循各自的许可条款。

2026-10-09：先公开发布占位说明。当前仓库尚不可用于运行模型。
