# 决策边界

所有模型共用 `contracts.py` 与 `observation.py`。状态仅接收可测量信息，拒绝特权地图、非有限数值、无效向量和倒序图像。RGB 经各模型官方视觉编码与语言融合后，只读取最后位置的语言模型词表 logits。

A–Z 必须在每个模型 tokenizer 中各为一个 token，否则初始化失败。softmax 仅覆盖本次候选 token；`label_mass` 给出这些 token 在整个词表内的概率质量。候选集变化会改变相对概率。

Gemma 3n 在 Transformers 4.57.1 依赖跨层共享 KV 的缓存，因此每次前向创建新缓存，使用 `use_cache=True`，请求间不复用缓存。关闭缓存会改变计算。语言头只计算最后位置，使用 `logits_to_keep=1`。

MiniCPM 与 InternVL 使用官方视觉 embedding 融合，随后调用原生语言模型。验证脚本对照官方 `generate(max_new_tokens=1)` 的候选 logits，不以自由文本解析代替决策。

四轴动作采用 body_FLU_yaw_aligned：xyz 七档为 -1.8、-0.9、-0.45、0、0.45、0.9、1.8 m/s，yaw 七档为 -0.9、-0.45、-0.15、0、0.15、0.45、0.9 rad/s。归一化比例为 2、2、2、pi/3。三帧执行窗口与时间步长 0.1 秒由外部控制器处理。

`safety.guard` 是独立检查函数，需要同一动作的传感器联合路径、最新定位与刹车距离；API 只输出建议，不自动执行该函数或连接飞控。它没有经过真实飞行验证。
