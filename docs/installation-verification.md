# 七模型部署验证

2026-10-10，在新的项目目录创建三个独立环境，执行初始化、服务启动、七模型图片请求、模型切换与停止流程。复用服务器已有官方权重和依赖下载缓存；本次没有重复验证七套权重的网络下载及新账号授权。下载步骤需按 README 提前获得上游访问权限。

验证设备为 NVIDIA A800-SXM4-80GB，驱动 550.163.01；Python 3.12.3，PyTorch 2.8.0+cu128。基础五模型环境使用 Transformers 4.57.1，Qwen3.5 与 Gemma 4 的独立环境使用 Transformers 5.19.0。

- `setup_all.sh` 完成三个项目环境的初始化；中断后重新执行也能继续安装。
- 启动后，基础后端、Qwen3.5 后端、Gemma 4 后端和网页网关全部响应健康检查。
- 13 项协议及模型切换检查全部通过。
- 七个模型依次接收同一张温室图片与文字条件，返回有效候选 ID 和归一化分布，视觉 token 数大于零。每次请求后仅所选模型驻留。
- `stop_all.sh` 停止本项目四个服务并清理进程记录；测试显卡回到测试前的显存占用。

七个通过的模型为 Qwen3.5-2B-Base、Gemma-4-26B-A4B-it、Gemma-3n-E2B-it、Gemma-3n-E4B-it、MiniCPM-V-4.5、InternVL3.5-8B 和 InternVL3.5-14B。[逐模型接口结果](../evidence/seven-model-installation-check.json)

这是部署功能检查，检查输入能进入原生视觉路径、输出格式有效、切换可释放其他驻留模型。它不衡量七模型的任务准确率；首轮请求包含权重加载时间，不能用作稳态延迟比较。

按照 README 下载全部模型并启动后，运行：

```bash
.venv/bin/python scripts/verify_installation.py
```

结果写入 `run/installation-check.json`。该检查依次加载七个模型；单卡需要满足最大基座的显存门槛。
