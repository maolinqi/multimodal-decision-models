# 上游模型与代码

本仓库不重新许可上游模型。下载和使用前查看各模型官方页面中的最新许可、使用条款和归属要求：

- [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/README.md)：本次固定官方模型卡声明 Apache-2.0。
- [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)：本次固定快照声明 Apache-2.0，仍应查看官方许可。
- [Gemma 3n E4B](https://huggingface.co/google/gemma-3n-E4B-it) / [E2B](https://huggingface.co/google/gemma-3n-E2B-it)：Gemma 条款与访问许可。
- [MiniCPM-V-4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5)：权重及自定义代码各自的许可。
- [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) / [14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B)：视觉与语言组件及模型卡的许可要求。

MiniCPM 与 InternVL 需要 `trust_remote_code=True`。本项目加载本地下载快照中的自定义代码；请审阅该代码并固定已验证的上游 revision。下载脚本支持 `--revision`。本仓库不打包这些官方自定义文件。
