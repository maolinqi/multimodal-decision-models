# 网页展示：精选正确示例

截图来自实际网页和模型接口返回，没有预填概率。这里精选答对的请求，用于展示交互效果。点击网页示例按钮载入输入和该截图使用的模型，你可以继续切换模型，再点击“计算概率并选择”运行所选模型；再次运行的结果可能不同。

单图按输入区宽度完整显示，多图最多两列；点击图片打开大图，Esc 或关闭按钮退出。手机上多图使用单列。

## 公开国考图形推理

使用 [华图公开汇编](https://gs.huatu.com/2024/0808/1771187.html) 中的原题图片，输入只含题图、问题和 A–D 候选，没有参考答案或解析。下面展示三个答对的请求；[完整测试记录](civil-reasoning-results.md) 单独保存全部结果。

### 旋转规律 · InternVL3.5-14B

2023国考市地级第73题（网友回忆）：选择 **A**，与来源参考答案一致。[接口记录](../evidence/console-demos/civil-internvl-2023-73.json)

![旋转题：InternVL3.5-14B 选择 A](images/civil-internvl-2023-73.png)

### 多元素规律 · InternVL3.5-14B

2022国考市地级第74题（网友回忆）：选择 **A**，与来源参考答案一致。[接口记录](../evidence/console-demos/civil-internvl-2022-74.json)

![多元素题：InternVL3.5-14B 选择 A](images/civil-internvl-2022-74.png)

### 复合规律 · MiniCPM-V-4.5

2019国考市地级第73题：选择 **D**，与来源参考答案一致。[接口记录](../evidence/console-demos/civil-2019-73.json)

![复合规律题：MiniCPM-V-4.5 选择 D](images/civil-2019-73.png)

## 合成视觉与状态示例

红球追踪：InternVL3.5-14B 选择 B（右侧杯子），相对概率约70.5%。四帧按时间顺序输入，正确参考由场景生成过程确定。

网页文字陷阱：同一模型选择 B（预约），相对概率约100%。这只是一个受控示例，不证明一般抗提示注入能力。

太空温室：同一模型选择 A（浇水），相对概率约98.8%。图中土壤含水率12%，文字阈值20%；二者共同定义应触发的条件。

[红球记录](../evidence/console-demos/red-ball.json) · [网页记录](../evidence/console-demos/web-trap.json) · [温室记录](../evidence/console-demos/space-greenhouse.json)

## 复现

运行安装章节后，网页可直接点击示例并提交；也可在项目根目录调用：

```bash
.venv/bin/python scripts/run_console_demos.py --model minicpm-v45
.venv/bin/python scripts/run_console_demos.py --model internvl35-14b --out evidence/console-demos-rerun-internvl
```

输入与哈希位于 `web/demos/*.json`，原图保留原始字节。程序只读取问题、状态、选项与图片字段，不读取参考答案。参考答案在 [汇总记录](../evidence/console-demos/summary.json) 中，便于独立核查。素材权属见 [NOTICE](../web/demos/NOTICE.md)。
