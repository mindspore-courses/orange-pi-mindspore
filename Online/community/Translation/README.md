# 文本摘要（Summarization）
本项目在 Ascend 架构的香橙派 AIpro 开发板上，实现一个机器翻译应用
本案例基于 MindSpore NLP + CANN 在香橙派 AIpro 开发板上运行，具体版本如下：
| 组件       | 版本          |
| ---------- | ------------- |
| MindSpore  | 2.6.0         |
| MindSpore NLP   | 0.4.1         |
| CANN       | 8.1.RC1   |
| Python     | 3.9           |
| 开发板型号 | Orange Pi AIpro（Ascend 310，8T16G） |
## 模型准备
T5 Small：0.1B的T5语言模型  

下载地址：https://huggingface.co/google-t5/t5-small

模型体积约 242 MB，满足 4GB 以下要求