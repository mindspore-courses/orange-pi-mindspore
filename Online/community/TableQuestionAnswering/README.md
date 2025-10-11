# 表格问答（Table Question Answering）
本项目在 Ascend 架构的香橙派 AIpro 开发板上，实现一个表格问答应用，支持两类问题：

可计算问题：SUM / COUNT / MAX / MIN 等（如“贡献者总数多少”“Stars 最多的是谁？”）

查值问题：基于自然语言的抽取式问答（如“Datasets 的编程语言是什么？”、“哪个仓库使用 Rust？”）
## 环境依赖说明
本案例基于 MindSpore + MindSpore NLP + CANN 在香橙派 AIpro 开发板上运行，具体版本如下：
| 组件       | 版本          |
| ---------- | ------------- |
| MindSpore  | 2.6.0         |
| MindSpore NLP    | 0.4.1         |
| CANN       | 8.0.0.beta1   |
| Python     | 3.9           |
| 开发板型号 | Orange Pi AIpro（Ascend 310，20T24G） |
## 模型准备
模型名称：distilbert-base-uncased-distilled-squad

下载地址：https://huggingface.co/distilbert-base-uncased-distilled-squad
## 方案要点
1. 表格 → 文本序列化：将每一行转写为简洁句子，对多值单元格（如“Rust, Python and NodeJS”）自动展开为多句等价表达，增强可抽取性。
2. 抽取式 QA 管线：使用`distilbert-base-uncased-distilled-squad`模型对序列化文本进行问答。
3. 轻量聚合：对合计/计数/最大/最小类问题直接在表上计算。
4. 分块推理：长表被按行切成多个上下文块，逐块问答并按分数择优，避免被模型输入长度截断。


