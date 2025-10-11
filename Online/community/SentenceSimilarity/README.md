# 中文句子相似度计算（Sentence Similarity）应用案例
本项目基于 MindSpore NLP 仓库提供的能力，使用 bert-base-chinese 模型，在香橙派 AIpro 开发板上实现了Sentence Similarity（中文句子相似度计算）任务。该应用通过提取句向量并计算余弦相似度，衡量两个句子在语义上的接近程度。

## 模型准备
模型名称：bert-base-chinese

下载地址：https://huggingface.co/google-bert/bert-base-chinese

说明：本模型体积约为 390MB，满足4GB 以下要求

## 推理方式
使用 MindSpore NLP 提供的 BertTokenizer 和 BertModel，将输入句子进行分词、编码，获取 `[CLS]` 向量表示句子语义，使用 NumPy 计算两个向量间的余弦相似度。

