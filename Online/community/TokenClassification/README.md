# 中文命名实体识别（Token Classification）应用案例
本项目基于 MindSpore NLP 仓库提供的能力，使用中文 RoBERTa 模型，在香橙派 AIpro 开发板上实现了 Token Classification（命名实体识别）任务。该模型已在 CLUENER2020 数据集上微调，能够识别地址、公司、人名、职位等多类中文实体。

## 模型准备
模型名称：uer/roberta-base-finetuned-cluener2020-chinese

下载地址：https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese

说明：
（1）此模型已在中文 NER 数据集 CLUENER2020 上微调，直接推理即可
（2）模型体积约 390MB，满足 4GB 以下要求  

