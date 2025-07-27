# 中文命名实体识别（Token Classification）应用案例
本项目基于 MindSpore NLP 仓库提供的能力，使用中文 RoBERTa 模型，在香橙派 AIpro 开发板上实现了 Token Classification（命名实体识别）任务。该模型已在 CLUENER2020 数据集上微调，能够识别地址、公司、人名、职位等多类中文实体。

## 模型准备
模型名称：uer/roberta-base-finetuned-cluener2020-chinese

下载地址：https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese

说明：
（1）此模型已在中文 NER 数据集 CLUENER2020 上微调，直接推理即可
（2）模型体积约 390MB，满足 4GB 以下要求  
（3）支持本地加载，适配 MindSpore 推理

下载方式：
只要下载以下5个文件即可
- `config.json`
- `pytorch_model.bin`
- `vocab.txt`
- `tokenizer_config.json`
- `special_tokens_map.json`

分别保存到文件夹：orange-pi-mindspore/Online/18-RoBERTaNER/roberta-cluener/

## 运行方式
（1）先按照上述方法准备模型

（2）直接运行mindspore_roberta_ner.ipynb

## 作者信息
开发者姓名：张子彤

github用户名：ddsfda99

完成实习任务token classification
