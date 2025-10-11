# Feature Extraction 
## 1. 任务简介
本应用演示了如何在香橙派 AIpro 开发板上运行一个文本特征抽取（Feature Extraction）任务。
选用 BAAI/bge-small-zh-v1.5 作为基座模型，可以将任意文本编码为 512 维数值向量，可直接用于语义检索、聚类、相似度计算等下游任务。

## 2. 环境要求
硬件：Orange Pi AIpro (20T24G)

操作系统：Ubuntu 镜像

CANN 版本：8.0.0.beta1

Python：3.9

MindSpore：2.6.0（Ascend 版本）

MindSpore NLP：0.4.1

依赖库：numpy, scikit-learn

## 3. 模型信息
模型名称：BAAI/bge-small-zh-v1.5

下载地址：https://huggingface.co/BAAI/bge-small-zh-v1.5

参数量：约 24M

模型文件大小：< 200 MB

用途：中文及中英混合文本向量化



