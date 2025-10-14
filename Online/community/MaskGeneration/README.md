# 掩码生成（Mask Generation）
本项目在 Ascend 架构的香橙派 AIpro 开发板上，实现一个掩码生成应用  
本案例基于 MindSpore NLP + CANN 在香橙派 AIpro 开发板上运行，具体版本如下：
| 组件       | 版本          |
| ---------- | ------------- |
| MindSpore  | 2.6.0         |
| MindSpore NLP    | 0.4.1         |
| CANN       | 8.1.RC1   |
| Python     | 3.9           |
| 开发板型号 | Orange Pi AIpro（Ascend 310，8T16G） |
## 模型准备
SegFormer 模型在 640x640 分辨率的 ADE20k 数据集上进行了微调。该模型由 Xie 等人在论文 SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers 中介绍  

下载地址：https://modelscope.cn/models/nv-community/segformer-b5-finetuned-ade-640-640  

模型体积约 339MB，满足 4GB 以下要求