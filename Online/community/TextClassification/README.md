# OrangePi AIpro 文本分类应用 (MindSpore + MindSpore NLP)

本目录包含部署在香橙派 AIpro (Ascend 310B) 上的中文新闻文本分类示例。

## 组件版本
- Python 3.9
- CANN Toolkit/Kernels 8.0.0.beta1
- MindSpore 2.6.0 (Ascend)
- MindSpore NLP 0.4.1

## 功能概要
- 预训练中文 Transformer直接进行推理
- 支持本地/镜像站 (hf-mirror) 加载
- 推理脚本可独立运行（无需 notebook 重新训练）

## Notebook 文件
`text_classification_transforme.ipynb` 将包含：
1. 环境与版本检测
2. 模型选择（小体积 <4GB）
3. 推理快速验证
4. 部署与推理脚本生成

## 快速部署步骤 (概述)
1. 安装依赖：参考 notebook 第1节输出命令
2. 放置/下载模型权重到本地缓存 (首次自动)
3. 运行 notebook，确认推理成功

## 运行效果
notebook运行后输出python脚本  
在当前文件夹下打开终端，输入指令：  
```
python inference_transformer_server.py <host> <port> [model_name] [--max-len 128] [--label-file export/label_mapping.json]
```
将客户端与香橙派置于同一局域网下，通过浏览器访问设定的网址，效果图如下：  
![alt text](PNG_1.png)

详情见 Notebook 内部说明。
