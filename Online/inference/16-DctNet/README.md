# DCT-net-MindSpore

基于MindSpore框架实现DCT-Net风格迁移网络："Domain-Calibrated Translation for Portrait Stylization".

## 介绍
DCT-Net是一种新颖的图像风格迁移模型，专为少样本人像风格迁移设计。该模型结构在给定风格样本（100个）的情况下，能够生成高质量的风格迁移结果，具备合成高保真内容的先进能力，并且在处理复杂场景（如遮挡和配饰）时表现出强大的泛化能力。

此外，它通过一个优雅的评估网络实现整图迁移，该网络是通过部分观察训练而成。基于少样本学习的风格迁移具有挑战性，因为学习到的模型容易在目标领域过拟合，原因是训练样本数量少导致的偏置分布。本文旨在通过采用“先校准，再迁移”的关键理念，结合局部聚焦迁移的增强全局结构，来应对这一挑战。

具体来说，所提出的 DCT-Net 包含三个模块：内容适配器，借用源照片的强大先验来校准目标样本的内容分布；几何扩展模块，利用仿射变换释放空间语义约束；以及纹理迁移模块，利用经过校准分布生成的样本来学习细粒度转换。实验结果表明，所提出的头部风格化方法在SOTA风格迁移方面优于现有的先进技术，并且在具有自适应变形的全图像迁移上也表现出良好的效果。

## 权重文件

MindSpore版ckpt权重文件获取地址：https://modelers.cn/models/zhaoyu/DctNet/tree/main

## Requirements
- MindSpore 2.4.0
- Python 3.9
- opencv-python == 4.8.0.74
- opencv-contrib-python == 4.8.0.74
- moviepy == 1.0.3

## 使用清华镜像加快下载速度
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用方法

```bash
# 运行项目
python inference.py --device_type CPU --camera yes --speed_first no

python inference.py --device_type Ascend --camera no --speed_first no --input_path ./images/gdg.png --output_path ./images/output.png 
python inference.py --device_type CPU --camera no --speed_first no --input_path ./images/input.mp4 --output_path ./images/output.mp4 
```
