# Object Detection

[【开源实习】针对任务类型Object Detection，开发可在香橙派AIpro开发板运行的应用](https://gitee.com/mindspore/community/issues/ICJ5UE)
任务编号：#ICJ5UE  

基于`MindSpore`框架和`facebook/detr-resnet-50`模型实现的Video Classification应用  

### 介绍
目标检测（Object Detection） 是计算机视觉中的核心任务之一，旨在在图像或视频中同时 定位和识别目标物体。与图像分类不同，它不仅输出类别标签，还需要给出目标在图像中的边界框（bounding box）。  
facebook/detr-resnet-50 是 Meta AI提出的 DETR（DEtection TRansformer）模型的经典版本，采用 ResNet-50 作为卷积主干网络提取图像特征，再通过 Transformer 编码器-解码器结构实现端到端的目标检测。与传统检测器（如 Faster R-CNN、YOLO）依赖手工设计的候选框不同，DETR 直接把目标检测建模为序列预测问题，利用匈牙利匹配（Hungarian matching）在预测框和真实框之间建立一一对应关系，从而实现统一的目标检测框架。


### 环境准备

开发者拿到香橙派开发板后，首先需要进行硬件资源确认，镜像烧录及CANN和MindSpore版本的升级，才可运行该案例，具体如下：

开发板：香橙派Aipro或其他同硬件开发板  
开发板镜像: Ubuntu镜像  
`CANN Toolkit/Kernels：8.0.0.beta1`  
`MindSpore: 2.6.0`  
`MindSpore NLP: 0.4.1`  
`Python: 3.9`

#### 镜像烧录

运行该案例需要烧录香橙派官网ubuntu镜像，烧录流程参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--镜像烧录](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html) 章节。

#### CANN升级

CANN升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--CANN升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

#### MindSpore升级

MindSpore升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--MindSpore升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

### requirements
```
Python == 3.9

MindSpore == 2.6.0

mindnlp == 0.4.1

opencv-python  == 4.12.0.88

pillow == 11.3.0

sympy  == 1.14.0

matplotlib == 3.9.4
```
## 快速使用

用户在准备好上述环境之后，逐步运行object_detection.ipynb文件即可，代码中模型加载部分会自动从huggingface镜像中下载模型。
使用时需将图片路径替换为要识别的图片路径，逐步运行后模型会返回检测结果。

## 预期输出
展示带有绘制检测结果的图像



