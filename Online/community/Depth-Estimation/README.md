# Depth Estimation

[【开源实习】针对任务类型Depth Estimation，开发可在香橙派AIpro开发板运行的应用](https://gitee.com/mindspore/community/issues/ICJ4YU)
任务编号：#ICJ4YU  

基于`MindSpore`框架和`ibaiGorordo/lap-depth-nyu`模型实现的Depth Estimation应用  

### 介绍
    深度估计（Depth Estimation）是计算机视觉中的核心任务之一，旨在从单张或多张二维图像中，推断出场景中物体与相机的相对距离（即深度信息），最终输出与输入图像尺寸匹配的 “深度图”—— 以像素级灰度 / 彩色映射直观反映不同区域的远近关系（如近距区域呈浅色调、远距区域呈深色调）。ibaiGorordo/lap-depth-nyu是基于 Laplacian Pyramid（拉普拉斯金字塔）结构设计的经典深度估计模型，专门针对 NYU Depth V2 等室内场景数据集优化。该模型通过 “多尺度特征融合” 思路，将低分辨率的全局语义特征（用于捕捉远距场景结构）与高分辨率的局部细节特征（用于优化近距物体边缘）结合，再通过拉普拉斯金字塔的层级解码过程，逐步恢复像素级的精准深度信息。与传统单尺度深度估计模型（如直接通过 CNN 输出固定尺寸深度图）相比，Lap-Depth 能有效平衡 “全局场景一致性” 与 “局部细节精度”，在室内复杂场景（如家具遮挡、多物体堆叠）中表现更优，尤其适合香橙派 AIpro 这类边缘设备的部署需求 —— 模型轻量化设计可在保证精度的同时，降低计算与内存开销。

### 环境准备

    开发者拿到香橙派开发板后，首先需要进行硬件资源确认，镜像烧录及CANN和MindSpore版本的升级，才可运行该案例，具体如下：

    开发板：香橙派Aipro或其他同硬件开发板  
    开发板镜像: Ubuntu镜像  
    `CANN Toolkit/Kernels：	8.1.RC1`  
    `MindSpore: 2.6.0` 
    `MindCV: 0.3.0`  
    `MindNLP: 0.4.1`  
    `Python: 3.9`

### 镜像烧录

    运行该案例需要烧录香橙派官网ubuntu镜像，烧录流程参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--镜像烧录](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html) 章节。

### CANN升级

    CANN升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--CANN升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

### MindSpore升级

    MindSpore升级参考[昇思MindSpore官网--香橙派开发专区--环境搭建指南--MindSpore升级](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/orange_pi/environment_setup.html)章节。

### requirements
```
Python == 3.9

MindSpore == 2.6.0

mindnlp == 0.4.1

mindcv == 0.3.0

pillow == 11.3.0

matplotlib == 3.9.4
```
### 快速使用

    用户在准备好上述环境之后，逐步运行Depth-estimation.ipynb文件即可，代码中使用mindcv中的vgg19作为特征提取器（第一次运行时需下载模型权重），且用mindspore实现了模型的decoder部分，模型权重文件已经训练并上传，开发者可直接使用，如需在自定义数据集上训练可参考depth-estimation.ipynb文件中最后注释的训练代码。
使用时需将图片路径替换为要识别的图片路径，逐步运行后模型会返回深度图结果。

### 预期输出
    展示对应输入图像及推理得到的深度图数据。



