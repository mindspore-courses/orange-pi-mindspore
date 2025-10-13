# Depth Estimation

[【开源实习】针对任务类型Depth Estimation，开发可在香橙派AIpro开发板运行的应用](https://gitee.com/mindspore/community/issues/ICJ4YU)
任务编号：#ICJ4YU  

基于`MindSpore`框架和`Intel/dpt-large`模型实现的Depth Estimation应用  

### 介绍
    深度估计是计算机视觉中的核心任务之一，旨在从单张或多张二维图像中推断出场景中物体与相机的相对距离，最终输出与输入图像尺寸匹配的 “深度图”—— 以像素级灰度 / 彩色映射直观反映不同区域的远近关系。Intel/dpt-large是 Intel 团队基于 Transformer 架构设计的高性能单目深度估计模型，在室内外多场景中均具备优异泛化能力，其核心优势在于 “Transformer 全局注意力 + CNN 局部特征提取” 的混合架构：首先通过预训练骨干网络如 ViT-B/ViT-L提取多尺度局部特征，保留高分辨率下的物体轮廓、纹理等细节信息，为近距物体深度精准度奠定基础；随后借助 Transformer 编码器的全局自注意力机制，捕捉图像长距离依赖关系，解决传统 CNN 在全局场景一致性上的不足；最后通过层级化解码模块将 “局部细节特征” 与 “全局语义特征” 逐步融合，结合深度值连续分布特性优化预测结果，输出与输入图像尺寸完全对齐的像素级深度图。该模型可适配香橙派 AIpro 等边缘设备，在降低内存开销的同时满足实时性需求。

### 环境准备

    开发者拿到香橙派开发板后，首先需要进行硬件资源确认，镜像烧录及CANN和MindSpore版本的升级，才可运行该案例，具体如下：

    开发板：香橙派Aipro或其他同硬件开发板  
    开发板镜像: Ubuntu镜像  
    `CANN Toolkit/Kernels：	8.1.RC1`  
    `MindSpore: 2.6.0` 
    `MindSpore NLP: 0.4.1`  
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

pillow >= 10.2.0

numpy == 1.22.4

requests == 2.32.5

```
### 快速使用

    用户在准备好上述环境之后，逐步运行Depth-estimation.ipynb文件即可，（第一次运行时需下载模型权重）使用时可将图片路径替换为要进行深度估计的图片路径，逐步运行后模型会返回深度图结果。

### 预期输出
    展示对应输入图像推理得到的深度图数据。



