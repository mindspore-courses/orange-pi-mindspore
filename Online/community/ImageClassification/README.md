# Image Classification

基于MindSpore框架和AltCLIP模型实现的图像分类

## 介绍

基于香橙派AIPro 24G 20T，利用AltCLIP，实现图像分类。

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

### 核心库版本
```
Python == 3.9
MindSpore == 2.6.0
mindnlp == 0.4.1
```
## 快速使用

在ipynb文件中逐步运行即可，可以修改图片url参数来对不同的图片进行分类。

## 预期输出

输出图片对应的类别以及置信度等信息。

