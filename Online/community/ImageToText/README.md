# Image-to-Text

基于MindSpore框架和Blip模型实现的图像生成文本应用

## 介绍

基于香橙派AIPro边缘计算硬件的低功耗、高算力特性，结合MindSpore开源框架的灵活部署能力，搭载BLIP图生文模型实现图像内容的精准语义转述。

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

### requirements.txt
```
Python == 3.9

MindSpore == 2.6.0

mindnlp == 0.4.1

opencv-python  == 4.12.0.88

pillow == 11.3.0

sympy  == 1.14.0
```
## 快速使用

用户在准备好上述环境之后，直接运行image-to-text.ipynb文件即可，代码中模型加载部分会自动从huggingface镜像中下载模型。
运行程序后输入图像路径，模型会返回识别结果；
输入stop终止程序

## 预期输出

``````
请输入图像路径
用户:  /opt/image2text/shixi/test.png
Blip: a tree with pink flowers against a blue sky
``````



