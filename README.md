# orange-pi-mindspore

本代码仓为基于昇思MindSpore+香橙派开发板案例仓，内包含带框架（Online，推荐）开发和离线（Offline）推理案例。

## 目录

- [orange-pi-mindspore](#orange-pi-mindspore)
  - [目录](#目录)
  - [昇思MindSpore香橙派能力介绍](#昇思mindspore香橙派能力介绍)
  - [最新动态](#最新动态)
  - [代码仓分支和版本兼容](#代码仓分支和版本兼容)
  - [案例与模型清单](#案例与模型清单)
    - [基于MindSpore开发（Online）](#基于mindspore开发online)
      - [官方案例（inference+training）](#官方案例inferencetraining)
      - [第三方应用案例(community)](#第三方应用案例community)
    - [离线推理（Offline）](#离线推理offline)
      - [官方案例（inference）](#官方案例inference)
      - [第三方应用案例(community)](#第三方应用案例community-1)
  - [学习资源](#学习资源)
  - [贡献指南](#贡献指南)
  - [问题答疑](#问题答疑)
## 昇思MindSpore香橙派能力介绍

- 开发友好：动态图易用性提升，类huggingface风格降低开发调试门槛
- 性能提升：mindspore.jit编译成图，一行代码实现推理性能提升一倍
- 全流程支持：在香橙派上支持模型训推全流程

## 最新动态

[《昇思+昇腾开发板：软硬结合玩转DeepSeek开发实战》](https://www.hiascend.com/developer/courses/detail/1925362775376744449)课程已上线，以DeepSeek蒸馏模型为例，讲解如何基于昇思MindSpore，在香橙派开发板上完成该模型的开发、微调、推理、性能提升，以及分享一些在开发板上实践的经验供大家参考。

欢迎开发者访问学习交流，如对课程有任何建议，或希望新增哪些内容的讲解，欢迎在课程评论区留下你宝贵的评论，或在本代码仓中提交`issue`。


## 代码仓分支和版本兼容

| branch | Online/Offline | CANN toolkit/kernel | MindSpore |
| :----- |:----- |:----- |:----- |
| r1.0 | Online | 8.0.0beta1 | 2.5.0 |
| r1.0 | Online | 8.0.RC3.alpha002 | 2.4.10 |
| r1.0 | Offline | 8.0.RC3.alpha002 | 2.2.14 |

## 案例与模型清单
### 基于MindSpore开发（Online）
#### 官方案例（inference+training）
| 模型 | 训练/推理 | CANN版本 | Mindspore版本 | 香橙派开发板型号 |
| :----- |:----- |:----- |:-----|:-----|
| [ResNet50](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/02-ResNet50) | 推理 | 8.1.RC1  | 2.6.0| 8T8G |
|[ViT](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/03-ViT)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[FCN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/04-FCN)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[ShuffleNet](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/05-ShuffleNet)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[SSD](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/06-SSD)| 推理 | 8.1.RC1  | 2.6.0| 8T8G |
|[RNN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/07-RNN)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[LSTM+CRF](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/08-LSTM%2BCRF)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[GAN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/09-GAN)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DCGAN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/10-DCGAN)|  推理 | 8.1.RC1  | 2.6.0| 8T8G |
|[Pix2Pix](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/11-Pix2Pix)|  推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[Diffusion](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/12-Diffusion)|  推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[ResNet50_transfer](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/13-ResNet50_transfer)|  推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[Qwen1.5-0.5b](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/14-qwen1.5-0.5b)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[TinyLlama-1.1B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/15-tinyllama)| 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DctNet](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/16-DctNet)  | 推理 | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DeepSeek-R1-Distill-Qwen-1.5B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/17-DeepSeek-R1-Distill-Qwen-1.5B)  | 推理 | 8.0.RC3.alpha002/8.0.0.beta1/8.1.RC1.beta1  | 2.4.10/2.5.0/2.6.0| 20T24G |
|[DeepSeek-R1-Distill-Qwen-1.5B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/training/01-DeepSeek-R1-Distill-Qwen-1.5B)  | 训练 | 8.0.0.beta1/8.1.RC1.beta1  | 2.5.0/2.6.0 | 20T24G |
|[DeepSeek-Janus-Pro-1B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/18-DeepSeek-Janus-Pro-1B)  | 推理 | 8.0.RC3.alpha002/8.0.0beta1 | 2.4.10/2.5.0| 20T24G |
|[MiniCPM3-4B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/19-MiniCPM3)  | 推理 | 8.0.0beta1 | 2.5.0| 20T24G |


#### 第三方应用案例(community)
| 模型 | 训练/推理 | CANN版本 | Mindspore版本 | 香橙派开发板型号 |
| :----- |:----- |:----- |:-----|:-----|
| [TokenClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TokenClassification) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [SentenceSimilarity](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/SentenceSimilarity) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [ImageToText](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/ImageToText) | 推理 | 8.0.0.beta1 | 2.6.0         | 8T16G            |
| [TextRanking](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TextRanking) | 推理 | 8.0.0.beta1 | 2.6.0         | 8T16G            |
| [FeatureExtraction](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/FeatureExtraction) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [TableQuestionAnswering](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TableQuestionAnswering) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [ImageClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/ImageClassification) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [TextClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TextClassification) | 推理 | 8.0.0.beta1  |2.6.0  |20T24G  |
| [Summarization](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/Summarization) | 推理 | 8.1.RC1 | 2.6.0 | 8T16G |
| [Translation](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/Translation) | 推理 | 8.1.RC1 | 2.6.0 | 8T16G |
| [ObjectDetection](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/ObjectDetection) | 推理 | 8.0.0.beta1   |2.6.0  |8T16G  |
| [VideoClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/VideoClassification) | 推理 | 8.0.0.beta1  | 2.6.0 |8T16G |
| [MaskGeneration](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/MaskGeneration) | 推理 | 8.1.RC1 | 2.6.0 | 8T16G |
| [DocumentQuestionAnswering](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/DocumentQuestionAnswering) | 推理 | 8.0.0.beta1 | 2.6.0         | 20T24G           |

> 注：在线案例指导请参考Online文件夹中的README文档

### 离线推理（Offline）
#### 官方案例（inference）
| 模型名 | 支持CANN版本 | 支持Mindspore版本 | 支持的香橙派开发板型号 |
|  ----  | ---- | ---- | ---- |
| [CNNCTC](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/01-CNNCTC) | 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[ResNet50](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/02-ResNet50)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[HDR](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/03-HDR)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[CycleGAN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/04-CycleGAN)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[Shufflenet](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/05-Shufflenet)|8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[FCN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/06-FCN)|8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[Pix2Pix](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/07-Pix2Pix)|8.0.RC2.alpha003  | 2.2.14| 8T16G |
|

#### 第三方应用案例(community)
| 模型名 | 支持CANN版本 | 支持Mindspore版本 | 支持的香橙派开发板型号 |
|  ----  | ---- | ---- | ---- |
[RingMoE](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/community/RingMoE-Classification)|8.0.0.beta1 | 	2.6.0 | 20T24G |
> 注：离线案例指导请参考Offline文件夹中的README文档

## 学习资源

| 阶段 | 描述 | 链接 |
| :----- |:----- |:----- |
| 镜像获取 | 香橙派官网-官方镜像 | [8T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro.html)</br>[20T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro(20T).html) |
| 环境搭建 | 昇思官网香橙派开发教程 | [香橙派开发](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/orange_pi/overview.html) | 
| 精品课程 | 《昇思+昇腾开发板：</br> 软硬结合玩转DeepSeek开发实战》课程  | [课程链接](https://www.hiascend.com/developer/courses/detail/1925362775376744449) | 
| 案例分享 | 昇腾开发板专区-案例分享 | [昇腾开发板专区](https://www.hiascend.com/developer/devboard) | 


## 贡献指南

欢迎各位开发者贡献基于昇思MindSpore+香橙派开发板的应用案例！开发者可通过向`Online/community`路径下提交`pull request`进行贡献，由工程师进行校验和合入。

案例贡献要求：

1. 保证应用案例在指定MindSpore版本要求下的香橙派环境中跑通，且输出达到预期。
2. 贡献需包含
    - **代码（必选）**：python文件或jupyter notebook文件均可，如仅单一文件建议携程jupyter notebook格式
    - **README（必选）**：需包含对版本、案例、模型、算法、如何启动运行、预期输出结果
    - **数据集（可选）**：如涉及数据集，欢迎提供数据集获取方式，数据集可开源至[魔乐社区](https://modelers.cn/)或[大模型平台](https://xihe.mindspore.cn/)
3. 在`Online`和`Online/community`路径下README文档中的`模型案例清单和版本兼容-第三方应用案例`中，新增案例信息
4. 对代码、README的详细要求，请见`Online`路径下README文档中`贡献指南`

## 问题答疑

如在基于昇思MindSpore+香橙派开发板开发过程中遇到任何问题，欢迎在本代码仓中提交`issue`，定期会有工程师进行答疑。
