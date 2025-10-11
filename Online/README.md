# 基于昇思MindSpore+香橙派开发板的应用实践案例

本路径下包含基于昇思MindSpore在香橙派AIpro开发板上开发的案例，共分为三类：

- inference：推理案例
- training：训推案例
- community：第三方贡献的香橙派开发板应用案例

欢迎广大开发者进行学习、交流和贡献，如对案例有任何疑问或建议，可以提交`issue`，会有工程师进行定期解答，如希望贡献案例，可提交`pull request`，将案例贡献在community路径下。

## 目录
- [基于昇思MindSpore+香橙派开发板的应用实践案例](#基于昇思mindspore香橙派开发板的应用实践案例)
  - [目录](#目录)
  - [模型案例清单和版本兼容](#模型案例清单和版本兼容)
    - [推理案例(inference)](#推理案例inference)
    - [训推案例(training)](#训推案例training)
    - [第三方应用案例(community)](#第三方应用案例community)
  - [学习资源](#学习资源)
  - [贡献指南](#贡献指南)
    - [贡献内容](#贡献内容)
    - [代码格式要求](#代码格式要求)
    - [README格式要求](#readme格式要求)
    - [案例自验](#案例自验)
    - [提交PR](#提交pr)
  - [问题答疑](#问题答疑)

## 模型案例清单和版本兼容

### 推理案例(inference)

| 模型名                                                       | CANN版本                                   | Mindspore版本      | 香橙派开发板型号 |
| :----------------------------------------------------------- | :----------------------------------------- | :----------------- | :--------------- |
| [ResNet50](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/02-ResNet50) | 8.1.RC1                                    | 2.6.0              | 8T8G             |
| [ViT](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/03-ViT) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [FCN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/04-FCN) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [ShuffleNet](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/05-ShuffleNet) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [SSD](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/06-SSD) | 8.1.RC1                                    | 2.6.0              | 8T8G             |
| [RNN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/07-RNN) | 8.1.RC1                                    | 2.6.0              | 8T8G             |
| [LSTM+CRF](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/08-LSTM%2BCRF) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [GAN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/09-GAN) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [DCGAN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/10-DCGAN) | 8.1.RC1                                    | 2.6.0              | 8T8G             |
| [Pix2Pix](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/11-Pix2Pix) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [Diffusion](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/12-Diffusion) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [ResNet50_transfer](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/13-ResNet50_transfer) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [Qwen1.5-0.5b](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/14-qwen1.5-0.5b) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [TinyLlama-1.1B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/15-tinyllama) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [DctNet](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/16-DctNet) | 8.0.RC3.alpha002                           | 2.4.10             | 8T16G            |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/17-DeepSeek-R1-Distill-Qwen-1.5B) | 8.0.RC3.alpha002/8.0.0.beta1/8.1.RC1.beta1 | 2.4.10/2.5.0/2.6.0 | 20T24G           |
| [DeepSeek-Janus-Pro-1B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/18-DeepSeek-Janus-Pro-1B) | 8.0.RC3.alpha002/8.0.0beta1                | 2.4.10/2.5.0       | 20T24G           |
| [MiniCPM3-4B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/inference/19-MiniCPM3) | 8.0.0beta1                                 | 2.5.0              | 20T24G           |

### 训推案例(training)

| 模型名                                                       | CANN版本                  | Mindspore版本 | 香橙派开发板型号 |
| :----------------------------------------------------------- | :------------------------ | :------------ | :--------------- |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/training/01-DeepSeek-R1-Distill-Qwen-1.5B) | 8.0.0.beta1/8.1.RC1.beta1 | 2.5.0/2.6.0   | 20T24G           |
| [minGPT](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/training/02-minGPT) | 8.1.RC1.beta1             | 2.6.0         | 20T24G           |
| [BERT](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/training/03-BERT) | 8.0.0.beta1               | 2.5.0         | 20T24G           |

### 第三方应用案例(community)


| 案例名称                                                     | CANN版本    | Mindspore版本 | 香橙派开发板型号 |
| :----------------------------------------------------------- | :---------- | :------------ | :--------------- |
| [TokenClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TokenClassification) | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [SentenceSimilarity](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/SentenceSimilarity) | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [ImageToText](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/ImageToText) | 8.0.0.beta1 | 2.6.0         | 8T16G            |
| [TextRanking](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TextRanking) | 8.0.0.beta1 | 2.6.0         | 8T16G            |
| [FeatureExtraction](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/FeatureExtraction) | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [TableQuestionAnswering](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TableQuestionAnswering) | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [ImageClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/ImageClassification) | 8.0.0.beta1 | 2.6.0         | 20T24G           |
| [TextClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/TextClassification) | 8.0.0.beta1  |2.6.0  |20T24G  |
| [Summarization](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/Summarization) | 8.1.RC1 | 2.6.0 | 8T16G |
| [Translation](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/Translation) | 8.1.RC1 | 2.6.0 | 8T16G |
| [ObjectDetection](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/ObjectDetection) | 8.0.0.beta1   |2.6.0  |8T16G  |
| [VideoClassification](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/VideoClassification) | 8.0.0.beta1  | 2.6.0 |8T16G |
| [MaskGeneration](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/MaskGeneration) | 8.1.RC1 | 2.6.0 | 8T16G |
| [DocumentQuestionAnswering](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Online/community/DocumentQuestionAnswering) | 8.0.0.beta1 | 2.6.0         | 20T24G           |



## 学习资源

| 阶段     | 描述                                                        | 链接                                                         |
| :------- | :---------------------------------------------------------- | :----------------------------------------------------------- |
| 镜像获取 | 香橙派官网-官方镜像                                         | [8T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro.html)</br>[20T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro(20T).html) |
| 环境搭建 | 昇思官网香橙派开发教程                                      | [香橙派开发](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/orange_pi/overview.html) |
| 精品课程 | 《昇思+昇腾开发板：</br> 软硬结合玩转DeepSeek开发实战》课程 | [课程链接](https://www.hiascend.com/developer/courses/detail/1925362775376744449) |
| 案例分享 | 昇腾开发板专区-案例分享                                     | [昇腾开发板专区](https://www.hiascend.com/developer/devboard) |

## 贡献指南

欢迎各位开发者贡献基于昇思MindSpore+香橙派开发板的应用案例！开发者可通过向`Online/community`路径下提交`pull request`进行贡献，由工程师进行校验和合入。

### 贡献内容

请在`Online/community`路径下单独创建文件夹，文件夹名称以案例名称命名，需为英文且尽量简洁。文件夹中需包含：

1. **代码（必选）**：python文件或jupyter notebook文件均可，如仅单一文件建议写成jupyter notebook格式
2. **README（必选）**：需包含对版本、案例、模型、算法、如何启动运行、预期输出结果
3. **数据集（可选）**：如涉及数据集，欢迎提供数据集获取方式，数据集可开源至[魔乐社区](https://modelers.cn/)或[大模型平台](https://xihe.mindspore.cn/)
4. 同时，请在Online路径的README文件中，模型案例清单和版本兼容-第三方应用案例(community)表格中，补充案例名称、CANN版本、MindSpore版本以及香橙派开发板型号

### 代码格式要求

- 建议写成jupyter notebook格式，并保留每个cell的输出结果
- 代码中需包含对模型、算法、数据集的详细说明，便于他人理解

### README格式要求

- 使用Markdown编写，层级清晰
- 包含案例名称、案例简介、所需依赖、版本、如何启动运行、预期输出结果等

### 案例自验

请开发者在提交PR前进行自验，保证应用案例在指定MindSpore版本要求下的香橙派环境中跑通，且输出达到预期。自验过程中请保留运行日志或截图，在提交PR时一并上传提供。

### 提交PR

提交PR时，除代码、README修改以外，请额外在评论区补充：
1. 案例开发使用的CANN、MindSpore和相关套件版本，以便于工程师进行快速验收合入
2. 自验通过保存的运行日志和截图
3. （可选）如涉及开源实习任务，请补充任务issue链接，以便于快速关联实习任务并进行闭环
4. （可选）开发过程中对MindSpore的建议，包括但不限于文档、教程、框架易用性、性能等等，一经采纳将作为评选昇思MindSpore优秀开发者的重要考核指标


## 问题答疑

如在基于昇思MindSpore+香橙派开发板开发过程中遇到任何问题，欢迎在本代码仓中提交`issue`，定期会有工程师进行答疑。
