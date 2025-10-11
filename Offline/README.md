# 基于昇思MindSpore+香橙派开发板的应用实践案例

本路径下包含基于昇思MindSpore在香橙派AIpro开发板上开发的案例，共分为两类：

- inference：推理案例
- community：第三方贡献的香橙派开发板应用案例

欢迎广大开发者进行学习、交流和贡献，如对案例有任何疑问或建议，可以提交`issue`，会有工程师进行定期解答，如希望贡献案例，可提交`pull request`，将案例贡献在community路径下。

## 目录
- [基于昇思MindSpore+香橙派开发板的应用实践案例](#基于昇思mindspore香橙派开发板的应用实践案例)
  - [目录](#目录)
  - [模型案例清单和版本兼容](#模型案例清单和版本兼容)
    - [推理案例(inference)](#推理案例inference)
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

| 模型名 | 支持CANN版本 | 支持Mindspore版本 | 支持的香橙派开发板型号 |
|  ----  | ---- | ---- | ---- |
| [CNNCTC](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/01-CNNCTC) | 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[ResNet50](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/02-ResNet50)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[HDR](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/03-HDR)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[CycleGAN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/04-CycleGAN)| 8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[Shufflenet](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/05-Shufflenet)|8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[FCN](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/06-FCN)|8.0.RC2.alpha003  | 2.2.14| 8T16G |
|[Pix2Pix](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/07-Pix2Pix)|8.0.RC2.alpha003  | 2.2.14| 8T16G |

### 第三方应用案例(community)

| 案例名称 | CANN版本 | Mindspore版本 | 香橙派开发板型号 |
| :----- |:----- |:----- |:-----|
|[RingMoE](https://github.com/mindspore-courses/orange-pi-mindspore/tree/master/Offline/community/RingMoE-Classification)|8.0.0.beta1 | 	2.6.0 | 20T24G |

## 学习资源

| 阶段 | 描述 | 链接 |
| :----- |:----- |:----- |
| 镜像获取 | 香橙派官网-官方镜像 | [8T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro.html)</br>[20T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro(20T).html) |
| 环境搭建 | 昇思官网香橙派开发教程 | [香橙派开发](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/orange_pi/overview.html) | 
| 精品课程 | 《昇思+昇腾开发板：</br> 软硬结合玩转DeepSeek开发实战》课程  | [课程链接](https://www.hiascend.com/developer/courses/detail/1925362775376744449) | 
| 案例分享 | 昇腾开发板专区-案例分享 | [昇腾开发板专区](https://www.hiascend.com/developer/devboard) |

## 贡献指南

欢迎各位开发者贡献基于昇思MindSpore+香橙派开发板的应用案例！开发者可通过向`Offline/community`路径下提交`pull request`进行贡献，由工程师进行校验和合入。

### 贡献内容

请在`Offline/community`路径下单独创建文件夹，文件夹名称以案例名称命名，需为英文且尽量简洁。文件夹中需包含：

1. **代码（必选）**：python文件或jupyter notebook文件均可，如仅单一文件建议写成jupyter notebook格式
2. **README（必选）**：需包含对版本、案例、模型、算法、如何启动运行、预期输出结果
3. **数据集（可选）**：如涉及数据集，欢迎提供数据集获取方式，数据集可开源至[魔乐社区](https://modelers.cn/)或[大模型平台](https://xihe.mindspore.cn/)
4. 同时，请在Offline路径的README文件中，模型案例清单和版本兼容-第三方应用案例(community)表格中，补充案例名称、CANN版本、MindSpore版本以及香橙派开发板型号

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
