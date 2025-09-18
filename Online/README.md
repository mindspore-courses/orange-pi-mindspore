# 基于昇思MindSpore+香橙派开发板的应用实践案例

本路径下包含基于昇思MindSpore在香橙派AIpro开发板上开发的案例，共分为两类：

- inference：推理案例
- training：训推案例

欢迎广大开发者进行学习、交流和贡献，如对案例有任何疑问或建议，可以提交`issue`，会有工程师进行定期解答。

## 目录
- [基于昇思MindSpore+香橙派开发板的应用实践案例](#基于昇思mindspore香橙派开发板的应用实践案例)
  - [目录](#目录)
  - [模型案例清单和版本兼容](#模型案例清单和版本兼容)
    - [推理案例(inference)](#推理案例inference)
    - [训推案例(training)](#训推案例training)
  - [学习资源](#学习资源)
  - [问题答疑](#问题答疑)

## 模型案例清单和版本兼容

### 推理案例(inference)

| 模型名 | CANN版本 | Mindspore版本 | 香橙派开发板型号 |
| :----- |:----- |:----- |:-----|
| [ResNet50](./inference/02-ResNet50) | 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[ViT](./inference/03-ViT)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[FCN](./inference/04-FCN)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[ShuffleNet](./inference/05-ShuffleNet)| 8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[SSD](./inference/06-SSD)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[RNN](./inference/07-RNN)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[LSTM+CRF](./inference/08-LSTM%2BCRF)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[GAN](./inference/09-GAN)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DCGAN](./inference/10-DCGAN)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[Pix2Pix](./inference/11-Pix2Pix)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[Diffusion](./inference/12-Diffusion)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[ResNet50_transfer](./inference/13-ResNet50_transfer)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[Qwen1.5-0.5b](./inference/14-qwen1.5-0.5b)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[TinyLlama-1.1B](./inference/15-tinyllama)|8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DctNet](./inference/16-DctNet)  |8.0.RC3.alpha002  | 2.4.10| 8T16G |
|[DeepSeek-R1-Distill-Qwen-1.5B](./inference/17-DeepSeek-R1-Distill-Qwen-1.5B) | 8.0.RC3.alpha002/8.0.0.beta1  | 2.4.10/2.5.0| 20T24G |
|[DeepSeek-Janus-Pro-1B](./inference/18-DeepSeek-Janus-Pro-1B) | 8.0.RC3.alpha002/8.0.0beta1 | 2.4.10/2.5.0| 20T24G |
|[MiniCPM3-4B](./inference/19-MiniCPM3) | 8.0.0beta1 | 2.5.0| 20T24G |

### 训推案例(training)

| 模型名 | CANN版本 | Mindspore版本 | 香橙派开发板型号 |
| :----- |:----- |:----- |:-----|
| [DeepSeek-R1-Distill-Qwen-1.5B](./training/01-DeepSeek-R1-Distill-Qwen-1.5B) | 8.0.0.beta1  | 2.5.0 | 20T24G |
|[BERT](./training/02-BERT)| 8.0.0.beta1  | 2.5.0 | 20T24G |

## 学习资源

| 阶段 | 描述 | 链接 |
| :----- |:----- |:----- |
| 镜像获取 | 香橙派官网-官方镜像 | [8T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro.html)</br>[20T](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro(20T).html) |
| 环境搭建 | 昇思官网香橙派开发教程 | [香橙派开发](https://www.mindspore.cn/docs/zh-CN/r2.5.0/orange_pi/index.html) | 
| 精品课程 | 《昇思+昇腾开发板：</br> 软硬结合玩转DeepSeek开发实战》课程  | [课程链接](https://www.hiascend.com/developer/courses/detail/1925362775376744449) | 
| 案例分享 | 昇腾开发板专区-案例分享 | [昇腾开发板专区](https://www.hiascend.com/developer/devboard) |

## 问题答疑

如在基于昇思MindSpore+香橙派开发板开发过程中遇到任何问题，欢迎在本代码仓中提交`issue`，定期会有工程师进行答疑。

