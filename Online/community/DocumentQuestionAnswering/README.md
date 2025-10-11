# Document Question Answering

基于MindSpore框架和Qwen1.5-0.5b模型实现的RAG文档问答系统

## 介绍

基于香橙派AIPro，利用大语言模型，解析和处理PDF 文档，构建一个智能问答系统。

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
sympy  == 1.14.0
jieba == 0.42.1
tokenizers == 0.21.4
```
## 快速使用

建议下载qwen1.5-0.5b模型至本地路径（如/home/HwHiAiUser）：

```
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat
```

同时下载paraphrase-multilingual-MiniLM-L12-v2模型至本地：

```
git clone https://www.modelscope.cn/Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2.git
```

在ipynb文件中根据模型路径修改对应step中的模型路径，然后逐步运行即可。最后query="什么是MindSpore？具体介绍一下"可以根据用户自己意愿修改query内容，体验不同问题的答案结果

## 预期输出

模型根据文档内容回答关于Mindspore的相关问题，减少幻觉产生。

