# 昇思+香橙派开发板的训推全流程实践

## 昇思+昇腾开发板：软硬结合玩转DeepSeek开发实战

以DeepSeek-R1-Distill-Qwen-1.5B模型为例，基于昇思MindSpore在以香橙派为代表的昇腾开发板上完成模型的开发、微调、推理、性能提升。对应课程：[《昇思+昇腾开发板：软硬结合玩转DeepSeek开发实战》](https://www.hiascend.com/developer/courses/detail/1925362775376744449)


| 阶段 | 描述 | 文件 |
| :----- |:----- |:----- |
| 模型微调 | 基于《甄嬛传》剧本中提取出的数据集，对模型进行LoRA微调 | [deepseek-r1-distill-qwen-1.5b-lora.py](./01-DeepSeek-R1-Distill-Qwen-1.5B/deepseek-r1-distill-qwen-1.5b-lora.py)|  |
| 模型推理 | 对比微调前后效果，拉起Gradio服务，构建对话机器人 | [deepseek-r1-distill-qwen-1.5b-gradio.py](./01-DeepSeek-R1-Distill-Qwen-1.5B/deepseek-r1-distill-qwen-1.5b-gradio.py) | |
| 性能提升 | 通过JIT即时编译，提升模型的推理性能 | [deepseek-r1-distill-qwen-1.5b-jit.py](./01-DeepSeek-R1-Distill-Qwen-1.5B/deepseek-r1-distill-qwen-1.5b-jit.py) | 
| 实验手册 | 实验指导手册，手把手完成从环境搭建到最后性能调优全流程 | [昇思+昇腾开发板：软硬结合玩转DeepSeek开发实战.pdf](./01-DeepSeek-R1-Distill-Qwen-1.5B/昇思+昇腾开发板：软硬结合玩转DeepSeek开发实战.pdf) | |
| 认证考试 | 【基础】补齐LoRA微调和推理代码 </br>【进阶】补齐代码，实现模型的解码工程 | 微调：[deepseek-r1-distill-qwen-1.5b-lora-question](./01-DeepSeek-R1-Distill-Qwen-1.5B/deepseek-r1-distill-qwen-1.5b-lora.py) </br> 推理：[deepseek-r1-distill-qwen-1.5b-gradio-question](./01-DeepSeek-R1-Distill-Qwen-1.5B/exam/deepseek-r1-distill-qwen-1.5b-gradio-question.py) </br> 解码工程：[昇思+昇腾开发板：软硬结合玩转大模型实践能力认证（中级）](./01-DeepSeek-R1-Distill-Qwen-1.5B/exam/昇思+昇腾开发板：软硬结合玩转大模型实践能力认证（中级）.ipynb) | 

### Referencce
1. huanhuan数据集：https://github.com/KMnO4-zx/huanhuan-chat

## minGPT

以gpt-nano模型为例，基于昇思MindSpore在以香橙派为代表的昇腾开发板上完成minGPT的开发、训练以及推理全流程。


| 阶段 | 描述 | 文件 |
| :----- |:----- |:----- |
| 模型开发 | minGPT中model和trainer的实现细节，以及一些工具模块 | [model.py](./02-minGPT/mingpt/model.py) </br>[trainer.py](./02-minGPT/mingpt/trainer.py) </br>[utils.py](./02-minGPT/mingpt/utils.py)|  |
| 模型训推 | GPT-nano数字排序训推全流程，通过run_demo.sh拉起demo.py进行训推 | [demo.py](./02-minGPT/demo.py) </br>[run_demo.sh](./02-minGPT/run_demo.sh) |  |
| 实验手册 | 待补充 | 待补充 | |
| 认证考试 | 补齐代码，实现minGPT模型搭建以及开发板上O2混合精度训练 | minGPT模型搭建：[model-question](./02-minGPT/exam/model-question.py) </br> 开发板上O2混合精度训练：[trainer-question](./02-minGPT/exam/trainer-question.py) </br> | 

### Reference
1. https://github.com/karpathy/minGPT