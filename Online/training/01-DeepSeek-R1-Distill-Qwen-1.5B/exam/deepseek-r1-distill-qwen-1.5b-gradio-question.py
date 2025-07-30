import gradio as gr
import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread

mindspore.set_context(pynative_synchronize=True)

# Load the tokenizer and model from MindNLP.
# Note: To use MindNLP, you need to install it first. Ensure you are using the master branch of MindNLP,
# which supports downloading the MindNLP-specific weights from Modelers.

# >>>>> 题目：实例化tokenizer和模型，镜像地址为modelers，模型ID为"MindSpore-Lab/DeepSeek-R1-Distill-Qwen-1.5B-FP16"，数据类型为float16
# >>>>> 补全实例化tokenizer和模型的代码
tokenizer = ________
model = ________


system_prompt = "You are a helpful and friendly chatbot"

def build_input_from_chat_history(chat_history, msg: str):
    messages = [{'role': 'system', 'content': system_prompt}]
    for user_msg, ai_msg in chat_history:
        # >>>>> 题目：补齐对话历史记录的输入格式，即字典中“content”对应的内容
        messages.append({'role': 'user', 'content': ________})
        messages.append({'role': 'assistant', 'content': ________})
    messages.append({'role': 'user', 'content': msg})
    return messages

# Function to generate model predictions.
def predict(message, history):
    history_transformer_format = history + [[message, ""]]

    # Formatting the input for the model.
    messages = build_input_from_chat_history(history, message)
    input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="ms",
            tokenize=True
        )
    streamer = TextIteratorStreamer(tokenizer, timeout=300, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        # >>>>> 题目：设置最大生成长度
        max_new_tokens=________,
        do_sample=True,
        # >>>>> 题目：设置top-p采样参数
        top_p=________,
        temperature=0.1,
        num_beams=1,
        # >>>>> 题目：设置重复惩罚系数
        repetition_penalty=________
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


# Setting up the Gradio chat interface.
gr.ChatInterface(predict,
                 title="DeepSeek-R1-Distill-Qwen-1.5B",
                 description="问几个问题",
                 examples=['你是谁？', '你能做什么？']
                 ).launch()  # Launching the web interface.
