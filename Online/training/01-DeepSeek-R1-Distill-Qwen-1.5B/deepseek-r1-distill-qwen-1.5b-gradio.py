import gradio as gr
import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread
from mindnlp.peft import PeftModel

# activate synchronization in pynative mode
# mindspore.set_context(pynative_synchronize=True)

# Load the tokenizer and model from MindNLP.
# Note: To use MindNLP, you need to install it first. Ensure you are using the master branch of MindNLP,
# which supports downloading the MindNLP-specific weights from Modelers.
tokenizer = AutoTokenizer.from_pretrained("MindSpore-Lab/DeepSeek-R1-Distill-Qwen-1.5B-FP16", mirror="modelers", ms_dtype=mindspore.float16)
model = AutoModelForCausalLM.from_pretrained("MindSpore-Lab/DeepSeek-R1-Distill-Qwen-1.5B-FP16", mirror="modelers", ms_dtype=mindspore.float16)
# model = PeftModel.from_pretrained(model, "./output/DeepSeek-R1-Distill-Qwen-1.5B/adapter_model_for_demo/") # adapter_model path

system_prompt = "You are a helpful and friendly chatbot"

def build_input_from_chat_history(chat_history, msg: str):
    messages = [{'role': 'system', 'content': system_prompt}]
    for user_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': ai_msg})
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
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        temperature=0.1,
        num_beams=1,
        repetition_penalty=1.2
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
