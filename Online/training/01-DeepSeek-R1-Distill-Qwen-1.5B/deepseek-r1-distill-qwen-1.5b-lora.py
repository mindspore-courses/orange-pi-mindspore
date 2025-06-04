import os

import mindspore
import mindnlp
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.engine import TrainingArguments, Trainer
from mindnlp.dataset import load_dataset
from mindnlp.transformers import GenerationConfig
from mindnlp.peft import LoraConfig, TaskType, get_peft_model, PeftModel

from mindnlp.engine.utils import PREFIX_CHECKPOINT_DIR
from mindnlp.configs import SAFE_WEIGHTS_NAME
from mindnlp.engine.callbacks import TrainerCallback, TrainerState, TrainerControl

from openmind_hub import om_hub_download

from mindspore._c_expression import disable_multi_thread
disable_multi_thread()  # 禁用多线程，提升微调性能

# 开启同步，在出现报错，定位问题时开启
# mindspore.set_context(pynative_synchronize=True)

# 从魔乐社区下载数据集
om_hub_download(
    repo_id="MindSpore-Lab/huanhuan",
    repo_type="dataset",
    filename="huanhuan.json",
    local_dir="./",
)

# 加载数据集
dataset = load_dataset(path="json", data_files="./huanhuan.json")


# 实例化tokenizer
tokenizer = AutoTokenizer.from_pretrained("MindSpore-Lab/DeepSeek-R1-Distill-Qwen-1.5B-FP16", mirror="modelers", use_fast=False, ms_type=mindspore.float16)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


# 定义数据处理逻辑

def process_func(instruction, input, output):
    MAX_SEQ_LENGTH = 64  # 最长序列长度
    input_ids, attention_mask, labels = [], [], []
    # 首先生成user和assistant的对话模板
    # User: instruction + input
    # Assistant: output
    formatted_instruction = tokenizer(f"User: {instruction}{input}\n\n", add_special_tokens=False)
    formatted_response = tokenizer(f"Assistant: {output}", add_special_tokens=False)
    # 最后添加 eos token，在deepseek-r1-distill-qwen的词表中， eos_token 和 pad_token 对应同一个token
    # User: instruction + input \n\n Assistant: output + eos_token
    input_ids = formatted_instruction["input_ids"] + formatted_response["input_ids"] + [tokenizer.pad_token_id]
    # 注意，我们在微调时仅考虑Assistant部分回答的内容，所以User部分提问的内容对应的标签为-100
    attention_mask = formatted_instruction["attention_mask"] + formatted_response["attention_mask"] + [1]
    labels = [-100] * len(formatted_instruction["input_ids"]) + formatted_response["input_ids"] + [tokenizer.pad_token_id]

    # 如超过最大长度，则进行截断
    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]

    # 如不足最大长度，则进行填充
    padding_length = MAX_SEQ_LENGTH - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    attention_mask = attention_mask + [0] * padding_length  # 填充的 attention_mask 为 0
    labels = labels + [-100] * padding_length  # 填充的 label 为 -100
    
    return input_ids, attention_mask, labels


formatted_dataset = dataset.map(operations=[process_func], 
                                input_columns=['instruction', 'input', 'output'], 
                                output_columns=["input_ids", "attention_mask", "labels"])

# 查看预处理后的数据
for input_ids, attention_mask, labels in formatted_dataset.create_tuple_iterator():
    print(tokenizer.decode(input_ids))
    break

# 为节约演示时间，将数据集裁剪
truncated_dataset = formatted_dataset.take(3)


# 实例化base model
model_id = "MindSpore-Lab/DeepSeek-R1-Distill-Qwen-1.5B-FP16"

base_model = AutoModelForCausalLM.from_pretrained(model_id, mirror="modelers", ms_dtype=mindspore.float16)
base_model.generation_config = GenerationConfig.from_pretrained(model_id, mirror="modelers")

base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id

# LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)

# 实例化LoRA模型
model = get_peft_model(base_model, config)
# 获取模型参与训练的参数，发现仅占总参数量的0.5%
model.trainable_params()

# Callback函数，随save_steps定义的步数保存LoRA adapter权重
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ): 
        # LoRA adapter权重保存路径
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        # 保存LoRA adapter权重
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path, safe_serialization=True)

        # 移除额外保存的base model的model.safetensors，节约空间
        base_model_path = os.path.join(checkpoint_folder, SAFE_WEIGHTS_NAME)
        os.remove(base_model_path) if os.path.exists(base_model_path) else None

        return control

# 训练超参
args = TrainingArguments(
    output_dir="./output/DeepSeek-R1-Distill-Qwen-1.5B",  # 输出保存路径
    per_device_train_batch_size=1,  # batch size
    logging_steps=1,  # 每多少步记录一次训练日志
    num_train_epochs=1,  # epoch数
    save_steps=3,  # 每多少步保存一次权重
    learning_rate=1e-4,  # 学习率
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=truncated_dataset,
    callbacks=[SavePeftModelCallback],
)

# 启动微调
trainer.train()