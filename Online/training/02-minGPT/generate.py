import mindspore
from mindspore import amp
from mindspore import ops
from download import download
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
import time
set_seed(3407)

# 设置同步，解除注释可以开启
# mindspore.set_context(pynative_synchronize=True)

# 模型设置
model_type = 'gpt2'
assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large'}

# 模型权重下载
url = "https://modelers.cn/coderepo/web/v1/file/MindSpore-Lab/minGPT/main/media/" + model_type + ".ckpt"
path = model_type + ".ckpt"
download(url=url, path=path, replace=False)
print(f"{model_type} 模型权重下载成功")

# 模型实例化
config = GPT.get_default_config()
config.model_type = model_type
config.vocab_size = 50257  # OpenAI 的模型词汇表大小
config.block_size = 1024   # OpenAI 的模型块大小
model = GPT(config)

# 模型权重加载
param_dict = mindspore.load_checkpoint(path)
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
# param_not_load为空，说明所有模型参数都完成加载
if len(param_not_load) == 0:
    print(f"{model_type} 模型权重加载成功")
else:
    print(f"[WARNING] 参数{param_not_load}未成功加载，建议排查后再进行实验")

# 设置O2模式混合精度
model = amp.auto_mixed_precision(model, 'O2')._backbone

# 开启推理模式
model.set_train(False)

def generate(prompt='', num_samples=10, steps=20, do_sample=False):      
    # 将输入提示标记化为整数输入序列
    tokenizer = BPETokenizer()
    if prompt == '':
        # 生成无条件样本……
        # 手动创建一个仅包含特殊标记的张量
        # 类似于 OpenAI 的代码在这里 https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py 所做的操作
        x = mindspore.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=mindspore.int32)
    else:
        x = tokenizer(prompt)
    
    # 一次性处理所有所需的样本数量，因此需要扩展“批次”维度。
    # 请注意，由于香橙派算子支持不够，请设置num_samples = 1，推荐在其他推理卡上扩大该参数
    x_expand = ops.broadcast_to(x, (num_samples, x.shape[1]))

    # 将该模型重复执行“步骤”次以获取样本，每次执行作为一个批次进行。
    print("start generating")
    start_time = time.time()
    y = model.generate(x_expand, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(num_samples):
        out = tokenizer.decode(y[i].squeeze())
        print('-'*80)
        print(out)
        
    end_time = time.time()
    print("time cost:", end_time - start_time)
           
generate(prompt='Andrej Karpathy, the', num_samples=1, steps=20)

