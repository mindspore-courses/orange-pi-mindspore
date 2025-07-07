# %% [markdown]
# A cute little demo showing the simplest usage of minGPT. Configured to run fine on Macbook Air in like a minute.

# %%
# !pip install git+https://github.com/mindspore-lab/mindnlp.git@0.4

# %%
import mindspore
from mindspore import load_checkpoint, load_param_into_net
mindspore.set_context(pynative_synchronize=True)
from mindspore import mint, ops
from mindspore.dataset import GeneratorDataset
from mingpt.utils import set_seed
set_seed(3407)

# %%
import pickle
import random
import numpy as np
import os

class SortDataset():
    """ 
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1
    
    def __getitem__(self, idx):
        while True:
            # 生成随机整数列表
            inp_list = [random.randint(0, self.num_digits - 1) for _ in range(self.length)]
            # 一半时间内，尝试增加重复数字较多的示例
            if random.random() < 0.5:
                if len(set(inp_list)) > self.length // 2:
                    continue
            # 根据 inp_list 的哈希值确定其属于训练集还是测试集
            h = hash(pickle.dumps(inp_list))
            inp_split = 'test' if h % 4 == 0 else 'train'
            if inp_split == self.split:
                break

        # 对输入序列进行排序
        sol_list = sorted(inp_list)

        # 拼接输入和输出序列
        cat_np = np.concatenate((inp_list, sol_list))

        # 生成偏移序列
        x_np = cat_np[:-1].copy()
        y_np = cat_np[1:].copy()

        # 掩蔽输入位置的损失
        y_np[:self.length-1] = -100

        # 生成 Transformer 的输入 x 和输出 y
        x = mindspore.tensor(x_np, dtype=mindspore.int32)
        y = mindspore.tensor(y_np, dtype=mindspore.int32)
        # x = mindspore.tensor(x_np, dtype=mindspore.int64)
        # y = mindspore.tensor(y_np, dtype=mindspore.int64)

        return x, y

    
train_dataset = SortDataset('train')
test_dataset = SortDataset('test')

x, y = train_dataset[0]
print(x.dtype, y.dtype)
for a, b in zip(x,y):
    print(int(a),int(b))


# create a GPT instance
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# 加载转换后的权重
param_dict = load_checkpoint("ms_nano.ckpt")

# 加载到 MindSpore 模型
param_not_load, ckpt_not_load = load_param_into_net(model, param_dict)
print('param_not_load:', param_not_load)
print('ckpt_not_load:', ckpt_not_load)


# %%
# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
train_config.max_iters = 300
train_config.num_workers = 1
trainer = Trainer(train_config, model, train_dataset)

# %%
def batch_end_callback(trainer):
    if trainer.iter_num < 10:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

# %%
# now let's perform some evaluation
# model.eval()
model.set_train(False)
print('end trainning! - predict start!')

# %%
def eval_split(trainer, split, max_batches):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    n = train_dataset.length # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    # loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    loader = GeneratorDataset(dataset, column_names=['x', 'y'], shuffle=False)
    loader = loader.batch(100, drop_remainder=False)
    for b, (x, y) in enumerate(loader):
        # x = x.to(trainer.device)
        # y = y.to(trainer.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(inp, n, do_sample=False) # using greedy argmax, not sampling
        sol_candidate = cat[:, n:] # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1) # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.shape[0]):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3: # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = mindspore.tensor(results, dtype=mindspore.float16)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

# run a lot of examples from both train and test through the model and verify the output correctness
# with torch.no_grad():
train_score = eval_split(trainer, 'train', max_batches=50)
test_score  = eval_split(trainer, 'test',  max_batches=50)


# %%
# let's run a random given sequence through the model as well
n = train_dataset.length # naugy direct access shrug
inp = mindspore.tensor([[0, 0, 2, 1, 0, 1]], dtype=mindspore.int32)#.to(trainer.device)
assert inp[0].nelement() == n
# with torch.no_grad():
cat = model.generate(inp, n, do_sample=False)
sol = ops.sort(inp[0])[0]
sol_candidate = cat[:, n:]
print('input sequence  :', inp.tolist())
print('predicted sorted:', sol_candidate.tolist())
print('gt sort         :', sol.tolist())
print('matches         :', bool((sol == sol_candidate).all()))


    



