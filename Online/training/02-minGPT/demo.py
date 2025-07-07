import mindspore
from mindspore import ops
from mindspore.dataset import GeneratorDataset
import pickle
import random
import numpy as np
from mingpt.utils import set_seed
set_seed(3407)

# 是否开启同步，测试使用，使用请取消注释
# mindspore.set_context(pynative_synchronize=True)

class SortDataset():
    """ 
    排序问题”的数据集。例如，对于问题长度为 6 的情况：
    输入：0 0 2 1 0 1 -> 输出：0 0 0 1 1 2
    这将被作为输入传递给transformer，并以以下形式进行拼接：
    输入： 0 0 2 1 0 1 0 0 0 1 1
    输出：I I I I I 0 0 0 1 1 2
    其中 I 表示“掩码”，因为转换器正在读取输入序列。
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # 输入到transformer中的序列的长度，包含连接后的输入和输出，设置为-1，因为transformer是从最后一个输入元素开始进行预测的
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
        y_np[:self.length-1] = -1
        # 生成 Transformer 的输入 x 和输出 y
        x = mindspore.tensor(x_np, dtype=mindspore.int32)
        y = mindspore.tensor(y_np, dtype=mindspore.int32)

        return x, y


train_dataset = SortDataset('train')
test_dataset = SortDataset('test')

x, y = train_dataset[0]
print(x.dtype, y.dtype)
for a, b in zip(x,y):
    print(int(a),int(b))

# 实例化GPT模型
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# 实例化trainer并开始训练
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4
train_config.max_iters = 2000
train_config.num_workers = 1
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

# 开始验证
model.set_train(False)

def eval_split(split, max_batches):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    n = train_dataset.length
    results = []
    mistakes_printed_already = 0
    loader = GeneratorDataset(dataset, column_names=['x', 'y'], shuffle=False)
    loader = loader.batch(100, drop_remainder=False)
    for b, (x, y) in enumerate(loader):
        inp = x[:, :n]
        sol = y[:, -n:]
        # 让模型对序列的其余部分进行采样
        cat = model.generate(inp, n, do_sample=False) # 采用贪心最大值选择法，而非抽样法
        sol_candidate = cat[:, n:] # 分离出已填入的序列
        # 将预测的序列与真实序列进行比较
        correct = (sol == sol_candidate).all(1)
        for i in range(x.shape[0]):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3: # 最多只打印出 3 处错误以助于理解
                mistakes_printed_already += 1
                print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = mindspore.tensor(results, dtype=mindspore.float32)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))

    return rt.sum()

# 对模型进行大量来自训练集和测试集的数据处理，并验证输出结果的正确性。
train_score = eval_split('train', max_batches=50)
test_score  = eval_split('test',  max_batches=50)

# 随机生成一条序列让模型进行推理
n = train_dataset.length
inp = mindspore.tensor([[0, 0, 2, 1, 0, 1]], dtype=mindspore.int32)
assert inp[0].nelement() == n
cat = model.generate(inp, n, do_sample=False)
sol = ops.sort(inp[0])[0]
sol_candidate = cat[:, n:]
print('input sequence  :', inp.tolist())
print('predicted sorted:', sol_candidate.tolist())
print('gt sort         :', sol.tolist())
print('matches         :', bool((sol == sol_candidate).all()))