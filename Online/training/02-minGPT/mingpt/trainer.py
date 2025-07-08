"""
这是一个可以应用于任何神经网络的训练简单循环样板，
所以这个文件中没有任何与GPT相关的内容
"""

import time
from collections import defaultdict
import mindspore
from mindspore import amp, ops, mint
from mindspore.dataset import GeneratorDataset, RandomSampler
from mindspore.ops import clip_by_norm
from mingpt.utils import CfgNode as CN


# 香橙派上没有ops.select方法的算子，需要手动实现后继承amp.DynamicLossScaler，对其adjust方法进行修改
def select(condition, input, other):
    return condition * input + (~condition) * other

class CustomDynamicLossScaler(amp.DynamicLossScaler):
    r"""
    一个替代ops的自定义动态损失缩放器。使用自定义Select方法进行选择。

    这个类继承了DynamicLossScaler的所有功能，但覆盖了
    方法以使用自定义选择实现。

    参数:
    scale_value (Union(float, int)): loss scale的初始值。
    scale_factor (int)：缩放因子。
    scale_window (int)：连续训练步数的最大值
        溢出增加损失规模。
    """

    def __init__(self, scale_value, scale_factor, scale_window):
        super().__init__(scale_value, scale_factor, scale_window)



    def adjust(self, grads_finite):
        """
        使用自定义的select方法而不是ops.select调整`scale_value`。

        参数:
            grads_finite (Tensor)：一个标量bool张量，表示梯度是否有限。
        """
        one = ops.ones((), self.scale_value.dtype)
        scale_mul_factor = self.scale_value * self.scale_factor

        scale_value = select(grads_finite, 
                             select(self.counter==(self.scale_window-1),
                                    select(ops.isfinite(scale_mul_factor),
                                           scale_mul_factor,self.scale_value),self.scale_value),
                                           ops.maximum(one,self.scale_value / self.scale_factor))
        ops.assign(self.scale_value, scale_value)
        counter = ((self.counter + 1) % self.scale_window) * grads_finite

        ops.assign(self.counter, counter)
        
        return True
        

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # 数据加载器参数
        C.num_workers = 4
        # 优化器参数
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # 仅应用于矩阵乘法权重
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # 分配给训练器类用于日志记录等操作的变量
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # 设置优化器
        self.optimizer = model.configure_optimizers(config)

        sampler = RandomSampler(replacement=True, num_samples=int(1e10))

        train_loader = GeneratorDataset(
            self.train_dataset,
            column_names=["x", "y"],
            sampler=sampler,
            num_parallel_workers=config.num_workers,
        )

        train_loader = train_loader.batch(config.batch_size, drop_remainder=True)

        # 香橙派训练必须使用设置O2模式的混合精度
        model = amp.auto_mixed_precision(model, 'O2')
        model.set_train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        # 动态调整损失缩放，具体信息请参考
        # https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/amp/mindspore.amp.DynamicLossScaler.html?highlight=dynamicloss#mindspore.amp.DynamicLossScaler
        loss_scaler = CustomDynamicLossScaler(scale_value=2**16, scale_factor=2, scale_window=50)
            
        def net_forward(x, y):
            logits = model(x)
            logits = logits.float()
            loss = mint.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-1)
            
            # 动态调整loss值
            return loss_scaler.scale(loss)
    
        grad_fn = mindspore.value_and_grad(net_forward, None, self.optimizer.parameters)

        while True:
            # 获取下一批数据（x，y），如果需要的话则重新初始化迭代器
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            x, y = batch
            
            loss, grads = grad_fn(x, y)
            # 把放大后的loss缩小回原始数值
            self.loss = loss_scaler.unscale(loss)
            
            # 判断是否有溢出
            is_finite = amp.all_finite(grads)
            if is_finite:
                unscaled_grads = loss_scaler.unscale(grads)
                unscaled_grads = clip_by_norm(unscaled_grads, config.grad_norm_clip) # 梯度裁剪
                self.optimizer(unscaled_grads)

            # 动态调整LossScaler数值
            loss_scaler.adjust(is_finite)

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # 终止条件
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
