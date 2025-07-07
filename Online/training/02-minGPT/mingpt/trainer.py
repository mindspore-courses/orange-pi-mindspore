"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import mindspore
# import torch
# from torch.utils.data.dataloader import DataLoader
from mindspore.dataset import GeneratorDataset, RandomSampler
from mindspore.ops import clip_by_norm
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        # C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # variables that will be assigned to trainer class later for logging and etc
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

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        sampler = RandomSampler(replacement=True, num_samples=int(1e10)) # torch中num_samples可以拓展样本数，但mindspore中num_samples必须小于等于数据集大小

        train_loader = GeneratorDataset(
            self.train_dataset,
            column_names=["x", "y"],
            # sampler=sampler,
            shuffle=False, # mindspore中shuffle=False时，会报错与RandomSampler冲突
            num_parallel_workers=config.num_workers,
        )
        train_loader = train_loader.batch(config.batch_size)

        model.set_train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        # print('iter len: ', sum(1 for _ in data_iter)) # 157
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            # batch = [t.to(self.device) for t in batch]
            # print(batch)
            x, y = batch

            from mindspore import amp
            loss_scaler = amp.StaticLossScaler(scale_value=2**10)
            
            def net_forward(x, y):
                logits, loss = model(x, y)
                return loss #loss_scaler.scale(loss)  # scale the loss for mixed precision training
                # return loss_scaler.scale(loss)
            
            # grad_fn = mindspore.value_and_grad(net_forward, None,  model.trainable_params())
            grad_fn = mindspore.value_and_grad(net_forward, None, self.optimizer.parameters)
            self.loss, grad = grad_fn(x, y)
            # unscaled_loss = loss_scaler.unscale(loss)
            # unscaled_grads = loss_scaler.unscale(grad)
            # unscaled_grads = clip_by_norm(unscaled_grads, config.grad_norm_clip) # 梯度裁剪
            # self.optimizer(unscaled_grads)
            # print('loss: ', unscaled_loss.asnumpy())
            grads = clip_by_norm(grad, config.grad_norm_clip) # 梯度裁剪
            # grads= clip_by_norm(unscaled_grads, config.grad_norm_clip)
            # print('grads: ', grad)
            self.optimizer(grads)
            # loss = net_forward(x, y)
            # print(loss)
            # print(type(loss))

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            # print(self.iter_num)
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
