import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, nn
from mindspore.ops import CustomOpBuilder

class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(2.0, requires_grad=True)
        self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops_online.cpp'], backend="Ascend").load()

    def construct(self, x, y):
        z = self.my_ops.mul(x, y)
        return self.my_ops.mul(z, self.p)


x = Tensor(1.0, ms.float32) * 2
y = Tensor(1.0, ms.float32) * 3
net = MyNet()
grad_op = ms.value_and_grad(net, grad_position=(0, 1), weights=net.trainable_params())
out, grads = grad_op(x, y)
print('out:', out)
print('grads[0]:', grads[0])
print('grads[1]:', grads[1])



