import mindspore as ms
import numpy as np
from mindspore import ops, nn, Tensor
from mindspore.ops import DataType, CustomRegOp
ms.runtime.launch_blocking() # Orange_Pi

class AddCustomNet(nn.Cell):
    def __init__(self, func, out_shape, out_dtype):
        super(AddCustomNet, self).__init__()
        reg_info = CustomRegOp("aclnnAddCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_add = ops.Custom(func=func, out_shape=out_shape, out_dtype=out_dtype, func_type="aot", bprop=None,
                                     reg_info=reg_info)

    def construct(self, x, y):
        res = self.custom_add(x, y)
        return res

ms.set_context(jit_config={"jit_level": "O0"})
ms.set_device("Ascend")
x = np.ones([8, 2048]).astype(np.float16)
y = np.ones([8, 2048]).astype(np.float16)

# 通过lambda实现infer shape函数
net = AddCustomNet("aclnnAddCustom", lambda x, _: x, lambda x, _: x)

print(net.custom_add(Tensor(x), Tensor(y)))