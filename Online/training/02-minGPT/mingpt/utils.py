import os
import sys
import json
import random
from ast import literal_eval
import numpy as np
import mindspore

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.manual_seed(seed)
    mindspore.set_seed(seed)
    mindspore.dataset.config.set_seed(seed)
    mindspore.set_deterministic(True)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # 如果工作目录尚未存在，则创建该目录
    os.makedirs(work_dir, exist_ok=True)
    # 记录参数
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # 记录配置本身
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ 需要有一个辅助工具来支持嵌套缩进，以便进行格式化打印 """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ 返回配置的字典形式表示 """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        根据预期从命令行传入的一系列字符串（即 sys.argv[1:]）来更新配置。
        这些参数预计将以“--arg=value”的形式出现，并且参数可以使用“.”来表示嵌套的子属性。示例：
        --模型层数=10 --训练器批次大小=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval

            # 将 val 转换为一个 Python 对象
            try:
                val = literal_eval(val)
                """
                这里需要做一些解释。
                - 如果 val 只是一个字符串，那么 literal_eval 函数将会抛出一个 ValueError 错误
                - 如果 val 非字符串（比如 3、3.14、[1,2,3]、False、None 等等），那么它将会被创建出来
                """
            except ValueError:
                pass

            # 找到合适的对象来将该属性插入其中
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # 确保此属性存在
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # 覆盖属性
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
