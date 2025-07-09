import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.mint as mint
from mindspore.common.initializer import initializer, Normal, Zero, One
from mingpt.utils import CfgNode as CN


# 基于mindspore.ops手动实现softmax(香橙派不支持nn.Softmax反向传播)
def manual_softmax(x, dim=-1):
    exp_x = ms.ops.exp(x - ms.ops.max(x, axis=dim, keepdims=True)[0])
    return exp_x / ms.ops.sum(exp_x, dim=dim, keepdim=True)


class NewGELU(nn.Cell):
    def construct(self, x):
        x = (0.5 * x * (1.0 + mint.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mint.pow(x, 3.0)))))
        return x


class CausalSelfAttention(nn.Cell):
    """
    一个带有末尾投影的纯多头掩码自注意力层。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 批处理所有头的key, query, value和projections
        self.c_attn = nn.Dense(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Dense(config.n_embd, config.n_embd)
        # 正则化
        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)
        # 因果掩码，用于确保注意力仅作用于输入序列的左侧部分
        self.bias = mint.tril(
            mint.ones((config.block_size, config.block_size), dtype=ms.int32)
        ).view(1, 1, config.block_size, config.block_size)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def construct(self, x):
        B, T, C = (
            x.shape
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # 计算批次中所有头的query, key, values，并将头向前移动以使其成为批次维度
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 因果自注意力；自注意力：(B, nh, T, hs) 与 (B, nh, hs,T) 相乘 -> (B, nh, T, T)
        # >>>>>>> 填空1 完成attention计算 <<<<<<< 
        att = _____
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = manual_softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class Block(nn.Cell):
    """Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm((config.n_embd,), epsilon=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm((config.n_embd,), epsilon=1e-5)
        self.mlp = nn.CellDict(
            {
                "c_fc": nn.Dense(config.n_embd, 4 * config.n_embd),
                "c_proj": nn.Dense(4 * config.n_embd, config.n_embd),
                "act": NewGELU(),
                "dropout": nn.Dropout(p=config.resid_pdrop),
            }
        )

        m = self.mlp
        self.mlpc = lambda x: m.dropout(
            m.c_proj(m.act(m.c_fc(x)))
        )  # MLP construct 函数

    def construct(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpc(self.ln_2(x))

        return x


class GPT(nn.Cell):
    """GPT模型实现"""

    @staticmethod
    def get_default_config():
        C = CN()
        # 在配置文件中，必须给出model_type或者(n_layer, n_head, n_embd)三元组
        C.model_type = "gpt"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # 这些选项必须在下游任务获取
        C.vocab_size = None
        C.block_size = None
        # dropout的超惨设置
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1

        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )
        assert type_given ^ params_given  # 其中有一个（异或关系）
        if type_given:
            # 将“模型类型”转换为“详细配置”
            config.merge_from_dict(
                {
                    # 名称遵循 HuggingFace 的命名规范
                    # GPT-1
                    "openai-gpt":  dict(n_layer=12, n_head=12, n_embd=768),   # 117M params
                    # GPT-2 configs
                    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
                    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                    # Gophers
                    "gopher-44m":  dict(n_layer=8, n_head=16, n_embd=512),
                    "gpt-mini":    dict(n_layer=6, n_head=6, n_embd=192),
                    "gpt-micro":   dict(n_layer=4, n_head=4, n_embd=128),
                    # >>>>>>> 填空2 添加一个名为gpt-nano的模型，它有3个block、3个头、词嵌入的维度为48 <<<<<<<
                    _____
                }[config.model_type]
            )

        self.transformer = nn.CellDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(p=config.embd_pdrop),
            }
        )

        self.h = nn.CellList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm((config.n_embd,), epsilon=1e-5)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)

        # 初始化所有权重，并对残差投影应用一种特殊的缩放初始化方法，这与 GPT-2 论文中的做法一致
        self.apply(self._init_weights)
        for pn, p in self.parameters_and_names():
            if pn.endswith("c_proj.weight"):
                p.set_data(
                    initializer(
                        Normal(sigma=0.02 / math.sqrt(2 * config.n_layer)),
                        p.shape,
                        p.dtype,
                    )
                )

        # 打印告参数的数量（请注意，我们不将语言模型头部中的解码器参数计入其中）
        n_transformer_params = sum(p.numel() for p in self.transformer.get_parameters())
        n_h_params = sum(p.numel() for p in self.h.get_parameters())
        n_params = n_transformer_params + n_h_params
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Dense):
            module.weight.set_data(
                initializer(
                    Normal(mean=0.0, sigma=0.02),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
            if module.bias is not None:
                module.bias.set_data(
                    initializer(Zero(), module.bias.shape, module.bias.dtype)
                )
        elif isinstance(module, nn.Embedding):
            module.embedding_table.set_data(
                initializer(Normal(mean=0.0, sigma=0.02), module.embedding_table.shape)
            )
        elif isinstance(module, nn.LayerNorm):
            module.beta.set_data(
                initializer(Zero(), module.beta.shape, module.beta.dtype)
            )
            module.gamma.set_data(
                initializer(One(), module.gamma.shape, module.gamma.dtype)
            )

    def configure_optimizers(self, train_config):
        """
        这段函数其实只是在做一件非常简单的事情：
        将模型的所有参数分为两个类别：那些会因正则化而经历权重衰减的参数，以及那些不会经历这种衰减的参数（包括偏置项以及层归一化/嵌入权重）
        然后返回了Mindspore的优化器对象
        """

        # 将所有参数分为两类：一类是会经历权重衰减正则化处理的参数，另一类是不会经历这一处理的参数
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Dense,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.cells_and_names():
            for pn, p in m.parameters_and_names():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                # 由于 named_modules 和 named_parameters 是递归的
                # 因此我们会多次看到相同的张量 p 。但采用这种方式我们能够知道任何张量 p 所归属的父模块是哪个
                if pn.endswith("bias"):
                    # 所有bias不会经历权重衰减正则化处理
                    no_decay.add(fpn)
                elif pn.endswith("beta"):
                    # nn.Dense中bias命名为beta
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重将进行权重衰减处理
                    decay.add(fpn)
                elif pn.endswith("gamma") and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重不会进行权重衰减处理
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("embedding_table") and isinstance(
                    m, blacklist_weight_modules
                ):
                    no_decay.add(fpn)

        # 确认已经处理了每一个参数
        param_dict = {pn: p for pn, p in self.parameters_and_names()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # 创建 Mindspore 优化器对象
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = nn.AdamWeightDecay(
            optim_groups,
            learning_rate=train_config.learning_rate,
            beta1=train_config.betas[0],
            beta2=train_config.betas[1],
            eps=1e-8,
            weight_decay=0.01,
        )
        return optimizer

    def construct(self, idx):
        b, t = idx.shape
        assert (t <= self.block_size), f"Cannot construct sequence of length {t}, block size is only {self.block_size}"
        pos = mint.arange(0, t, dtype=ms.int32).unsqueeze(0)  # (1, t)

        # GPT模型的构建
        tok_emb = self.transformer.wte(idx)  # token embeddings的shape： (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings的shape： (1, t, n_embd)

        # >>>>>>> 填空3 完成模型输入x，x应为token编码与位置编码之和，并通过一个dropout层，防止过拟合 <<<<<<<
        x = _____

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        给定一个索引序列 idx（形状为 (b,t) 的长整型张量），重复执行一个条件序列 max_new_tokens 次，每次将预测结果反馈给模型
        需要确定模型处于验证模式！！！
        """
        for _ in range(max_new_tokens):
            # 如果序列的长度增长过长，我们就必须在达到block_size时对其进行截断。
            idx_cond = (
                idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size :]
            )
            # 构建模型以获取序列中该索引对应的logits
            logits = self(idx_cond)
            # 在最后一步提取logits，并根据所需的温度进行缩放
            logits = logits[:, -1, :] / temperature
            # （可选地）将对数输出值裁剪为仅前 k 个选项的值
            if top_k is not None:
                v, _ = mint.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # 应用 softmax 函数将对数输出转换为（归一化的）概率值
            probs = mint.nn.functional.softmax(logits, dim=-1)
            # 要么从该分布中抽取一个样本，要么选取最有可能的元素
            # >>>>>>> 填空4 完成上述代码，抽取样本参考mint.multinomial接口，选取最有可能的元素参考mint.topk接口 <<<<<<<
            _____
            
            # 将采样的索引添加到运行序列中，并继续进行
            idx = mint.cat((idx, idx_next.to(idx.dtype)), dim=1)

        return idx
