"""
BPE 是“字节对编码器”的缩写。它将任意的 UTF-8 字符串转换为一系列整数，其中每个整数代表常见字符的小块组合。
此实现基于 OpenAI 的 GPT2 编码器.py 文件：https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import os
import json
import regex as re
import requests
import mindspore as ms

# -----------------------------------------------------------------------------

def bytes_to_unicode():
    """
    OpenAI 将每一个可能的字节（实际上是一个 0 到 255 之间的整数）映射为一个能直观呈现其含义的 Unicode 字符。
    有些字节因其不会造成任何问题而保留了其原有外观。这些字节的列表定义在 bs 中。例如：
    chr(33) 返回“！”，因此在返回的字典中，我们只需有 d[33] -> “！”。
    然而，例如 chr(0) 是 "\x00"，这看起来很糟糕。
    所以 OpenAI 将这些字节映射到一个新的字符范围内，在这个范围内，chr() 会返回一个单独的美观字符。
    因此，在最终的字典中，我们有 d[0] -> 'Ā'，而不是之前提到的 'Ā'，这是因为 d[0] 等同于 chr(0 + 2^8)。
    特别地，空格字符是 32，通过 ord(' ') 可以看到这一点。相反，此函数会将空格（32）向右移动 256 位到 288，所以 d[32] -> 'Ġ'。
    所以这只是将 0 到 255 这些字节简单地一对一映射到“看起来美观”的 Unicode 字符上，
    这些字符可以以它们原始形式呈现，或者像 'Ā' 或 'Ġ' 这样的有趣移位字符呈现，等等。
    """
    # 这 188 个数字原本就是正常的，无需进行任何调整。
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] # 在输出字典中，所有在 bs 中的整数 b 都会直接映射为 chr(b) 。
    # 现在获取其余 68 个需要进行移位操作的整数的表示形式
    # 每个元素都将被映射为 chr(256 + n) 的形式，其中 n 在循环中从 0 到 67 依次递增。
    n = 0
    for b in range(2**8):
        if b not in bs:
            # 如果这个字节是"ugly"的，那就将其转换为下一个可用的"nice"的字符。
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d

def get_pairs(word):
    """
    将所有双词组合以元组的形式返回，这些元组包含可迭代字符串中的连续元素。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges):
        # 字节编码器/解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        # BPE token编码器/解码器
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        # BPE 合并列表，该列表定义了 BPE 的“树”，其中包含若干元组 (a， b) ，它们将合并为一个词项 ab 。
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 用于预分词的拆分模式
        # 应该添加 re.IGNORECASE 这个参数，这样就能对缩写词的变体（即大写形式）进行合并操作了。——原始的 OpenAI 评论
        """
        那么，这个正则表达式到底在寻找什么呢？
        Python 的 re 模块参考：https://docs.python.org/3/library/re.html
        - 横线“|”表示“或”，所以 re.findall 会按照匹配的部分从左到右将文本分块
        - "\'"会将诸如"Andrej's"这样的内容拆分成"(Andrej， 's)"
        - "？\p{L}"：可选的空格后跟 1 个或多个属于“字母”类别的 Unicode 代码点
        - "？\p{N}"：可选的空格后跟 1 个或多个属于“数字”类别的 Unicode 代码点
        - "？[^\s\p{L}\p{N}]+"：可选的空格，然后是 1 个或多个不是空白、字母或数字的字符
        - "\s+(？!\S)"：1 个或多个空白字符（例如空格、制表符等），除非它们后面是非空白字符
        所以它会依次消费序列中的空白字符，但会排除序列中的最后一个空白字符。
        这个最后一个空白字符有机会在前面的模式中匹配早期模式中的可选"？"。
        - "\s+"：1 个或多个空白字符，可能旨在捕获字符串末尾的完整空白序列
        简而言之：
        - 我们对一些常见的撇号构造（"s"、"t"、"re"等）进行了特殊处理，并将其转换为单独的标记然后，我们将字符串按以下顺序分成连续的几部分：
        1）字母；2）数字；3）非字母数字字符；4）空格。
        """
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        """
        此函数利用 self.bpe_ranks 参数，通过递归方式在树结构中合并所有可能的 BPE 词元。
        这里的“token”是指经过正则表达式分词处理后的一个单独“单词”（经过字节编码后，例如“Ġthere”）。
        """
        # “token”是指一个单独的“单词”组成的字符串，经过字节编码后，例如“Ġthere”

        # 记忆化，以提高效率
        if token in self.cache:
            return self.cache[token]

        word = tuple(token) # 构成该标记的各个字符，以元组的形式呈现
        pairs = get_pairs(word) # 获取所有双词组合

        if not pairs:
            return token

        while True:

            # 找到可以合并的下一个最低层级的双词组合
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # 不再有三字组合能够被合并了

            first, second = bigram

            # 现在我们将把列表中所有出现的“(第一项，第二项)”内容全部替换掉。
            # 将单词合并为一个合并后的标记 first_second，然后将其添加到输出列表 new_words 中。
            new_word = []
            i = 0
            while i < len(word):

                # 在当前单词序列中找到“first”这一词的下一次出现位置
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # 如果这种情况再次发生，那么就把这两者合并为一个。
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # 所有出现的“(第一，第二)”的情况都已合并为“第一_第二”了。
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # 将所有单词连接成一个字符串，并使用""作为分隔符。请注意
        # 至此，所有字符均已进行字节编码，确保在实际数据中不会使用""，而它则是一个"special"的分隔符字符。
        word = ' '.join(word)

        # cache the result and return
        self.cache[token] = word
        return word

    def encode(self, text):
        """ string goes in, list of integers comes out """
        bpe_idx = []
        # 将输入文本预分词为字符串标记（大致来说就是单词）
        tokens = re.findall(self.pat, text)
        # 将每个标记转换为 BPE 整数形式
        for token in tokens:
            # 将该token编码为一个字节（b' '）对象
            token_bytes = token.encode('utf-8')
            # 将所有字节转换为其对应的 Unicode 字符串表示形式，并进行展平处理。
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            # 根据 self.bpe_ranks 的值执行所有适用的 BPE 合并操作
            token_merged = self.bpe(token_translated).split(' ')
            # 将所有 BPE token转换为整数
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # 将列出的所有输出整数的清单进一步扩充
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        """ debugging function, same as encode but returns all intermediate work """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })
        out = {
            'bpe_idx': bpe_idx, # 实际的输出序列
            'tokens': tokens, # 预分词的结果
            'parts': parts, # 每个词元部分的中间产物
        }
        return out

    def decode(self, bpe_idx):
        """ list of integers comes in, string comes out """
        # 将整数映射为相应的标记符
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # 反转字节编码器，例如将"Ġ"恢复为" "，从而获取字节。
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # 恢复完整的 UTF-8 字符串
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text

def get_file(local_file, remote_file):
    """ downloads remote_file to local_file if necessary """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)

def get_encoder():
    """
    Returns an instance of the GPT BPE Encoder/Decoder
    and handles caching of "database" files.
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    # 加载名为“encoder.json”的文件，该文件包含了从“词元”到“BPE 编索引”的原始映射关系。
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == 50257 # 256 个独立的字节标记，50,000 个合并标记，以及 1 个特殊标记

    # 加载名为“vocab.bpe”的文件，该文件包含了 BPE 合并规则，即 BPE 树结构。
    # 以元组的形式表示（a, b），这表示（a， b）将被合并为一个标记“ab”
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # light postprocessing: 第一行和最后一行去掉版本信息，其余部分保持不变。
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000 # 50,000 个合并后的token

    # 构建编码器对象并返回
    enc = Encoder(encoder, bpe_merges)
    return enc

# -----------------------------------------------------------------------------

class BPETokenizer:
    """ 一个与 MindSpore 相兼容的类，用于封装上述的编码器 """

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors='pt'):
        # 仅使用 MindSpore；这是因为希望与 mindnlp/mindformers 的接口保持一致。
        assert return_tensors == 'pt'
        # 目前仅支持单个字符串输入，未来可能会支持字符串列表输入。
        assert isinstance(text, str)
        # 进行编码并创建一个“批次维度”值为 1 的设置
        idx = [self.encoder.encode(text)]
        # 转换为 MindSpore 张量
        out = ms.tensor(idx, dtype=ms.int32)
        return out

    def decode(self, idx):
        # 确保使用一个简单的一维张量
        assert idx.ndim == 1
        # 将索引解码为文本
        text = self.encoder.decode(idx.tolist())
        return text


if __name__ == '__main__':

    # 这是个编码示例
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("Original text is:")
    print(text)
    print("First the text gets pre-tokenized, broken up into chunks, the outcome is:")
    print(r['tokens'])
    # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    print("Then we iterate over each chunk and process them in turn...")
    for part in r['parts']:
        print(part)
    # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    print("and the final outcome is concatenating and flattening all the token_ix:")
    print(r['bpe_idx'])
    # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # 这样一来，它就会成为输入到transformer中的整数序列。
    print("ready to feed into a Transformer!")
