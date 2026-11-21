# 词的相似性和类比
:label:`sec_synonyms`

在 :numref:`sec_word2vec_pretraining` 中，
我们在一个小数据集上训练了一个 word2vec 模型，
并将其应用于
为输入词查找语义相近的词。
在实践中，
在大型语料库上预训练的
词向量可以
应用于下游的
自然语言处理任务，
这将在后面的
:numref:`chap_nlp_app` 中介绍。
为了以直接的方式展示
来自大型语料库的预训练词向量的
语义特性，
让我们把它们应用于
词的相似性和类比任务。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## 加载预训练词向量

下面列出了维度为 50、100 和 300 的预训练 GloVe 嵌入，
它们可以从 [GloVe 网站](https://nlp.stanford.edu/projects/glove/)下载。
预训练的 fastText 嵌入有多种语言版本。
这里我们考虑一个英文版本（300 维的"wiki.en"），可以从
[fastText 网站](https://fasttext.cc/)下载。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

为了加载这些预训练的 GloVe 和 fastText 嵌入，我们定义以下`TokenEmbedding`类。

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

下面我们加载
50 维的 GloVe 嵌入
（在维基百科子集上预训练）。
创建`TokenEmbedding`实例时，
如果指定的嵌入文件尚未下载，
则必须先下载它。

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

输出词表大小。该词表包含 400000 个词（词元）以及一个特殊的未知词元。

```{.python .input}
#@tab all
len(glove_6b50d)
```

我们可以获取一个词在词表中的索引，反之亦然。

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 应用预训练词向量

使用加载的 GloVe 向量，
我们将通过把它们应用于
下面的词的相似性和类比任务，
来展示它们的语义特性。


### 词的相似性

与 :numref:`subsec_apply-word-embed` 类似，
为了基于
词向量之间的余弦相似度，
为输入词查找语义相近的词，
我们实现下面的`knn`（$k$ 近邻）函数。

```{.python .input}
#@tab mxnet
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

然后，我们
使用预训练词向量
从`TokenEmbedding`实例`embed`中
搜索相似的词。

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d`中预训练词向量的
词表包含 400000 个词以及一个特殊的未知词元。
排除输入词和未知词元，
让我们在这个词表中
找出与单词"chip"
语义最相近的
三个词。

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

下面输出与"baby"和"beautiful"相似的词。

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 词的类比

除了查找相似的词，
我们还可以将词向量
应用于词的类比任务。
例如，
“man”∶“woman”∷“son”∶“daughter”
是一种词类比的形式：
“man”之于“woman”就相当于“son”之于“daughter”。
具体来说，
词类比补全任务
可以定义为：
对于词类比
$a : b :: c : d$，给定前三个词 $a$、$b$ 和 $c$，求 $d$。
用 $\textrm{vec}(w)$ 表示词 $w$ 的向量。
为了完成类比，
我们要找出与
$\textrm{vec}(c)+\textrm{vec}(b)-\textrm{vec}(a)$
的结果最相似的词
的向量。

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

让我们使用加载的词向量来验证“男性-女性”类比。

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

下面完成一个
“首都-国家”类比：
“beijing”∶“china”∷“tokyo”∶“japan”。
这展示了
预训练词向量中的语义特性。

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

对于
“形容词-形容词最高级”类比，
例如
“bad”∶“worst”∷“big”∶“biggest”，
我们可以看到预训练词向量
能够捕捉句法信息。

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

为了展示预训练词向量所捕获的
过去时概念，
我们可以用
“现在时-过去时”类比来测试句法：“do”∶“did”∷“go”∶“went”。

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 小结

* 在实践中，在大型语料库上预训练的词向量可以应用于下游的自然语言处理任务。
* 预训练词向量可以应用于词的相似性和类比任务。


## 练习

1. 使用`TokenEmbedding('wiki.en')`测试 fastText 的结果。
1. 当词表非常大时，我们怎样才能更快地找到相似的词或完成词的类比？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1336)
:end_tab:
