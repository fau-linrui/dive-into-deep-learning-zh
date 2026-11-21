# 预训练 word2vec
:label:`sec_word2vec_pretraining`


我们继续实现 :numref:`sec_word2vec` 中定义的跳元模型。
然后，
我们将在 PTB 数据集上使用负采样预训练 word2vec。
首先，
让我们调用 :numref:`sec_word2vec_data` 中描述的`d2l.load_data_ptb`
函数，来获取该数据集的
数据迭代器和词表。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## 跳元模型

我们使用嵌入层和批量矩阵乘法
来实现跳元模型。
首先，让我们回顾一下
嵌入层是如何工作的。


### 嵌入层

如 :numref:`sec_seq2seq` 所述，
嵌入层
将词元的索引映射到其特征向量。
该层的权重
是一个矩阵，其行数等于
词表大小（`input_dim`），
列数等于
每个词元的向量维度（`output_dim`）。
词嵌入模型训练完成后，
这个权重正是我们所需要的。

```{.python .input}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

嵌入层的输入是
词元（单词）的索引。
对于任意词元索引 $i$，
其向量表示
可以从嵌入层中
权重矩阵的
第 $i^\textrm{th}$ 行获得。
由于向量维度（`output_dim`）
被设置为 4，
对于形状为
(2, 3) 的词元索引小批量，
嵌入层
返回形状为 (2, 3, 4) 的向量。

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### 定义前向传播

在前向传播中，
跳元模型的输入
包括
形状为 (批量大小, 1) 的
中心词索引`center`
以及
形状为 (批量大小, `max_len`) 的
连结的上下文词和噪声词索引`contexts_and_negatives`，
其中`max_len`
定义在
:numref:`subsec_word2vec-minibatch-loading` 中。
这两个变量首先通过嵌入层
从词元索引
变换为向量，
然后它们的批量矩阵乘法
（在 :numref:`subsec_batch_dot` 中描述）
返回
形状为 (批量大小, 1, `max_len`) 的输出。
输出中的每个元素是
中心词向量与上下文词或噪声词向量的点积。

```{.python .input}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

让我们为一些示例输入打印这个`skip_gram`函数的输出形状。

```{.python .input}
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## 训练

在训练带负采样的跳元模型之前，
让我们先定义它的损失函数。


### 二元交叉熵损失

根据 :numref:`subsec_negative-sampling` 中
负采样损失函数的定义，
我们将使用
二元交叉熵损失。

```{.python .input}
#@tab mxnet
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

回想一下我们在 :numref:`subsec_word2vec-minibatch-loading` 中
对掩码变量
和标签变量的描述。
下面计算
给定变量的
二元交叉熵损失。

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

下面展示了
以上结果是如何（以较低效率的方式）
在二元交叉熵损失中
使用
sigmoid 激活函数
计算出来的。
我们可以把这两个输出
看作两个规范化的损失，
它们分别是对非掩码预测取平均得到的。

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### 初始化模型参数

我们分别为词表中的所有词
作为中心词
和作为上下文词的情况
定义两个嵌入层。
词向量维度
`embed_size`被设置为 100。

```{.python .input}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### 定义训练循环

训练循环定义如下。由于存在填充，损失函数的计算与之前的训练函数略有不同。

```{.python .input}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

现在我们可以使用负采样训练跳元模型了。

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## 应用词嵌入
:label:`subsec_apply-word-embed`


训练 word2vec 模型后，
我们可以利用训练所得模型中
词向量的余弦相似度，
从词表中
找出与输入词
语义最相近的词。

```{.python .input}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## 小结

* 我们可以使用嵌入层和二元交叉熵损失来训练带负采样的跳元模型。
* 词嵌入的应用包括：基于词向量的余弦相似度，为给定词查找语义相近的词。


## 练习

1. 使用训练好的模型，为其他输入词查找语义相近的词。你能通过调超参数来改进结果吗？
1. 当训练语料库很大时，我们往往在当前小批量的中心词*更新模型参数时*采样上下文词和噪声词。换言之，同一个中心词在不同训练轮中可能有不同的上下文词或噪声词。这种方法有什么好处？尝试实现这种训练方法。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1335)
:end_tab:
