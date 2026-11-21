# 情感分析及数据集
:label:`sec_sentiment`


随着在线社交媒体
和评论平台的激增，
大量
带有观点倾向的数据
被记录下来，
它们具有支持
决策过程的巨大潜力。
*情感分析*
研究人们
在其生成的文本（如产品评论、
博客评论
和
论坛讨论）中
所表达的情感。
它具有广泛的应用，
涉及政治（例如，分析公众对政策的情绪）、
金融（例如，分析市场情绪）
和
市场营销（例如，产品调研和品牌管理）等众多领域。

由于情感
可以被划分为
离散的极性或尺度（例如，积极和消极），
我们可以把
情感分析
看作一个文本分类任务，
它将变长的文本序列
转换为固定长度的文本类别。
在本章中，
我们将使用斯坦福的[大型电影评论数据集](https://ai.stanford.edu/%7Eamaas/data/sentiment/)
进行情感分析。
该数据集由训练集和测试集组成，
两者各包含从 IMDb 下载的 25000 条电影评论。
在这两个数据集中，
“积极”和“消极”标签的数量
相等，
表示不同的情感极性。

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

## 读取数据集

首先，在路径`../data/aclImdb`中下载并提取这个 IMDb 评论数据集。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

接下来，读取训练数据集和测试数据集。每个样例都是一条评论及其标签：1 表示“积极”，0 表示“消极”。

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## 预处理数据集

将每个词视为一个词元，
并过滤掉出现次数少于 5 次的词，
我们基于训练数据集创建一个词表。

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

词元化之后，
我们绘制评论长度（以词元计）的直方图。

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

正如我们所预期的，
评论的长度各不相同。
为了每次处理
一个小批量的这类评论，
我们通过截断和填充将每条评论的长度设为 500，
这类似于 :numref:`sec_machine_translation` 中
机器翻译数据集的
预处理步骤。

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## 创建数据迭代器

现在我们可以创建数据迭代器了。
在每次迭代中，都会返回一个小批量的样例。

```{.python .input}
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## 整合代码

最后，我们将上述步骤封装到`load_data_imdb`函数中。
它返回训练数据迭代器、测试数据迭代器以及 IMDb 评论数据集的词表。

```{.python .input}
#@tab mxnet
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## 小结

* 情感分析研究人们在生成的文本中所表达的情感，它被视为一个文本分类问题：将变长的文本序列转换为固定长度的文本类别。
* 经过预处理，我们可以将斯坦福的大型电影评论数据集（IMDb 评论数据集）加载到带有词表的数据迭代器中。


## 练习


1. 本节中哪些超参数可以修改，以加速情感分析模型的训练？
1. 能否实现一个函数，将 [Amazon 评论](https://snap.stanford.edu/data/web-Amazon.html) 数据集加载到数据迭代器和标签中，用于情感分析？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1387)
:end_tab:
