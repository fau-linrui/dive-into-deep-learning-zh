# 预训练词嵌入的数据集
:label:`sec_word2vec_data`

既然我们已经了解了 word2vec 模型
和近似训练方法的技术细节，
让我们走一遍它们的实现。
具体来说，
我们将以 :numref:`sec_word2vec` 中的跳元模型
和 :numref:`sec_approx_train` 中的负采样
为例。
在本节中，
我们从预训练词嵌入模型的
数据集开始：
数据的原始格式
将被转换为
训练期间可以迭代的
小批量。

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## 读取数据集

我们这里使用的数据集是
[Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42)。
该语料库采样自
《华尔街日报》的文章，
被划分为训练集、验证集和测试集。
在原始格式中，
文本文件的每一行
代表一个由空格分隔的单词组成的句子。
这里我们将每个单词视为一个词元。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

读取训练集之后，
我们为语料库构建词表，
其中出现次数
少于 10 次的任何单词都会被
替换为"&lt;unk&gt;"词元。
请注意，原始数据集
也包含表示罕见（未知）词的"&lt;unk&gt;"词元。

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## 子采样

文本数据
通常含有高频词，
如"the"、"a"和"in"：
在非常大的语料库中，它们甚至可能出现数十亿次。
然而，
这些词往往在上下文窗口中
与许多不同的词共现，
提供的有用信号很少。
例如，
考虑上下文窗口中的单词"chip"：
直观地说，
它与低频词"intel"的共现
在训练中比
与高频词"a"的共现
更有用。
此外，用大量（高频）词进行训练
速度很慢。
因此，在训练词嵌入模型时，
高频词可以被*子采样* :cite:`Mikolov.Sutskever.Chen.ea.2013`。
具体来说，
数据集中的每个
索引词 $w_i$
将按以下概率被丢弃：


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

其中 $f(w_i)$ 是
单词 $w_i$ 的数量
与数据集中单词总数之比，
常数 $t$ 是一个超参数
（实验中取 $10^{-4}$）。
我们可以看到，只有当
相对频率
$f(w_i) > t$ 时，（高频）词 $w_i$ 才可能被丢弃，
并且词的相对频率越高，
被丢弃的概率越大。

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

以下代码片段
绘制了子采样前后
每个句子的
词元数量直方图。
正如预期的那样，
子采样通过丢弃高频词
显著缩短了句子长度，
这将带来训练加速。

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

对于单个词元，高频词"the"的采样率不到 1/20。

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

相比之下，
低频词"join"被完全保留。

```{.python .input}
#@tab all
compare_counts('join')
```

子采样之后，我们将语料库中的词元映射到它们的索引。

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 提取中心词和上下文词


下面的`get_centers_and_contexts`
函数从`corpus`中
提取所有
中心词及其上下文词。
它随机均匀地采样一个 1 到`max_window_size`
之间的整数作为上下文窗口大小。
对于任意中心词，
那些与它的距离
不超过采样得到的
上下文窗口大小的词
是它的上下文词。

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

接下来，我们创建一个包含两个句子的人工数据集，分别有 7 个和 3 个词。
设最大上下文窗口大小为 2，
并打印所有中心词及其上下文词。

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

在 PTB 数据集上训练时，
我们将最大上下文窗口大小设为 5。
下面提取数据集中所有的中心词及其上下文词。

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## 负采样

我们使用负采样进行近似训练。
为了根据预定义分布
采样噪声词，
我们定义以下`RandomGenerator`类，
其中（可能未规范化的）采样分布
通过参数`sampling_weights`传入。

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

例如，
我们可以如下在索引 1、2 和 3 之间
抽取 10 个随机变量 $X$，
其采样概率分别为 $P(X=1)=2/9, P(X=2)=3/9$ 和 $P(X=3)=4/9$。

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

对于一对中心词和上下文词，
我们随机采样`K`个（实验中为 5 个）噪声词。根据 word2vec 论文中的建议，
噪声词 $w$
的采样概率 $P(w)$
被设为其在
词表中的相对频率
的 0.75 次幂 :cite:`Mikolov.Sutskever.Chen.ea.2013`。

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## 以小批量加载训练样例
:label:`subsec_word2vec-minibatch-loading`

在提取出
所有中心词
及其
上下文词和采样得到的噪声词之后，
它们将被转换为
可以在训练期间
迭代加载的
样例小批量。



在一个小批量中，
第 $i^\textrm{th}$ 个样例包含一个中心词
及其 $n_i$ 个上下文词和 $m_i$ 个噪声词。
由于上下文窗口大小不同，
$n_i+m_i$ 随不同的 $i$ 而变化。
因此，
对于每个样例，
我们将其上下文词和噪声词连结在
`contexts_negatives`变量中，
并填充零，直到连结长度
达到 $\max_i n_i+m_i$（`max_len`）。
为了在计算损失时
排除填充，
我们定义一个掩码变量`masks`。
`masks`中的元素与`contexts_negatives`中的元素
之间存在一一对应关系，
其中`masks`中的零（否则为一）对应`contexts_negatives`中的填充。


为了区分正例和负例，
我们通过`labels`变量将`contexts_negatives`中的上下文词与噪声词分开。
与`masks`类似，
`labels`中的元素与`contexts_negatives`中的元素
之间也存在一一对应关系，
其中`labels`中的一（否则为零）对应`contexts_negatives`中的上下文词（正例）。


上述思想在下面的`batchify`函数中实现。
它的输入`data`是一个长度
等于批量大小的列表，
其中每个元素是一个样例，
由
中心词`center`、其上下文词`context`和其噪声词`negative`组成。
该函数返回
一个小批量，可在训练期间
加载用于计算，
例如包含掩码变量。

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

让我们用两个样例组成的小批量来测试这个函数。

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## 整合在一起

最后，我们定义`load_data_ptb`函数，它读取 PTB 数据集并返回数据迭代器和词表。

```{.python .input}
#@tab mxnet
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

让我们打印数据迭代器的第一个小批量。

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## 小结

* 高频词在训练中可能不那么有用。我们可以对它们进行子采样，以加速训练。
* 为了计算效率，我们以小批量加载样例。我们可以定义其他变量来区分填充与非填充，以及正例与负例。



## 练习

1. 如果不使用子采样，本节代码的运行时间会如何变化？
1. `RandomGenerator`类缓存`k`个随机采样结果。将`k`设为其他值，看看它如何影响数据加载速度。
1. 本节代码中还有哪些超参数可能影响数据加载速度？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1330)
:end_tab:
