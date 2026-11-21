# 将原始文本转换为序列数据
:label:`sec_text-sequence`

在本书中，
我们经常要处理表示为
单词、字符或词片序列的文本数据。
为了开始工作，我们需要一些基本工具，
用于把原始文本转换为
适当形式的序列。
典型的预处理流水线
执行以下步骤：

1. 将文本作为字符串加载到内存中。
1. 将字符串切分为词元（例如单词或字符）。
1. 构造词表字典，把每个词表元素与一个数值索引关联起来。
1. 将文本转换为数值索引序列。

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## 读取数据集

这里，我们将使用H. G. Wells的
[《时间机器》](http://www.gutenberg.org/ebooks/35)，
这本书仅包含三万余词。
虽然真实应用通常
涉及大得多的数据集，
但这已足以演示
预处理流水线。
下面的`_download`方法
(**将原始文本读入一个字符串**)。

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

为简单起见，我们在预处理原始文本时忽略标点和大小写。

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## 词元化

*词元*是文本的原子（不可再分）单位。
每个时间步对应1个词元，
但究竟什么才算作一个词元是一种设计选择。
例如，我们可以把句子
“Baby needs a new pair of shoes”
表示为一个由7个单词组成的序列，
其中所有单词构成的集合形成
一个很大的词表（通常有数万到数十万个单词）。
或者，我们也可以把同一句话
表示为长得多的30个字符的序列，
使用小得多的词表
（不同的ASCII字符只有256个）。
下面，我们把预处理后的文本词元化
为字符序列。

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## 词表

这些词元仍然是字符串。
然而，我们模型的输入
最终必须由
数值输入组成。
[**接下来，我们引入一个类，
用于构造*词表*（vocabulary），
即把每个不同的词元值
与唯一索引关联起来的对象。**]
首先，我们确定训练*语料库*中不同词元的集合。
然后为每个不同词元分配一个数值索引。
为方便起见，很少出现的词表元素往往会被丢弃。
每当我们在训练或测试时遇到一个
以前未见过、或已被从词表中丢弃的词元时，
就用一个特殊的“&lt;unk&gt;”词元来表示它，
表示这是一个*未知*值。

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

现在我们为数据集[**构造一个词表**]，
把字符串序列
转换为数值索引列表。
请注意，我们没有丢失任何信息，
可以轻松地把数据集
转换回其原始（字符串）表示。

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## 整合所有内容

使用上面的类和方法，
我们[**把所有内容打包到
`TimeMachine`类的以下`build`方法中**]，
它返回`corpus`（一个词元索引列表）和`vocab`
（*《时间机器》*语料库的词表）。
我们在这里所做的修改是：
(i) 把文本词元化为字符而不是单词，
以简化后续各节中的训练；
(ii) `corpus`是单个列表，而不是词元列表的列表，
因为*《时间机器》*数据集中的每一行文本
不一定是一个句子或段落。

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## 探索性语言统计
:label:`subsec_natural-lang-stat`

使用真实的语料库以及基于单词定义的`Vocab`类，
我们可以考察语料库中单词使用情况的基本统计。
下面，我们根据*《时间机器》*中使用的单词构造词表，
并打印其中出现频率最高的十个单词。

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

注意，(**十个最频繁的单词**)
并不都具有很好的区分性。
你甚至可以想象，
如果我们任意选一本书，也可能会看到非常相似的列表。
像“the”和“a”这样的冠词，
像“i”和“my”这样的代词，
以及像“of”、“to”和“in”这样的介词
出现得很频繁，因为它们承担着常见的句法角色。
这类常见但并不特别具有区分性的词
通常称为(***停用词***)，
在上一代基于所谓词袋表示的文本分类器中，
它们大多会被过滤掉。
然而，它们是有含义的，
在处理现代基于RNN和
Transformer的神经模型时，没有必要把它们过滤掉。
如果你继续往下看列表，
你会注意到
词频衰减得很快。
排在第$10^{\textrm{th}}$位的单词，
其常见程度不到最频繁单词的$1/5$。
当我们沿排名往下看时，
词频倾向于服从幂律分布
（具体来说是齐普夫分布）。
为了更直观地了解，我们[**绘制单词频率的图**]。

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

把前几个词作为例外处理之后，
其余所有单词在双对数图上大致遵循一条直线。
这一现象由*齐普夫定律*（Zipf's law）刻画，
它指出，排在第$i^\textrm{th}$位的单词，
其频率$n_i$为：

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

这等价于

$$\log n_i = -\alpha \log i + c,$$

其中$\alpha$是刻画该分布的指数，
$c$是常数。
如果我们要通过计数统计来对单词建模，
这一点就足以让我们停下来思考。
毕竟，我们会显著高估尾部（即不常出现的词）的频率。但是，[**其他词组合又如何呢，例如两个连续单词（二元语法）和三个连续单词（三元语法）**]，以及更长的组合呢？
让我们看看二元语法的频率是否与单个单词（一元语法）的频率表现出相同的行为。

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

这里有一点值得注意。在十个最频繁的词对中，有九个都是由两个停用词组成的，只有一个与本书真正相关——“the time”。此外，让我们看看三元语法的频率是否表现出相同的行为。

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

现在，让我们[**可视化这三种模型中的词元频率**]：一元语法、二元语法和三元语法。

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

这张图相当令人兴奋。
首先，除了一元语法的单词之外，单词序列
似乎也遵循齐普夫定律，
只不过根据序列长度的不同，
:eqref:`eq_zipf_law` 中的指数$\alpha$更小。
其次，不同$n$元语法的数量
并没有那么大。
这让我们有理由相信语言中存在相当多的结构。
第三，许多$n$元语法出现得非常少。
这使得某些方法不适合语言建模，
并促使我们使用深度学习模型。
我们将在下一节讨论这一点。


## 小结

文本是深度学习中遇到的最常见的序列数据形式之一。
词元的常见选择包括字符、单词和词片。
为预处理文本，我们通常 (i) 把文本切分为词元；(ii) 构造词表，把词元字符串映射为数值索引；(iii) 把文本数据转换为词元索引，供模型处理。
在实践中，单词的频率往往遵循齐普夫定律。这不仅对单个单词（一元语法）成立，对$n$元语法也同样成立。


## 练习

1. 在本节的实验中，把文本词元化为单词，并改变`Vocab`实例的`min_freq`参数值。定性地刻画`min_freq`的变化如何影响所得词表的大小。
1. 估计该语料库中一元语法、二元语法和三元语法对应齐普夫分布的指数。
1. 寻找一些其他数据来源（下载一个标准机器学习数据集、挑选另一本公有领域的书、
   抓取一个网站等）。对每个数据来源，分别在单词和字符两个级别上对数据做词元化。在`min_freq`取值相同时，词表大小与*《时间机器》*语料库相比如何？估计这些语料库对应一元语法和二元语法分布的齐普夫分布指数。它们与你对*《时间机器》*语料库观察到的值相比如何？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/18011)
:end_tab:
