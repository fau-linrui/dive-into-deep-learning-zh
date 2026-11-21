# 子词嵌入
:label:`sec_fasttext`

在英语中，
"helps"、"helped"和"helping"这样的单词
都是同一个单词"help"的
屈折形式。
"dog"与"dogs"之间的
关系
等同于
"cat"与"cats"之间的
关系，
并且
"boy"与"boyfriend"之间的
关系
等同于
"girl"与"girlfriend"之间的
关系。
在法语和西班牙语等
其他语言中，
许多动词有 40 多种屈折形式，
而在芬兰语中，
一个名词最多可以有 15 种格。
在语言学中，
形态学研究词的构成和词与词之间的关系。
然而，
词的内部结构
在 word2vec 中
和在 GloVe 中
都没有被探究。

## fastText 模型

回想一下 word2vec 中词是如何表示的。
在跳元模型和
连续词袋模型中，
同一个词的不同屈折形式
直接由不同的向量表示，
不共享参数。
为了利用形态学信息，
*fastText* 模型
提出了*子词嵌入*方法，
其中子词是一个字符 $n$ 元语法 :cite:`Bojanowski.Grave.Joulin.ea.2017`。
fastText 不是学习词级别的向量表示，
而是可以看作
子词级别的跳元模型，
其中每个*中心词*由其
子词向量之和表示。

让我们以单词"where"为例，
说明如何获得 fastText 中
每个中心词的子词。
首先，在单词的开头和结尾添加特殊字符“&lt;”和“&gt;”，
以区分前缀、后缀与其他子词。
然后，从单词中提取字符 $n$ 元语法。
例如，当 $n=3$ 时，
我们得到所有长度为 3 的子词："&lt;wh"、"whe"、"her"、"ere"、"re&gt;"，以及特殊子词"&lt;where&gt;"。


在 fastText 中，对于任意词 $w$，
用 $\mathcal{G}_w$ 表示
其长度在 3 到 6 之间的所有子词
及其特殊子词的并集。
词表
是所有词的子词的并集。
令 $\mathbf{z}_g$
为词表中子词 $g$ 的向量，
则跳元模型中
词 $w$ 作为中心词时的
向量 $\mathbf{v}_w$
是其子词向量之和：

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

fastText 的其余部分与跳元模型相同。与跳元模型相比，
fastText 中的词表更大，
导致模型参数更多。
此外，
为了计算一个词的表示，
必须对其所有子词向量
求和，
这导致更高的计算复杂度。
然而，
得益于结构相似的词之间共享的子词参数，
生僻词甚至词表外的词
在 fastText 中
可能获得更好的向量表示。



## 字节对编码
:label:`subsec_Byte_Pair_Encoding`

在 fastText 中，所有提取的子词都必须是指定长度的，例如 $3$ 到 $6$，因此词表大小无法预先确定。
为了在固定大小的词表中允许可变长度的子词，
我们可以应用一种称为*字节对编码*（byte pair encoding，BPE）的压缩算法
来提取子词 :cite:`Sennrich.Haddow.Birch.2015`。

字节对编码对训练数据集进行统计分析，以发现词内的常见符号，
例如任意长度的连续字符。
从长度为 1 的符号开始，
字节对编码迭代地合并最频繁的连续符号对，以产生新的更长的符号。
请注意，为了效率，不考虑跨越词边界的符号对。
最后，我们可以使用这样的符号作为子词来切分单词。
字节对编码及其变体已被用于 GPT-2 :cite:`Radford.Wu.Child.ea.2019` 和 RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` 等流行的自然语言处理预训练模型的输入表示。
下面，我们将说明字节对编码是如何工作的。

首先，我们将符号词表初始化为所有英文字母小写字符、一个特殊的词尾符号`'_'`和一个特殊的未知符号`'[UNK]'`。

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

由于我们不考虑跨越单词边界的符号对，
我们只需要一个字典`raw_token_freqs`，它将单词映射到其在
数据集中的频率（出现次数）。
请注意，特殊符号`'_'`被附加到每个单词上，以便
我们可以轻松地从输出符号序列（例如"a_ tall er_ man"）
恢复单词序列（例如"a taller man"）。
由于我们从只包含单个字符和特殊符号的词表开始合并过程，单词内的每对连续字符之间都会插入空格（字典`token_freqs`的键）。
换句话说，空格是单词内符号之间的分隔符。

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

我们定义以下`get_max_freq_pair`函数，
它返回单词内最频繁的连续符号对，
其中单词来自输入字典`token_freqs`的键。

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

作为一种基于连续符号频率的贪心方法，
字节对编码将使用以下`merge_symbols`函数来合并最频繁的连续符号对，以产生新的符号。

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

现在我们对字典`token_freqs`的键迭代地执行字节对编码算法。在第一次迭代中，最频繁的连续符号对是`'t'`和`'a'`，因此字节对编码将它们合并，产生一个新符号`'ta'`。在第二次迭代中，字节对编码继续合并`'ta'`和`'l'`，得到另一个新符号`'tal'`。

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

经过 10 次字节对编码迭代后，我们可以看到列表`symbols`现在包含另外 10 个由其他符号迭代合并而来的符号。

```{.python .input}
#@tab all
print(symbols)
```

对于字典`raw_token_freqs`的键所指定的同一数据集，
作为字节对编码算法的结果，数据集中的每个单词现在都被子词"fast_"、"fast"、"er_"、"tall_"和"tall"切分。
例如，单词"faster_"和"taller_"分别被切分为"fast er_"和"tall er_"。

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

请注意，字节对编码的结果取决于所使用的数据集。
我们也可以使用从一个数据集学习到的子词
来切分另一个数据集的单词。
作为一种贪心方法，下面的`segment_BPE`函数尝试将单词拆分为来自输入参数`symbols`的最长可能子词。

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

下面，我们使用列表`symbols`中从上述数据集学习到的子词，
来切分代表另一个数据集的`tokens`。

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## 小结

* fastText 模型提出了子词嵌入方法。它基于 word2vec 中的跳元模型，将中心词表示为其子词向量之和。
* 字节对编码对训练数据集进行统计分析，以发现词内的常见符号。作为一种贪心方法，字节对编码迭代地合并最频繁的连续符号对。
* 子词嵌入可以提高生僻词和词表外的词的表示质量。

## 练习

1. 举例来说，英语中大约有 $3\times 10^8$ 种可能的 $6$ 元语法。当子词太多时会有什么问题？如何解决该问题？提示：参见 fastText 论文 :cite:`Bojanowski.Grave.Joulin.ea.2017` 第 3.2 节的末尾部分。
1. 如何基于连续词袋模型设计一个子词嵌入模型？
1. 当初始符号词表大小为 $n$ 时，要得到大小为 $m$ 的词表，需要多少次合并操作？
1. 如何将字节对编码的思想扩展到提取短语？



:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/4587)
:end_tab:
