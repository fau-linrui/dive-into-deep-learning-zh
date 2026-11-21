```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 机器翻译与数据集
:label:`sec_machine_translation`

推动人们对现代 RNN 产生广泛兴趣的重大突破之一，
是应用领域统计*机器翻译*（machine translation）的一项重大进展。
这里，模型接收一种语言中的句子，
并且必须预测另一种语言中对应的句子。
请注意，这里的句子长度可能不同，
而且由于两种语言的语法结构差异，
两个句子中对应的词可能不会以相同的顺序出现。

许多问题都具有这种在两个“未对齐”序列之间进行映射的特点。
例子包括从对话提示到回复的映射，
或者从问题到答案的映射。
广义上说，这类问题称为*序列到序列*（sequence-to-sequence，seq2seq）问题，
它们既是本章余下内容的重点，
也是 :numref:`chap_attention-and-transformers` 大部分内容的重点。

在本节中，我们介绍机器翻译问题，
以及后续示例中将使用的一个示例数据集。
几十年来，语言间翻译的统计形式化方法一直很流行
:cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`，
甚至在研究者让神经网络方法奏效之前就是如此
（这些方法常被统称为*神经机器翻译*（neural machine translation））。

首先，我们需要一些新代码来处理数据。
与 :numref:`sec_language-model` 中看到的语言建模不同，
这里的每个样例都由两个独立的文本序列组成：
一个属于源语言，另一个（译文）属于目标语言。
下面的代码片段将展示如何把预处理后的数据加载为小批量以供训练。

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## [**下载和预处理数据集**]

首先，我们下载一个英法数据集，
它由 [Tatoeba 项目的双语句子对](http://www.manythings.org/anki/)组成。
数据集中的每一行都是一个制表符分隔的句子对，
由一个英文文本序列（*源*（source））
和对应的法文译文文本序列（*目标*（target））组成。
请注意，每个文本序列可以只是一个句子，
也可以是由多个句子组成的段落。

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

下载数据集之后，我们对原始文本数据[**执行若干预处理步骤**]。
例如，我们把不间断空格替换为空格，
把大写字母转换为小写字母，
并在单词与标点符号之间插入空格。

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## [**词元化**]

与 :numref:`sec_language-model` 中字符级的词元化不同，
对于机器翻译，这里我们更倾向于词级别的词元化
（当今最先进的模型使用更复杂的词元化技术）。
下面的 `_tokenize` 方法对前 `max_examples` 个文本序列对进行词元化，
其中每个词元要么是一个单词，要么是一个标点符号。
我们在每个序列末尾追加特殊的“&lt;eos&gt;”词元，
以表示序列的结束。
当模型通过逐个生成词元来进行预测时，
生成“&lt;eos&gt;”词元可以表明输出序列已经完成。
最后，下面的方法返回两个由词元列表组成的列表：
`src` 和 `tgt`。
具体来说，`src[i]` 是源语言（这里是英语）中
第 $i^\textrm{th}$ 个文本序列的词元列表，
而 `tgt[i]` 是目标语言（这里是法语）中的对应列表。

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

让我们[**绘制每个文本序列词元数的直方图。**]
在这个简单的英法数据集中，
大多数文本序列的词元数少于 20。

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## 加载固定长度的序列
:label:`subsec_loading-seq-fixed-len`

回想一下，在语言建模中，[**每个样例序列**]，
无论是一个句子的一段，
还是跨越多个句子的一个片段，
都(**具有固定长度。**)
这是由 :numref:`sec_language-model` 中的 `num_steps`
（时间步数或词元数）参数指定的。
在机器翻译中，每个样例是一对源文本序列和目标文本序列，
其中两个文本序列的长度可能不同。

为了提高计算效率，
我们仍然可以通过*截断*和*填充*，
一次处理一个小批量的文本序列。
假设同一小批量中的每个序列都应具有相同的长度 `num_steps`。
如果某个文本序列的词元数少于 `num_steps`，
我们就在其末尾不断追加特殊的“&lt;pad&gt;”词元，
直到其长度达到 `num_steps`。
否则，我们就对文本序列进行截断，
只保留其前 `num_steps` 个词元，丢弃剩余部分。
这样，每个文本序列都将具有相同的长度，
从而可以加载为形状相同的小批量。
此外，我们还会记录源序列除去填充词元之后的长度。
后面的某些模型将需要这一信息。

由于机器翻译数据集由成对的语言组成，
我们可以分别为源语言和目标语言构建两个词表。
使用词级别的词元化时，
词表大小会显著大于使用字符级词元化时的大小。
为了缓解这一问题，
这里我们把出现次数少于两次的罕见词元
视为同一个未知（“&lt;unk&gt;”）词元。
正如我们后面会解释的（:numref:`fig_seq2seq`），
用目标序列训练时，
解码器输出（标签词元）可以是同一个解码器输入（目标词元）平移一个词元得到的；
而特殊的序列起始“&lt;bos&gt;”词元
将用作预测目标序列的第一个输入词元（:numref:`fig_seq2seq_predict`）。

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**读取数据集**]

最后，我们定义 `get_dataloader` 方法以返回数据迭代器。

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

让我们[**从英法数据集中读取第一个小批量。**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

下面我们展示一对经过上述 `_build_arrays` 方法处理后的源序列和目标序列（以字符串格式）。

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## 小结

在自然语言处理中，*机器翻译*指的是自动把
表示*源*语言中一段文本的字符串
映射为表示*目标*语言中一个合理译文的字符串的任务。
使用词级别的词元化时，词表大小会显著大于字符级词元化，
但序列长度会短得多。
为了缓解词表过大的问题，
我们可以把罕见词元视为某个“未知”词元。
我们可以对文本序列进行截断和填充，
使它们都具有相同的长度，以便加载为小批量。
现代实现通常把长度相近的序列分桶，
以避免在填充上浪费过多计算。

## 练习

1. 尝试 `_tokenize` 方法中 `max_examples` 参数的不同取值。这对源语言和目标语言的词表大小有什么影响？
1. 像中文和日文这样的语言，其文本中没有词边界标志（例如空格）。在这种情况下，词级别的词元化仍然是个好主意吗？为什么是或为什么不是？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/3863)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/18020)
:end_tab:
