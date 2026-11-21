# 自然语言推断及数据集
:label:`sec_natural-language-inference-and-dataset`

在 :numref:`sec_sentiment` 中，我们讨论了情感分析问题。
该任务旨在将单个文本序列分类到预定义的类别中，
例如一组情感极性。
然而，当需要判定一个句子能否从另一个句子推断出来，
或者通过识别语义等价的句子来消除冗余时，
仅知道如何对单个文本序列进行分类是不够的。
相反，我们需要能够对文本序列对进行推理。


## 自然语言推断

*自然语言推断*（natural language inference）研究的是能否从*前提*（premise）推断出*假设*（hypothesis），其中两者都是文本序列。
换言之，自然语言推断确定一对文本序列之间的逻辑关系。
这种关系通常分为三种类型：

* *蕴含*（entailment）：假设可以从前提中推断出来。
* *矛盾*（contradiction）：假设的否定可以从前提中推断出来。
* *中性*（neutral）：所有其他情况。

自然语言推断也称为识别文本蕴含任务。
例如，下面这一对句子将被标记为*蕴含*，因为假设中的“表达爱意”可以从前提中的“互相拥抱”推断出来。

> 前提：两个女人在互相拥抱。

> 假设：两个女人在表达爱意。

下面的例子是*矛盾*，因为“运行代码示例”意味着“没有在睡觉”，而不是“在睡觉”。

> 前提：一个男人正在运行《动手学深度学习》的代码示例。

> 假设：这个男人在睡觉。

第三个例子展示了*中性*关系，因为从“正在为我们表演”这一事实中，既不能推断出“著名”，也不能推断出“不著名”。

> 前提：音乐家们正在为我们表演。

> 假设：音乐家们很有名。

自然语言推断一直是理解自然语言的核心话题。
它的应用范围广泛，从
信息检索到开放域问答。
为了研究这个问题，我们将首先考察一个流行的自然语言推断基准数据集。


## 斯坦福自然语言推断（SNLI）数据集

[**斯坦福自然语言推断（SNLI）语料库**]包含超过 500000 对带标签的英语句子对 :cite:`Bowman.Angeli.Potts.ea.2015`。
我们下载并存储提取的 SNLI 数据集到路径`../data/snli_1.0`中。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**读取数据集**]

原始 SNLI 数据集包含的信息远比我们实验中实际需要的丰富得多。因此，我们定义了一个函数`read_snli`，只提取数据集的一部分，然后返回前提、假设及其标签的列表。

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

现在让我们[**打印前 3 对**]前提和假设，以及它们的标签（“0”、“1”和“2”分别对应“蕴含”、“矛盾”和“中性”）。

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

训练集约有 550000 对，
测试集约有 10000 对。
下面表明，在
训练集和测试集中，
三个[**标签“蕴含”、“矛盾”和“中性”是平衡的**]。

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**定义加载数据集的类**]

下面我们定义一个用于加载 SNLI 数据集的类，它继承自 Gluon 中的`Dataset`类。类构造函数中的参数`num_steps`指定文本序列的长度，使每个小批量的序列具有相同的形状。
换言之，
对于较长的序列，第一个`num_steps`个词元之后的词元会被截断，而对于较短的序列，特殊词元“&lt;pad&gt;”将被追加到末尾，直到其长度变为`num_steps`。
通过实现`__getitem__`函数，我们可以用索引`idx`任意访问前提、假设和标签。

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**整合代码**]

现在我们可以调用`read_snli`函数和`SNLIDataset`类来下载 SNLI 数据集，并返回训练集和测试集的`DataLoader`实例，以及训练集的词表。
值得注意的是，我们必须使用由训练集构建的词表
作为测试集的词表。
因此，测试集中的任何新词元对于在训练集上训练的模型来说都是未知的。

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

这里我们将批量大小设为 128、序列长度设为 50，
并调用`load_data_snli`函数获取数据迭代器和词表。
然后打印词表大小。

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

现在我们打印第一个小批量的形状。
与情感分析相反，
我们有两个输入`X[0]`和`X[1]`，分别表示成对的前提和假设。

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## 小结

* 自然语言推断研究能否从前提推断出假设，其中两者都是文本序列。
* 在自然语言推断中，前提和假设之间的关系包括蕴含、矛盾和中性。
* 斯坦福自然语言推断（SNLI）语料库是一个流行的自然语言推断基准数据集。


## 练习

1. 长期以来，机器翻译一直是基于输出译文与真实译文之间表面的 $n$ 元语法匹配来评估的。能否设计一种利用自然语言推断来评估机器翻译结果的度量方法？
1. 我们如何改变超参数来减小词表大小？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1388)
:end_tab:
