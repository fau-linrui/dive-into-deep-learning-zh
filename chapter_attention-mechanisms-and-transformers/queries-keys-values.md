```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# 查询、键和值
:label:`sec_queries-keys-values`

到目前为止，我们回顾过的所有网络都严重依赖于输入具有明确定义的大小。例如，ImageNet 中的图像大小为$224 \times 224$像素，而 CNN 正是针对这一尺寸专门调校的。即使在自然语言处理中，RNN 的输入大小也是明确定义且固定的。变长问题通过一次处理一个词元，或者通过专门设计的卷积核 :cite:`Kalchbrenner.Grefenstette.Blunsom.2014` 来解决。当输入确实长度可变、信息量也可变时（例如 :numref:`sec_seq2seq` 中文本的变换 :cite:`Sutskever.Vinyals.Le.2014`），这种做法可能导致严重的问题。特别是对于长序列，要跟踪网络已经生成甚至已经查看过的所有内容，会变得相当困难。即使是 :citet:`yang2016neural` 提出的显式跟踪启发式方法，也只能提供有限的帮助。

请与数据库做个对比。数据库最简单的形式是键（$k$）和值（$v$）的集合。例如，我们的数据库$\mathcal{D}$可能由元组\{("Zhang", "Aston"), ("Lipton", "Zachary"), ("Li", "Mu"), ("Smola", "Alex"), ("Hu", "Rachel"), ("Werness", "Brent")\}组成，其中姓氏是键，名字是值。我们可以对$\mathcal{D}$进行操作，例如对"Li"进行精确查询（$q$）会返回值"Mu"。如果("Li", "Mu")不是$\mathcal{D}$中的一条记录，就不会有有效答案。如果我们也允许近似匹配，那么我们就会改为检索到("Lipton", "Zachary")。这个非常简单且平凡的例子仍然教给我们一些有用的东西：

* 我们可以设计查询$q$，使其作用于（$k$,$v$）对时的有效性不依赖于数据库的大小。
* 同一个查询可以根据数据库的内容得到不同的答案。
* 作用于大状态空间（数据库）的“代码”可以非常简单（例如精确匹配、近似匹配、top-$k$）。
* 无需压缩或简化数据库，就能使这些操作有效。

显然，如果不是为了解释深度学习，我们不会在这里引入一个简单的数据库。事实上，这引出了过去十年深度学习中最令人兴奋的概念之一：*注意力机制*（attention mechanism） :cite:`Bahdanau.Cho.Bengio.2014`。我们稍后将介绍它应用于机器翻译的具体细节。现在，只需考虑以下内容：用$\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$表示由$m$个*键*（key）和*值*（value）元组构成的数据库。此外，用$\mathbf{q}$表示一个*查询*（query）。于是，我们可以把在$\mathcal{D}$上的*注意力*（attention）定义为

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$
:eqlabel:`eq_attention_pooling`

其中$\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$（$i = 1, \ldots, m$）是标量注意力权重。该运算本身通常称为*注意力汇聚*（attention pooling）。*注意力*这一名称源于以下事实：该运算特别关注权重$\alpha$显著（即较大）的那些项。因此，在$\mathcal{D}$上的注意力生成数据库中值的线性组合。事实上，它把上面的例子作为特例包含在内，此时除一个权重外其余权重均为零。我们有若干特例：

* 权重$\alpha(\mathbf{q}, \mathbf{k}_i)$非负。此时注意力机制输出的结果位于值$\mathbf{v}_i$张成的凸锥内。
* 权重$\alpha(\mathbf{q}, \mathbf{k}_i)$构成凸组合，即$\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$，且$\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0$对所有的$i$都成立。这是深度学习中最常见的设定。
* 恰好其中一个权重$\alpha(\mathbf{q}, \mathbf{k}_i)$为$1$，而其余权重均为$0$。这类似于传统的数据库查询。
* 所有权重都相等，即$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{1}{m}$对所有的$i$都成立。这相当于在整个数据库上取平均，在深度学习中也称为平均汇聚。

确保权重之和为$1$的一个常见策略是通过下式对它们进行规范化：

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{{\sum_j} \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

特别地，为了确保权重还非负，可以借助指数化。这意味着，我们现在可以选取*任意*函数$a(\mathbf{q}, \mathbf{k})$，然后通过下式对它应用用于多项模型的 softmax 运算：

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}. $$
:eqlabel:`eq_softmax_attention`

该运算在所有深度学习框架中都可以直接使用。它可微，而且其梯度永不消失，这些都是模型中期望具备的性质。不过请注意，上面引入的注意力机制并不是唯一的选择。例如，我们可以设计一种不可微的注意力模型，并用强化学习方法 :cite:`Mnih.Heess.Graves.ea.2014` 来训练它。正如人们所料，训练这样的模型相当复杂。因此，现代注意力研究的主体遵循 :numref:`fig_qkv` 所勾勒的框架。所以我们把阐述重点放在这一族可微机制上。

![注意力机制通过注意力汇聚计算值$\mathbf{v}_\mathit{i}$的线性组合，
其中权重根据查询$\mathbf{q}$与键$\mathbf{k}_\mathit{i}$之间的相容性导出。](../img/qkv.svg)
:label:`fig_qkv`

相当值得注意的是，作用于键值集合的实际“代码”，即查询，可以相当简洁，即便要操作的空间很大。这对网络层来说是一个理想的性质，因为它不需要学习太多参数。同样方便的是，注意力可以作用于任意大的数据库，而无需改变注意力汇聚运算的执行方式。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=2}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## 可视化

注意力机制的好处之一是它可以相当直观，尤其是当权重非负且求和为$1$时。在这种情况下，我们可以把较大的权重*解读*为模型挑选相关成分的一种方式。尽管这是一个很好的直觉，但要记住它仅此而已，只是一种*直觉*。无论如何，在应用各种不同的查询时，我们可能希望可视化它对给定键集合的影响。这个函数稍后会很方便。

于是我们定义`show_heatmaps`函数。请注意，它的输入不是（注意力权重的）矩阵，而是一个具有四个轴的张量，从而可以容纳一组不同的查询和权重。因此，输入`matrices`的形状为（显示的行数、显示的列数、查询数、键数）。稍后当我们想要可视化即将设计的 Transformer 的内部工作机制时，这会派上用场。

```{.python .input  n=17}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if tab.selected('jax'):
                pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

作为一个快速的合理性检查，我们来可视化单位矩阵，它表示仅当查询与键相同时注意力权重才为$1$的情形。

```{.python .input  n=20}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

## 小结

注意力机制使我们能够聚合来自许多（键，值）对的数据。到目前为止，我们的讨论还相当抽象，只是描述了汇聚数据的一种方式。我们还没有解释这些神秘的查询、键和值可能来自哪里。这里一些直觉或许有帮助：例如，在回归设定中，查询可能对应于应当执行回归的位置。键是过去观测到数据的位置，而值是（回归的）值本身。这就是所谓的 Nadaraya--Watson 估计量 :cite:`Nadaraya.1964,Watson.1964`，我们将在下一节研究它。

按照设计，注意力机制提供了一种*可微*的控制手段，神经网络借此可以从集合中选择元素，并构造相关的表示加权和。

## 练习

1. 假设你想重新实现经典数据库中使用的近似（键，查询）匹配，你会选择哪个注意力函数？
1. 假设注意力函数为$a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$，且$\mathbf{k}_i = \mathbf{v}_i$对$i = 1, \ldots, m$成立。用$p(\mathbf{k}_i; \mathbf{q})$表示使用 :eqref:`eq_softmax_attention` 中的 softmax 规范化时在键上的概率分布。证明$\nabla_{\mathbf{q}} \mathop{\textrm{Attention}}(\mathbf{q}, \mathcal{D}) = \textrm{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i]$。
1. 使用注意力机制设计一个可微的搜索引擎。
1. 回顾 Squeeze and Excitation 网络 :cite:`Hu.Shen.Sun.2018` 的设计，并从注意力机制的角度对它们加以解读。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1710)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/18024)
:end_tab:
