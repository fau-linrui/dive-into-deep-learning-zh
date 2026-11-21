# 基于相似性的注意力汇聚

:label:`sec_attention-pooling`

既然我们已经介绍了注意力机制的主要组成部分，下面就让我们把它们用于一个相当经典的场景，即通过核密度估计 :cite:`Nadaraya.1964,Watson.1964` 进行回归和分类。这段绕路只是提供额外的背景知识：它完全是可选的，必要时可以跳过。
Nadaraya--Watson 估计量的核心在于依赖某种相似性核$\alpha(\mathbf{q}, \mathbf{k})$，它把查询$\mathbf{q}$与键$\mathbf{k}$联系起来。一些常见的核如下

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

我们还有很多可供选择的核。关于更全面的综述，以及核的选择如何与核密度估计（有时也称为*Parzen 窗*（Parzen Windows）:cite:`parzen1957consistent`）相关联，可参见[维基百科条目](https://en.wikipedia.org/wiki/Kernel_(statistics))。所有这些都是启发式核，并且可以调参。例如，我们不仅可以全局地调整宽度，甚至可以按坐标逐一调整宽度。无论如何，它们都会导出下式，对回归和分类同样适用：

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

在（标量）回归情形下，观测数据为特征和标签$(\mathbf{x}_i, y_i)$，此时$\mathbf{v}_i = y_i$是标量，$\mathbf{k}_i = \mathbf{x}_i$是向量，查询$\mathbf{q}$表示应当计算$f$的新位置。在（多类）分类情形下，我们对$y_i$使用独热编码来得到$\mathbf{v}_i$。该估计量的一个便利性质是它无需训练。更进一步，如果随着数据量增加而适当地收窄核，该方法是一致的 :cite:`mack1982weak`，即它会收敛到某种统计最优解。让我们先来考察一些核。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**核与数据**]

本节定义的所有核$\alpha(\mathbf{k}, \mathbf{q})$都是*平移和旋转不变的*；也就是说，如果我们以同样的方式平移和旋转$\mathbf{k}$和$\mathbf{q}$，$\alpha$的值保持不变。为简单起见，我们因此选取标量参数$k, q \in \mathbb{R}$，并选取键$k = 0$作为原点。这得到：

```{.python .input}
%%tab all
# Define some kernels
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

不同的核对应不同的范围和光滑程度概念。例如，Boxcar 核只关注距离在$1$（或按其他方式定义的某个超参数）以内的观测，而且不加区分地关注所有这些观测。

为了看看 Nadaraya--Watson 估计的实际效果，让我们定义一些训练数据。下面我们使用依赖关系

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

其中$\epsilon$从均值为零、方差为一的正态分布中抽取。我们抽取 40 个训练样例。

```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**通过 Nadaraya--Watson 回归进行注意力汇聚**]

现在有了数据和核，我们只需要一个计算核回归估计值的函数。请注意，为了进行一些简单的诊断，我们还想得到相对的核权重。因此，我们先计算所有训练特征（协变量）`x_train`与所有验证特征`x_val`之间的核。这会得到一个矩阵，随后我们对其规范化。当它与训练标签`y_train`相乘时，我们就得到了估计值。

回想 :eqref:`eq_attention_pooling` 中的注意力汇聚。令每个验证特征为一个查询，每个训练特征-标签对为一个键-值对。于是，规范化后的相对核权重（下面的`attention_w`）就是*注意力权重*。

```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Each column/row corresponds to each query/key
    k = d2l.astype(kernel(dists), d2l.float32)
    # Normalization over keys for each query
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

让我们看看不同核所产生的估计结果。

```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

首先引人注意的是，三个非平凡核（Gaussian、Boxcar 和 Epanechikov）都产生了相当可用的估计，与真实函数相差不太远。只有导致平凡估计$f(x) = \frac{1}{n} \sum_i y_i$的常数核给出了相当不切实际的结果。让我们更仔细地考察注意力权重：

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

可视化清楚地显示了为什么 Gaussian、Boxcar 和 Epanechikov 的估计非常相似：毕竟，尽管核的函数形式不同，它们却由非常相似的注意力权重导出。这引出了一个问题：是否总是如此。

## [**调整注意力汇聚**]

我们可以把高斯核换成宽度不同的核。也就是说，我们可以使用$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$，其中$\sigma^2$决定核的宽度。让我们看看这是否会影响结果。

```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

显然，核越窄，估计就越不平滑。与此同时，它也能更好地适应局部变化。让我们看看相应的注意力权重。

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

正如我们所料，核越窄，较大注意力权重的范围也越窄。同样清楚的是，选取相同的宽度可能并不理想。事实上，:citet:`Silverman86` 提出了一种依赖局部密度的启发式方法。人们还提出了更多这类“技巧”。例如，:citet:`norelli2022asif` 使用了类似的最近邻插值技术来设计跨模态的图像与文本表示。

敏锐的读者可能会想，我们为什么要对一个已有半个多世纪之久的方法做如此深入的探讨。第一，它是现代注意力机制最早的先驱之一。第二，它非常适合可视化。第三，同样重要的是，它展示了手工设计注意力机制的局限。一个更好的策略是通过学习查询和键的表示来*学习*这一机制。这正是我们将在后续小节中着手做的事情。


## 小结

Nadaraya--Watson 核回归是当前注意力机制的早期先驱。
它几乎不需要训练或调参就可以直接用于分类或回归。
注意力权重根据查询与键之间的相似性（或距离）来分配，同时也根据可用的相似观测的数量来分配。

## 练习

1. Parzen 窗密度估计由$\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$给出。证明：对于二元分类，由 Parzen 窗得到的函数$\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$等价于 Nadaraya--Watson 分类。
1. 实现随机梯度下降，以学习 Nadaraya--Watson 回归中核宽度的合适取值。
    1. 如果直接使用上述估计来最小化$(f(\mathbf{x_i}) - y_i)^2$会发生什么？提示：$y_i$是用于计算$f$的项的一部分。
    1. 把$(\mathbf{x}_i, y_i)$从$f(\mathbf{x}_i)$的估计中移除，然后对核宽度进行优化。你还会观察到过拟合吗？
1. 假设所有$\mathbf{x}$都位于单位球面上，即都满足$\|\mathbf{x}\| = 1$。你能简化指数中的$\|\mathbf{x} - \mathbf{x}_i\|^2$项吗？提示：我们稍后会看到，这与点积注意力密切相关。
1. 回想 :citet:`mack1982weak` 证明了 Nadaraya--Watson 估计是一致的。随着数据增多，你应该以多快的速度减小注意力机制的尺度？请为你的答案提供一些直觉解释。它依赖于数据的维数吗？如何依赖？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/3866)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/18026)
:end_tab:
