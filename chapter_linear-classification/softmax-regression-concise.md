```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# softmax 回归的简洁实现
:label:`sec_softmax_concise`



正如高级深度学习框架
使线性回归的实现变得更加容易
（参见 :numref:`sec_linear_concise`），
它们在这里同样方便。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## 定义模型

与 :numref:`sec_linear_concise` 中一样，
我们用内置层来构造全连接层。
当我们需要把网络应用于某些输入时，
内置的`__call__`方法就会调用`forward`。

:begin_tab:`mxnet`
尽管输入`X`是一个四阶张量，
内置的`Dense`层
仍会自动把`X`转换为二阶张量，
同时保持沿第一个轴的维数不变。
:end_tab:

:begin_tab:`pytorch`
我们使用`Flatten`层，
在保持沿第一个轴的维数不变的情况下，
把四阶张量`X`转换为二阶张量。

:end_tab:

:begin_tab:`tensorflow`
我们使用`Flatten`层，
在保持沿第一个轴的维数不变的情况下
转换四阶张量`X`。
:end_tab:

:begin_tab:`jax`
Flax 允许用户使用`@nn.compact`装饰器
以更紧凑的方式编写网络类。
借助`@nn.compact`，用户
可以直接把所有网络逻辑写在一个“前向传播”
方法中，
而不需要在数据类中定义标准的`setup`方法。
:end_tab:

```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## 重新审视 softmax
:label:`subsec_softmax-implementation-revisited`

在 :numref:`sec_softmax_scratch` 中，我们计算了模型的输出
并应用了交叉熵损失。虽然这在数学上完全合理，
但在计算上却存在风险，因为
取指数时会出现数值下溢和上溢。

回想一下，softmax 函数通过
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ 计算概率。
如果某些 $o_k$ 非常大，即非常正，
那么 $\exp(o_k)$ 可能大于
某些数据类型所能表示的最大数。
这称为*上溢*（overflow）。
同样，如果每个参数都是非常大的负数，
我们就会得到*下溢*（underflow）。
例如，单精度浮点数大约
覆盖 $10^{-38}$ 到 $10^{38}$ 的范围。
因此，如果 $\mathbf{o}$ 中最大的项
位于区间 $[-90, 90]$ 之外，结果就不会稳定。
绕过这个问题的一种方法是
从所有项中减去 $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$：

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

根据构造我们知道，对所有 $j$ 都有 $o_j - \bar{o} \leq 0$。
因此，对于 $q$ 类
分类问题，分母落在区间 $[1, q]$ 内。
此外，分子永远不会超过 $1$，
从而避免了数值上溢。
数值下溢只会在
$\exp(o_j - \bar{o})$ 的数值计算结果为 $0$ 时发生。
然而，在后续的某一步中，
当我们想要计算 $\log \hat{y}_j$ 时，
可能会因为 $\log 0$ 而陷入麻烦。
具体来说，在反向传播中，
我们可能会发现自己面对
满屏可怕的`NaN`（Not a Number）结果。

幸运的是，尽管我们计算的是指数函数，
但最终打算对它们取对数
（在计算交叉熵损失时），
这一事实拯救了我们。
通过把 softmax 和交叉熵结合起来，
我们可以完全避开数值稳定性问题。我们有：

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

这同时避免了上溢和下溢。
我们仍会希望把常规的 softmax 函数留在手边，
以备我们想要评估模型输出的概率时使用。
但我们不再把 softmax 概率传入新的损失函数，
而是[**直接传入 logit，
并在交叉熵损失函数内部一次性计算 softmax 及其对数，**]
这样做会采用诸如 [“LogSumExp 技巧”](https://en.wikipedia.org/wiki/LogSumExp) 之类的巧妙做法。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # To be used later (e.g., for batch norm)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## 训练

接下来我们训练模型。我们使用展平为 784 维特征向量的 Fashion-MNIST 图像。

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

和之前一样，该算法收敛到一个
相当准确的解，
只不过这次使用的代码行数比之前更少。


## 小结

高级 API 非常便于向用户隐藏潜在的危险方面，例如数值稳定性。此外，它们允许用户用很少的代码行数简洁地设计模型。这既是福也是祸。显而易见的好处是，它让这些内容变得高度易用，甚至对从未修过一门统计学课程的工程师也是如此（事实上，他们正是本书的目标读者之一）。但隐藏那些尖锐的棱角也是有代价的：这会抑制你自行添加新的、不同的组件，因为你对这样做缺乏肌肉记忆。此外，每当框架的保护性填充无法完全覆盖所有边角情况时，它也会让你更难*修复*问题。这同样是由于缺乏熟悉度。

因此，我们强烈建议你回顾后续许多实现的*两个*版本：从零实现的基础版和优雅版。虽然我们强调易于理解，但这些实现通常仍然具有相当好的性能（卷积是这里的一大例外）。我们的用意是让你在发明出任何框架都无法提供的新东西时，能够以这些实现为基础。


## 练习

1. 深度学习使用许多不同的数字格式，包括 FP64 双精度（极少使用）、
FP32 单精度、BFLOAT16（适合压缩表示）、FP16（非常不稳定）、TF32（NVIDIA 推出的一种新格式）以及 INT8。请计算指数函数在不导致数值下溢或上溢的情况下所能接受的最小和最大参数。
1. INT8 是一种非常受限的格式，由 $1$ 到 $255$ 的非零数组成。如何在不使用更多比特的情况下扩展其动态范围？标准的乘法和加法还能正常工作吗？
1. 增加训练的轮数。为什么验证准确率可能在一段时间后下降？我们该如何修正这一点？
1. 增大学习率时会发生什么？比较若干学习率下的损失曲线。哪一个效果更好？在什么时候？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/260)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17983)
:end_tab:
