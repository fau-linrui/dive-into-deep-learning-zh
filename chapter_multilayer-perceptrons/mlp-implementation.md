```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 多层感知机的实现
:label:`sec_mlp-implementation`

多层感知机（MLP）的实现并不比简单的线性模型复杂多少。关键的概念性
区别在于，我们现在把多个层连结（concatenate）起来。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
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
import jax
from jax import numpy as jnp
```

## 从零开始实现

让我们再次从零开始实现这样一个网络。

### 初始化模型参数

回想一下，Fashion-MNIST 包含 10 个类别，
而且每张图像由 $28 \times 28 = 784$
个灰度像素值组成的网格构成。
和之前一样，我们暂时忽略
像素之间的空间结构，
因此可以把这看作一个
具有 784 个输入特征和 10 个类别的分类数据集。
首先，我们将[**实现一个
含一个隐藏层、256 个隐藏单元的 MLP。**]
层数及其宽度都是可调的
（它们被视为超参数）。
通常，我们选择的层宽度是 2 的较大幂次的倍数。
由于内存分配和寻址方式在硬件中的特点，
这样做在计算上是高效的。

同样，我们将用几个张量来表示参数。
注意，*对于每一层*，我们都必须跟踪
一个权重矩阵和一个偏置向量。
和以往一样，我们为损失关于这些参数的梯度
分配内存。

:begin_tab:`mxnet`
在下面的代码中，我们先定义并初始化参数，
然后启用梯度跟踪。
:end_tab:

:begin_tab:`pytorch`
在下面的代码中，我们使用 `nn.Parameter`
把一个类属性自动注册为
由 `autograd` 跟踪的参数（:numref:`sec_autograd`）。
:end_tab:

:begin_tab:`tensorflow`
在下面的代码中，我们使用 `tf.Variable`
来定义模型参数。
:end_tab:

:begin_tab:`jax`
在下面的代码中，我们使用 `flax.linen.Module.param`
来定义模型参数。
:end_tab:

```{.python .input}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### 模型

为了确保我们明白一切是如何运作的，
我们将自己[**实现 ReLU 激活**]，
而不是直接调用内置的 `relu` 函数。

```{.python .input}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

由于我们忽略了空间结构，
我们用 `reshape` 把每张二维图像变成
长度为 `num_inputs` 的平坦向量。
最后，我们只用几行代码就(**实现了我们的模型**)。由于我们使用框架内置的自动微分，这么多代码就够了。

```{.python .input}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### 训练

幸运的是，[**MLP 的训练循环
与 softmax 回归完全一样。**]我们定义模型、数据和训练器，最后在模型和数据上调用 `fit` 方法。

```{.python .input}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 简洁实现

正如你可能预料的那样，借助高层 API，我们可以更简洁地实现 MLP。

### 模型

与我们简洁实现
softmax 回归时
（:numref:`sec_softmax_concise`）相比，
唯一的区别是：以前我们只添加*一个*全连接层，
而现在我们添加*两个*。
第一个是[**隐藏层**]，
第二个是输出层。

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

之前，我们为模型定义了 `forward` 方法，用模型参数来变换输入。
这些运算本质上是一条流水线：
你取一个输入，
施加一个变换（例如，
与权重做矩阵乘法，再加上偏置），
然后反复把当前变换的输出
作为下一个变换的输入。
然而，你可能已经注意到，
这里并没有定义 `forward` 方法。
事实上，`MLP` 从 `Module` 类继承了 `forward` 方法（:numref:`subsec_oo-design-models`），
它只是调用 `self.net(X)`（`X` 是输入），
而后者现在通过 `Sequential` 类
被定义为一连串变换。
`Sequential` 类抽象了前向过程，
使我们能够专注于各种变换。
我们将在 :numref:`subsec_model-construction-sequential` 中进一步讨论 `Sequential` 类是如何工作的。


### 训练

[**训练循环**]与我们实现 softmax 回归时
完全一样。
这种模块化使我们能够把
与模型架构相关的事项
与彼此正交的考虑分离开来。

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## 小结

现在我们在设计深度网络方面有了更多实践，从单层深度网络到多层深度网络的这一步，不再构成什么重大挑战。特别是，我们可以复用训练算法和数据加载器。不过要注意，从零开始实现 MLP 仍然很杂乱：给模型参数命名并跟踪它们，使得模型难以扩展。例如，设想一下在模型的第 42 层和第 43 层之间插入另一层。那它现在也许得叫第 42b 层，除非我们愿意按顺序重命名。此外，如果我们从零实现网络，框架就更难进行有意义的性能优化。

尽管如此，你现在已经达到了 20 世纪 80 年代末的最先进水平，当时全连接深度网络是神经网络建模的首选方法。我们接下来在概念上要迈出的一步是考虑图像。在此之前，我们需要回顾一些统计学基础，以及关于如何高效计算模型的细节。


## 练习

1. 改变隐藏单元数 `num_hiddens`，并画出它的数量如何影响模型的准确率。这个超参数的最佳取值是多少？
1. 试着增加一个隐藏层，看看它如何影响结果。
1. 为什么插入一个只有一个神经元的隐藏层是个坏主意？可能会出什么问题？
1. 改变学习率会如何改变你的结果？在所有其他参数固定的情况下，哪个学习率能给你最好的结果？这与轮数有什么关系？
1. 让我们对所有超参数联合进行优化，即学习率、轮数、隐藏层数以及每层的隐藏单元数。
    1. 对所有这些超参数进行优化，你能取得的最好结果是什么？
    1. 为什么处理多个超参数要困难得多？
    1. 描述一种联合优化多个参数的高效策略。
1. 在一个有挑战性的问题上，比较框架实现和从零实现的运行速度。它随网络复杂度的变化如何变化？
1. 测量对齐良好和未对齐的矩阵做张量--矩阵乘法的速度。例如，测试维度为 1024、1025、1026、1028 和 1032 的矩阵。
    1. 这在 GPU 和 CPU 之间有何不同？
    1. 确定你的 CPU 和 GPU 的内存总线宽度。
1. 尝试不同的激活函数。哪一个效果最好？
1. 网络的权重初始化不同有区别吗？这重要吗？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/227)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17985)
:end_tab:
