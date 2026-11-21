```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 自定义层

深度学习成功背后的一个因素，
是存在种类繁多的层，
它们可以以创造性的方式组合，
来设计适合各种任务的架构。
例如，研究人员发明了专门用于处理图像、文本、
遍历序列数据
以及执行动态规划的层。
迟早你会需要
一个深度学习框架中尚不存在的层。
在这些情况下，你必须构建一个自定义层。
在本节中，我们将向你展示如何做到这一点。

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
import jax
from jax import numpy as jnp
```

## (**不带参数的层**)

作为开始，我们构造一个
本身没有任何参数的自定义层。
如果你还记得我们在 :numref:`sec_model_construction` 中对模块的介绍，
这应该看起来很熟悉。
下面的 `CenteredLayer` 类只是
从它的输入中减去均值。
要构建它，我们只需继承
基类层并实现前向传播函数。

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

我们向层中传入一些数据，
来验证它按预期工作。

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

现在我们可以[**把我们的层作为一个组件
纳入更复杂模型的构造中。**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

作为一个额外的健全性检查，我们可以把随机数据
传入网络，并检查均值是否确实为0。
由于我们处理的是浮点数，受量化影响，
我们可能仍会看到一个非常小的非零数。

:begin_tab:`jax`
这里我们使用 `init_with_output` 方法，它同时返回网络的输出
以及参数。在这种情况下，我们只关注
输出。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**带参数的层**]

既然我们知道了如何定义简单的层，
接下来我们定义带有可以通过训练调整的
参数的层。
我们可以使用内置函数来创建参数，
这些函数提供了一些基本的日常管理功能。
特别是，它们管理参数的访问、初始化、
共享、保存和加载。
这样，除了其他好处之外，我们无需
为每个自定义层编写自定义的序列化程序。

现在让我们实现自己版本的全连接层。
回想一下，这个层需要两个参数，
一个表示权重，另一个表示偏置。
在这个实现中，我们内置了ReLU激活作为默认。
这个层需要两个输入参数：`in_units` 和 `units`，
分别表示输入和输出的数量。

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
接下来，我们实例化 `MyDense` 类
并访问其模型参数。
:end_tab:

:begin_tab:`pytorch`
接下来，我们实例化 `MyLinear` 类
并访问其模型参数。
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

我们可以[**直接使用自定义层执行前向传播计算。**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

我们还可以(**使用自定义层构造模型。**)
一旦有了它，我们就可以像使用内置全连接层一样使用它。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## 小结

我们可以通过基本的层类来设计自定义层。这让我们可以定义灵活的新层，其行为与库中任何现有的层都不同。
一旦定义好，自定义层就可以在任意的上下文和架构中调用。
层可以拥有局部参数，这些参数可以通过内置函数创建。


## 练习

1. 设计一个层，它接收输入并计算一种张量缩减，
   即返回 $y_k = \sum_{i, j} W_{ijk} x_i x_j$。
1. 设计一个层，返回数据的傅里叶系数中前一半。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17993)
:end_tab:
