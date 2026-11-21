```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 参数初始化

既然我们已经知道如何访问参数，
接下来看看如何正确地初始化它们。
我们在 :numref:`sec_numerical_stability` 中讨论过正确初始化的必要性。
深度学习框架为其各层提供了默认的随机初始化。
然而，我们常常希望按照各种其他方案来初始化权重。
框架提供了最常用的方案，
也允许创建自定义的初始化器。

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

:begin_tab:`mxnet`
默认情况下，MXNet 通过从均匀分布 $U(-0.07, 0.07)$ 中随机抽取来初始化权重参数，
并把偏置参数置零。
MXNet 的 `init` 模块提供了多种
预置的初始化方法。
:end_tab:

:begin_tab:`pytorch`
默认情况下，PyTorch 通过从根据输入和输出维度计算出的范围中均匀抽取，
来初始化权重矩阵和偏置矩阵。
PyTorch 的 `nn.init` 模块提供了多种
预置的初始化方法。
:end_tab:

:begin_tab:`tensorflow`
默认情况下，Keras 通过从根据输入和输出维度计算出的范围中均匀抽取来初始化权重矩阵，偏置参数则全部置零。
TensorFlow 在根模块和 `keras.initializers` 模块中都提供了多种初始化方法。
:end_tab:

:begin_tab:`jax`
默认情况下，Flax 使用 `jax.nn.initializers.lecun_normal` 初始化权重，
即从以0为中心、标准差设为 $1 / \textrm{fan}_{\textrm{in}}$ 的平方根的截断正态分布中抽取样本，
其中 `fan_in` 是权重张量中输入单元的数量。偏置
参数全部置零。
JAX 的 `nn.initializers` 模块提供了多种
预置的初始化方法。
:end_tab:

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])
X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**内置初始化**]

让我们从调用内置初始化器开始。
下面的代码把所有可调权重参数
初始化为标准差为0.01的高斯随机变量，
同时把偏置参数清零。

```{.python .input}
%%tab mxnet
# Here force_reinit ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

我们也可以把所有参数
初始化为某个给定的常数（比如1）。

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**我们还可以对某些块应用不同的初始化器。**]
例如，下面我们用 Xavier 初始化器
初始化第一层，
并把第二层初始化为
常数42。

```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**自定义初始化**]

有时，我们需要的初始化方法
并未由深度学习框架提供。
在下面的例子中，我们为任意权重参数 $w$
定义一个使用如下奇怪分布的初始化器：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
这里我们定义 `Initializer` 类的一个子类。
通常，我们只需要实现 `_init_weight` 函数，
它接受一个张量参数（`data`），
并把期望的初始值赋给它。
:end_tab:

:begin_tab:`pytorch`
同样，我们实现一个 `my_init` 函数并应用到 `net` 上。
:end_tab:

:begin_tab:`tensorflow`
这里我们定义 `Initializer` 的一个子类，并实现 `__call__`
函数，它在给定形状和数据类型时返回期望的张量。
:end_tab:

:begin_tab:`jax`
JAX 初始化函数接受 `PRNGKey`、`shape` 和
`dtype` 作为参数。这里我们实现函数 `my_init`，
它在给定形状和数据类型时返回期望的张量。
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
请注意，我们总是可以选择
直接设置参数。
:end_tab:

:begin_tab:`jax`
在 JAX 和 Flax 中初始化参数时，返回的参数字典
具有 `flax.core.frozen_dict.FrozenDict` 类型。在 JAX 生态系统中
不建议直接修改数组的值，
因此这些数据类型通常是不可变的。可以使用 `params.unfreeze()` 来进行修改。
:end_tab:

```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## 小结

我们可以使用内置初始化器和自定义初始化器来初始化参数。

## 练习

查阅在线文档以了解更多内置初始化器。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/8091)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17991)
:end_tab:
