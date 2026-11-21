```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 延迟初始化
:label:`sec_lazy_init`

到目前为止，我们似乎一直
在搭建网络时马虎从事而没出什么问题。
具体来说，我们做了以下反直觉的事情，
它们看起来似乎不该行得通：

* 我们定义了网络架构，
  却没有指定输入维度。
* 我们添加了层，却没有指定
  前一层的输出维度。
* 我们甚至在提供足够的信息来确定
  模型应包含多少参数之前，
  就“初始化”了这些参数。

你可能会惊讶，我们的代码竟然能运行。
毕竟，深度学习框架无法知道
一个网络的输入维度会是多少。
这里的技巧在于，框架会*推迟初始化*，
一直等到我们第一次把数据传入模型时，
才动态地推断每一层的大小。


稍后，当我们处理卷积神经网络时，
这一技术会变得更加方便，
因为输入维度
（例如图像的分辨率）
会影响后续每一层的维度。
因此，在编写代码时无需知道维度值
就能设置参数，
这可以大大简化指定
以及随后修改模型的任务。
接下来，我们深入探讨初始化的机制。

```{.python .input}
%%tab mxnet
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

作为开始，让我们实例化一个MLP。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

此时，网络不可能知道
输入层权重的维度，
因为输入维度仍然未知。

:begin_tab:`mxnet, pytorch, tensorflow`
因此，框架还没有初始化任何参数。
下面我们尝试访问参数来确认这一点。
:end_tab:

:begin_tab:`jax`
如 :numref:`subsec_param-access` 所述，在 JAX 和 Flax 中，
参数与网络定义是解耦的，二者都由用户手动处理。
Flax 模型是无状态的，
因此没有 `parameters` 属性。
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
请注意，虽然参数对象存在，
但每一层的输入维度都列为 -1。
MXNet 使用特殊值 -1 来表示
参数维度仍然未知。
此时，尝试访问 `net[0].weight.data()`
会触发运行时错误，提示必须先初始化网络，
才能访问参数。
现在让我们看看，当我们尝试通过 `initialize` 方法
初始化参数时会发生什么。
:end_tab:

:begin_tab:`tensorflow`
请注意，每个层对象都存在，但权重是空的。
使用 `net.get_weights()` 会抛出一个错误，
因为权重尚未初始化。
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
如我们所见，什么也没有改变。
当输入维度未知时，
调用 initialize 并不会真正初始化参数。
相反，这个调用只是向 MXNet 登记：
我们希望（并且可选地，按照哪种分布）
初始化参数。
:end_tab:

接下来，让我们把数据传入网络，
让框架最终初始化参数。

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

一旦我们知道了输入维度，
即 20，
框架就能通过代入 20 这个值
确定第一层权重矩阵的形状。
识别出第一层的形状后，框架会继续
处理第二层，
如此沿着计算图继续下去，
直到所有形状都已知。
请注意，在这种情况下，
只有第一层需要延迟初始化，
但框架会按顺序初始化。
一旦所有参数形状都已知，
框架最终就能初始化参数。

:begin_tab:`pytorch`
下面的方法
通过网络传入虚拟输入，
进行一次空运行，
以推断所有参数形状，
并随后初始化参数。
当不希望使用默认随机初始化时，
就会用到它。
:end_tab:

:begin_tab:`jax`
Flax 中的参数初始化总是手动完成的，
并由用户来处理。下面的方法接受一个虚拟输入和一个键字典作为参数。
这个键字典包含用于初始化模型参数的 rng，
以及为带暂退层的模型生成暂退掩码的 dropout rng。
关于暂退法的更多内容将在后面的 :numref:`sec_dropout` 中介绍。
最终，这个方法会初始化模型并返回参数。
我们在前面的小节中也一直在底层使用它。
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## 小结

延迟初始化可以很方便，它让框架自动推断参数形状，从而使修改架构变得容易，并消除一个常见的错误来源。
我们可以把数据传入模型，让框架最终初始化参数。


## 练习

1. 如果你为第一层指定了输入维度，但没有为后续层指定，会发生什么？你会得到立即初始化吗？
1. 如果你指定了不匹配的维度，会发生什么？
1. 如果你有维度可变的输入，需要做什么？提示：看看参数绑定。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17992)
:end_tab:
