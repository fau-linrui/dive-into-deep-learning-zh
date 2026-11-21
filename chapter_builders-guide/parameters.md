```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 参数管理

一旦我们选好了架构
并设置好超参数，
就进入训练循环，
目标是找到使损失函数最小化的参数值。
训练之后，我们需要这些参数
来做出未来的预测。
此外，有时我们还希望
提取这些参数，
或许是为了在其他场景中复用它们，
为了把模型保存到磁盘
以便在其他软件中执行，
或者为了检查它们，
以期获得科学上的理解。

大多数时候，我们能够
忽略参数如何声明和操纵的
繁琐细节，
依赖深度学习框架来完成繁重的工作。
然而，当我们离开由标准层堆叠而成的架构时，
有时就需要深入参数声明与操纵的细节。
在本节中，我们讨论以下内容：

* 为调试、诊断和可视化而访问参数。
* 在不同模型组件之间共享参数。

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

(**我们先关注一个带有一个隐藏层的MLP。**)

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
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

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

## [**参数访问**]
:label:`subsec_param-access`

让我们从如何访问你已经熟悉的模型中的参数开始。

:begin_tab:`mxnet, pytorch, tensorflow`
当模型通过 `Sequential` 类定义时，
我们可以先像访问列表一样通过索引
访问任意层。
每一层的参数都方便地
位于其属性中。
:end_tab:

:begin_tab:`jax`
你可能已经在前面的模型定义中观察到，Flax 和 JAX 将模型与参数解耦。
当模型通过 `Sequential` 类定义时，
我们首先需要初始化网络，以生成
参数字典。我们可以通过这个字典的键
访问任意层的参数。
:end_tab:

我们可以如下检查第二个全连接层的参数。

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

可以看到，这个全连接层
包含两个参数，
分别对应这一层的
权重和偏置。


### [**目标参数**]

请注意，每个参数都表示
为参数类的实例。
要对参数做任何有用的事情，
我们首先需要访问底层的数值。
有几种方法可以做到这一点。
有些更简单，有些更通用。
下面的代码从第二个神经网络层中提取偏置，
它返回一个参数类实例，
并进一步访问该参数的值。

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
参数是复杂的对象，
包含值、梯度
以及其他信息。
这就是为什么我们需要显式地请求值。

除了值之外，每个参数还允许我们访问梯度。由于我们还没有对这个网络调用反向传播，所以梯度处于初始状态。
:end_tab:

:begin_tab:`jax`
与其他框架不同，JAX 不跟踪神经网络参数上的梯度，
而是把参数与网络解耦。
它允许用户把计算表达为一个 Python 函数，
并用 `grad` 变换来实现同样的目的。
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**一次性访问所有参数**]

当我们需要对所有参数执行操作时，
逐个访问它们会变得很繁琐。
当我们处理更复杂的（例如嵌套的）模块时，
情况会变得尤其难以应付，
因为我们需要递归地遍历整棵树
来提取每个子模块的参数。下面我们演示如何访问所有层的参数。

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**参数绑定**]

通常，我们希望跨多个层共享参数。
让我们看看如何优雅地做到这一点。
在下面，我们分配一个全连接层，
然后专门使用它的参数
来设置另一个层的参数。
这里我们需要先运行前向传播
`net(X)`，然后再访问参数。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

这个例子表明，
第二层和第三层的参数是绑定的。
它们不仅相等，而且
由同一个确切的张量表示。
因此，如果我们改变其中一个参数，
另一个也会随之改变。

:begin_tab:`mxnet, pytorch, tensorflow`
你可能会问，
当参数被绑定时，
梯度会怎样？
由于模型参数包含梯度，
在反向传播时，第二个隐藏层和
第三个隐藏层的梯度会相加。
:end_tab:


## 小结

我们有几种访问和绑定模型参数的方法。


## 练习

1. 使用 :numref:`sec_model_construction` 中定义的 `NestMLP` 模型，访问各个层的参数。
1. 构造一个包含共享参数层的MLP并训练它。在训练过程中，观察每一层的模型参数和梯度。
1. 为什么共享参数是一个好主意？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17990)
:end_tab:
