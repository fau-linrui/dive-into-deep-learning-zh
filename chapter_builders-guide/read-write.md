```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 读写文件

到目前为止，我们已经讨论了如何处理数据，
以及如何构建、训练和测试深度学习模型。
然而，总有一天，我们会庆幸自己
对学到的模型足够满意，
希望把结果保存下来，供以后在各种场景中使用
（甚至可能用于在部署中做预测）。
此外，在运行长时间的训练过程时，
最佳实践是定期保存中间结果（检查点），
以确保如果我们不小心绊到服务器的电源线，
也不会丢失几天的工作量。
因此，现在是时候学习如何加载和存储
单个权重向量和整个模型了。
本节将讨论这两个问题。

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**加载和保存张量**)

对于单个张量，我们可以直接
调用 `load` 和 `save` 函数
分别读写它们。
这两个函数都要求我们提供一个名称，
而 `save` 要求把要保存的变量作为输入。

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

现在我们可以把存储文件中的数据读回内存。

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

我们可以[**存储一个张量列表，并把它们读回内存。**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

我们甚至可以[**读写一个从字符串映射到张量的
字典。**]
当我们想要读取或写入模型中的所有
权重时，这会很方便。

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**加载和保存模型参数**]

保存单个权重向量（或其他张量）是有用的，
但如果我们想保存（并在以后加载）整个模型，
就会变得非常繁琐。
毕竟，我们可能有数百个
散布在各处的参数组。
因此，深度学习框架提供了内置功能
来加载和保存整个网络。
需要注意的一个重要细节是，
这保存的是模型*参数*，而不是整个模型。
例如，如果我们有一个3层MLP，
我们需要单独指定架构。
原因是模型本身可以包含任意代码，
因此无法自然地序列化。
因此，为了重建一个模型，我们需要
在代码中生成架构，
然后从磁盘加载参数。
(**让我们从熟悉的MLP开始。**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

接下来，我们[**把模型的参数存储为一个文件**]，文件名为“mlp.params”。

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

为了恢复模型，我们实例化
原始MLP模型的一个副本。
我们不是随机初始化模型参数，
而是[**直接读取文件中存储的参数**]。

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

由于两个实例具有相同的模型参数，
相同输入 `X` 的计算结果应该相同。
让我们来验证这一点。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## 小结

`save` 和 `load` 函数可用于对张量对象执行文件I/O。
我们可以通过网络参数字典来保存和加载整个网络的参数集。
保存架构必须在代码中完成，而不是在参数中完成。

## 练习

1. 即使不需要把训练好的模型部署到不同的设备上，存储模型参数有什么实际好处？
1. 假设我们只想复用网络的一部分，把它纳入一个具有不同架构的网络。你会如何把例如前一个网络的前两层，用到一个新网络中？
1. 你会如何保存网络架构和参数？你会对架构施加哪些限制？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17994)
:end_tab:
