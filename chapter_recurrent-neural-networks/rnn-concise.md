# 循环神经网络的简洁实现
:label:`sec_rnn-concise`

与大多数从零开始的实现一样，:numref:`sec_rnn-scratch` 的设计目的
是让我们深入了解每个组件是如何工作的。
但是，当你每天都要使用 RNN 或者编写生产代码时，
你会希望更多地依赖库：
这些库既能减少实现时间（为常见模型和函数提供库代码），
又能减少计算时间（对这些库实现进行极致的优化）。
本节将展示如何利用深度学习框架提供的高级 API，
更高效地实现同一个语言模型。
与之前一样，我们首先加载*The Time Machine*数据集。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
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
from jax import numpy as jnp
```

## [**定义模型**]

我们使用高级 API 实现的 RNN 来定义下面的类。

:begin_tab:`mxnet`
具体来说，为了初始化隐藏状态，
我们调用成员方法 `begin_state`。
它返回一个列表，列表中含有小批量中每个样例的初始隐藏状态，
其形状为（隐藏层数，批量大小，隐藏单元数）。
对于之后要介绍的一些模型（例如长短期记忆网络），
这个列表还会包含其他信息。
:end_tab:

:begin_tab:`jax`
截至目前，Flax 还没有提供用于简洁实现普通 RNN 的 RNNCell。
对于 LSTM 和 GRU 等更高级的 RNN 变体，
则可以使用 Flax 的 `linen` API。
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

下面的 `RNNLM` 类继承 :numref:`sec_rnn-scratch` 中的 `RNNLMScratch` 类，
定义了一个完整的基于 RNN 的语言模型。
请注意，我们需要单独创建一个全连接输出层。

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## 训练与预测

在训练模型之前，让我们[**用一个随机初始化权重的模型进行预测。**]
由于我们还没有训练网络，它生成的预测会毫无意义。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

接下来，我们[**利用高级 API 训练模型**]。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

与 :numref:`sec_rnn-scratch` 相比，
这个模型达到了相当的困惑度，
但由于实现经过了优化，运行速度更快。
与之前一样，我们可以按照指定的前缀字符串生成预测的词元。

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## 小结

深度学习框架的高级 API 提供了标准 RNN 的实现。
这些库帮助我们避免浪费时间重新实现标准模型。
此外，框架中的实现通常经过高度优化，
与从零实现的版本相比，可以带来显著的（计算）性能提升。

## 练习

1. 你能用高级 API 使这个 RNN 模型过拟合吗？
1. 用 RNN 实现 :numref:`sec_sequence` 中的自回归模型。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/18015)
:end_tab:
