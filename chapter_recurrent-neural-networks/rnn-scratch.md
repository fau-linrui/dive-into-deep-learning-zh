# 循环神经网络的从零开始实现
:label:`sec_rnn-scratch`

现在，我们已经准备好从零开始实现一个循环神经网络（RNN）。
具体来说，我们将训练这个 RNN，使其充当字符级语言模型
（参见 :numref:`sec_rnn`），
并按照 :numref:`sec_text-sequence` 中概述的数据处理步骤，
在由 H. G. Wells 的*The Time Machine*（《时间机器》）全文组成的语料库上训练它。
我们首先加载数据集。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## RNN 模型

我们首先定义一个类来实现 RNN 模型
（:numref:`subsec_rnn_w_hidden_states`）。
请注意，隐藏单元数 `num_hiddens` 是一个可调的超参数。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """The RNN model implemented from scratch."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**下面的 `forward` 方法定义：给定当前输入和模型在上一时间步的状态，
如何计算任意时间步的输出和隐藏状态。**]
请注意，RNN 模型会循环遍历 `inputs` 的最外层维度，
一次一个时间步地更新隐藏状态。
这里的模型使用 $\tanh$ 激活函数（:numref:`subsec_tanh`）。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

我们可以像下面这样，向 RNN 模型输入一个小批量的输入序列。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

让我们检查一下 RNN 模型产生的结果的形状是否正确，
以确保隐藏状态的维度保持不变。

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## 基于 RNN 的语言模型

下面的 `RNNLMScratch` 类定义了一个基于 RNN 的语言模型，
我们通过 `__init__` 方法的 `rnn` 参数传入 RNN。
训练语言模型时，输入和输出都来自同一个词表。
因此，它们的维度相同，且都等于词表大小。
请注意，我们使用困惑度来评价模型。
正如 :numref:`subsec_perplexity` 中所讨论的，这样可以确保不同长度的序列具有可比性。

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**独热编码**]

回想一下，每个词元都由一个数值索引表示，
该索引指明对应的单词、字符或子词在词表中的位置。
你可能会想构建一个（在每个时间步）只有一个输入节点的神经网络，
把索引作为标量值输入。
当处理价格或温度之类的数值输入时，这种做法是可行的，
因为这些输入中任意两个足够接近的值都应被类似地对待。
但在这里，这种做法并不太合理。
我们词表中的第 $45^{\textrm{th}}$ 个词和第 $46^{\textrm{th}}$ 个词
恰好是“their”和“said”，二者的含义毫不相近。

处理这类类别数据时，最常见的策略是用*独热编码*（one-hot encoding）表示每个条目
（回想 :numref:`subsec_classification-problem`）。
独热编码是一个向量，其长度由词表大小 $N$ 决定：
除与当前词元对应的条目置为 $1$ 之外，其余所有条目都置为 $0$。
例如，若词表有 5 个元素，则索引 0 和 2 对应的独热向量如下所示。

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**我们在每次迭代中采样得到的小批量，形状为（批量大小，时间步数）。
一旦把每个输入表示为独热向量，我们就可以把每个小批量看作一个三维张量，
其中第三个轴的长度由词表大小（`len(vocab)`）给出。**)
我们通常对输入进行转置，从而得到形状为（时间步数，批量大小，词表大小）的输出。
这样，我们就可以更方便地遍历最外层维度，
逐个时间步地更新一个小批量的隐藏状态
（例如，在上面的 `forward` 方法中）。

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### 变换 RNN 输出

语言模型使用一个全连接输出层，
在每个时间步把 RNN 的输出变换为词元预测。

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

让我们[**检查前向计算产生的输出形状是否正确。**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**梯度裁剪**]

你可能已经习惯于把神经网络看作“深层”的，
因为即使在单个时间步内，也有许多层把输入和输出分隔开；
而序列的长度又引入了“深度”的新概念。
除了沿输入到输出的方向穿过网络之外，
第一个时间步的输入还必须沿时间步穿过 $T$ 层的链路，
才能影响模型在最后一个时间步的输出。
从反向的角度看，在每次迭代中，我们都通过时间反向传播梯度，
这会产生长度为 $\mathcal{O}(T)$ 的矩阵乘积链。
正如 :numref:`sec_numerical_stability` 中所提到的，
这可能导致数值不稳定，使梯度爆炸或消失，
具体取决于权重矩阵的性质。

处理梯度消失和梯度爆炸是设计 RNN 时的一个根本问题，
它也启发了现代神经网络架构中一些最大的进展。
在下一章中，我们将讨论一些专门的架构，
它们的设计初衷就是缓解梯度消失问题。
然而，即使是现代的 RNN，也常常受到梯度爆炸的困扰。
一种不优雅但普遍存在的解决方案，就是简单地裁剪梯度，
迫使得到的“裁剪后”梯度取较小的值。

一般来说，当用梯度下降优化某个目标时，
我们会迭代地更新感兴趣的参数，比如向量 $\mathbf{x}$，
但更新是把它推向负梯度 $\mathbf{g}$ 的方向
（在随机梯度下降中，我们在随机采样的小批量上计算这个梯度）。
例如，在学习率 $\eta > 0$ 下，每次更新的形式为
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$。
我们进一步假设目标函数 $f$ 足够光滑。
形式化地说，如果目标函数是*利普希茨连续*（Lipschitz continuous）的，常数为 $L$，
就意味着对任意 $\mathbf{x}$ 和 $\mathbf{y}$，都有

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

可以看到，当我们通过减去 $\eta \mathbf{g}$ 来更新参数向量时，
目标函数值的变化取决于学习率、梯度的范数以及 $L$，具体如下：

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

换句话说，目标函数的变化不能超过 $L \eta \|\mathbf{g}\|$。
这个上界取较小值，既可能被视为好事，也可能被视为坏事。
不利的一面是，我们限制了目标函数值下降的速度；
有利的一面是，它限制了我们在任何一次梯度步骤中可能犯下多大错误。

当我们说梯度爆炸时，意思是指 $\|\mathbf{g}\|$ 变得过大。
在这种最坏情况下，我们在单次梯度步骤中可能造成巨大破坏，
以至于抵消掉数千次训练迭代中取得的所有进展。
当梯度可能如此之大时，神经网络训练往往会发散，
无法降低目标函数值。
在另一些情况下，训练最终会收敛，
但由于损失出现大幅尖峰而变得不稳定。

限制 $L \eta \|\mathbf{g}\|$ 大小的一种方法，是把学习率 $\eta$ 缩小到极小值。
这样做的好处是不会使更新产生偏差。
但如果我们只是*很少*遇到大梯度呢？
这种极端做法会在所有步骤上拖慢我们的进展，
仅仅是为了应对罕见的梯度爆炸事件。
一种流行的替代方案是采用*梯度裁剪*（gradient clipping）启发式，
把梯度 $\mathbf{g}$ 投影到给定半径 $\theta$ 的球上，如下所示：

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

这确保了梯度的范数永远不会超过 $\theta$，
并且更新后的梯度与原来的 $\mathbf{g}$ 方向完全一致。
它还有一个理想的副作用：
限制任意给定小批量（以及其中的任意给定样本）对参数向量可能施加的影响。
这赋予了模型一定程度的鲁棒性。
需要明确的是，这是一种技巧。
梯度裁剪意味着我们并不总是沿着真实梯度前进，
并且很难分析可能产生的副作用。
然而，它是一种非常有用的技巧，
并且被大多数深度学习框架中的 RNN 实现广泛采用。

下面我们定义一个裁剪梯度的方法，
它由 `d2l.Trainer` 类的 `fit_epoch` 方法调用
（参见 :numref:`sec_linear_scratch`）。
请注意，在计算梯度范数时，
我们将所有模型参数连接起来，把它们当作一个巨大的参数向量。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## 训练

我们使用*The Time Machine*数据集（`data`），
训练一个基于从零实现的 RNN（`rnn`）的字符级语言模型（`model`）。
请注意，我们先计算梯度，然后裁剪梯度，
最后使用裁剪后的梯度更新模型参数。

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## 解码

语言模型训练好之后，我们不仅可以用来预测下一个词元，
还可以继续逐个预测后续的每一个词元，
把之前预测出的词元当作输入中的下一个词元。
有时我们只想从文档开头开始生成文本。
然而，通常更有用的是让语言模型以用户提供的前缀为条件。
例如，如果我们在为搜索引擎开发自动补全功能，
或者帮助用户撰写电子邮件，
我们就会希望把用户到目前为止写下的内容（前缀）输入模型，
然后生成一个可能的后续内容。

[**下面的 `predict` 方法在接收用户提供的 `prefix` 之后，一次一个字符地生成后续内容**]。
在循环遍历 `prefix` 中的字符时，我们不断把隐藏状态传递到下一个时间步，
但不产生任何输出。
这称为*预热*（warm-up）期。
在接收完前缀之后，我们就准备好开始输出后续字符了，
其中每个字符都会作为下一个时间步的输入反馈给模型。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

在下面，我们指定前缀，让它额外生成 20 个字符。

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

虽然从零实现上面的 RNN 模型很有启发性，但并不方便。
在下一节中，我们将看到如何利用深度学习框架，
用标准架构快速构建 RNN，
并依靠高度优化的库函数来获得性能提升。

## 小结

我们可以训练基于 RNN 的语言模型，让它按照用户提供的文本前缀生成文本。
一个简单的 RNN 语言模型由输入编码、RNN 建模和输出生成三部分组成。
在训练过程中，梯度裁剪可以缓解梯度爆炸问题，
但并不能解决梯度消失问题。
在实验中，我们实现了一个简单的 RNN 语言模型，
并在按字符级词元化的文本序列上，用梯度裁剪训练它。
以某个前缀为条件，我们可以用语言模型生成可能的后续内容，
这在许多应用中都被证明很有用，例如自动补全功能。

## 练习

1. 所实现的语言模型是否基于*The Time Machine*中直到第一个词元为止的所有过去词元来预测下一个词元？
1. 哪个超参数控制用于预测的历史长度？
1. 证明独热编码等价于为每个对象挑选一个不同的嵌入。
1. 调整超参数（例如轮数、隐藏单元数、小批量中的时间步数以及学习率）来改善困惑度。在坚持使用这种简单架构的前提下，你能把它降到多低？
1. 用可学习的嵌入替换独热编码。这会带来更好的性能吗？
1. 做实验，确定这个在*The Time Machine*上训练的语言模型在 H. G. Wells 的其他书（例如*The War of the Worlds*）上表现如何。
1. 再做另一个实验，评价这个模型在其他作者所写书籍上的困惑度。
1. 修改预测方法，使其使用采样，而不是挑选最可能的下一个字符。
    * 会发生什么？
    * 让模型偏向更可能的输出，例如从 $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ 中采样，其中 $\alpha > 1$。
1. 运行本节代码但不对梯度进行裁剪。会发生什么？
1. 把本节使用的激活函数替换为 ReLU，并重复本节的实验。我们还需要梯度裁剪吗？为什么？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/18014)
:end_tab:
