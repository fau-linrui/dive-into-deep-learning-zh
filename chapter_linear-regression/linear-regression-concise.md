```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 线性回归的简洁实现
:label:`sec_linear_concise`

在过去十年里，深度学习经历了某种意义上的寒武纪大爆发。
各种技术、应用和算法的数量远远超过了此前数十年的进展。
这得益于多种因素的偶然结合，其中之一便是众多开源深度学习框架所提供的强大免费工具。
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`、
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`
和 Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`
可以说是第一代获得广泛采用的此类模型。
与更早的（开创性）工作相比，例如提供类 Lisp 编程体验的
SN2（Simulateur Neuristique）:cite:`Bottou.Le-Cun.1988`，
现代框架提供了自动微分以及 Python 带来的便利。
这些框架使我们能够将实现基于梯度的学习算法这类重复性工作自动化、模块化。

在 :numref:`sec_linear_scratch` 中，我们仅仅依赖：
(i) 用于数据存储和线性代数的张量；
(ii) 用于计算梯度的自动微分。
在实践中，由于数据迭代器、损失函数、优化器和神经网络层都如此常见，
现代库也为我们实现了这些组件。
本节中，(**我们将展示如何实现**):numref:`sec_linear_scratch` 中的线性回归模型，
(**即借助深度学习框架的高级 API 简洁地实现**)。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## 定义模型

当我们在 :numref:`sec_linear_scratch` 中从零实现线性回归时，
我们显式地定义了模型参数，
并编写了利用基本线性代数运算生成输出的计算过程。
你*应该*知道怎样做这件事。
但是，一旦模型变得更加复杂，一旦你几乎每天都要做这件事，
你就会乐于得到这样的帮助。
这种情形类似于从零编写自己的博客：
做一两次既有收获又有教益，
但如果你花一个月时间去重新发明轮子，
那就算不上一个好的 Web 开发者了。

对于标准操作，
我们可以[**使用框架预定义的层**]，
这样就能把注意力集中在用于构建模型的层上，
而不必担心它们的实现。
回想一下 :numref:`fig_single_neuron` 中描述的单层网络架构。
该层被称为*全连接层*（fully connected layer），
因为它的每个输入都通过矩阵-向量乘法与每个输出相连。

:begin_tab:`mxnet`
在 Gluon 中，全连接层定义在`Dense`类中。
由于我们只想生成单个标量输出，
所以将该数字设为 1。
值得注意的是，为了方便起见，
Gluon 不要求我们指定每一层的输入形状。
因此，我们不需要告诉 Gluon
有多少个输入进入这个线性层。
当我们第一次将数据传入模型时，
例如稍后执行`net(X)`时，
Gluon 将自动推断每一层的输入数量，
从而实例化出正确的模型。
我们将在后面更详细地描述它的工作原理。
:end_tab:

:begin_tab:`pytorch`
在 PyTorch 中，全连接层定义在`Linear`和`LazyLinear`类中（自 1.8.0 版本起可用）。
后者允许用户*仅仅*指定输出维度，
而前者还会要求指定该层有多少个输入。
指定输入形状并不方便，而且可能需要繁琐的计算
（例如在卷积层中）。
因此，为简单起见，只要能够使用，我们就会使用这种“惰性”层。
:end_tab:

:begin_tab:`tensorflow`
在 Keras 中，全连接层定义在`Dense`类中。
由于我们只想生成单个标量输出，
所以将该数字设为 1。
值得注意的是，为了方便起见，
Keras 不要求我们指定每一层的输入形状。
我们不需要告诉 Keras
有多少个输入进入这个线性层。
当我们第一次尝试将数据传入模型时，
例如稍后执行`net(X)`时，
Keras 将自动推断每一层的输入数量。
我们将在后面更详细地描述它的工作原理。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

在`forward`方法中，我们只是调用预定义层内置的`__call__`方法来计算输出。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## 定义损失函数

:begin_tab:`mxnet`
`loss`模块定义了许多有用的损失函数。
为了速度和便利，我们放弃自己实现，
转而选择内置的`loss.L2Loss`。
由于它返回的`loss`是每个样例的平方误差，
因此我们使用`mean`在小批量上对损失求平均。
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss`类计算均方误差（不含 :eqref:`eq_mse` 中的 $1/2$ 因子）。**]
默认情况下，`MSELoss`返回各样例上的平均损失。
它比我们自己实现更快（也更容易使用）。
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError`类计算均方误差（不含 :eqref:`eq_mse` 中的 $1/2$ 因子）。
默认情况下，它返回各样例上的平均损失。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## 定义优化算法

:begin_tab:`mxnet`
小批量随机梯度下降是优化神经网络的标准工具，
因此 Gluon 通过其`Trainer`类支持该算法及其多种变体。
请注意，Gluon 的`Trainer`类代表优化算法，
而我们在 :numref:`sec_oo-design` 中创建的`Trainer`类包含训练方法，
即重复调用优化器来更新模型参数。
当我们实例化`Trainer`时，
我们指定待优化的参数（可通过`net.collect_params()`从模型`net`中获得）、
我们希望使用的优化算法（`sgd`），
以及优化算法所需的超参数字典。
:end_tab:

:begin_tab:`pytorch`
小批量随机梯度下降是优化神经网络的标准工具，
因此 PyTorch 在`optim`模块中支持该算法及其多种变体。
当我们(**实例化一个`SGD`实例**)时，
我们指定待优化的参数（可通过`self.parameters()`从模型中获得），
以及优化算法所需的学习率（`self.lr`）。
:end_tab:

:begin_tab:`tensorflow`
小批量随机梯度下降是优化神经网络的标准工具，
因此 Keras 在`optimizers`模块中支持该算法及其多种变体。
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## 训练

你可能已经注意到，通过深度学习框架的高级 API 来表达我们的模型
所需的代码行数更少。
我们不必逐个分配参数、定义损失函数，也不必实现小批量随机梯度下降。
一旦我们开始处理复杂得多的模型，
高级 API 的优势就会显著增加。

既然所有基本组件都已就位，
[**训练循环本身与我们从头实现的那一个相同。**]
因此，我们只需调用`fit`方法（在 :numref:`oo-design-training` 中介绍），
它依赖 :numref:`sec_linear_scratch` 中`fit_epoch`方法的实现，
来训练我们的模型。

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

下面，我们
[**比较在有限数据上训练得到的模型参数与生成数据集的真实参数**]。
为了访问参数，
我们访问所需层的权重和偏置。
与我们从零实现时一样，
请注意我们估计出的参数接近其真实值。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## 小结

本节包含了（本书中）第一个利用现代深度学习框架所提供便利的深度网络实现，
这些框架包括 MXNet :cite:`Chen.Li.Li.ea.2015`、
JAX :cite:`Frostig.Johnson.Leary.2018`、
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`
和 Tensorflow :cite:`Abadi.Barham.Chen.ea.2016`。
加载数据、定义层、损失函数、优化器和训练循环时，我们都使用了框架的默认实现。
只要框架提供了所有必要的特性，使用它们通常是个好主意，
因为这些组件的库实现往往针对性能进行了大量优化，
并经过了充分的可靠性测试。
同时，尽量不要忘记，这些模块*是可以*直接实现的。
对于希望站在模型开发前沿的有志研究者来说，这一点尤其重要，
因为你将在那里发明当前任何库中都不可能存在的新组件。

:begin_tab:`mxnet`
在 Gluon 中，`data`模块提供数据处理工具，
`nn`模块定义了大量的神经网络层，
`loss`模块定义了许多常见的损失函数。
此外，`initializer`提供了许多参数初始化的选择。
对用户来说很方便的是，维度和存储会被自动推断出来。
这种延迟初始化的一个结果是：
在参数被实例化（并初始化）之前，你绝不能试图访问它们。
:end_tab:

:begin_tab:`pytorch`
在 PyTorch 中，`data`模块提供数据处理工具，
`nn`模块定义了大量的神经网络层和常见的损失函数。
我们可以通过用以`_`结尾的方法替换参数值来初始化参数。
请注意，我们需要指定网络的输入维度。
虽然目前这微不足道，
但当我们想要设计具有许多层的复杂网络时，它可能会产生重大的连锁影响。
为了使网络具有可移植性，我们需要仔细考虑如何参数化这些网络。
:end_tab:

:begin_tab:`tensorflow`
在 TensorFlow 中，`data`模块提供数据处理工具，
`keras`模块定义了大量的神经网络层和常见的损失函数。
此外，`initializers`模块提供了多种模型参数初始化的方法。
网络的维度和存储会被自动推断出来
（但要注意，在参数被初始化之前不要试图访问它们）。
:end_tab:

## 练习

1. 如果把小批量上的损失总和换成小批量上损失的平均值，你需要如何改变学习率？
1. 查阅框架文档，看看提供了哪些损失函数。特别地，用 Huber 鲁棒损失函数替换平方损失。也就是说，使用如下损失函数
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$
1. 如何访问模型权重的梯度？
1. 如果改变学习率和训练轮数，会对解产生什么影响？它会一直改进吗？
1. 当改变生成的数据量时，解会如何变化？
    1. 绘制估计误差 $\hat{\mathbf{w}} - \mathbf{w}$ 和 $\hat{b} - b$ 随数据量变化的图。提示：以对数方式而不是线性方式增加数据量，即取 5、10、20、50、……、10000，而不是 1000、2000、……、10000。
    2. 为什么提示中的建议是恰当的？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17977)
:end_tab:
