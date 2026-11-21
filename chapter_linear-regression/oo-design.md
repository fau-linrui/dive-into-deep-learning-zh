```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 面向对象的设计与实现
:label:`sec_oo-design`

在介绍线性回归时，
我们逐一讲解了
数据、模型、损失函数
以及优化算法等各个组成部分。
的确，
线性回归是
最简单的机器学习模型之一。
然而，训练它
所用的许多组件，也是本书中其他模型所需要的。
因此，
在深入实现细节之前，
有必要先
设计一些
贯穿全书使用的API。
把深度学习中的组件
当作对象来处理，
我们就可以先
为这些对象定义类
以及它们之间的交互方式。
这种面向对象的
设计思路
将大大简化
我们的讲解，你甚至可能想在自己的项目中采用它。


受[PyTorch Lightning](https://www.pytorchlightning.ai/)等开源库的启发，
从高层来看，
我们希望有三个类：
（i）`Module`包含模型、损失和优化方法；
（ii）`DataModule`提供用于训练和验证的数据加载器；
（iii）这两个类通过`Trainer`类组合起来，使我们能够
在各种硬件平台上训练模型。
本书中的大部分代码都改编自`Module`和`DataModule`。我们只在讨论GPU、CPU、并行训练和优化算法时才会涉及`Trainer`类。

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## 工具函数
:label:`oo-design-utilities`

为了简化Jupyter 笔记本中的面向对象编程，我们需要几个工具函数。其中一个挑战是，类的定义往往是相当长的代码块。笔记本的可读性要求代码片段简短，并穿插在解释之间，这与Python库中常见的编程风格并不相容。第一个
工具函数让我们可以在创建类*之后*把函数注册为该类的方法。事实上，我们甚至可以在已经创建了该类的实例*之后*再这样做！这样就能把一个类的实现拆分到多个代码块中。

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

让我们快速看看如何使用它。我们打算实现一个带有方法`do`的类`A`。与其把`A`和`do`的代码放在同一个代码块中，我们可以先声明类`A`，并创建一个实例`a`。

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

接下来，我们像通常那样定义方法`do`，只是它不在类`A`的作用域内。取而代之的是，我们用`add_to_class`来装饰这个方法，并以类`A`作为其参数。这样一来，该方法就能访问`A`的成员变量，就如同它原本就包含在`A`的定义中一样。让我们看看在实例`a`上调用它时会发生什么。

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

第二个是一个工具类，它把类`__init__`方法中的所有参数都保存为类的属性。这让我们无需额外编写代码，就能隐式地扩展构造函数的调用签名。

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

我们将其实现推迟到 :numref:`sec_utils` 中介绍。要使用它，我们定义自己的类，使其继承`HyperParameters`，并在`__init__`方法中调用`save_hyperparameters`。

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

最后一个工具函数让我们能够在实验进行过程中交互式地绘制实验进度。为了向功能强大得多（也复杂得多）的[TensorBoard](https://www.tensorflow.org/tensorboard)致敬，我们把它命名为`ProgressBoard`。其实现推迟到 :numref:`sec_utils` 中介绍。现在，让我们先看看它的实际效果。

`draw`方法在图中绘制点`(x, y)`，`label`指定该点在图例中的标签。可选的`every_n`只显示图中$1/n$的点，从而对折线进行平滑。这些点的值是在原图中由$n$个相邻点平均得到的。

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

在下面的例子中，我们用不同的平滑程度绘制`sin`和`cos`。如果你运行这个代码块，就会看到线条以动画方式逐渐绘出。

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## 模型
:label:`subsec_oo-design-models`

`Module`类是我们将要实现的所有模型的基类。至少需要三个方法。第一个是`__init__`，用于存储可学习的参数；`training_step`方法接受一个数据批量并返回损失值；最后，`configure_optimizers`返回用于更新可学习参数的优化方法，或者优化方法的列表。我们还可以选择性地定义`validation_step`来报告评价指标。
有时，我们会把计算输出的代码放到一个单独的`forward`方法中，以提高代码的可复用性。

:begin_tab:`jax`
随着Python 3.7引入[数据类](https://docs.python.org/3/library/dataclasses.html)，
用`@dataclass`装饰的类会自动添加
`__init__`和`__repr__`等魔法方法。成员变量
使用类型注解来定义。所有Flax模块都是Python 3.7的数据类。
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # No need for save_hyperparam when using Python dataclass
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # Use default_factory to make sure new plots are generated on each run
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX & Flax do not have a forward-method-like syntax. Flax uses setup
        # and built-in __call__ magic methods for forward pass. Adding here
        # for consistency
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """To be defined later in :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
你可能注意到，`Module`是`nn.Block`的子类，后者是Gluon中神经网络的基类。
它为处理神经网络提供了便利的特性。例如，如果我们定义了一个`forward`方法，比如`forward(self, X)`，那么对于实例`a`，我们可以通过`a(X)`来调用这个方法。这之所以可行，是因为它在内置的`__call__`方法中调用了`forward`方法。你可以在 :numref:`sec_model_construction` 中找到关于`nn.Block`的更多细节和示例。
:end_tab:

:begin_tab:`pytorch`
你可能注意到，`Module`是`nn.Module`的子类，后者是PyTorch中神经网络的基类。
它为处理神经网络提供了便利的特性。例如，如果我们定义了一个`forward`方法，比如`forward(self, X)`，那么对于实例`a`，我们可以通过`a(X)`来调用这个方法。这之所以可行，是因为它在内置的`__call__`方法中调用了`forward`方法。你可以在 :numref:`sec_model_construction` 中找到关于`nn.Module`的更多细节和示例。
:end_tab:

:begin_tab:`tensorflow`
你可能注意到，`Module`是`tf.keras.Model`的子类，后者是TensorFlow中神经网络的基类。
它为处理神经网络提供了便利的特性。例如，它会在内置的`__call__`方法中调用`call`方法。这里我们把`call`重定向到`forward`方法，并把它的参数保存为类属性。我们这样做，是为了让我们的代码更接近其他框架的实现。
:end_tab:

:begin_tab:`jax`
你可能注意到，`Module`是`linen.Module`的子类，后者是Flax中神经网络的基类。
它为处理神经网络提供了便利的特性。例如，它负责处理模型参数，提供`nn.compact`装饰器来简化代码，并且还会调用`__call__`方法等等。
这里我们同样把`__call__`重定向到`forward`方法。我们这样做，是为了让我们的代码更接近其他框架的实现。
:end_tab:

##  数据
:label:`oo-design-data`

`DataModule`类是所有数据的基类。`__init__`方法经常被用来准备数据，包括必要时的下载和预处理。`train_dataloader`返回训练数据集的数据加载器。数据加载器是一个（Python）生成器，每次使用时都会产生一个数据批量。这个批量随后被送入`Module`的`training_step`方法以计算损失。还有一个可选的`val_dataloader`，用于返回验证数据集加载器。它的行为方式相同，只不过它为`Module`中的`validation_step`方法产生数据批量。

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## 训练
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
`Trainer`类用`DataModule`中指定的数据来训练`Module`类中的可学习参数。关键方法是`fit`，它接受两个参数：`model`（`Module`的一个实例）和`data`（`DataModule`的一个实例）。然后它遍历整个数据集`max_epochs`次来训练模型。和之前一样，我们将把这个方法的实现推迟到后面的章节。
:end_tab:

:begin_tab:`jax`
`Trainer`类用`DataModule`中指定的数据来训练可学习参数`params`。关键方法是`fit`，它接受三个参数：`model`（`Module`的一个实例）、`data`（`DataModule`的一个实例）和`key`（JAX的一个`PRNGKeyArray`）。这里我们把`key`参数设为可选，以简化接口，但在JAX和Flax中，建议始终传递根密钥并用它初始化模型参数。然后它遍历整个数据集`max_epochs`次来训练模型。和之前一样，我们将把这个方法的实现推迟到后面的章节。
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Here batch_stats will be used later (e.g., for batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax uses optax under the hood for a single state obj TrainState.
            # More will be discussed later in the dropout and batch
            # normalization section
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## 小结

为了突出我们未来深度学习实现中的面向对象设计，
上面的类只是展示了它们的对象
如何存储数据以及彼此如何交互。
在本书余下的部分，我们会
通过`@add_to_class`等方式，
不断丰富这些类的实现。
此外，
这些完整实现的类
保存在[D2L库](https://github.com/d2l-ai/d2l-en/tree/master/d2l)中，
这是一个*轻量级工具包*，使深度学习的结构化建模变得容易。
特别地，它便于在项目之间复用许多组件，而几乎无需任何改动。例如，我们可以只替换优化器、只替换模型、只替换数据集等等；
这种程度的模块化在全书中以简洁和简单的方式带来回报（这也是我们加入它的原因），它同样能为你自己的项目带来好处。


## 练习

1. 在[D2L库](https://github.com/d2l-ai/d2l-en/tree/master/d2l)中找到上述类的完整实现。我们强烈建议你在对深度学习建模有了更多熟悉之后，再仔细研究其实现。
1. 删除`B`类中的`save_hyperparameters`语句。你还能打印`self.a`和`self.b`吗？可选：如果你已经深入研究了`HyperParameters`类的完整实现，你能解释原因吗？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17974)
:end_tab:
