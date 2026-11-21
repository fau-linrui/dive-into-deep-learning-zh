```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 线性回归的从零开始实现
:label:`sec_linear_scratch`

我们现在已经准备好，
来实现一个功能完整的线性回归。
在本节中，
(**我们将从零开始实现整个方法，
包括：(i) 模型；(ii) 损失函数；
(iii) 小批量随机梯度下降优化器；
(iv) 把所有这些部分串联起来的训练函数。**)
最后，我们将运行
 :numref:`sec_synthetic-regression-data` 中的合成数据生成器，
并把我们的模型应用到
生成的数据集上。
虽然现代深度学习框架
几乎可以自动完成所有这些工作，
但从零开始实现是
确保你真正理解自己在做什么的唯一途径。
此外，当需要定制模型、
定义自己的层或损失函数时，
理解底层的工作原理将会很有帮助。
在本节中，我们将只依赖
张量和自动微分。
稍后，我们将介绍一种更简洁的实现，
它利用深度学习框架的各种花哨功能，
同时保留下面这种实现的结构。

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## 定义模型

在[**能够通过小批量SGD开始优化模型参数之前**]，
(**我们需要先有一些参数。**)
下面，我们通过从均值为0、
标准差为0.01的正态分布中抽取随机数
来初始化权重。
这个魔法数字0.01在实践中往往效果很好，
但你可以通过参数`sigma`
指定一个不同的值。
此外，我们把偏置设为0。
请注意，为了遵循面向对象的设计，
我们把代码添加到`d2l.Module`子类的`__init__`方法中（该类在 :numref:`subsec_oo-design-models` 中介绍）。

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

接下来，我们必须[**定义我们的模型，
把它的输入和参数与输出关联起来。**]
采用与 :eqref:`eq_linreg-y-vec` 相同的记号，
对于线性模型，我们只需取输入特征$\mathbf{X}$
与模型权重$\mathbf{w}$的矩阵-向量乘积，
并为每个样例加上偏移量$b$。
乘积$\mathbf{Xw}$是一个向量，而$b$是一个标量。
由于广播机制
（参见 :numref:`subsec_broadcasting`），
当我们把一个向量和一个标量相加时，
标量会被加到向量的每个分量上。
由此得到的`forward`方法
通过`add_to_class`注册到`LinearRegressionScratch`类中
（该函数在 :numref:`oo-design-utilities` 中介绍）。

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## 定义损失函数

由于[**更新模型需要计算损失函数的梯度，**]
我们应当(**先定义损失函数。**)
这里我们使用 :eqref:`eq_mse` 中的平方损失函数。
在实现中，我们需要把真实值`y`
变换成预测值的形状`y_hat`。
下面的方法返回的结果
也将具有与`y_hat`相同的形状。
我们还返回小批量中
所有样例的平均损失值。

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## 定义优化算法

正如 :numref:`sec_linear_regression` 中所讨论的，
线性回归有闭式解。
然而，我们这里的目标是说明
如何训练更一般的神经网络，
而这要求我们教你
如何使用小批量SGD。
因此，我们将借此机会
介绍你的第一个可运行的SGD示例。
在每一步中，使用从数据集中
随机抽取的一个小批量，
我们估计损失
关于参数的梯度。
接下来，我们沿着
可能降低损失的方向更新参数。

下面的代码在给定一组参数和学习率`lr`的情况下
执行更新。
由于我们的损失是在小批量上求平均计算得到的，
因此不需要根据批量大小调整学习率。
在后面的章节中，我们将研究
当分布式大规模学习中出现非常大的小批量时，
应该如何调整学习率。
目前，我们可以忽略这种依赖关系。

:begin_tab:`mxnet`
我们定义自己的`SGD`类，
它是`d2l.HyperParameters`的子类（该类在 :numref:`oo-design-utilities` 中介绍），
使其API与内置的SGD优化器相似。
我们在`step`方法中更新参数。
该方法接受一个可以忽略的`batch_size`参数。
:end_tab:

:begin_tab:`pytorch`
我们定义自己的`SGD`类，
它是`d2l.HyperParameters`的子类（该类在 :numref:`oo-design-utilities` 中介绍），
使其API与内置的SGD优化器相似。
我们在`step`方法中更新参数。
`zero_grad`方法把所有梯度设为0，
它必须在反向传播步骤之前运行。
:end_tab:

:begin_tab:`tensorflow`
我们定义自己的`SGD`类，
它是`d2l.HyperParameters`的子类（该类在 :numref:`oo-design-utilities` 中介绍），
使其API与内置的SGD优化器相似。
我们在`apply_gradients`方法中更新参数。
它接受一个参数与梯度对的列表。
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # The key transformation of Optax is the GradientTransformation
    # defined by two methods, the init and the update.
    # The init initializes the state and the update transforms the gradients.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

接下来，我们定义`configure_optimizers`方法，它返回`SGD`类的一个实例。

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## 训练

现在我们已经把所有部分都准备好了
（参数、损失函数、模型和优化器），
可以[**实现主要的训练循环了。**]
完全理解这段代码至关重要，
因为对于本书中介绍的
每一个其他深度学习模型，
你都会使用类似的训练循环。
在每个*轮*（epoch）中，我们遍历
整个训练数据集，
对每个样例各过一遍
（假设样例数
能被批量大小整除）。
在每个*迭代*（iteration）中，我们抓取一个小批量的训练样例，
并通过模型的`training_step`方法计算其损失。
然后我们计算损失关于每个参数的梯度。
最后，我们调用优化算法
来更新模型参数。
总之，我们将执行下面的循环：

* 初始化参数$(\mathbf{w}, b)$
* 重复直到完成
    * 计算梯度$\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 更新参数$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
回想一下，我们在 :numref:``sec_synthetic-regression-data`` 中生成的
合成回归数据集并不提供验证数据集。
然而在大多数情况下，
我们会希望有一个验证数据集
来衡量模型的质量。
这里我们在每一轮中
把验证数据加载器过一遍，以衡量模型性能。
按照我们的面向对象设计，
`prepare_batch`和`fit_epoch`方法
注册在`d2l.Trainer`类中
（该类在 :numref:`oo-design-training` 中介绍）。

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # Mutable states will be used later (e.g., for batch norm)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

我们几乎可以训练模型了，
但首先需要一些训练数据。
这里我们使用`SyntheticRegressionData`类，
并传入一些真实参数。
然后我们用学习率`lr=0.03`
训练模型，并设置`max_epochs=3`。
请注意，一般来说，轮数
和学习率都是超参数。
通常，设置超参数是件棘手的事，
我们往往希望使用三路划分：
一份用于训练，
第二份用于超参数选择，
第三份留作最终评估。
我们暂时省略这些细节，
但稍后会重新讨论。

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

由于数据集是我们自己合成的，
所以我们确切地知道真实参数是什么。
因此，我们可以[**通过比较真实参数与
我们学到的参数，来评估训练的成功程度**]，
而比较正是通过我们的训练循环完成的。
的确，它们的结果非常接近。

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

我们不应把精确恢复
真实参数的能力视为理所当然。
一般来说，对于深度模型，
参数的唯一解并不存在，
而且即便是线性模型，
也只有在没有任何特征
与其他特征线性相关时，
才可能精确恢复参数。
然而，在机器学习中，
我们往往并不那么关心
恢复真实的潜在参数，
而更关心那些
能带来高度准确预测的参数 :cite:`Vapnik.1992`。
幸运的是，即使在困难的优化问题上，
随机梯度下降也常常能找到非常好的解，
部分原因在于，对于深度网络，
存在许多参数配置
能够带来高度准确的预测。


## 小结

在本节中，我们朝着设计深度学习系统
迈出了重要的一步，
实现了一个功能完整的
神经网络模型和训练循环。
在此过程中，我们构建了数据加载器、
模型、损失函数、优化过程，
以及一个可视化与监控工具。
我们是通过组合一个包含训练模型
所需的全部相关组件的Python对象来做到这一点的。
虽然这还不是专业级的实现，
但它完全可用，而且这样的代码
已经能帮助你快速解决一些小问题。
在接下来的几节中，我们将看到如何做到这一点：
既*更简洁*（避免样板代码），
又*更高效*（充分发挥GPU的潜力）。



## 练习

1. 如果我们把权重初始化为零，会发生什么？算法还能正常工作吗？如果我们
   把参数初始化为方差$1000$而不是$0.01$，又会怎样？
1. 假设你是[Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm)，正试图提出一个
   将电压与电流联系起来、用于描述电阻的模型。你能利用自动
   微分来学习模型的参数吗？
1. 你能利用[普朗克定律](https://en.wikipedia.org/wiki/Planck%27s_law)，通过光谱能量密度来确定物体的温度
   吗？作为参考，黑体发出的辐射的光谱密度$B$为
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$。这里
   $\lambda$是波长，$T$是温度，$c$是光速，$h$是普朗克常数，$k$是
   玻尔兹曼常数。你测量了不同波长$\lambda$下的能量，现在需要把光谱
   密度曲线拟合到普朗克定律。
1. 如果你想计算损失的二阶导数，可能会遇到哪些问题？你会
   如何解决它们？
1. 为什么`loss`函数中需要用到`reshape`方法？
1. 用不同的学习率做实验，找出损失函数值下降得有多快。你能通过
   增加训练的轮数来减小误差吗？
1. 如果样例数不能被批量大小整除，在一轮结束时`data_iter`会发生什么？
1. 尝试实现另一种损失函数，比如绝对值损失`(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`。
    1. 检查在普通数据上会发生什么。
    1. 如果你主动扰动$\mathbf{y}$中的某些项（例如$y_5 = 10000$），检查行为是否有差异。
    1. 你能想出一个廉价的方案，把平方损失和绝对值损失的优点结合起来吗？
       提示：如何避免非常大的梯度值？
1. 为什么我们需要重新打乱数据集？你能设计一个例子，说明否则恶意构造的数据集会如何破坏优化算法吗？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17976)
:end_tab:
