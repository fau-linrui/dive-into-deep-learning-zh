```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 分类模型基类
:label:`sec_classification`

你可能已经注意到，在回归的情形下，从零开始的实现和使用框架功能的简洁实现非常相似。分类也是如此。由于本书中的许多模型都涉及分类，值得添加一些专门支持这一场景的功能。本节提供了一个分类模型的基类，以简化后续的代码。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## `Classifier` 类

:begin_tab:`pytorch, mxnet, tensorflow`
下面我们定义`Classifier`类。在`validation_step`中，我们报告验证批次上的损失值和分类准确率。我们每处理`num_val_batches`个批次才绘制一次更新。这样做的好处是能生成整个验证数据上的平均损失和平均准确率。如果最后一个批次包含的样例较少，这些平均值并不完全正确，但为了保持代码简单，我们忽略这一微小差异。
:end_tab:


:begin_tab:`jax`
下面我们定义`Classifier`类。在`validation_step`中，我们报告验证批次上的损失值和分类准确率。我们每处理`num_val_batches`个批次才绘制一次更新。这样做的好处是能生成整个验证数据上的平均损失和平均准确率。如果最后一个批次包含的样例较少，这些平均值并不完全正确，但为了保持代码简单，我们忽略这一微小差异。

我们还为 JAX 重新定义了`training_step`方法，因为此后所有将继承`Classifier`的模型都会有一个返回辅助数据的损失。
这些辅助数据可用于带有批量规范化层的模型（将在 :numref:`sec_batch_norm` 中解释），而在其他所有情况下，
我们让损失同时返回一个占位符（空字典）来表示辅助数据。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def training_step(self, params, batch, state):
        # Here value is a tuple since models with BatchNorm layers require
        # the loss to return auxiliary data
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard the second returned value. It is used for training models
        # with BatchNorm layers since loss also returns auxiliary data
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

默认情况下，我们使用作用于小批量的随机梯度下降优化器，就像我们在线性回归中所做的那样。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## 准确率

给定预测的概率分布`y_hat`，
每当我们必须输出一个硬预测时，
我们通常选择预测概率最高的类别。
的确，许多应用都要求我们做出选择。
例如，Gmail 必须把一封电子邮件归入“主要”“社交”“更新”“论坛”或“垃圾邮件”。
它可能在内部估计概率，
但归根结底必须在这些类别中选择一个。

当预测结果与标签类别`y`一致时，该预测就是正确的。
分类准确率是所有预测中正确预测所占的比例。
虽然直接优化准确率可能很困难（它不可微），
但它往往是我们最关心的性能指标。在基准测试中，它常常*就是*那个相关量。
因此，在训练分类器时，我们几乎总会报告它。

准确率的计算方式如下。
首先，如果`y_hat`是一个矩阵，
我们假设第二维存储的是每个类别的预测分数。
我们用`argmax`取得每一行中最大元素的下标，作为预测出的类别。
然后我们[**将预测类别与真实标签`y`逐元素比较。**]
由于相等运算符`==`对数据类型敏感，
我们把`y_hat`的数据类型转换为与`y`一致。
结果是一个包含 0（假）和 1（真）的张量。
对其求和就得到正确预测的数量。

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## 小结

分类是一个非常常见的问题，因此值得为它提供专门的便捷函数。分类中最为核心的是分类器的*准确率*。请注意，虽然我们往往主要关心准确率，但出于统计和计算上的原因，我们训练分类器时优化的却是各种其他目标。然而，无论训练期间最小化的是哪个损失函数，有一个便捷方法来根据经验评估分类器的准确率都是很有用的。


## 练习

1. 用 $L_\textrm{v}$ 表示验证损失，并令 $L_\textrm{v}^\textrm{q}$ 表示本节中通过对损失函数取平均而计算出的粗略估计（quick and dirty estimate）。最后，用 $l_\textrm{v}^\textrm{b}$ 表示最后一个小批量上的损失。请用 $L_\textrm{v}^\textrm{q}$、$l_\textrm{v}^\textrm{b}$ 以及样本数和小批量大小来表示 $L_\textrm{v}$。
1. 证明该粗略估计 $L_\textrm{v}^\textrm{q}$ 是无偏的，即证明 $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$。既然如此，为什么你仍然想改用 $L_\textrm{v}$？
1. 给定多类分类损失，用 $l(y,y')$ 表示在观察到 $y$ 时估计 $y'$ 所付出的代价，并给定概率 $p(y \mid x)$，请给出最优选择 $y'$ 的规则。提示：用 $l$ 和 $p(y \mid x)$ 表示期望损失。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17981)
:end_tab:
