# 优化与深度学习
:label:`sec_optimization-intro`

本节将讨论优化与深度学习之间的关系，以及深度学习中使用优化时所面临的挑战。
对于深度学习问题，我们通常会先定义一个*损失函数*（loss function）。一旦有了损失函数，就可以使用优化算法来尝试最小化损失。
在优化中，损失函数通常被称为优化问题的*目标函数*（objective function）。按照传统和惯例，大多数优化算法关注的是*最小化*。如果我们需要最大化某个目标，有一个简单的解决办法：只要翻转目标的符号即可。

## 优化的目标

尽管优化提供了一种最小化深度学习损失函数的方法，但本质上，优化和深度学习的目标是根本不同的。
前者主要关注最小化一个目标，而后者关注在给定有限数据量的情况下找到一个合适的模型。
在 :numref:`sec_generalization_basics` 中，我们详细讨论了这两个目标之间的区别。
例如，训练误差和泛化误差一般是不同的：由于优化算法的目标函数通常是基于训练数据集的损失函数，因此优化的目标是减少训练误差。
然而，深度学习（或者更宽泛地说，统计推断）的目标是减少泛化误差。
为了实现后者，除了使用优化算法减少训练误差外，我们还需要关注过拟合。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

为了说明上述目标的不同，让我们考虑经验风险和风险。
如 :numref:`subsec_empirical-risk-and-risk` 所述，经验风险是训练数据集上的平均损失，
而风险是在整个数据总体上的期望损失。
下面我们定义两个函数：风险函数 `f` 和经验风险函数 `g`。
假设我们只有有限数量的训练数据。
因此，这里的 `g` 不如 `f` 光滑。

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

下图说明，训练数据集上经验风险的最小值可能位于与风险（泛化误差）的最小值不同的位置。

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## 深度学习中的优化挑战

本章我们将专门关注优化算法在最小化目标函数方面的性能，而不是模型的泛化误差。
在 :numref:`sec_linear_regression` 中，我们区分了优化问题中的解析解和数值解。
在深度学习中，大多数目标函数都很复杂，没有解析解。
相反，我们必须使用数值优化算法。
本章中的优化算法都属于这一类。

深度学习优化中存在许多挑战。其中最令人头疼的包括局部最小值、鞍点和梯度消失。
下面我们来看看这些挑战。


### 局部最小值

对任意目标函数$f(x)$，如果$x$处$f(x)$的值小于$x$附近任何其他点处$f(x)$的值，那么$f(x)$可能是局部最小值。
如果$x$处$f(x)$的值是目标函数在整个定义域上的最小值，那么$f(x)$是全局最小值。

例如，给定函数

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

我们可以近似地求出该函数的局部最小值和全局最小值。

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

深度学习模型的目标函数通常有许多局部最优解。
当优化问题的数值解接近局部最优时，最终迭代得到的数值解可能只是*局部*地最小化目标函数，而不是*全局*地最小化，因为目标函数解的梯度接近或变为零。
只有一定程度的噪声才可能把参数从局部最小值中震出。事实上，这是小批量随机梯度下降（minibatch stochastic gradient descent）的有益性质之一：
梯度在小批量之间的自然变化能够把参数从局部最小值中推出。


### 鞍点

除了局部最小值之外，鞍点也是梯度消失的另一个原因。*鞍点*（saddle point）是函数的梯度全部消失、但既不是全局最小值也不是局部最小值的位置。
考虑函数$f(x) = x^3$。它的一阶和二阶导数在$x=0$处都为零。优化过程可能在该点停滞，尽管它并不是最小值。

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

如下例所示，高维空间中的鞍点更加隐蔽。考虑函数$f(x, y) = x^2 - y^2$。它的鞍点位于$(0, 0)$。
这是一个关于$y$的极大值，同时是关于$x$的极小值。
此外，它*看起来*就像一个马鞍，这一数学性质的名称正来源于此。

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

假设函数的输入是$k$维向量，输出是标量，那么其黑塞矩阵（Hessian matrix）将具有$k$个特征值。
在函数梯度为零的位置，函数的解可能是局部最小值、局部最大值或鞍点：

* 当函数在零梯度位置的黑塞矩阵的特征值全部为正时，函数在该处取得局部最小值。
* 当函数在零梯度位置的黑塞矩阵的特征值全部为负时，函数在该处取得局部最大值。
* 当函数在零梯度位置的黑塞矩阵的特征值有正有负时，函数在该处存在鞍点。

对于高维问题，至少*部分*特征值为负的可能性相当高。这使得鞍点比局部最小值更可能出现。在下一节介绍凸性时，我们将讨论这种情况的一些例外。简而言之，凸函数（convex function）是指黑塞矩阵的特征值从不为负的函数。然而遗憾的是，大多数深度学习问题并不属于这一类。尽管如此，它仍然是研究优化算法的一个很好的工具。

### 梯度消失

我们可能遇到的最隐蔽的问题是梯度消失（vanishing gradient）。
回想一下 :numref:`subsec_activation-functions` 中常用的激活函数及其导数。
例如，假设我们想最小化函数$f(x) = \tanh(x)$，并且恰好从$x = 4$开始。可以看到，$f$的梯度接近于零。
更具体地说，$f'(x) = 1 - \tanh^2(x)$，因此$f'(4) = 0.0013$。
于是，在我们取得进展之前，优化会长时间停滞不前。事实证明，这也是在引入 ReLU 激活函数之前训练深度学习模型相当棘手的原因之一。

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

正如我们所看到的，深度学习的优化充满了挑战。幸运的是，存在一系列鲁棒的算法，它们表现良好，而且即使对初学者来说也易于使用。此外，并不真正需要找到*那个*最优解。局部最优解，甚至它们的近似解，仍然非常有用。

## 小结

* 最小化训练误差并*不能*保证我们找到使泛化误差最小的最佳参数集合。
* 优化问题可能有许多局部最小值。
* 问题可能具有更多的鞍点，因为一般这些问题都不是凸的。
* 梯度消失可能导致优化停滞。通常对问题重新参数化会有帮助。好的参数初始化也可能是有益的。


## 练习

1. 考虑一个简单的多层感知机（MLP），它只有单个隐藏层，隐藏层维度为（比如）$d$，输出为单个值。证明：对于任意局部最小值，都存在至少$d!$个行为完全相同的等价解。
1. 假设有一个对称随机矩阵$\mathbf{M}$，其元素
   $M_{ij} = M_{ji}$各自取自某个概率分布
   $p_{ij}$。进一步假设$p_{ij}(x) = p_{ij}(-x)$，即该
   分布是对称的（例如，详见 :citet:`Wigner.1958`）。
    1. 证明特征值的分布也是对称的。也就是说，对于任意特征向量$\mathbf{v}$，其关联的特征值$\lambda$满足$P(\lambda > 0) = P(\lambda < 0)$。
    1. 为什么上述结论*不*意味着$P(\lambda > 0) = 0.5$？
1. 你还能想到深度学习优化中涉及的其他哪些挑战？
1. 假设你想把一个（真实的）球平衡在一个（真实的）马鞍上。
    1. 为什么这很难？
    1. 你能把这种效应也用于优化算法吗？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/489)
:end_tab:
