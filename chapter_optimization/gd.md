# 梯度下降
:label:`sec_gd`

本节我们将介绍*梯度下降*（gradient descent）背后的基本概念。
尽管梯度下降在深度学习中很少被直接使用，但理解梯度下降是理解随机梯度下降算法的关键。
例如，优化问题可能因学习率过大而发散。这一现象在梯度下降中就已经可以看到。同样，预处理是梯度下降中的一种常用技术，并且可以沿用到更高级的算法中。
让我们从一个简单的特例开始。


## 一维梯度下降

一维梯度下降是说明梯度下降算法为何可能降低目标函数值的一个很好的例子。考虑某个连续可微的实值函数$f: \mathbb{R} \rightarrow \mathbb{R}$。利用泰勒展开，我们得到

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

也就是说，在一阶近似下，$f(x+\epsilon)$由函数值$f(x)$和$x$处的一阶导数$f'(x)$给出。假设对于小的$\epsilon$，沿负梯度方向移动会减小$f$，这并非不合理。为简单起见，我们选取固定步长$\eta > 0$，并令$\epsilon = -\eta f'(x)$。将其代入上面的泰勒展开，得到

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

如果导数$f'(x) \neq 0$不为零，我们就能取得进展，因为$\eta f'^2(x)>0$。此外，我们总可以选择足够小的$\eta$，使高阶项变得无关紧要。因此我们得到

$$f(x - \eta f'(x)) \lessapprox f(x).$$

这意味着，如果我们使用

$$x \leftarrow x - \eta f'(x)$$

来迭代$x$，函数$f(x)$的值可能会下降。因此，在梯度下降中，我们首先选择一个初始值$x$和一个常数$\eta > 0$，然后用它们不断迭代$x$，直到达到停止条件，例如梯度的模$|f'(x)|$足够小，或者迭代次数达到某个值。

为简单起见，我们选择目标函数$f(x)=x^2$来说明如何实现梯度下降。尽管我们知道$x=0$是使$f(x)$最小化的解，我们仍然使用这个简单的函数来观察$x$如何变化。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

接下来，我们使用$x=10$作为初始值，并假设$\eta=0.2$。用梯度下降对$x$迭代 10 次，可以看到，最终$x$的值会接近最优解。

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

对$x$进行优化时的进展可以绘制如下。

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 学习率
:label:`subsec_gd-learningrate`

学习率$\eta$可以由算法设计者设定。如果我们使用太小的学习率，会导致$x$更新非常缓慢，需要更多次迭代才能得到更好的解。为了展示这种情况下会发生什么，考虑在同一个优化问题中取$\eta = 0.05$时的进展。可以看到，即使经过 10 步，我们仍然离最优解很远。

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

相反，如果我们使用过高的学习率，$\left|\eta f'(x)\right|$对一阶泰勒展开公式来说可能太大。也就是说，:eqref:`gd-taylor-2` 中的项$\mathcal{O}(\eta^2 f'^2(x))$可能变得显著。在这种情况下，我们无法保证对$x$的迭代能够降低$f(x)$的值。例如，当我们将学习率设为$\eta=1.1$时，$x$会越过最优解$x=0$并逐渐发散。

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 局部最小值

为了说明非凸函数会发生什么，考虑$f(x) = x \cdot \cos(cx)$的情形，其中$c$为某个常数。该函数有无穷多个局部最小值。取决于我们对学习率的选择，以及取决于问题的良态程度，我们最终可能得到众多解中的一个。下面的例子说明了（不切实际的）高学习率如何导致一个很差的局部最小值。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 多元梯度下降

现在我们对一维情形有了更好的直觉，接下来考虑$\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$的情形。也就是说，目标函数$f: \mathbb{R}^d \to \mathbb{R}$将向量映射为标量。相应地，它的梯度也是多元的。它是由$d$个偏导数组成的向量：

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

梯度中的每个偏导数元素$\partial f(\mathbf{x})/\partial x_i$表示$f$在$\mathbf{x}$处关于输入$x_i$的变化率。正如前面在一维情形中那样，我们可以使用多元函数相应的泰勒近似来了解应该怎么做。特别地，我们有

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

换句话说，精确到$\boldsymbol{\epsilon}$的二阶项，最速下降方向由负梯度$-\nabla f(\mathbf{x})$给出。选择合适的学习率$\eta > 0$，就得到典型的梯度下降算法：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

为了观察算法在实际中的表现，让我们构造一个目标函数$f(\mathbf{x})=x_1^2+2x_2^2$，其输入是二维向量$\mathbf{x} = [x_1, x_2]^\top$，输出是标量。梯度由$\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$给出。我们将从初始位置$[-5, -2]$出发，观察$\mathbf{x}$在梯度下降下的轨迹。

首先，我们还需要两个辅助函数。第一个使用一个更新函数，并将其作用于初始值 20 次。第二个辅助函数可视化$\mathbf{x}$的轨迹。

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

接下来，我们观察优化变量$\mathbf{x}$在学习率$\eta = 0.1$下的轨迹。可以看到，经过 20 步后，$\mathbf{x}$的值接近其在$[0, 0]$处的最小值。进展相当平稳，尽管相当缓慢。

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## 自适应方法

正如我们在 :numref:`subsec_gd-learningrate` 中所看到的，把学习率$\eta$调得“恰到好处”是很棘手的。如果选得太小，我们进展甚微。如果选得太大，解会来回振荡，最坏情况下甚至可能发散。如果我们能自动确定$\eta$，或者完全不必选择学习率，会怎么样呢？
二阶方法不仅考察目标函数的值和梯度，
还考察其*曲率*，在这种情况下可以提供帮助。虽然由于计算代价，这些方法不能直接应用于深度学习，但它们为如何设计高级优化算法提供了有用的直觉，这些算法模仿了下面概述的算法的许多理想性质。


### 牛顿法

回顾某个函数$f: \mathbb{R}^d \rightarrow \mathbb{R}$的泰勒展开，我们没有必要在第一项之后就停下来。事实上，我们可以将其写成

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

为避免繁琐的记号，我们定义$\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$为$f$的黑塞矩阵，它是一个$d \times d$矩阵。对于小的$d$和简单的问题，$\mathbf{H}$很容易计算。另一方面，对于深度神经网络，由于存储$\mathcal{O}(d^2)$个元素的代价，$\mathbf{H}$可能大得难以承受。此外，通过反向传播来计算它可能过于昂贵。目前让我们忽略这些考虑，来看看我们会得到什么算法。

毕竟，$f$的最小值满足$\nabla f = 0$。
按照 :numref:`subsec_calculus-grad` 中的微积分法则，
对 :eqref:`gd-hot-taylor` 关于$\boldsymbol{\epsilon}$求导并忽略高阶项，我们得到

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

也就是说，作为优化问题的一部分，我们需要对黑塞矩阵$\mathbf{H}$求逆。

作为一个简单的例子，对于$f(x) = \frac{1}{2} x^2$，我们有$\nabla f(x) = x$且$\mathbf{H} = 1$。因此对任意$x$，我们得到$\epsilon = -x$。换句话说，*单*步就足以完美收敛，无需任何调整！唉，我们这里有点走运：泰勒展开是精确的，因为$f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$。

让我们看看在其他问题中会发生什么。
给定某个常数$c$下的凸双曲余弦函数$f(x) = \cosh(cx)$，我们可以看到，
经过几次迭代后
就到达了$x=0$处的全局最小值。

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

现在让我们考虑一个*非凸*函数，例如$f(x) = x \cos(c x)$，其中$c$为某个常数。毕竟要注意，在牛顿法中我们最终要除以黑塞矩阵。这意味着如果二阶导数是*负*的，我们可能走向*增大*$f$值的方向。
这是该算法的一个致命缺陷。
让我们看看在实践中会发生什么。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

结果错得离谱。我们怎样才能修正它呢？一种方法是通过取绝对值来“修正”黑塞矩阵。另一种策略是重新引入学习率。这似乎违背了初衷，但也不尽然。拥有二阶信息使我们能够在曲率较大时保持谨慎，而在目标函数较平坦时迈出更大的步长。
让我们看看用稍小一点的学习率，比如$\eta = 0.5$，会怎样。可以看到，我们得到了一个相当高效的算法。

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 收敛性分析

我们只分析牛顿法对某个凸且三次可微的目标函数$f$的收敛速率，其中二阶导数非零，即$f'' > 0$。多元情形的证明是下面一维论证的直接推广，由于它对直觉帮助不大，因此从略。

用$x^{(k)}$表示$x$在第$k^\textrm{th}$次迭代时的值，并令$e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$为第$k^\textrm{th}$次迭代时与最优点的距离。由泰勒展开，条件$f'(x^*) = 0$可以写成

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

该式对某个$\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$成立。将上述展开式除以$f''(x^{(k)})$，得到

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

回想一下，我们有更新公式$x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$。
代入这个更新方程并对两边取绝对值，我们得到

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

因此，只要我们处于$\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$有界的区域内，误差就会二次减小

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$


顺带一提，优化研究者称这种收敛为*线性*收敛，而像$\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$这样的条件则被称为*常数*收敛速率。
注意，这一分析带有若干注意事项。
首先，我们并不能真正保证何时会到达快速收敛的区域。我们只知道一旦到达该区域，收敛会非常快。其次，这一分析要求$f$在更高阶导数方面表现良好。归根结底，这要确保$f$在取值如何变化方面没有“令人惊讶”的性质。



### 预处理

毫不意外，计算和存储完整的黑塞矩阵非常昂贵。因此，人们希望寻找替代方案。改善这一问题的一种方法是*预处理*（preconditioning）。它避免计算完整的黑塞矩阵，而只计算*对角*元素。这引出了如下形式的更新算法：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$


虽然这不如完整的牛顿法好，但仍然比不使用它要好得多。
要理解为什么这可能是个好主意，考虑这样一种情形：一个变量以毫米为单位表示高度，另一个变量以千米为单位表示高度。假设两者自然的尺度都是米，那么我们在参数化上就存在严重的不匹配。幸运的是，使用预处理可以消除这一点。实际上，带预处理的梯度下降相当于为每个变量（向量$\mathbf{x}$的坐标）选择不同的学习率。
正如我们稍后将看到的，预处理推动了随机梯度下降优化算法中的一些创新。


### 梯度下降和线搜索

梯度下降的关键问题之一是，我们可能越过目标，或者进展不足。该问题的一个简单修正方法是把线搜索（line search）与梯度下降结合起来使用。也就是说，我们使用$\nabla f(\mathbf{x})$给出的方向，然后进行二分搜索，以确定哪个学习率$\eta$能使$f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$最小。

该算法收敛很快（关于分析和证明，例如见 :citet:`Boyd.Vandenberghe.2004`）。然而，就深度学习的目的而言，这并不太可行，因为线搜索的每一步都需要我们在整个数据集上计算目标函数。这样做的代价太高了。

## 小结

* 学习率很重要。太大就会发散，太小则无法取得进展。
* 梯度下降可能陷入局部最小值。
* 在高维情况下，调整学习率很复杂。
* 预处理有助于进行尺度调整。
* 一旦在凸问题中正常发挥作用，牛顿法要快得多。
* 对于非凸问题，要谨慎在没有任何调整的情况下使用牛顿法。

## 练习

1. 对梯度下降使用不同的学习率和目标函数进行实验。
1. 实现线搜索以最小化区间$[a, b]$上的凸函数。
    1. 二分搜索需要导数吗？也就是说，决定选取$[a, (a+b)/2]$还是$[(a+b)/2, b]$时需要导数吗？
    1. 该算法的收敛速率有多快？
    1. 实现该算法，并将其用于最小化$\log (\exp(x) + \exp(-2x -3))$。
1. 设计一个定义在$\mathbb{R}^2$上的目标函数，使梯度下降极为缓慢。提示：以不同的方式缩放不同的坐标。
1. 使用预处理实现牛顿法的轻量版本：
    1. 使用对角黑塞矩阵作为预处理器。
    1. 使用它的绝对值，而不是实际的（可能带符号的）值。
    1. 将其应用于上面的问题。
1. 将上述算法应用于若干目标函数（凸的或非凸的）。如果你将坐标旋转$45$度，会发生什么？

[讨论](https://discuss.d2l.ai/t/351)
