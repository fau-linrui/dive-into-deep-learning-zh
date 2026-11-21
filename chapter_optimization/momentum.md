# 动量法
:label:`sec_momentum`

在 :numref:`sec_sgd` 中，我们回顾了执行随机梯度下降时会发生什么，即当只能获得梯度的噪声版本时执行优化会发生什么。特别地，我们注意到，对于带噪声的梯度，在噪声面前选择学习率时需要格外谨慎。如果我们把学习率降得太快，收敛会停滞。如果我们过于宽松，就无法收敛到足够好的解，因为噪声会不断把我们驱离最优性。

## 基础

在本节中，我们将探索更有效的优化算法，特别是针对实践中常见的某些类型的优化问题。


### 泄漏平均值

上一节我们讨论了小批量 SGD 作为加速计算的一种手段。它还有一个很好的副作用，即对梯度取平均减小了方差。小批量随机梯度下降可以计算为：

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

为保持记号简单，这里我们用$\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$表示使用在时间$t-1$更新的权重时对样本$i$的随机梯度下降。
如果我们能够从方差减小的效果中受益，甚至超出在小批量上对梯度取平均的效果，那就太好了。完成这一任务的一种选择是用“泄漏平均值”（leaky average）来替代梯度计算：

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

其中$\beta \in (0, 1)$。这有效地把瞬时梯度替换为对多个*过去*梯度取平均的结果。$\mathbf{v}$称为*速度*（velocity）。它积累过去的梯度，类似于一个重球沿目标函数地形滚下时对过去力的积分。为了更详细地了解发生了什么，让我们把$\mathbf{v}_t$递归地展开为

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

较大的$\beta$相当于长程平均，而较小的$\beta$相对于梯度法只是轻微的修正。新的梯度替换不再指向特定实例上的最速下降方向，而是指向过去梯度的加权平均方向。这使我们能够获得在一个批量上取平均的大部分好处，而无需实际计算该批量上梯度的代价。稍后我们将更详细地重新审视这一平均过程。

上述推理构成了现在所谓的*加速*梯度方法（例如带动量的梯度）的基础。它们还有一个额外的好处：当优化问题条件不佳时（即存在某些方向，在这些方向上的进展比在其他方向上慢得多，就像一条狭窄的峡谷），它们要有效得多。此外，它们允许我们对后续梯度取平均，以获得更稳定的下降方向。诚然，即使对于无噪声的凸问题，加速这一点也是动量有效并且效果如此之好的关键原因之一。

正如人们所期望的那样，由于其有效性，动量是深度学习及其后优化中一个被深入研究的主题。关于深入的分析和交互式动画，例如见 :citet:`Goh.2017` 那篇精彩的[阐述性文章](https://distill.pub/2017/momentum/)。它由 :citet:`Polyak.1964` 提出。:citet:`Nesterov.2018` 在凸优化的背景下作了详细的理论讨论。深度学习中动量有益早已为人所知。详见例如 :citet:`Sutskever.Martens.Dahl.ea.2013` 的讨论。

### 条件不佳的问题

为了更好地理解动量法的几何性质，我们重新审视梯度下降，不过使用一个明显不那么令人愉快的目标函数。回想一下，在 :numref:`sec_gd` 中我们使用了$f(\mathbf{x}) = x_1^2 + 2 x_2^2$，即一个中等程度变形的椭球目标。我们通过在$x_1$方向上拉伸它来进一步变形这个函数：

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

和之前一样，$f$在$(0, 0)$处取得最小值。这个函数在$x_1$方向上*非常*平坦。让我们看看像之前那样对这个新函数执行梯度下降会发生什么。我们选取学习率$0.4$。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

根据构造，$x_2$方向上的梯度比水平的$x_1$方向上的梯度*大得多*，而且变化快得多。因此我们陷入两个不可取的选择之间：如果选择较小的学习率，我们确保解在$x_2$方向上不发散，但会受困于$x_1$方向上缓慢的收敛。相反，使用较大的学习率时，我们在$x_1$方向上进展迅速，但在$x_2$方向上发散。下面的例子说明了即使学习率从$0.4$略微增大到$0.6$会发生什么。$x_1$方向上的收敛有所改善，但整体解的质量差得多。

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### 动量法

动量法使我们能够解决上面描述的梯度下降问题。
观察上面的优化轨迹，我们可能会直觉地认为对过去的梯度取平均会效果很好。毕竟，在$x_1$方向上，这会聚集对齐良好的梯度，从而增大我们每步覆盖的距离。相反，在梯度振荡的$x_2$方向上，聚合梯度会因振荡相互抵消而减小步长。
使用$\mathbf{v}_t$代替梯度$\mathbf{g}_t$，得到如下更新方程：

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

注意，当$\beta = 0$时，我们恢复为普通的梯度下降。在深入探讨数学性质之前，让我们快速看看该算法在实际中的表现。

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

正如我们所看到的，即使使用与之前相同的学习率，动量法仍然能很好地收敛。让我们看看减小动量参数时会发生什么。将其减半为$\beta = 0.25$，得到的轨迹几乎完全不收敛。尽管如此，它比没有动量时（此时解发散）要好得多。

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

注意，我们可以把动量法与随机梯度下降结合起来，特别是与小批量随机梯度下降结合。唯一的改变是，在这种情况下我们用$\mathbf{g}_t$替换梯度$\mathbf{g}_{t, t-1}$。最后，为方便起见，我们在时间$t=0$时初始化$\mathbf{v}_0 = 0$。让我们看看泄漏平均值实际上对更新做了什么。

### 有效样本权重

回想一下，$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$。在极限情况下，这些项之和为$\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$。换句话说，在梯度下降或随机梯度下降中，我们不是迈出大小为$\eta$的一步，而是迈出大小为$\frac{\eta}{1-\beta}$的一步，同时处理一个可能表现好得多的下降方向。这是一举两得。为了说明对于不同的$\beta$选择，加权是如何表现的，考虑下图。

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 实际实验

让我们看看动量法在实际中如何发挥作用，即在合适的优化器环境中使用时。为此，我们需要一个更具可扩展性的实现。

### 从零开始实现

与（小批量）随机梯度下降相比，动量法需要维护一组辅助变量，即速度。它与梯度的形状相同（也与优化问题的变量形状相同）。在下面的实现中，我们把这些变量称为`states`。

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

让我们看看这在实际中如何运作。

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

当我们将动量超参数`momentum`增大到 0.9 时，这相当于有效样本量显著增大为$\frac{1}{1 - 0.9} = 10$。我们把学习率略微降低到$0.01$，以保持情况可控。

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

进一步降低学习率可以解决非光滑优化问题的任何问题。将其设为$0.005$可以得到良好的收敛性质。

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 简洁实现

在 Gluon 中几乎不需要做什么，因为标准的`sgd`求解器已经内置了动量。设置匹配的参数会得到非常相似的轨迹。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## 理论分析

到目前为止，$f(x) = 0.1 x_1^2 + 2 x_2^2$这个二维例子似乎相当人为。我们现在将看到，它实际上相当具有代表性，代表了一类可能遇到的问题，至少对于最小化凸二次目标函数的情形是如此。

### 二次凸函数

考虑函数

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

这是一个一般的二次函数。对于正定矩阵$\mathbf{Q} \succ 0$，即特征值为正的矩阵，它在$\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$处有最小化子，最小值为$b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。因此我们可以把$h$改写为

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

梯度由$\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$给出。也就是说，它由$\mathbf{x}$与最小化子之间的距离乘以$\mathbf{Q}$给出。因此，速度也是项$\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$的线性组合。

由于$\mathbf{Q}$是正定的，它可以通过其特征系统分解为$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$，其中$\mathbf{O}$是正交（旋转）矩阵，$\boldsymbol{\Lambda}$是由正特征值构成的对角矩阵。这允许我们进行变量替换，从$\mathbf{x}$变换到$\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$，从而得到一个大大简化的表达式：

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

这里$b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。由于$\mathbf{O}$只是一个正交矩阵，这不会以有意义的方式扰动梯度。用$\mathbf{z}$表示时，梯度下降变为

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

这个表达式中的重要事实是，梯度下降在不同特征空间之间*不会混合*。也就是说，当用$\mathbf{Q}$的特征系统来表示时，优化问题按坐标方式逐个进行。这也适用于

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

在这样做时，我们刚刚证明了以下定理：对于凸二次函数，带或不带动量的梯度下降都分解为沿着二次矩阵特征向量方向的坐标式优化。

### 标量函数

给定上述结果，让我们看看最小化函数$f(x) = \frac{\lambda}{2} x^2$时会发生什么。对于梯度下降，我们有

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

只要$|1 - \eta \lambda| < 1$，该优化就以指数速率收敛，因为经过$t$步后我们有$x_t = (1 - \eta \lambda)^t x_0$。这说明当我们增大学习率$\eta$直到$\eta \lambda = 1$时，收敛速率最初是如何改善的。超过这一点后情况会发散，而对于$\eta \lambda > 2$，优化问题会发散。

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

为了分析动量情形下的收敛性，我们首先用两个标量重写更新方程：一个用于$x$，一个用于速度$v$。这得到：

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

我们用$\mathbf{R}$来表示这个$2 \times 2$的、决定收敛行为的矩阵。经过$t$步后，初始选择$[v_0, x_0]$变为$\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$。因此，收敛速度取决于$\mathbf{R}$的特征值。关于出色的动画，见 :citet:`Goh.2017` 的 [Distill 文章](https://distill.pub/2017/momentum/)，详细分析见 :citet:`Flammarion.Bach.2015`。可以证明，当$0 < \eta \lambda < 2 + 2 \beta$时速度收敛。与梯度下降的$0 < \eta \lambda < 2$相比，这是一个更大的可行参数范围。这也表明，一般来说较大的$\beta$值是可取的。进一步的细节需要相当多的技术内容，我们建议感兴趣的读者查阅原始文献。

## 小结

* 动量法用过去梯度的泄漏平均值来代替梯度。这显著加速了收敛。
* 它对无噪声的梯度下降和（带噪声的）随机梯度下降都是可取的。
* 动量法防止了优化过程停滞，而这种情况在随机梯度下降中更可能发生。
* 由于对过去数据按指数方式降权，梯度的有效数量为$\frac{1}{1-\beta}$。
* 在凸二次问题情形中，可以对此进行详细显式的分析。
* 实现相当直接，但它需要我们额外存储一个状态向量（速度$\mathbf{v}$）。

## 练习

1. 使用动量超参数和学习率的其他组合，观察并分析不同的实验结果。
1. 对具有多个特征值的二次问题尝试梯度下降和动量法，即$f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$，例如$\lambda_i = 2^{-i}$。画出在初始化$x_i = 1$时$x$的值如何减小。
1. 推导$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$的最小值和最小化子。
1. 当我们执行带动量的随机梯度下降时，会有什么变化？当我们使用带动量的小批量随机梯度下降时会发生什么？对参数进行实验？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1071)
:end_tab:
