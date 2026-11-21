# 随机梯度下降
:label:`sec_sgd`

在前面的章节中，我们在训练过程中一直使用随机梯度下降，但没有解释它为什么有效。
为了阐明这一点，
我们刚刚在 :numref:`sec_gd` 中描述了梯度下降的基本原理。
在本节中，我们将更详细地讨论
*随机梯度下降*（stochastic gradient descent）。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## 随机梯度更新

在深度学习中，目标函数通常是训练数据集中每个样例的损失函数的平均值。
给定包含$n$个样例的训练数据集，
我们假设$f_i(\mathbf{x})$是关于索引为$i$的训练样例的损失函数，
其中$\mathbf{x}$是参数向量。
于是我们得到目标函数

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

目标函数在$\mathbf{x}$处的梯度计算为

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

如果使用梯度下降，每次独立变量迭代的计算代价为$\mathcal{O}(n)$，它随$n$线性增长。因此，当训练数据集较大时，梯度下降每次迭代的代价会更高。

随机梯度下降（SGD）降低了每次迭代的计算代价。在随机梯度下降的每次迭代中，我们随机均匀地采样一个数据样例的索引$i\in\{1,\ldots, n\}$，并计算梯度$\nabla f_i(\mathbf{x})$来更新$\mathbf{x}$：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

其中$\eta$是学习率。我们可以看到，每次迭代的计算代价从梯度下降的$\mathcal{O}(n)$降到常数$\mathcal{O}(1)$。此外，我们要强调，随机梯度$\nabla f_i(\mathbf{x})$是对完整梯度$\nabla f(\mathbf{x})$的无偏估计，因为

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

这意味着，平均而言，随机梯度是对梯度的良好估计。

现在，我们将通过向梯度添加均值为 0、方差为 1 的随机噪声来模拟随机梯度下降，从而将其与梯度下降进行比较。

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

正如我们所看到的，随机梯度下降中变量的轨迹比我们在 :numref:`sec_gd` 中观察到的梯度下降的轨迹噪声大得多。这是由于梯度的随机性质。也就是说，即使我们到达最小值附近，我们仍然会受到瞬时梯度通过$\eta \nabla f_i(\mathbf{x})$注入的不确定性的影响。即使经过 50 步，质量仍然不太好。更糟糕的是，在更多的步骤之后它也不会改善（我们鼓励你用更多的步数进行实验来确认这一点）。这让我们只剩下唯一的选择：改变学习率$\eta$。然而，如果我们把它选得太小，一开始就不会有任何有意义的进展。另一方面，如果我们把它选得太大，如上所见，我们也得不到好的解。解决这些相互冲突的目标的唯一方法是随着优化的进行*动态地*降低学习率。

这也是在`sgd`步骤函数中加入学习率函数`lr`的原因。在上面的例子中，由于我们将关联的`lr`函数设为常数，学习率调度的任何功能都处于休眠状态。

## 动态学习率

用与时间相关的学习率$\eta(t)$替换$\eta$，增加了控制优化算法收敛性的复杂度。特别是，我们需要弄清楚$\eta$应该衰减多快。如果衰减太快，我们会在优化过早停止。如果衰减太慢，我们会在优化上浪费太多时间。以下是随时间调整$\eta$时使用的几种基本策略（稍后我们将讨论更高级的策略）：

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

在第一种*分段常数*场景中，我们降低学习率，例如每当优化进展停滞时。这是训练深度网络的一种常见策略。或者，我们可以通过*指数衰减*更积极地降低学习率。不幸的是，这通常会导致在算法收敛之前的过早停止。一种流行的选择是$\alpha = 0.5$的*多项式衰减*。在凸优化情形中，有许多证明表明这种速率表现良好。

让我们看看指数衰减在实际中的样子。

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

正如预期的那样，参数的方差显著减小。然而，这是以无法收敛到最优解$\mathbf{x} = (0, 0)$为代价的。即使经过 1000 次迭代步骤，我们仍然离最优解非常远。事实上，该算法根本没有收敛。另一方面，如果我们使用多项式衰减，让学习率按步数的平方根的倒数衰减，那么仅仅 50 步之后收敛就会变得更好。

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

如何设置学习率还有多得多的选择。例如，我们可以从较小的学习率开始，然后迅速增大，再将其调小，尽管调小的速度更慢。我们甚至可以在较小和较大的学习率之间交替。这类调度方案多种多样。目前让我们专注于能够进行完整理论分析的学习率调度，即在凸设定下的学习率。对于一般的非凸问题，很难获得有意义的收敛保证，因为一般情况下最小化非线性非凸问题是 NP 难的。关于综述，例如见 Tibshirani 2015 的精彩[讲义](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf)。

## 凸目标的收敛性分析

以下关于随机梯度下降在凸目标函数上的收敛性分析是可选的，主要用于传达对该问题的更多直觉。
我们只给出其中最简单的证明之一 :cite:`Nesterov.Vial.2000`。
存在显著更高级的证明技巧，例如当目标函数表现得特别好时。


假设目标函数$f(\boldsymbol{\xi}, \mathbf{x})$对所有$\boldsymbol{\xi}$都是关于$\mathbf{x}$的凸函数。
更具体地说，
我们考虑如下随机梯度下降更新：

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

其中$f(\boldsymbol{\xi}_t, \mathbf{x})$
是关于在步骤$t$从某个分布中抽取的训练样例$\boldsymbol{\xi}_t$的目标函数，
而$\mathbf{x}$是模型参数。
记

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

为期望风险，$R^*$为其关于$\mathbf{x}$的最小值。最后令$\mathbf{x}^*$为最小化子（我们假设它存在于$\mathbf{x}$有定义的区域内）。在这种情况下，我们可以跟踪时间$t$的当前参数$\mathbf{x}_t$与风险最小化子$\mathbf{x}^*$之间的距离，并观察它是否随时间改善：

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

我们假设随机梯度$\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$的$\ell_2$范数被某个常数$L$所界，因此我们有

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`


我们主要关心$\mathbf{x}_t$与$\mathbf{x}^*$之间的距离*在期望意义下*如何变化。事实上，对于任意特定的步骤序列，距离很可能会增大，这取决于我们遇到的是哪个$\boldsymbol{\xi}_t$。因此我们需要界定这个点积。
由于对任意凸函数$f$，对所有$\mathbf{x}$和$\mathbf{y}$都有
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$，
所以由凸性我们有

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

将两个不等式 :eqref:`eq_sgd-L` 和 :eqref:`eq_sgd-f-xi-xstar` 代入 :eqref:`eq_sgd-xt+1-xstar`，我们得到时间$t+1$时参数之间距离的一个界，如下所示：

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

这意味着，只要当前损失与最优损失之间的差超过$\eta_t L^2/2$，我们就在取得进展。由于这个差必定收敛到零，因此学习率$\eta_t$也需要*消失*。

接下来，我们对 :eqref:`eqref_sgd-xt-diff` 取期望。这得到

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

最后一步涉及对$t \in \{1, \ldots, T\}$的不等式求和。由于该和会相消，并且通过舍弃较低的一项，我们得到

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

注意，我们利用了$\mathbf{x}_1$是给定的这一事实，因此可以去掉期望。最后定义

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

由于

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

根据詹森不等式（在 :eqref:`eq_jensens-inequality` 中令$i=t$、$\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$）以及$R$的凸性，可以推出$E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$，因此

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

将其代入不等式 :eqref:`eq_sgd-x1-xstar`，得到界

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

其中$r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$是初始参数选择与最终结果之间距离的界。简而言之，收敛速度取决于随机梯度的范数如何被界定（$L$），以及初始参数值离最优性有多远（$r$）。注意，这个界是用$\bar{\mathbf{x}}$而不是$\mathbf{x}_T$来表示的。之所以如此，是因为$\bar{\mathbf{x}}$是优化路径的平滑版本。
当$r, L$和$T$已知时，我们可以选取学习率$\eta = r/(L \sqrt{T})$。这给出的上界是$rL/\sqrt{T}$。也就是说，我们以$\mathcal{O}(1/\sqrt{T})$的速率收敛到最优解。





## 随机梯度与有限样本

到目前为止，在谈论随机梯度下降时，我们有些随意。我们假设抽取实例$x_i$（通常带有标签$y_i$），它们来自某个分布$p(x, y)$，并且我们以某种方式用它来更新模型参数。特别地，对于有限的样本量，我们只是论证了离散分布$p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$
（其中$\delta_{x_i}$和$\delta_{y_i}$是某些函数）
允许我们在其上执行随机梯度下降。

然而，这并不是我们真正所做的。在本节的玩具例子中，我们只是给本来非随机的梯度添加了噪声，也就是说，我们假装有数据对$(x_i, y_i)$。事实证明，在这里这是合理的（详细讨论见练习）。更令人不安的是，在之前的所有讨论中，我们显然没有这样做。相反，我们对所有实例*恰好迭代一次*。要理解为什么这更可取，考虑相反的情形，即我们从离散分布中*有放回地*采样$n$个观测。随机选中元素$i$的概率是$1/n$。因此，*至少*选中它一次的概率是

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

类似的推理表明，选中某个样本（即训练样例）*恰好一次*的概率为

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

与*无放回*采样相比，有放回采样会导致方差增大、数据效率降低。因此，在实践中我们采用后者（这也是本书自始至终的默认选择）。最后请注意，重复遍历训练数据集时，会以*不同的*随机顺序遍历它。


## 小结

* 对于凸问题，我们可以证明，对于广泛选择的学习率，随机梯度下降都会收敛到最优解。
* 对于深度学习，这通常不成立。然而，对凸问题的分析为我们如何着手优化提供了有用的洞见，即逐步降低学习率，但不要降得太快。
* 学习率太小或太大都会产生问题。在实践中，往往只有经过多次实验才能找到合适的学习率。
* 训练数据集中样例越多，梯度下降每次迭代的计算代价就越高，因此在这些情况下更倾向于使用随机梯度下降。
* 在非凸情形下，一般无法获得随机梯度下降的最优性保证，因为需要检查的局部最小值的数量很可能是指数级的。




## 练习

1. 用不同的学习率调度和不同的迭代次数对随机梯度下降进行实验。特别地，画出与最优解$(0, 0)$的距离随迭代次数变化的函数图像。
1. 证明：对于函数$f(x_1, x_2) = x_1^2 + 2 x_2^2$，向梯度添加正态噪声等价于最小化损失函数$f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$，其中$\mathbf{x}$取自正态分布。
1. 比较从$\{(x_1, y_1), \ldots, (x_n, y_n)\}$中有放回采样和无放回采样时随机梯度下降的收敛情况。
1. 如果某个梯度（或者更确切地说，与之相关的某个坐标）一直大于所有其他梯度，你会如何修改随机梯度下降求解器？
1. 假设$f(x) = x^2 (1 + \sin x)$。$f$有多少个局部最小值？你能否改变$f$，使得要最小化它就必须评估所有局部最小值？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1067)
:end_tab:
