```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("gpytorch")
```

# 高斯过程推断

本节将展示如何使用上一节介绍的高斯过程先验进行后验推断并做出预测。我们从回归开始，回归中的推断可以用 _闭式_ 解完成。这一节是“高斯过程速览”，可帮助你快速在实践中上手高斯过程。我们将先从零开始编写所有基本操作的代码，然后介绍 [GPyTorch](https://gpytorch.ai/)，它能让使用最先进的高斯过程以及与深度神经网络集成变得更加方便。我们将在下一节深入讨论这些更高级的主题。在那一节中，我们还会考虑需要近似推断的情形——分类、点过程或任何非高斯似然。

## 回归中的后验推断

_观测_ 模型把我们要学习的函数 $f(x)$ 与观测 $y(x)$ 联系起来，二者都由某个输入 $x$ 索引。在分类中，$x$ 可以是图像的像素，$y$ 可以是相关的类别标签。在回归中，$y$ 通常表示连续输出，例如地表温度、海平面、$CO_2$ 浓度等。

在回归中，我们常假设输出由一个潜在的、无噪声的函数 $f(x)$ 加上独立同分布的高斯噪声 $\epsilon(x)$ 给出：

$$y(x) = f(x) + \epsilon(x),$$
:eqlabel:`eq_gp-regression`

其中 $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$。令 $\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$ 为我们的训练观测向量，$\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$ 为在训练输入 $X = {x_1, \dots, x_n}$ 处查询得到的潜在无噪声函数值向量。

我们假设 $f(x) \sim \mathcal{GP}(m,k)$，这意味着任意一组函数值 $\textbf{f}$ 具有联合多元高斯分布，其均值向量为 $\mu_i = m(x_i)$，协方差矩阵为 $K_{ij} = k(x_i,x_j)$。RBF 核 $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$ 会是协方差函数的一个标准选择。为记号简洁，我们假设均值函数 $m(x)=0$；我们的推导之后可以很容易地推广。

假设我们想在一组输入 $$X_* = x_{*1},x_{*2},\dots,x_{*m}.$$ 处做预测。那么我们想要求出 $x^2$ 和 $p(\mathbf{f}_* | \mathbf{y}, X)$。在回归情形下，在求出 $\mathbf{f}_* = f(X_*)$ 与 $\mathbf{y}$ 上的联合分布之后，我们可以方便地利用高斯恒等式求出这个分布。

如果在训练输入 $X$ 处计算式 :eqref:`eq_gp-regression`，我们得到 $\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$。根据高斯过程的定义（见上一节），$\mathbf{f} \sim \mathcal{N}(0,K(X,X))$，其中 $K(X,X)$ 是一个 $n \times n$ 矩阵，由在所有可能的输入对 $x_i, x_j \in X$ 处计算我们的协方差函数（又称 _核函数_）得到。$\mathbf{\epsilon}$ 就是由从 $\mathcal{N}(0,\sigma^2)$ 中独立同分布采样得到的向量，因此服从分布 $\mathcal{N}(0,\sigma^2I)$。于是 $\mathbf{y}$ 是两个相互独立的多元高斯变量之和，因而服从分布 $\mathcal{N}(0, K(X,X) + \sigma^2I)$。还可以证明 $\textrm{cov}(\mathbf{f}_*, \mathbf{y}) = \textrm{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$，其中 $K(X_*,X)$ 是一个 $m \times n$ 矩阵，由在所有测试输入与训练输入对处计算核函数得到。

$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}
\sim
\mathcal{N}\left(0, 
\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_*) \\
K(X_*,X) & K(X_*,X_*)
\end{bmatrix}
\right)
$$

然后，我们可以利用标准的高斯恒等式从联合分布求出条件分布（例如，参见 Bishop 第 2 章），
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$，其中 $m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$，且 $S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$。

通常，我们不需要使用完整的预测协方差矩阵 $S$，而是用 $S$ 的对角线来表示每个预测的不确定性。出于这个原因，我们常常写出单个测试点 $x_*$ 的预测分布，而不是一组测试点的预测分布。

核矩阵带有我们希望估计的参数 $\theta$，例如上面 RBF 核的振幅 $a$ 和长度尺度 $\ell$。为此我们使用 _边际似然_ $p(\textbf{y} | \theta, X)$，我们在推导边际分布以求出 $\textbf{y},\textbf{f}_*$ 上的联合分布时已经得到了它。正如我们将看到的，边际似然可分解为模型拟合项和模型复杂度项，并自动为学习超参数编码了一种奥卡姆剃刀的观念。完整的讨论见 MacKay 第 28 章 :cite:`mackay2003information`，以及 Rasmussen 和 Williams 第 5 章 :cite:`rasmussen2006gaussian`。

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os

d2l.set_figsize()
```

## 高斯过程回归中做预测与学习核超参数的公式

这里我们列出你在高斯过程回归中学习超参数和做预测时将要用到的公式。同样，我们假设回归目标向量 $\textbf{y}$ 由输入 $X = \{x_1,\dots,x_n\}$ 索引，并希望在测试输入 $x_*$ 处做出预测。我们假设存在独立同分布的、均值为零、方差为 $\sigma^2$ 的加性高斯噪声。我们对潜在的无噪声函数使用高斯过程先验 $f(x) \sim \mathcal{GP}(m,k)$，其均值函数为 $m$、核函数为 $k$。核本身带有我们想要学习的参数 $\theta$。例如，如果使用 RBF 核 $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$，我们想要学习 $\theta = \{a^2, \ell^2\}$。令 $K(X,X)$ 表示一个 $n \times n$ 矩阵，对应于在所有可能的 $n$ 个训练输入对处计算核函数。令 $K(x_*,X)$ 表示一个 $1 \times n$ 向量，由计算 $k(x_*, x_i)$（$i=1,\dots,n$）得到。令 $\mu$ 表示在所有训练点 $x$ 处计算均值函数 $m(x)$ 所得到的均值向量。

通常，在处理高斯过程时，我们遵循两步流程。
1. 通过关于核超参数最大化边际似然，学习核超参数 $\hat{\theta}$。
2. 把预测均值用作点预测器，用两倍预测标准差构成 95\% 置信集，以这些学习到的超参数 $\hat{\theta}$ 为条件。

对数边际似然就是一个对数高斯密度，其形式为：
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

预测分布的形式为：
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## 学习与预测公式的解读

关于高斯过程的预测分布，有几个要点需要注意：

* 尽管模型类别很灵活，但高斯过程回归仍可以 _闭式_ 地做 _精确的_ 贝叶斯推断。除了学习核超参数之外，并没有 _训练_ 过程。我们可以确切地写出要用来做预测的公式。高斯过程在这方面相当特殊，这极大地成就了它们的便利性、多功能性和持久流行。

* 预测均值 $a_*$ 是训练目标 $\textbf{y}$ 的线性组合，其权重由核 $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$ 给出。正如我们将看到的，核（及其超参数）因而在模型的泛化性质中起着至关重要的作用。

* 预测均值显式依赖于目标值 $\textbf{y}$，而预测方差不依赖。相反，预测不确定性会随着测试输入 $x_*$ 远离目标位置 $X$ 而增大，其规律由核函数决定。不过，不确定性会通过从数据中学习到的核超参数 $\theta$ 隐式地依赖于目标值 $\textbf{y}$ 的取值。

* 边际似然可分解为模型拟合项和模型复杂度（对数行列式）项。边际似然倾向于选择那些能给出与数据仍然一致的最简单拟合的超参数。

* 关键的计算瓶颈在于求解一个线性系统，以及为 $n$ 个训练点计算一个 $n \times n$ 对称正定矩阵 $K(X,X)$ 上的对数行列式。朴素地看，这些操作各自需要 $\mathcal{O}(n^3)$ 次计算，并且需要 $\mathcal{O}(n^2)$ 的存储来保存核（协方差）矩阵的每个元素，通常还要从 Cholesky 分解开始。历史上，这些瓶颈把高斯过程限制在训练点少于约 10,000 个的问题上，并给高斯过程带来了“速度慢”的名声——但这种说法在近十年里已经不再准确了。在高级主题中，我们将讨论如何把高斯过程扩展到有数百万个点的问题。

* 对于流行的核函数选择，$K(X,X)$ 往往接近奇异，这在进行 Cholesky 分解或求解线性系统的其他操作时可能引起数值问题。幸运的是，在回归中我们通常处理的是 $K_{\theta}(X,X)+\sigma^2I$，这样噪声方差 $\sigma^2$ 就被加到 $K(X,X)$ 的对角线上，显著改善了其条件数。如果噪声方差很小，或者我们在做无噪声回归，通常的做法是在对角线上加入少量“抖动”（jitter），量级约为 $10^{-6}$，以改善条件数。

## 从零开始的完整示例

让我们生成一些回归数据，然后用高斯过程拟合这些数据，同时从零开始实现每一步。
我们将从
$$y(x) = \sin(x) + \frac{1}{2}\sin(4x) + \epsilon,$$ 采样数据，其中 $\epsilon \sim \mathcal{N}(0,\sigma^2)$。我们希望找到的无噪声函数是 $f(x) = \sin(x) + \frac{1}{2}\sin(4x)$。我们首先使用噪声标准差 $\sigma = 0.25$。

```{.python .input}
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("Observations y", fontsize=20)
d2l.plt.show()
```

这里，带噪声的观测显示为圆圈，我们想找到的无噪声函数显示为蓝色。

现在，让我们为潜在的无噪声函数 $f(x)\sim \mathcal{GP}(m,k)$ 指定一个高斯过程先验。我们使用均值函数 $m(x) = 0$，以及 RBF 协方差函数（核函数）
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right).$$

```{.python .input}
mean = np.zeros(test_x.shape[0])
cov = d2l.rbfkernel(test_x, test_x, ls=0.2)
```

我们从长度尺度 0.2 开始。在拟合数据之前，重要的是考虑我们是否指定了一个合理的先验。让我们可视化来自该先验的一些样本函数，以及 95\% 置信集（我们相信真实函数有 95\% 的概率位于该区域内）。

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
d2l.plt.plot(test_x, mean, linewidth=2.)
d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), 
                 alpha=0.25)
d2l.plt.show()
```

这些样本看起来合理吗？这些函数的高层性质与我们试图建模的数据类型相符吗？

现在，让我们构造任意测试点 $x_*$ 处后验预测分布的均值和方差。

$$
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$

$$
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$

在做出预测之前，我们应该学习核超参数 $\theta$ 和噪声方差 $\sigma^2$。让我们把长度尺度初始化为 0.75，因为与正在拟合的数据相比，我们的先验函数看起来变化得太快。我们还会猜测噪声标准差 $\sigma$ 为 0.75。

为了学习这些参数，我们将关于这些参数最大化边际似然。

$$
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$
$$
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$


也许我们的先验函数变化得太快了。让我们猜测长度尺度为 0.4。我们还会猜测噪声标准差为 0.75。这些只是超参数的初始化——我们将从边际似然中学习这些参数。

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = d2l.rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1] ** 2 * \
                                         np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)


learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est,post_sig_est]), 
                                   bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]
```

在这个例子中，我们学习到的长度尺度为 0.299，噪声标准差为 0.24。注意，学到的噪声与真实噪声极其接近，这有助于说明我们的高斯过程对该问题有很好的设定。

一般来说，仔细思考如何选择核函数以及如何初始化超参数至关重要。虽然边际似然优化对初始化相对稳健，但它并非不受糟糕初始化的影响。试着用各种各样的初始化运行上述脚本，看看你会得到什么结果。

现在，让我们用这些学到的超参数做预测。

```{.python .input}
K_x_xstar = d2l.rbfkernel(train_x, test_x, ls=ell)
K_x_x = d2l.rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = d2l.rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
d2l.plt.show()
```

我们看到橙色的后验均值几乎完美地匹配真实的无噪声函数！注意，我们展示的 95\% 置信集是针对潜在的 _无噪声_（真实）函数的，而不是针对数据点的。我们看到该置信集完全包含真实函数，而且看起来既不过宽也不过窄。我们不会希望也不应期望它包含数据点。如果我们希望得到观测的置信集，就应计算

```{.python .input}
lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
```

不确定性有两个来源：_认知_ 不确定性，表示 _可约_ 不确定性；以及 _偶然_（aleatoric）或 _不可约_ 不确定性。这里的 _认知_ 不确定性表示关于无噪声函数真实值的不确定性。当我们远离数据点时，这种不确定性应当增大，因为在远离数据的地方，与我们的数据一致的函数值有更多种可能。随着我们观测到越来越多的数据，我们对真实函数的信念会变得更加确定，认知不确定性随之消失。此情形下的 _偶然_ 不确定性就是观测噪声，因为数据是带着这种噪声给到我们的，它无法被减小。

数据中的 _认知_ 不确定性由潜在无噪声函数的方差 np.diag(post\_cov) 刻画。_偶然_ 不确定性则由噪声方差 post_sig_est**2 刻画。

不幸的是，人们在如何表示不确定性方面往往不够严谨，许多论文展示的误差棒完全没有明确的定义，不清楚可视化的是认知不确定性还是偶然不确定性，还是两者兼有，也常常把噪声方差与噪声标准差、标准差与标准误、置信区间与置信集等混为一谈。如果不对不确定性所表示的含义加以明确，它基本上是没有意义的。

本着密切关注我们的不确定性究竟表示什么的宗旨，有一点至关重要：我们是对无噪声函数的方差估计取 _两倍_ 的 _平方根_。由于我们的预测分布是高斯分布，这个量使我们能够构造 95\% 置信集，表示我们对“该区间有 95\% 的可能性包含真实函数”这一信念。噪声 _方差_ 则处在完全不同的尺度上，可解释性差得多。

最后，让我们看看 20 个后验样本。这些样本告诉我们，事后看来，我们认为哪些类型的函数可能拟合我们的数据。

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
d2l.plt.show()
```

在基础回归应用中，最常见的是把后验预测均值和标准差分别用作点预测器和不确定性度量。在更高级的应用中，例如使用蒙特卡洛采集函数的贝叶斯优化，或用于基于模型的强化学习的高斯过程，往往需要取后验样本。然而，即使在基础应用中并非严格需要，这些样本也能让我们对数据的拟合情况有更多直觉，并且往往适合放进可视化中。

## 用 GPyTorch 让工作更轻松

正如我们所见，完全从零开始实现基本的高斯过程回归其实相当容易。然而，一旦我们想要探索各种核函数选择、考虑近似推断（即使对分类也是必需的）、把高斯过程与神经网络结合，或者数据集甚至超过约 10,000 个点，从零开始的实现就变得笨拙而繁琐。一些最有效的可扩展高斯过程推断方法（例如 SKI，也称为 KISS-GP）可能需要数百行代码来实现高级数值线性代数例程。

在这些情况下，_GPyTorch_ 库会让我们的生活轻松很多。我们将在后续关于高斯过程数值方法和高级方法的笔记本中进一步讨论 GPyTorch。GPyTorch 库包含[许多示例](https://github.com/cornellius-gp/gpytorch/tree/master/examples)。为了对这个包有一个直观感受，我们将走一遍[简单回归示例](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb)，展示如何修改它以便用 GPyTorch 复现我们上面的结果。仅仅为了复现上面的基本回归，这段代码看起来可能很多，从某种意义上说确实如此。但我们可以只改动下面的几行代码，就能立刻使用各种核函数、可扩展推断技术和近似推断，而不必编写可能成千上万行的新代码。

```{.python .input}
# First let's convert our data into tensors for use with PyTorch
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

# We are using exact GP inference with a zero mean and RBF kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

这个代码块把数据整理成 GPyTorch 所需的格式，并指明我们使用的是精确推断，以及我们想使用的均值函数（零）和核函数（RBF）。只要调用例如 gpytorch.kernels.matern_kernel() 或 gpyotrch.kernels.spectral_mixture_kernel()，我们就能非常容易地使用任何其他核函数。到目前为止，我们只讨论了精确推断，即可以在不做任何近似的情况下推断预测分布。对于高斯过程，只有当似然为高斯时——更具体地说，当我们假设观测是由高斯过程表示的无噪声函数加上高斯噪声生成时——我们才能进行精确推断。在后续笔记本中，我们将考虑其他情形，例如分类，此时我们无法做出这些假设。

```{.python .input}
# Initialize Gaussian likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
# Find optimal model hyperparameters
model.train()
likelihood.train()
# Use the adam optimizer, includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# Set our loss as the negative log GP marginal likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
```

这里，我们显式指定了想要使用的似然（高斯）、用于训练核超参数的目标（此处为边际似然），以及用于优化该目标的过程（此处为 Adam）。我们注意到，虽然我们用的是 Adam 这个“随机”优化器，但在这种情况下它其实是全批量 Adam。由于边际似然不按数据实例分解，我们不能使用基于数据“小批量”的优化器并保证收敛。GPyTorch 也支持 L-BFGS 等其他优化器。与标准深度学习不同，做好边际似然的优化与良好的泛化有很强的对应关系，这往往使我们倾向于使用 L-BFGS 这类强大的优化器——前提是它们的开销不至于过高。

```{.python .input}
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()
```

这里我们实际运行优化过程，每 10 次迭代输出一次损失值。

```{.python .input}
# Get into evaluation (predictive posterior) mode
test_x = torch.tensor(test_x)
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x)) 
```

上面的代码块使我们能够在测试输入上做出预测。

```{.python .input}
with torch.no_grad():
    # Initialize plot
    f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower bounds for 95\% credible set (in this case, in
    # observation space)
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
               '95% Credible Set'])
```

最后，我们画出拟合结果。

我们看到两者的拟合结果几乎完全相同。有几点需要注意：GPyTorch 使用 _平方后的_ 长度尺度和观测噪声。例如，我们在从零开始的代码中学到的噪声标准差约为 0.283，而 GPyTorch 找到的噪声方差是 $0.81 \approx 0.283^2$。在 GPyTorch 的图中，我们展示的是 _观测空间_ 中的置信集，而不是潜在函数空间中的置信集，以说明它们确实覆盖了观测到的数据点。

## 小结

我们可以把高斯过程先验与数据结合起来形成后验，并用后验做出预测。我们还可以构造边际似然，它对于自动学习核超参数很有用，而这些超参数控制着高斯过程的变化速率等性质。对于回归，构造后验和学习核超参数的操作很简单，大约只需十几行代码。对于任何想快速上手高斯过程的读者来说，本笔记本是一份很好的参考。我们还介绍了 GPyTorch 库。虽然用于基本回归的 GPyTorch 代码相对较长，但只要稍加修改，就可以用于其他核函数，或用于我们将在后续笔记本中讨论的更高级功能，例如可扩展推断，或用于分类的非高斯似然。


## 练习

1. 我们强调了 _学习_ 核超参数的重要性，以及超参数和核函数对高斯过程泛化性质的影响。试着跳过学习超参数这一步，改为猜测各种长度尺度和噪声方差，并检查它们对预测的影响。使用较大的长度尺度会怎样？较小的长度尺度呢？较大的噪声方差？较小的噪声方差？
2. 我们说过，边际似然不是凸目标，但像长度尺度和噪声方差这样的超参数在高斯过程回归中可以被可靠地估计。这通常是成立的——事实上，与空间统计中拟合经验自相关函数（“协方差图”）的传统方法相比，边际似然在学习长度尺度超参数方面要好 _得多_。可以说，机器学习对高斯过程研究的最大贡献（至少在对可扩展推断的近期研究之前）就是引入了用于超参数学习的边际似然。

_然而_，即便是这些参数的不同组合，也会为许多数据集提供可解释性不同的合理说明，从而导致我们的目标中出现局部最优。如果我们使用较大的长度尺度，我们就假设真实的潜在函数是缓慢变化的。如果观测数据 _确实_ 变化显著，那么要合理地使用较大的长度尺度，唯一的办法就是使用较大的噪声方差。另一方面，如果我们使用较小的长度尺度，我们的拟合将对数据中的变化非常敏感，几乎没有余地用噪声（偶然不确定性）来解释变化。

试着看看你能否找到这些局部最优：用很大的长度尺度配合较大的噪声初始化，以及用较小的长度尺度配合较小的噪声初始化。你会收敛到不同的解吗？
  
3. 我们说过，贝叶斯方法的一个根本优势在于它能够自然地表示 _认知_ 不确定性。在上面的例子中，我们无法完全看到认知不确定性的效果。改为用 `test_x = np.linspace(0, 10, 1000)` 进行预测试试。当预测超出数据范围时，95\% 置信集会怎样？它在该区间内覆盖真实函数吗？如果在该区域只可视化偶然不确定性，会怎样？

4. 试着用 10,000、20,000 和 40,000 个训练点运行上面的例子，并测量运行时间。训练时间如何随规模变化？或者，运行时间如何随测试点数量变化？对预测均值和预测方差来说情况是否不同？请既通过理论推导训练和测试的时间复杂度来回答这个问题，也通过用不同数量的点运行上面的代码来回答。

5. 试着用不同的协方差函数（例如 Matern 核）运行 GPyTorch 示例。结果如何变化？GPyTorch 库中的谱混合核呢？有些核是否比其他核更容易训练边际似然？有些核是否对长程预测比短程预测更有价值？

6. 在我们的 GPyTorch 示例中，我们绘制了包含观测噪声的预测分布，而在“从零开始”的示例中，我们只包含了认知不确定性。重新做一遍 GPyTorch 示例，但这次只绘制认知不确定性，并与从零开始的结果进行比较。现在两个预测分布看起来一样吗？（应该一样。）

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/12117)
:end_tab:
