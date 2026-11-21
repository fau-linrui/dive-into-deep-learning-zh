```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch'])
```

# 高斯过程先验

理解高斯过程（GP）对于推理模型构造与泛化、以及在包括主动学习和深度学习中的超参数调优在内的各种应用中达到最先进性能，都非常重要。高斯过程无处不在，了解它们是什么以及我们如何使用它们符合我们的利益。

本节介绍函数上的高斯过程 _先验_。在下一个笔记本中，我们将展示如何使用这些先验进行 _后验推断_ 并做出预测。下一节可以看作“高斯过程速览”，快速给出在实践中应用高斯过程所需的内容。

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix

d2l.set_figsize()
```

## 定义

高斯过程定义为 _一组随机变量，其中任意有限多个随机变量都具有联合高斯分布_。如果函数 $f(x)$ 是一个高斯过程，其 _均值函数_ 为 $m(x)$、_协方差函数_ 或 _核函数_ 为 $k(x,x')$，即 $f(x) \sim \mathcal{GP}(m, k)$，那么在任意一组输入点 $x$（时间、空间位置、图像像素等）处查询得到的任意一组函数值，都具有均值向量 $\mu$、协方差矩阵为 $K$ 的联合多元高斯分布：$f(x_1),\dots,f(x_n) \sim \mathcal{N}(\mu, K)$，其中 $\mu_i = E[f(x_i)] = m(x_i)$，$K_{ij} = \textrm{Cov}(f(x_i),f(x_j)) = k(x_i,x_j)$。

这个定义可能显得抽象难懂，但高斯过程实际上是非常简单的对象。任何函数

$$f(x) = w^{\top} \phi(x) = \langle w, \phi(x) \rangle,$$ :eqlabel:`eq_gp-function`

只要 $w$ 取自高斯（正态）分布，$\phi$ 是任意基函数向量——例如 $\phi(x) = (1, x, x^2, ..., x^d)^{\top}$——就是一个高斯过程。此外，任何高斯过程 f(x) 都可以表示为式 :eqref:`eq_gp-function` 的形式。让我们考虑几个具体例子，开始熟悉高斯过程，之后我们就能体会它们实际上多么简单和有用。

## 一个简单的高斯过程

假设 $f(x) = w_0 + w_1 x$，且 $w_0, w_1 \sim \mathcal{N}(0,1)$，其中 $w_0, w_1, x$ 都在一维空间中。我们也可以把这个函数等价地写成内积 $f(x) = (w_0, w_1)(1, x)^{\top}$。在上面的 :eqref:`eq_gp-function` 中，$w = (w_0, w_1)^{\top}$，$\phi(x) = (1,x)^{\top}$。

对任意 $x$，$f(x)$ 都是两个高斯随机变量之和。由于高斯分布对加法封闭，所以对任意 $x$，$f(x)$ 也是高斯随机变量。事实上，对任意特定的 $x$，我们都可以算出 $f(x)$ 服从 $\mathcal{N}(0,1+x^2)$。类似地，对任意一组输入 $x_1,\dots,x_n$，任意一组函数值 $(f(x_1),\dots,f(x_n))$ 的联合分布都是多元高斯分布。因此 $f(x)$ 是一个高斯过程。

简言之，$f(x)$ 是一个 _随机函数_，或者说是 _函数上的分布_。我们可以通过反复采样 $w_0, w_1$ 的值，并把对应的函数 $f(x)$（这些函数是斜率与截距各不相同的直线）可视化，来对这个分布获得一些直观认识，如下所示：

```{.python .input}
def lin_func(x, n_sample):
    preds = np.zeros((n_sample, x.shape[0]))
    for ii in range(n_sample):
        w = np.random.normal(0, 1, 2)
        y = w[0] + w[1] * x
        preds[ii, :] = y
    return preds

x_points = np.linspace(-5, 5, 50)
outs = lin_func(x_points, 10)
lw_bd = -2 * np.sqrt((1 + x_points ** 2))
up_bd = 2 * np.sqrt((1 + x_points ** 2))

d2l.plt.fill_between(x_points, lw_bd, up_bd, alpha=0.25)
d2l.plt.plot(x_points, np.zeros(len(x_points)), linewidth=4, color='black')
d2l.plt.plot(x_points, outs.T)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("f(x)", fontsize=20)
d2l.plt.show()
```

如果 $w_0$ 和 $w_1$ 改为从 $\mathcal{N}(0,\alpha^2)$ 中抽取，你想象一下改变 $\alpha$ 会如何影响函数上的分布？

## 从权重空间到函数空间

在上图中，我们看到了模型参数上的分布如何诱导出函数上的分布。虽然我们对于想要建模的函数往往有一些想法——它们是否光滑、是否具有周期性、是否快速变化，等等——但要针对这些参数进行推理却相当繁琐，因为参数在很大程度上难以解释。幸运的是，高斯过程提供了一种简单的机制，可以 _直接_ 对函数进行推理。由于高斯分布完全由其前两阶矩（均值和协方差矩阵）确定，高斯过程也就可以由其均值函数和协方差函数来确定。

在上面的例子中，均值函数为

$$m(x) = E[f(x)] = E[w_0 + w_1x] = E[w_0] + E[w_1]x = 0+0 = 0.$$

类似地，协方差函数为

$$k(x,x') = \textrm{Cov}(f(x),f(x')) = E[f(x)f(x')]-E[f(x)]E[f(x')] = E[w_0^2 + w_0w_1x' + w_1w_0x + w_1^2xx'] = 1 + xx'.$$

现在，我们无需从参数上的分布中采样，就可以直接指定函数上的分布并从中采样。例如，要从 $f(x)$ 中抽样，我们只需构造与我们想查询的任意一组 $x$ 相关的多元高斯分布，然后直接从中采样即可。我们很快就会发现这种表述方式有多么优越。

首先我们注意到，上面针对简单直线模型所做的推导基本可以原封不动地用于求出 _任意_ 形如 $f(x) = w^{\top} \phi(x)$（其中 $w \sim \mathcal{N}(u,S)$）的模型的均值函数和协方差函数。此时，均值函数为 $m(x) = u^{\top}\phi(x)$，协方差函数为 $k(x,x') = \phi(x)^{\top}S\phi(x')$。由于 $\phi(x)$ 可以表示任意非线性基函数组成的向量，因此我们考虑的是一类非常一般的模型，甚至包括具有 _无穷_ 多个参数的模型。

## 径向基函数（RBF）核

_径向基函数_（radial basis function，RBF）核是高斯过程乃至核机器中最常用的协方差函数。
该核函数的形式为 $k_{\textrm{RBF}}(x,x') = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$，其中 $a$ 是振幅参数，$\ell$ 是 _长度尺度_ 超参数。

让我们从权重空间出发推导这个核函数。考虑函数

$$f(x) = \sum_{i=1}^J w_i \phi_i(x), w_i  \sim \mathcal{N}\left(0,\frac{\sigma^2}{J}\right), \phi_i(x) = \exp\left(-\frac{(x-c_i)^2}{2\ell^2 }\right).$$

$f(x)$ 是若干径向基函数之和，这些径向基函数的宽度为 $\ell$，中心位于点 $c_i$ 处，如下图所示。


我们可以看出 $f(x)$ 具有 $w^{\top} \phi(x)$ 的形式，其中 $w = (w_1,\dots,w_J)^{\top}$，$\phi(x)$ 是一个包含各个径向基函数的向量。于是该高斯过程的协方差函数为

$$k(x,x') = \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x').$$

现在考虑当参数（以及基函数）的数量趋于无穷时会发生什么。令 $c_J = \log J$、$c_1 = -\log J$、$c_{i+1}-c_{i} = \Delta c = 2\frac{\log J}{J}$，并令 $J \to \infty$。协方差函数变为黎曼和：

$$k(x,x') = \lim_{J \to \infty} \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x') = \int_{c_0}^{c_\infty} \phi_c(x)\phi_c(x') dc.$$

通过令 $c_0 = -\infty$、$c_\infty = \infty$，我们把无穷多个基函数铺展到整条实轴上，相邻基函数之间相隔
的距离为 $\Delta c \to 0$：

$$k(x,x') = \int_{-\infty}^{\infty} \exp(-\frac{(x-c)^2}{2\ell^2}) \exp(-\frac{(x'-c)^2}{2\ell^2 }) dc = \sqrt{\pi}\ell \sigma^2 \exp(-\frac{(x-x')^2}{2(\sqrt{2} \ell)^2}) \propto k_{\textrm{RBF}}(x,x').$$

值得花一点时间消化我们在这里所做的推导。通过进入函数空间表示，我们推导出了如何用有限的计算量来表示一个具有 _无穷_ 多个参数的模型。具有 RBF 核的高斯过程是一个 _通用近似器_，能够以任意精度表示任何连续函数。从上面的推导中，我们可以直观地看出原因。我们可以取 $\ell \to 0$，把每个径向基函数坍缩为一个点质量，并给每个点质量赋予任意高度。

因此，具有 RBF 核的高斯过程是一个具有无穷多个参数、且比任何有限神经网络都灵活得多的模型。也许关于 _过参数化_ 神经网络的那些大肆讨论都放错了重点。正如我们将看到的，具有 RBF 核的高斯过程不会过拟合，事实上在小数据集上还能提供格外有说服力的泛化性能。此外，:cite:`zhang2021understanding` 中的那些例子——例如能够完美拟合随机标签的图像、却仍能在结构化问题上取得良好泛化——（可以借助高斯过程完美复现） :cite:`wilson2020bayesian`。神经网络并不像我们以为的那样与众不同。

通过直接从函数上的分布中采样，我们可以对具有 RBF 核的高斯过程以及 _长度尺度_ 等超参数建立更多的直觉。与之前一样，这包括一个简单的过程：

1. 选择我们想要查询高斯过程的输入 $x$ 点：$x_1,\dots,x_n$。
2. 计算 $m(x_i)$、$i = 1,\dots,n$，并对 $i,j = 1,\dots,n$ 计算 $k(x_i,x_j)$，分别构成均值向量和协方差矩阵 $\mu$ 与 $K$，其中 $(f(x_1),\dots,f(x_n)) \sim \mathcal{N}(\mu, K)$。
3. 从该多元高斯分布中采样，得到样本函数值。
4. 多采样几次，以可视化更多在这些点上查询得到的样本函数。

下图展示了这一过程。

```{.python .input}
def rbfkernel(x1, x2, ls=4.):  #@save
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls / 2) * (dist ** 2))

x_points = np.linspace(0, 5, 50)
meanvec = np.zeros(len(x_points))
covmat = rbfkernel(x_points,x_points, 1)

prior_samples= np.random.multivariate_normal(meanvec, covmat, size=5);
d2l.plt.plot(x_points, prior_samples.T, alpha=0.5)
d2l.plt.show()
```

## 神经网络核

机器学习中关于高斯过程的研究，是由神经网络的研究引发的。Radford Neal 当时在研究越来越大的贝叶斯神经网络，最终在 1994 年（后来于 1996 年发表，因为这是 NeurIPS 最臭名昭著的拒稿之一）证明了这类具有无穷多个隐藏单元的神经网络会变成具有特定核函数的高斯过程 :cite:`neal1996bayesian`。人们对这一推导的兴趣再次兴起，例如神经正切核等思想被用于研究神经网络的泛化性质 :cite:`matthews2018gaussian` :cite:`novak2018bayesian`。我们可以如下推导神经网络核。

考虑一个具有一个隐藏层的神经网络函数 $f(x)$：

$$f(x) = b + \sum_{i=1}^{J} v_i h(x; u_i).$$

$b$ 是偏置，$v_i$ 是从隐藏层到输出层的权重，$h$ 是任意有界隐藏单元传递函数，$u_i$ 是输入到隐藏层的权重，$J$ 是隐藏单元的数量。令 $b$ 与 $v_i$ 相互独立、均值为零，方差分别为 $\sigma_b^2$ 和 $\sigma_v^2/J$，并令 $u_i$ 具有独立同分布。于是我们可以利用中心极限定理证明：任意一组函数值 $f(x_1),\dots,f(x_n)$ 都具有联合多元高斯分布。

对应高斯过程的均值函数和协方差函数为：

$$m(x) = E[f(x)] = 0$$

$$k(x,x') = \textrm{cov}[f(x),f(x')] = E[f(x)f(x')] = \sigma_b^2 + \frac{1}{J} \sum_{i=1}^{J} \sigma_v^2 E[h_i(x; u_i)h_i(x'; u_i)]$$

在某些情况下，我们实际上可以闭式地计算这个协方差函数。令 $h(x; u) = \textrm{erf}(u_0 + \sum_{j=1}^{P} u_j x_j)$，其中 $\textrm{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} dt$，且 $u \sim \mathcal{N}(0,\Sigma)$。此时 $k(x,x') = \frac{2}{\pi} \textrm{sin}(\frac{2 \tilde{x}^{\top} \Sigma \tilde{x}'}{\sqrt{(1 + 2 \tilde{x}^{\top} \Sigma \tilde{x})(1 + 2 \tilde{x}'^{\top} \Sigma \tilde{x}')}})$。

RBF 核是 _平稳的_，这意味着它具有 _平移不变性_，因此可以写成 $\tau = x-x'$ 的函数。直观上，平稳性意味着函数的高层性质（例如变化速率）不会随着我们在输入空间中移动而改变。然而，神经网络核是 _非平稳的_。下面我们展示从具有该核的高斯过程中采样的样本函数。可以看到，该函数在原点附近看起来有质的不同。

## 小结


进行贝叶斯推断的第一步是设定先验。高斯过程可用于指定一个完整的函数先验。从传统的“权重空间”建模视角出发，我们可以从模型的函数形式入手，并引入其参数上的分布，从而诱导出函数上的先验。我们也可以直接在函数空间中指定先验分布，其性质由核函数控制。函数空间方法有许多优点。我们可以构建实际上对应无穷多个参数的模型，却只使用有限的计算量！此外，虽然这些模型具有极大的灵活性，但它们也对哪类函数在 _先验上_ 更可能做出强假设，从而在小数据集上带来相对较好的泛化。

函数空间中模型的假设直观上由核函数控制，核函数往往编码了函数的高层性质，例如光滑性和周期性。许多核函数是平稳的，即具有平移不变性。从具有平稳核的高斯过程中抽取的函数，无论我们在输入空间的哪个位置观察，都具有大致相同的高层性质（例如变化速率）。

高斯过程是一类相对一般的模型，其中包含我们已经熟悉的许多模型，包括多项式、傅里叶级数等，只要参数上具有高斯先验即可。它们还包括具有无穷多个参数的神经网络，甚至不需要参数服从高斯分布。Radford Neal 发现的这一联系，促使机器学习研究者从神经网络转向高斯过程。


## 练习

1. 从具有 Ornstein-Uhlenbeck（OU）核 $k_{\textrm{OU}}(x,x') = \exp\left(-\frac{1}{2\ell}||x - x'|\right)$ 的高斯过程中抽取样本先验函数。如果固定相同的长度尺度 $\ell$，这些函数与来自具有 RBF 核的高斯过程的样本函数相比有何不同？

2. 改变 RBF 核的 _振幅_ $a^2$ 会如何影响函数上的分布？

3. 假设我们构造 $u(x) = f(x) + 2g(x)$，其中 $f(x) \sim \mathcal{GP}(m_1,k_1)$、$g(x) \sim \mathcal{GP}(m_2,k_2)$。$u(x)$ 是高斯过程吗？如果是，它的均值函数和协方差函数是什么？

4. 假设我们构造 $g(x) = a(x)f(x)$，其中 $f(x) \sim \mathcal{GP}(0,k)$、$a(x) = x^2$。$g(x)$ 是高斯过程吗？如果是，它的均值函数和协方差函数是什么？$a(x)$ 起什么作用？从 $g(x)$ 中抽取的样本函数是什么样子？

5. 假设我们构造 $u(x) = f(x)g(x)$，其中 $f(x) \sim \mathcal{GP}(m_1,k_1)$、$g(x) \sim \mathcal{GP}(m_2,k_2)$。$u(x)$ 是高斯过程吗？如果是，它的均值函数和协方差函数是什么？

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/12116)
:end_tab:
