# Adagrad
:label:`sec_adagrad`

让我们首先考虑特征出现不频繁的学习问题。


## 稀疏特征与学习率

想象我们正在训练一个语言模型。为了获得良好的准确率，我们通常希望在持续训练的同时降低学习率，通常按照$\mathcal{O}(t^{-\frac{1}{2}})$或更慢的速率。现在考虑一个在稀疏特征上训练的模型，即只偶尔出现的特征。这在自然语言中很常见，例如我们遇到单词*预处理*（preconditioning）的可能性比*学习*（learning）低得多。然而，这在其他领域也很常见，例如计算广告和个性化协同过滤。毕竟，有许多东西只有少数人感兴趣。

与不频繁特征相关联的参数只有在这些特征出现时才会收到有意义的更新。在递减的学习率下，我们可能陷入这样一种情形：常见特征的参数相当快地收敛到其最优值，而对于不频繁的特征，我们在其最优值能够被确定之前，仍然没有足够频繁地观测到它们。换句话说，学习率要么对频繁特征衰减得太慢，要么对不频繁特征衰减得太快。

纠正这一问题的一种可能的权宜之计是，统计我们看到某个特定特征的次数，并将其作为调整学习率的时钟。也就是说，与其选择$\eta = \frac{\eta_0}{\sqrt{t + c}}$形式的学习率，我们可以使用$\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$。这里$s(i, t)$统计截至时间$t$我们观测到的特征$i$的非零元素个数。这实际上非常容易实现，且没有显著开销。然而，每当我们并不真正拥有稀疏性，而只是梯度往往很小、只有偶尔才很大时，这就会失效。毕竟，很难界定在哪里划定某个东西是否算作已观测特征的界线。

:citet:`Duchi.Hazan.Singer.2011` 提出的 Adagrad 通过用先前观测到的梯度的平方的聚合量来替代这个相当粗糙的计数器$s(i, t)$，从而解决了这一问题。特别地，它使用$s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$作为调整学习率的手段。这有两个好处：第一，我们不再需要判断梯度何时才算足够大。第二，它会随梯度的大小自动缩放。经常对应于大梯度的坐标会被显著缩小，而其他具有小梯度的坐标则得到温和得多的处理。在实践中，这为计算广告及相关问题带来了一种非常有效的优化过程。但这掩盖了 Adagrad 固有的一些额外好处，这些好处在预处理的背景下最容易理解。


## 预处理

凸优化问题很适合分析算法的特性。毕竟，对于大多数非凸问题，很难推导出有意义的理论保证，但*直觉*和*洞见*往往可以沿用。让我们看看最小化$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$的问题。

正如我们在 :numref:`sec_momentum` 中所看到的，可以用特征分解$\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$来重写这个问题，从而得到一个大大简化的问题，其中每个坐标都可以单独求解：

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

这里我们使用了$\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$，从而$\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$。修改后的问题的最小化子为$\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$，最小值为$-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$。这要容易计算得多，因为$\boldsymbol{\Lambda}$是包含$\mathbf{Q}$的特征值的对角矩阵。

如果我们稍微扰动$\mathbf{c}$，我们希望$f$的最小化子只有微小的变化。不幸的是，事实并非如此。虽然$\mathbf{c}$的微小变化会导致$\bar{\mathbf{c}}$同样微小的变化，但$f$（以及相应的$\bar{f}$）的最小化子却并非如此。当特征值$\boldsymbol{\Lambda}_i$很大时，我们只会看到$\bar{x}_i$以及$\bar{f}$的最小值有很小的变化。相反，对于小的$\boldsymbol{\Lambda}_i$，$\bar{x}_i$的变化可能非常剧烈。最大特征值与最小特征值之比称为优化问题的条件数（condition number）。

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

如果条件数$\kappa$很大，就很难准确地求解优化问题。我们需要确保在正确处理很大的动态范围取值时保持谨慎。我们的分析引出一个显而易见但略显天真的问题：我们难道不能简单地通过扭曲空间，使所有特征值都为$1$来“修正”这个问题吗？理论上这相当容易：我们只需要$\mathbf{Q}$的特征值和特征向量，就能把问题从$\mathbf{x}$重新缩放到$\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$。在新的坐标系中，$\mathbf{x}^\top \mathbf{Q} \mathbf{x}$可以简化为$\|\mathbf{z}\|^2$。唉，这是一个相当不切实际的建议。计算特征值和特征向量通常比求解实际问题*昂贵得多*。

虽然精确计算特征值可能很昂贵，但猜测它们，甚至只是近似地计算它们，可能就已经比完全不做任何事好得多。特别地，我们可以使用$\mathbf{Q}$的对角元素并据此重新缩放。这比计算特征值*便宜得多*。

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

在这种情况下，我们有$\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$，特别地，对所有$i$都有$\tilde{\mathbf{Q}}_{ii} = 1$。在大多数情况下，这会大大简化条件数。例如，在我们之前讨论的情形中，由于问题与坐标轴对齐，这将完全消除手头的问题。

不幸的是，我们还面临另一个问题：在深度学习中，我们通常甚至无法获得目标函数的二阶导数：对于$\mathbf{x} \in \mathbb{R}^d$，即使在小批量上计算二阶导数也可能需要$\mathcal{O}(d^2)$的空间和计算量，从而使其实际上不可行。Adagrad 的巧妙想法是，用某个既能相对廉价地计算又有效的代理量来代替黑塞矩阵那个难以捉摸的对角线——即梯度本身的大小。

为了理解为什么这有效，让我们看看$\bar{f}(\bar{\mathbf{x}})$。我们有

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

其中$\bar{\mathbf{x}}_0$是$\bar{f}$的最小化子。因此，梯度的大小既取决于$\boldsymbol{\Lambda}$，也取决于到最优性的距离。如果$\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$不变，那么这就足够了。毕竟，在这种情况下梯度$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$的大小就足够了。由于 AdaGrad 是一种随机梯度下降算法，即使在最优性处我们也会看到非零方差的梯度。因此，我们可以安全地把梯度的方差用作黑塞矩阵尺度的一种廉价代理。深入的分析超出了本节的范围（那将需要好几页）。关于细节，我们请读者参考 :cite:`Duchi.Hazan.Singer.2011`。

## 算法

让我们把上面的讨论形式化。我们使用变量$\mathbf{s}_t$来累积过去的梯度方差，如下所示。

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

这里运算按坐标逐元素进行。也就是说，$\mathbf{v}^2$的元素是$v_i^2$。同样，$\frac{1}{\sqrt{v}}$的元素是$\frac{1}{\sqrt{v_i}}$，$\mathbf{u} \cdot \mathbf{v}$的元素是$u_i v_i$。和之前一样，$\eta$是学习率，$\epsilon$是一个加性常数，确保我们不会除以$0$。最后，我们初始化$\mathbf{s}_0 = \mathbf{0}$。

正如在动量法情形中一样，我们需要跟踪一个辅助变量，在这里是为了允许每个坐标有各自的学习率。相对于 SGD，这并不会显著增加 Adagrad 的代价，原因很简单：主要代价通常是计算$l(y_t, f(\mathbf{x}_t, \mathbf{w}))$及其导数。

注意，在$\mathbf{s}_t$中累积平方梯度意味着$\mathbf{s}_t$基本上以线性速率增长（在实践中略慢于线性，因为梯度最初会减小）。这导致了一个$\mathcal{O}(t^{-\frac{1}{2}})$的学习率，不过是在每个坐标的基础上进行调整的。对于凸问题，这完全足够了。然而，在深度学习中，我们可能希望学习率衰减得更慢一些。这引出了若干 Adagrad 变体，我们将在后续章节中讨论。现在让我们看看它在一个二次凸问题上的表现。我们使用和之前相同的问题：

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

我们将使用与之前相同的学习率，即$\eta = 0.4$，来实现 Adagrad。正如我们所看到的，自变量的迭代轨迹更平滑。然而，由于$\boldsymbol{s}_t$的累积效应，学习率不断衰减，因此自变量在迭代的后期阶段移动得没有那么多。

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

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

当我们将学习率增大到$2$时，我们看到好得多的行为。这已经表明，即使是在无噪声的情况下，学习率的衰减也可能相当激进，我们需要确保参数适当地收敛。

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## 从零开始实现

与动量法一样，Adagrad 需要维护一个与参数形状相同的状态变量。

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

与 :numref:`sec_minibatch_sgd` 中的实验相比，我们使用
更大的学习率来训练模型。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 简洁实现

使用算法`adagrad`的`Trainer`实例，我们可以在 Gluon 中调用 Adagrad 算法。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## 小结

* Adagrad 在每个坐标的基础上动态地降低学习率。
* 它使用梯度的大小作为调整取得进展速度的手段——梯度大的坐标会用较小的学习率来补偿。
* 由于内存和计算约束，在深度学习问题中精确计算二阶导数通常是不可行的。梯度可以作为一种有用的代理量。
* 如果优化问题具有相当不均匀的结构，Adagrad 可以帮助减轻这种扭曲。
* Adagrad 对稀疏特征特别有效，对于不频繁出现的项，学习率需要衰减得更慢。
* 在深度学习问题上，Adagrad 有时在降低学习率方面可能过于激进。我们将在 :numref:`sec_adam` 的背景下讨论缓解这一问题的策略。

## 练习

1. 证明对于正交矩阵$\mathbf{U}$和向量$\mathbf{c}$，下式成立：$\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$。为什么这意味着在正交变量替换之后，扰动的幅度不变？
1. 对$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$以及将目标函数旋转 45 度后的$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$分别尝试 Adagrad。它的行为会有所不同吗？
1. 证明[盖尔什戈林圆盘定理](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)：矩阵$\mathbf{M}$的特征值$\lambda_i$至少对某个$j$的选择满足$|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$。
1. 盖尔什戈林定理告诉我们关于对角预处理矩阵$\textrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \textrm{diag}^{-\frac{1}{2}}(\mathbf{M})$的特征值的什么信息？
1. 对合适的深度网络尝试 Adagrad，例如将 :numref:`sec_lenet` 应用于 Fashion-MNIST。
1. 要获得不那么激进的学习率衰减，你需要如何修改 Adagrad？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1073)
:end_tab:
