# Adam
:label:`sec_adam`

在直到本节为止的讨论中，我们遇到了许多用于高效优化的技术。下面我们详细回顾一下：

* 我们看到，在求解优化问题时，:numref:`sec_sgd` 比梯度下降更有效，例如，这是因为它对冗余数据具有内在的鲁棒性。
* 我们看到，:numref:`sec_minibatch_sgd` 借助向量化带来了显著额外的效率，即在一个小批量中使用更大的观测集。这是实现高效的多机、多 GPU 以及整体并行处理的关键。
* :numref:`sec_momentum` 增加了一种聚合过去梯度历史的机制，以加速收敛。
* :numref:`sec_adagrad` 使用逐坐标缩放，得到了计算上高效的预条件子。
* :numref:`sec_rmsprop` 将逐坐标缩放与学习率调整解耦。

Adam :cite:`Kingma.Ba.2014` 把所有这些技术结合成一个高效的学习算法。正如预期的那样，该算法作为深度学习中较为稳健、有效的优化算法之一，已经变得相当流行。不过，它也并非没有问题。特别地，:cite:`Reddi.Kale.Kumar.2019` 表明，在某些情形下，由于方差控制不佳，Adam 可能发散。在后续工作 :citet:`Zaheer.Reddi.Sachan.ea.2018` 中，作者提出了一个针对 Adam 的快速修复，称为 Yogi，用来解决这些问题。稍后我们会详细介绍。现在先来回顾 Adam 算法。

## 算法

Adam 的关键组成部分之一是，它使用指数加权移动平均（exponential weighted moving average，也称为泄漏平均值）来估计动量以及梯度的二阶矩。也就是说，它使用如下状态变量

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

这里 $\beta_1$ 和 $\beta_2$ 是非负的权重参数。常见的取法是 $\beta_1 = 0.9$ 和 $\beta_2 = 0.999$。也就是说，方差估计的移动*比动量项慢得多*。请注意，如果我们初始化 $\mathbf{v}_0 = \mathbf{s}_0 = 0$，那么初始阶段会有相当程度的偏差，数值偏向较小的一侧。这可以利用 $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$ 这一事实来重新规范化各项，从而加以修正。相应地，规范化后的状态变量为

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

有了正确的估计，我们现在可以写出更新方程。首先，我们以与 RMSProp 非常相似的方式对梯度进行缩放，得到

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

与 RMSProp 不同，我们的更新使用动量 $\hat{\mathbf{v}}_t$，而不是梯度本身。此外，还有一个细微的表面差别：缩放使用的是 $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$，而不是 $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$。前者在实践中可以说效果略好，因此这里与 RMSProp 有所不同。通常我们取 $\epsilon = 10^{-6}$，以在数值稳定性与保真度之间取得良好的折中。

至此，计算更新所需的所有部件都已就位。这一结果略显平淡：我们得到一个形式简单的更新

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

回顾 Adam 的设计，其灵感清晰可见。动量和缩放都清楚地体现在状态变量中。它们颇为特殊的定义迫使我们做偏差修正（用稍有不同的初始化方式和更新条件本可以避免这一点）。其次，有了 RMSProp 作基础，两者的组合同样直截了当。最后，显式的学习率 $\eta$ 使我们能够控制步长，以应对收敛问题。

## 实现

从头实现 Adam 并不太难。为方便起见，我们把时间步计数器 $t$ 存放在 `hyperparams` 字典中。除此之外，一切都很直接。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

现在可以使用 Adam 来训练模型了。我们使用学习率 $\eta = 0.01$。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

更简洁的实现也很直接，因为 `adam` 是 Gluon `trainer` 优化库中提供的算法之一。因此，我们只需为 Gluon 中的实现传入配置参数。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Adam 的问题之一是，当 $\mathbf{s}_t$ 中的二阶矩估计爆炸时，即使在凸设置下它也可能无法收敛。作为一种修复，:citet:`Zaheer.Reddi.Sachan.ea.2018` 提出对 $\mathbf{s}_t$ 进行改进的更新（以及初始化）。为了理解其中的道理，让我们把 Adam 的更新改写如下：

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

当 $\mathbf{g}_t^2$ 的方差很大或更新稀疏时，$\mathbf{s}_t$ 可能会过快忘记过去的值。对此的一种可能修复是把 $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ 替换为 $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$。这样，更新幅度就不再取决于偏差的大小。由此得到 Yogi 更新

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

作者还建议在更大的初始批量上初始化动量，而不仅仅使用逐点的初始估计。我们略去这些细节，因为它们对当前讨论并不重要，而且即使没有这一点，收敛效果依然相当不错。

```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## 小结

* Adam 把许多优化算法的特性结合成了一条相当稳健的更新规则。
* Adam 在 RMSProp 的基础上构建，同样对小批量随机梯度使用指数加权移动平均（EWMA）。
* Adam 使用偏差修正，以调整估计动量和二阶矩时启动缓慢的问题。
* 对于方差显著的梯度，我们可能遇到收敛问题。这些问题可以通过采用更大的小批量，或改用对 $\mathbf{s}_t$ 的改进估计来缓解。Yogi 提供了这样一种替代方案。

## 练习

1. 调整学习率，观察并分析实验结果。
1. 你能否改写动量与二阶矩的更新，使其不需要偏差修正？
1. 为什么随着收敛的进行，需要减小学习率 $\eta$？
1. 尝试构造一个 Adam 发散而 Yogi 收敛的例子？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1078)
:end_tab:


:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1079)
:end_tab:
