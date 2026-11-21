# RMSProp
:label:`sec_rmsprop`


:numref:`sec_adagrad` 中的一个关键问题是，学习率按预先设定的调度策略下降，其量级实际上为 $\mathcal{O}(t^{-\frac{1}{2}})$。这对凸问题通常是合适的，但对非凸问题（例如深度学习中遇到的问题）可能并不理想。然而，Adagrad 的按坐标自适应性作为一种预条件子（preconditioner）是非常理想的。

:citet:`Tieleman.Hinton.2012` 提出了 RMSProp 算法，作为一种简单的修正，把学习率调度与按坐标自适应的学习率解耦。问题在于，Adagrad 将梯度 $\mathbf{g}_t$ 的平方累积到状态向量 $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ 中。结果，由于缺少规范化，$\mathbf{s}_t$ 会无界地持续增长，在算法收敛时基本上呈线性增长。

解决这个问题的一种方法是用 $\mathbf{s}_t / t$。对于 $\mathbf{g}_t$ 的合理分布，这样做会收敛。遗憾的是，由于该过程会记住数值的完整轨迹，可能要经过很长时间，极限行为才开始起作用。另一种方法是使用泄漏平均值（leaky average），与我们在动量法中使用的方式相同，即 $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$，其中参数 $\gamma > 0$。保持其他所有部分不变，就得到了 RMSProp 算法。

## 算法

让我们详细写出这些方程。

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

常数 $\epsilon > 0$ 通常取为 $10^{-6}$，以确保不会出现除以零或步长过大的情况。有了这一展开式，我们现在就可以独立于按坐标施加的缩放来控制学习率 $\eta$。就泄漏平均值而言，我们可以采用之前在动量法中应用的同样推理。展开 $\mathbf{s}_t$ 的定义可得

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

正如前面在 :numref:`sec_momentum` 中那样，我们使用 $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$。因此，权重之和被规范化为 $1$，一次观测的半衰期为 $\gamma^{-1}$。下面我们针对不同的 $\gamma$ 取值，可视化过去 40 个时间步的权重。

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
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## 从头实现

和之前一样，我们用二次函数 $f(\mathbf{x})=0.1x_1^2+2x_2^2$ 来观察 RMSProp 的轨迹。回想一下，在 :numref:`sec_adagrad` 中，当使用学习率为 0.4 的 Adagrad 时，由于学习率下降得太快，变量在算法后期移动得非常缓慢。由于 $\eta$ 是单独控制的，RMSProp 不会出现这种情况。

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

接下来，我们实现用于深度网络的 RMSProp。这同样简单直白。

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

我们将初始学习率设为 0.01，权重项 $\gamma$ 设为 0.9。也就是说，$\mathbf{s}$ 平均聚合了过去 $1/(1-\gamma) = 10$ 次梯度平方的观测值。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 简洁实现

由于 RMSProp 是相当流行的算法，它也可以在 `Trainer` 实例中使用。我们只需使用名为 `rmsprop` 的算法来实例化它，并把 $\gamma$ 赋给参数 `gamma1`。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## 小结

* RMSProp 与 Adagrad 非常相似，两者都用梯度的平方来缩放系数。
* RMSProp 与动量法一样使用泄漏平均值。但是，RMSProp 使用该技术来调整逐系数的预条件子。
* 在实践中，学习率需要由实验者进行调度。
* 系数 $\gamma$ 决定了调整逐坐标缩放时历史记录有多长。

## 练习

1. 如果设 $\gamma = 1$，实验上会发生什么？为什么？
1. 旋转优化问题以最小化 $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$。收敛性会发生什么变化？
1. 在真实的机器学习问题上试验 RMSProp，例如在 Fashion-MNIST 上训练。尝试采用不同的学习率调整方案。
1. 你会随着优化的推进而调整 $\gamma$ 吗？RMSProp 对此有多敏感？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1075)
:end_tab:
