# Adadelta
:label:`sec_adadelta`

Adadelta 是 AdaGrad 的又一个变体（:numref:`sec_adagrad`）。其主要区别在于，它减小了学习率按坐标自适应的幅度。此外，传统上人们认为它没有学习率，因为它使用变化量本身作为未来变化的校准。该算法由 :citet:`Zeiler.2012` 提出。有了前面这些算法的讨论，它的内容相当直截了当。

## 算法

简而言之，Adadelta 使用两个状态变量：$\mathbf{s}_t$ 存储梯度二阶矩的泄漏平均值（leaky average），$\Delta\mathbf{x}_t$ 存储模型本身参数变化的二阶矩的泄漏平均值。请注意，我们沿用作者原始的记号与命名，以便与其他文献和实现保持兼容（在动量法、Adagrad、RMSProp 和 Adadelta 中用不同的希腊字母表示用途相同的参数，并没有其他真正的理由）。

下面给出 Adadelta 的技术细节。给定当前的参数为 $\rho$，我们与 :numref:`sec_rmsprop` 类似地得到如下泄漏更新：

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

与 :numref:`sec_rmsprop` 的不同之处在于，我们使用重新缩放的梯度 $\mathbf{g}_t'$ 进行更新，即

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

那么重新缩放的梯度 $\mathbf{g}_t'$ 是什么呢？我们可以按如下方式计算它：

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

其中 $\Delta \mathbf{x}_{t-1}$ 是重新缩放梯度的平方 $\mathbf{g}_t'$ 的泄漏平均值。我们将 $\Delta \mathbf{x}_{0}$ 初始化为 $0$，并在每一步用 $\mathbf{g}_t'$ 更新它，即

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

其中加入 $\epsilon$（一个如 $10^{-5}$ 的小值）以保持数值稳定性。



## 实现

Adadelta 需要为每个变量维护两个状态变量：$\mathbf{s}_t$ 和 $\Delta\mathbf{x}_t$。由此得到如下实现。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

选择 $\rho = 0.9$ 相当于每次参数更新的半衰期为 10。这往往效果相当好。我们得到如下行为。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

对于简洁实现，我们只需使用高层 API 中的 Adadelta 算法。由此得到如下一行代码，调用一下子就紧凑多了。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it is converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## 小结

* Adadelta 没有学习率参数。相反，它使用参数本身的变化率来自适应学习率。
* Adadelta 需要两个状态变量来存储梯度和参数变化的二阶矩。
* Adadelta 使用泄漏平均值来维护相应统计量的滑动估计。

## 练习

1. 调整 $\rho$ 的值。会发生什么？
1. 说明如何在不使用 $\mathbf{g}_t'$ 的情况下实现该算法。为什么这可能是个好主意？
1. Adadelta 真的不需要学习率吗？你能找到使 Adadelta 失效的优化问题吗？
1. 比较 Adadelta 与 Adagrad 和 RMSProp，讨论它们的收敛行为。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1077)
:end_tab:
