```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 数值稳定性与初始化
:label:`sec_numerical_stability`


到目前为止，我们实现的每个模型
都要求按照某个预先指定的分布
来初始化其参数。
在此以前，我们对初始化方案习以为常，
忽略了这些选择是如何做出的细节。
你甚至可能产生过这样的印象：这些选择
并不特别重要。
恰恰相反，初始化方案的选择
在神经网络学习中起着重要作用，
而且它对保持数值稳定性可能至关重要。
此外，这些选择还会以有趣的方式
与非线性激活函数的选择联系在一起。
我们选择哪个函数以及如何初始化参数，
可以决定我们的优化算法收敛得有多快。
这里选择不当，可能会让我们在训练时
遇到梯度爆炸或梯度消失。
在本节中，我们更深入地探讨这些主题，
并讨论一些有用的经验法则，
你会发现它们在深度学习的整个职业生涯中
都很有用。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## 梯度消失与梯度爆炸

考虑一个深达 $L$ 层的网络，
输入为 $\mathbf{x}$，输出为 $\mathbf{o}$。
设每一层 $l$ 由变换 $f_l$ 定义，
该变换以权重 $\mathbf{W}^{(l)}$ 为参数，
其隐藏层输出为 $\mathbf{h}^{(l)}$（令 $\mathbf{h}^{(0)} = \mathbf{x}$），
那么我们的网络可以表示为：

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ and thus } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

如果所有隐藏层输出和输入都是向量，
我们就可以把 $\mathbf{o}$ 关于
任意一组参数 $\mathbf{W}^{(l)}$ 的梯度写成如下形式：

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

换句话说，这个梯度是
$L-l$ 个矩阵
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
与梯度向量 $\mathbf{v}^{(l)}$ 的乘积。
因此，我们容易遭遇
数值下溢的同样问题——这类问题在
把过多概率相乘在一起时经常出现。
处理概率时一个常用的技巧是
切换到对数空间，即把
压力从数值表示的尾数转移到指数上。
不幸的是，我们上面的问题更为严重：
初始时矩阵 $\mathbf{M}^{(l)}$ 的特征值可能种类繁多。
它们可能很小也可能很大，而
它们的乘积可能*非常大*或者*非常小*。

不稳定梯度造成的风险
超出了数值表示的范围。
大小不可预测的梯度
还会威胁我们优化算法的稳定性。
我们可能面临这样的参数更新：
(i) 过大，毁掉我们的模型
（*梯度爆炸*问题）；
或者 (ii) 过小
（*梯度消失*问题），
使得学习变得不可能，因为每次更新时
参数几乎不动。


### (**梯度消失**)

造成梯度消失问题的一个常见元凶，
是紧接在每层线性运算之后所附加的
激活函数 $\sigma$ 的选择。
历史上，sigmoid 函数
$1/(1 + \exp(-x))$（在 :numref:`sec_mlp` 中已经介绍过）
曾很受欢迎，因为它类似于一个阈值函数。
由于早期的人工神经网络受到
生物神经网络的启发，
神经元要么*完全*激活、要么*完全不*激活
（像生物神经元那样）的想法看起来很有吸引力。
让我们更仔细地考察 sigmoid，
看看它为何会导致梯度消失。

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

如你所见，(**sigmoid 的梯度
在其输入很大和很小时都会消失**)。
此外，当反向传播穿过许多层时，
除非我们恰好处在“金发姑娘区”
（即许多 sigmoid 的输入都接近零的区域），
否则整个乘积的梯度可能会消失。
当我们的网络有很多层时，
除非格外小心，否则梯度
很可能在某一层被截断。
的确，这个问题曾困扰深度网络的训练。
因此，更稳定
（但在神经科学上不那么合理）的 ReLU
已成为实践者的默认选择。


### [**梯度爆炸**]

相反的问题——梯度爆炸——
同样可能令人烦恼。
为了更好地说明这一点，
我们画出 100 个高斯随机矩阵，
并把它们与某个初始矩阵相乘。
对于我们选取的尺度
（选择方差 $\sigma^2=1$），
矩阵乘积会爆炸。
如果这种情况是由深度网络的初始化
引起的，那么梯度下降优化器就绝无收敛的可能。

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # Generate PRNG keys
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### 打破对称性

神经网络设计中的另一个问题，
是其参数化中固有的对称性。
假设我们有一个简单的 MLP，
含一个隐藏层和两个单元。
在这种情况下，我们可以置换第一层的权重 $\mathbf{W}^{(1)}$，
并同样地置换
输出层的权重，
从而得到同一个函数。
第一个和第二个隐藏单元之间
没有任何特别之处。
换句话说，每一层的隐藏单元之间存在
置换对称性。

这不仅仅是一个理论上的麻烦。
考虑前面提到的
含两个隐藏单元的单隐藏层 MLP。
为便于说明，
假设输出层把这两个隐藏单元变换为仅一个输出单元。
设想一下，如果我们将隐藏层的
所有参数都初始化为 $\mathbf{W}^{(1)} = c$（$c$ 为某个常数），会发生什么。
在这种情况下，在前向传播中，
两个隐藏单元接收相同的输入和参数，
产生相同的激活值，
并将其馈送给输出单元。
在反向传播中，
对输出单元关于参数 $\mathbf{W}^{(1)}$ 求导，得到的梯度所有元素都取相同的值。
因此，在基于梯度的迭代（例如小批量随机梯度下降）之后，
$\mathbf{W}^{(1)}$ 的所有元素仍然取相同的值。
这样的迭代本身
永远不会*打破对称性*，
我们可能永远无法发挥
网络的表达能力。
隐藏层的行为
就好像它只有一个单元。
注意，虽然小批量随机梯度下降不会打破这种对称性，
但暂退法正则化（将在后面介绍）会！


## 参数初始化

解决——或至少缓解——
上述问题的一种方式，就是谨慎地初始化。
正如我们后面将会看到的，
优化过程中的额外注意
以及合适的正则化可以进一步增强稳定性。


### 默认初始化

在前面的章节中，例如在 :numref:`sec_linear_concise` 中，
我们使用正态分布
来初始化权重的取值。
如果我们不指定初始化方法，框架将
使用默认的随机初始化方法，对于中等规模的问题，
这在实践中往往效果很好。






### Xavier 初始化
:label:`subsec_xavier`

让我们来看一看*不带非线性*的某个全连接层
输出 $o_{i}$ 的尺度分布。
设该层有 $n_\textrm{in}$ 个输入 $x_j$
及与之关联的权重 $w_{ij}$，
一个输出由下式给出

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

权重 $w_{ij}$ 都
独立地从同一分布中抽取。
此外，假设该分布
均值为零、方差为 $\sigma^2$。
注意这并不意味着分布必须是高斯分布，
只是要求均值和方差存在。
现在，假设该层的输入 $x_j$
也具有零均值和方差 $\gamma^2$，
并且它们与 $w_{ij}$ 独立、彼此之间也独立。
在这种情况下，我们可以计算 $o_i$ 的均值：

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

以及方差：

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

保持方差不变的一种方式，
是令 $n_\textrm{in} \sigma^2 = 1$。
现在考虑反向传播。
在那里我们面临类似的问题，
只不过梯度是从更靠近输出的层传播而来的。
利用与前向传播相同的推理，
我们发现，除非 $n_\textrm{out} \sigma^2 = 1$，
否则梯度的方差可能会爆炸，
其中 $n_\textrm{out}$ 是该层的输出个数。
这让我们陷入两难：
我们不可能同时满足这两个条件。
于是，我们干脆设法满足：

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

这就是现在已成为标准、
且在实践中有益的*Xavier 初始化*背后的推理，
它以该方法的创造者中的第一作者命名 :cite:`Glorot.Bengio.2010`。
通常，Xavier 初始化
从均值为零、方差为
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$ 的高斯分布中
采样权重。
我们也可以把这一点推广到
从均匀分布中采样权重时
对方差的选择上。
注意均匀分布 $U(-a, a)$ 的方差为 $\frac{a^2}{3}$。
把 $\frac{a^2}{3}$ 代入我们对 $\sigma^2$ 的条件，
促使我们按照下式初始化

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

尽管上述数学推理中
不存在非线性的假设
在神经网络中很容易被违反，
但 Xavier 初始化方法
在实践中效果很好。


### 拓展

上面的推理仅仅触及了
现代参数初始化方法的皮毛。
一个深度学习框架往往实现了十几种不同的经验法则。
此外，参数初始化仍然是
深度学习中基础研究的一个热门领域。
其中包括针对
绑定（共享）参数、超分辨率、
序列模型以及其他情形的
专门经验法则。
例如，
:citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018` 通过使用一种精心设计的初始化方法，
展示了训练
10000 层神经网络而不使用架构技巧的可能性。

如果你对这个主题感兴趣，我们建议
深入研读本模块的内容，
阅读提出并分析每种经验法则的论文，
然后探索关于该主题的最新文献。
也许你会偶然发现、甚至发明
一个巧妙的想法，并为深度学习框架贡献一份实现。


## 小结

梯度消失和梯度爆炸是深度网络中的常见问题。需要在参数初始化上格外小心，以确保梯度和参数保持在良好受控的范围内。
需要初始化经验法则来确保初始梯度既不太大也不太小。
随机初始化是确保在优化之前打破对称性的关键。
Xavier 初始化表明，对于每一层，任何输出的方差都不受输入个数的影响，任何梯度的方差都不受输出个数的影响。
ReLU 激活函数缓解了梯度消失问题。这可以加快收敛。

## 练习

1. 除了 MLP 各层中的置换对称性之外，你还能设计出其他需要打破的对称性情形吗？
1. 我们能否把线性回归或 softmax 回归中的所有参数都初始化为同一个值？
1. 查阅关于两个矩阵乘积特征值的解析界。这对确保梯度良态有什么启示？
1. 如果我们知道某些项会发散，能否在事后修正？可以查阅关于逐层自适应速率缩放的论文以获取灵感 :cite:`You.Gitman.Ginsburg.2017`。


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17986)
:end_tab:
