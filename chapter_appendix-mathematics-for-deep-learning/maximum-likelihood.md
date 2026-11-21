# 最大似然
:label:`sec_maximum_likelihood`

机器学习中最常见的思考方式之一是最大似然视角。这个观点的核心是：当我们面对一个参数未知的概率模型时，那些能使数据出现概率最高的参数，就是最可能的参数。

## 最大似然原理

对最大似然可以给出一种贝叶斯式的解释，这种解释有助于我们思考。假设我们有一个参数为$\boldsymbol{\theta}$的模型，以及一组数据样例$X$。为具体起见，我们可以设想$\boldsymbol{\theta}$是表示硬币抛出后正面朝上概率的单个数值，而$X$是一系列独立的抛硬币结果。稍后我们将深入考察这个例子。

如果我们想找到模型参数的最可能取值，那就意味着我们想求解

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

根据贝叶斯定理，这等价于

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

表达式$P(X)$是与参数无关的生成数据概率，它根本不依赖于$\boldsymbol{\theta}$，因此可以在不改变$\boldsymbol{\theta}$最佳选择的前提下将其丢弃。类似地，我们现在可以假设：对于哪一组参数更好，我们没有任何先验假设。于是也可以声明$P(\boldsymbol{\theta})$同样不依赖于该参数！例如，在我们的抛硬币例子中这样做是合理的：正面朝上的概率可以是$[0,1]$中的任意值，而我们对硬币是否公平没有任何先验信念（这种先验通常称为*无信息先验*（uninformative prior））。因此我们看到，应用贝叶斯定理表明：$\boldsymbol{\theta}$的最佳选择就是$\boldsymbol{\theta}$的最大似然估计：

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

按照通行的术语，给定参数时数据的概率（$P(X \mid \boldsymbol{\theta})$）称为*似然*（likelihood）。

### 一个具体例子

让我们看一个具体例子，弄清它是如何工作的。假设我们只有一个参数$\theta$，它表示硬币抛出后正面朝上的概率。那么抛出背面朝上的概率就是$1-\theta$。因此，如果观测数据$X$是一个含有$n_H$个正面和$n_T$个背面的序列，我们就可以利用独立概率相乘的性质得到

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

如果我们抛$13$次硬币，得到序列“HHHTHTTHHHHHT”，该序列中$n_H = 9$、$n_T = 4$，于是

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

这个例子有一个好处：我们事先就知道答案。事实上，如果口头说“我抛了 13 次硬币，其中 9 次正面朝上，那么这枚硬币正面朝上的概率最优猜测是多少？”，每个人都会正确地猜出$9/13$。最大似然方法给我们的，是一种从基本原理出发得到这个数值的方式，而这种方式可以推广到远远更为复杂的情形。

在我们的例子中，$P(X \mid \theta)$的曲线如下：

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

该函数的最大值出现在我们预期的$9/13 \approx 0.7\ldots$附近。要判断最大值是否恰好在那里，我们可以求助于微积分。注意在最大值处，函数的梯度是平坦的。因此，我们可以通过找出导数为零的$\theta$值，并从中选出使概率最大的那个，来求得最大似然估计 :eqref:`eq_max_like`。我们计算：

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

该式有三个解：$0$、$1$和$9/13$。前两个显然是极小值而不是极大值，因为它们给我们的序列分配的概率为$0$。最后一个值*不会*给我们的序列分配零概率，因此它必定是最大似然估计$\hat \theta = 9/13$。

## 数值优化与负对数似然

上一个例子很好，但如果我们有数十亿个参数和数据样例，又该怎么办呢？

首先注意，若假设所有数据样例相互独立，那么在实践上我们就不能再直接考虑似然本身了，因为它是许多概率的乘积。事实上，每个概率都落在$[0,1]$中，通常取值约为$1/2$，而$(1/2)^{1000000000}$的乘积远远低于机器精度。我们无法直接对其进行计算。

不过，请回忆一下，对数能把乘积变成求和，于是

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

这个数值即使放在单精度$32$位浮点数中也完全能表示。因此，我们应当考虑*对数似然*（log-likelihood），即

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

由于函数$x \mapsto \log(x)$是递增的，最大化似然等价于最大化对数似然。事实上，在 :numref:`sec_naive_bayes` 中，我们将看到这一推理被应用于朴素贝叶斯分类器这一具体例子。

我们经常使用损失函数，并希望最小化损失。通过取$-\log(P(X \mid \boldsymbol{\theta}))$，我们可以把最大似然问题转化为最小化损失问题，这个量就是*负对数似然*（negative log-likelihood）。

为了说明这一点，考虑之前的抛硬币问题，并假装我们不知道闭式解。我们可以计算出

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

该式可以写成代码，即使对数十亿次抛硬币也能自由优化。

```{.python .input}
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

数值上的便利并不是人们偏爱负对数似然的唯一原因。它之所以更可取，还有其他几个理由。



第二个理由是对微积分法则的运用得到了简化。如上所述，由于独立性假设，我们在机器学习中遇到的大多数概率都是各个概率的乘积。

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

这意味着，如果我们直接应用乘积法则来计算导数，就会得到

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

这需要$n(n-1)$次乘法以及$(n-1)$次加法，因此其复杂度与输入规模成二次方关系！足够巧妙的项分组技巧可以把复杂度降到线性，但这需要费一番思考。而对负对数似然，我们有

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

于是得到

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

这只需要$n$次除法和$n-1$次求和，因此复杂度与输入规模成线性关系。

考虑负对数似然的第三个也是最后一个理由，是它与信息论的关系，我们将在 :numref:`sec_information_theory` 中详细讨论。信息论是一门严谨的数学理论，它给出了度量随机变量中所含信息量或随机性的方法。该领域的核心研究对象是熵，即

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

它度量一个信息源的随机性。注意它只不过是平均$-\log$概率，因此如果我们取负对数似然并除以数据样例数，就会得到一个与熵相关的量，称为交叉熵。仅凭这一理论解释，就足以有力地促使我们把数据集上的平均负对数似然作为度量模型性能的一种方式。

## 连续变量的最大似然

迄今为止我们所做的一切都假定了我们处理的是离散随机变量，但如果我们想处理连续随机变量，又该怎么办呢？

简短的回答是：一切都不变，只是要把所有出现概率的地方替换成概率密度。回想一下，我们用小写$p$表示密度，这意味着例如现在我们说

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

问题就变成了：“为什么这是可以的？”毕竟，我们引入密度的理由正是：取到某个具体结果的概率本身为零，那么对任意参数集合，生成我们数据的概率不也都是零吗？

的确如此。要理解为什么我们可以转向密度，就得追踪一下 epsilon 发生了什么变化。

让我们先重新定义目标。假设对连续随机变量，我们不再想计算恰好取到正确值的概率，而是想计算落在某个范围$\epsilon$之内的概率。为简单起见，假设我们的数据是对同分布随机变量$X_1, \ldots, X_N$的重复观测$x_1, \ldots, x_N$。正如前面所见，这可以写成

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

因此，如果对它取负对数，就得到

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

观察这个表达式可以发现，$\epsilon$只出现在加性常数$-N\log(\epsilon)$中。这一项根本不依赖于参数$\boldsymbol{\theta}$，所以$\boldsymbol{\theta}$的最优选择并不取决于我们如何选取$\epsilon$！无论我们要求四位数字还是四百位数字，$\boldsymbol{\theta}$的最佳选择都保持不变，因此我们可以自由地丢掉 epsilon，从而看出我们要优化的目标是

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

由此我们看到，只要把概率替换成概率密度，最大似然视角就能同样轻松地处理连续随机变量和离散随机变量。

## 小结
* 最大似然原理告诉我们：对给定数据集，最佳拟合模型就是生成该数据概率最高的那个模型。
* 人们常常改用负对数似然，原因有多个：数值稳定性、把乘积转化为求和（以及由此带来的梯度计算简化），还有与信息论的理论联系。
* 虽然在离散情形下最容易说明其动机，但它也可以自由地推广到连续情形，方法是最大化分配给数据点的概率密度。

## 练习
1. 假设已知一个非负随机变量的密度为$\alpha e^{-\alpha x}$，其中$\alpha>0$为某个值。你从该随机变量得到一个观测值，数值为$3$。$\alpha$的最大似然估计是多少？
2. 假设你有一个由样本$\{x_i\}_{i=1}^N$构成的数据集，这些样本抽自一个均值未知但方差为$1$的高斯分布。均值的最大似然估计是什么？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1097)
:end_tab:
