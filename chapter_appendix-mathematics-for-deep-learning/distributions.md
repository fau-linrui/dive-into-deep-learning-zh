# 分布
:label:`sec_distributions`

既然我们已经学会了如何在离散和连续两种情形下处理概率，接下来让我们认识一些常见的分布。根据机器学习领域的不同，我们可能需要熟悉多得多的分布，而在深度学习的某些领域中可能一个都不需要。不过，下面这份清单是一份值得熟悉的基础清单。让我们先导入一些常用库。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## 伯努利分布

这是通常遇到的最简单的随机变量。这个随机变量编码了一次抛硬币：以概率$p$出现$1$，以概率$1-p$出现$0$。如果有一个服从该分布的随机变量$X$，我们记作

$$
X \sim \textrm{Bernoulli}(p).
$$

其累积分布函数为

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

概率质量函数如下图所示。

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

现在，让我们画出累积分布函数 :eqref:`eq_bernoulli_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

如果$X \sim \textrm{Bernoulli}(p)$，那么：

* $\mu_X = p$，
* $\sigma_X^2 = p(1-p)$。

我们可以如下从伯努利随机变量中采样任意形状的数组。

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 离散均匀分布

下一个常见随机变量是离散均匀分布。在此处的讨论中，我们假设它的支撑集是整数$\{1, 2, \ldots, n\}$，不过也可以自由选择任何其他取值集合。在这个语境中，*均匀*（uniform）一词的含义是每个可能取值出现的可能性相同。每个取值$i \in \{1, 2, 3, \ldots, n\}$的概率为$p_i = \frac{1}{n}$。我们把服从该分布的随机变量$X$记作

$$
X \sim U(n).
$$

其累积分布函数为

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \textrm{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

让我们先画出概率质量函数。

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

现在，让我们画出累积分布函数 :eqref:`eq_discrete_uniform_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

如果$X \sim U(n)$，那么：

* $\mu_X = \frac{1+n}{2}$，
* $\sigma_X^2 = \frac{n^2-1}{12}$。

我们可以如下从离散均匀随机变量中采样任意形状的数组。

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 连续均匀分布

接下来，让我们讨论连续均匀分布。这个随机变量背后的想法是：如果增大离散均匀分布中的$n$，再对它进行缩放到区间$[a, b]$内，我们就会趋近于一个连续随机变量，它在$[a, b]$中任意取一个值，且所有取值等可能。我们把这个分布记作

$$
X \sim U(a, b).
$$

其概率密度函数为

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

其累积分布函数为

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

让我们先画出概率密度函数 :eqref:`eq_cont_uniform_pdf`。

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

现在，让我们画出累积分布函数 :eqref:`eq_cont_uniform_cdf`。

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

如果$X \sim U(a, b)$，那么：

* $\mu_X = \frac{a+b}{2}$，
* $\sigma_X^2 = \frac{(b-a)^2}{12}$。

我们可以如下从均匀随机变量中采样任意形状的数组。注意它默认从$U(0,1)$中采样，所以如果我们需要不同的取值范围，就要对它进行缩放。

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 二项分布

让我们把问题变得稍复杂一些，考察*二项*（binomial）随机变量。这个随机变量源自如下过程：进行$n$次独立实验，每次实验以概率$p$成功，并问我们期望看到多少次成功。

让我们用数学语言表达这一点。每次实验是一个独立随机变量$X_i$，我们用$1$表示成功、用$0$表示失败。由于每次实验都是成功概率为$p$的独立抛硬币，我们可以说$X_i \sim \textrm{Bernoulli}(p)$。于是二项随机变量为

$$
X = \sum_{i=1}^n X_i.
$$

在这种情况下，我们记作

$$
X \sim \textrm{Binomial}(n, p).
$$

要得到累积分布函数，我们需要注意到：恰好取得$k$次成功可以有$\binom{n}{k} = \frac{n!}{k!(n-k)!}$种方式，而每种方式发生的概率都是$p^k(1-p)^{n-k}$。因此累积分布函数为

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \textrm{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

让我们先画出概率质量函数。

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

现在，让我们画出累积分布函数 :eqref:`eq_binomial_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

如果$X \sim \textrm{Binomial}(n, p)$，那么：

* $\mu_X = np$，
* $\sigma_X^2 = np(1-p)$。

这可由期望值对$n$个伯努利随机变量之和的线性性，以及独立随机变量之和的方差等于各方差之和这一事实推出。可以如下进行采样。

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## 泊松分布
现在让我们做一个思想实验。我们站在公交车站，想知道接下来一分钟内会有多少辆公交车到达。让我们先从$X^{(1)} \sim \textrm{Bernoulli}(p)$开始考虑，它不过是一分钟时间窗内有一辆公交车到达的概率。对于远离市中心的公交车站，这可能是一个相当好的近似。在一分钟内我们可能永远都不会看到超过一辆公交车。

然而，如果我们在一个繁忙地区，那么两辆公交车到达是可能的，甚至是很有可能的。我们可以通过把随机变量按前 30 秒和后 30 秒分成两部分来对此建模。在这种情况下，我们可以写

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

其中$X^{(2)}$是总和，而$X^{(2)}_i \sim \textrm{Bernoulli}(p/2)$。于是总分布为$X^{(2)} \sim \textrm{Binomial}(2, p/2)$。

为什么要停在这里呢？让我们继续把这一分钟分成$n$份。按照与上面相同的推理，我们看到

$$X^{(n)} \sim \textrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

考虑这些随机变量。根据上一节，我们知道 :eqref:`eq_eq_poisson_approx` 的均值为$\mu_{X^{(n)}} = n(p/n) = p$，方差为$\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$。如果取$n \rightarrow \infty$，可以看到这些数值稳定到$\mu_{X^{(\infty)}} = p$，方差稳定到$\sigma_{X^{(\infty)}}^2 = p$。这表明，在这个无限细分极限中*可能存在*某个我们能定义的随机变量。

这应该不会太令人意外，因为在现实世界中我们只要数一数公交车到达的数目即可；不过，看到我们的数学模型定义良好还是一件好事。这一讨论可以形式化为*稀有事件定律*（law of rare events）。

仔细地按这一推理推下去，我们就可以得到如下模型。我们说$X \sim \textrm{Poisson}(\lambda)$，如果它是一个取值为$\{0,1,2, \ldots\}$的随机变量，其概率为

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

$\lambda > 0$这个值称为*速率*（rate）（或*形状*参数（shape parameter）），它表示我们在单位时间内预期会看到的平均到达数。

我们可以对这个概率质量函数求和，得到累积分布函数。

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \textrm{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

让我们先画出概率质量函数 :eqref:`eq_poisson_mass`。

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

现在，让我们画出累积分布函数 :eqref:`eq_poisson_cdf`。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

如上面所见，均值和方差尤为简洁。如果$X \sim \textrm{Poisson}(\lambda)$，那么：

* $\mu_X = \lambda$，
* $\sigma_X^2 = \lambda$。

可以如下进行采样。

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## 高斯分布
现在让我们尝试一个不同但相关的实验。假设我们同样在进行$n$次独立的$\textrm{Bernoulli}(p)$测量$X_i$。这些测量之和的分布为$X^{(n)} \sim \textrm{Binomial}(n, p)$。我们不去取$n$增大、$p$减小的极限，而是固定$p$，然后令$n \rightarrow \infty$。在这种情况下，$\mu_{X^{(n)}} = np \rightarrow \infty$，$\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$，所以没有理由认为这个极限会是定义良好的。

然而，并非全无希望！我们只要通过定义

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

让均值和方差都表现良好即可。可以看出它的均值为零、方差为一，因此我们有理由相信它会收敛到某个极限分布。如果把这些分布的样子画出来，我们就会更加确信它可行。

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

需要注意一点：与泊松情形相比，我们现在除以了标准差，这意味着我们正在把可能的结果挤压到越来越小的区域里。这表明我们的极限将不再是离散的，而是连续的。

对这一过程如何发生的推导超出了本文档的范围，但*中心极限定理*（central limit theorem）指出，当$n \rightarrow \infty$时，这将产生高斯分布（有时也称正态分布）。更明确地说，对任意$a, b$：

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

其中，我们说一个随机变量服从给定均值$\mu$和方差$\sigma^2$的正态分布，记作$X \sim \mathcal{N}(\mu, \sigma^2)$，如果$X$的密度为

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

让我们先画出概率密度函数 :eqref:`eq_gaussian_pdf`。

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

现在，让我们画出累积分布函数。这超出了本附录的范围，但高斯分布的累积分布函数无法用更初等的函数写成闭式公式。我们将使用`erf`，它提供了对这种积分进行数值计算的一种方式。

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

眼尖的读者会认出其中一些项。事实上，我们在 :numref:`sec_integral_calculus` 中遇到过这个积分。要看出这个$p_X(x)$的总面积为 1、因而是一个合法的密度，正需要那一计算。

我们选择抛硬币来讨论使计算更简短，但这个选择本身并无本质之处。事实上，如果我们取任意一组独立同分布的随机变量$X_i$，并构造

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

那么

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

将近似服从高斯分布。要使它成立还需要一些额外条件，最常见的是$E[X^4] < \infty$，但其思想是清楚的。

中心极限定理正是高斯分布在概率论、统计学和机器学习中具有基础地位的原因。每当我们能够说所测量的量是许多微小独立贡献之和时，就可以假定所测量的量接近高斯分布。

高斯分布还有许多迷人的性质，这里我们想再讨论一个。高斯分布就是所谓的*最大熵分布*（maximum entropy distribution）。我们将在 :numref:`sec_information_theory` 中更深入地讨论熵，此处我们只需要知道它是一种随机性的度量。在严格的数学意义上，我们可以把高斯分布看作在固定均值和方差下*最*随机的随机变量选择。因此，如果我们已知随机变量的均值和方差，那么从某种意义上说，高斯分布是我们可以作出的最保守的分布选择。

在结束本节之前，让我们回顾一下：如果$X \sim \mathcal{N}(\mu, \sigma^2)$，那么：

* $\mu_X = \mu$，
* $\sigma_X^2 = \sigma^2$。

我们可以如下从高斯分布（或标准正态分布）中采样。

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## 指数族
:label:`subsec_exponential_family`

上面列出的所有分布有一个共同性质：它们都属于所谓的*指数族*（exponential family）。指数族是一类分布，其密度可以表示为如下形式：

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

由于这个定义可能有点微妙，让我们仔细考察一下。

首先，$h(\mathbf{x})$称为*底层测度*（underlying measure）或*基测度*（base measure）。它可以看作是我们在其基础上用指数权重进行修改的原始测度选择。

其次，我们有向量$\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in
\mathbb{R}^l$，称为*自然参数*（natural parameters）或*典范参数*（canonical parameters）。它们
定义基测度将如何被修改。自然参数进入新测度的方式是：把这些参数与
某个函数$T(\cdot)$在$\mathbf{x}= (x_1, x_2, ..., x_n) \in
\mathbb{R}^n$处的取值作点积，然后再取指数。向量$T(\mathbf{x})= (T_1(\mathbf{x}),
T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$
称为$\boldsymbol{\eta}$的*充分统计量*（sufficient statistics）。之所以用这个名字，是因为
由$T(\mathbf{x})$表示的信息足以计算出概率密度，不再需要样本$\mathbf{x}$的其他任何
信息。

第三，我们有$A(\boldsymbol{\eta})$，它称为*累积量函数*（cumulant function），用于保证上述分布 :eqref:`eq_exp_pdf` 积分为一，即

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \exp
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

为具体起见，让我们考虑高斯分布。假设$\mathbf{x}$是单变量，我们已经看到它的密度为

$$
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \exp \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

这与指数族的定义相匹配，其中：

* *底层测度*：$h(x) = \frac{1}{\sqrt{2 \pi}}$，
* *自然参数*：$\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$，
* *充分统计量*：$T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$，以及
* *累积量函数*：$A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$。

值得注意的是，上述各项的确切选择在某种程度上是任意的。事实上，重要的特征是分布可以表示成这种形式，而不是这种确切形式本身。

正如我们在 :numref:`subsec_softmax_and_derivatives` 中暗示的那样，一种广泛使用的技术是假定最终输出$\mathbf{y}$服从指数族分布。指数族是机器学习中经常遇到的一类常见而强大的分布族。


## 小结
* 伯努利随机变量可用于对具有是/否结果的事件建模。
* 离散均匀分布对从有限可能集合中进行选择建模。
* 连续均匀分布从区间中进行选择。
* 二项分布对一系列伯努利随机变量建模，并统计成功次数。
* 泊松随机变量对稀有事件的发生建模。
* 高斯随机变量对大量独立随机变量相加的结果建模。
* 上述所有分布都属于指数族。

## 练习

1. 若随机变量是两个独立二项随机变量$X, Y \sim \textrm{Binomial}(16, 1/2)$之差$X-Y$，它的标准差是多少？
2. 如果我们取一个泊松随机变量$X \sim \textrm{Poisson}(\lambda)$，并考虑当$\lambda \rightarrow \infty$时的$(X - \lambda)/\sqrt{\lambda}$，可以证明它近似变为高斯分布。为什么这是合理的？
3. 两个在$n$个元素上的离散均匀随机变量之和的概率质量函数是什么？


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1099)
:end_tab:
