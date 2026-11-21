# 统计学
:label:`sec_statistics`

毫无疑问，要成为一名顶尖的深度学习从业者，训练出最先进且高准确率的模型的能力至关重要。然而，我们往往不清楚某项改进到何时才是显著的，或者它只是训练过程中随机波动的结果。为了能够讨论估计值中的不确定性，我们必须学习一些统计学知识。


*统计学*（statistics）最早的文献记载可以追溯到$9^{\textrm{th}}$世纪的阿拉伯学者 Al-Kindi，他详细描述了如何使用统计和频率分析来破译加密信息。800 年之后，现代统计学于 1700 年代兴起于德国，当时研究者的关注点是对人口与经济数据的收集与分析。如今，统计学是一门关于数据的收集、处理、分析、解释与可视化的科学。此外，统计学的核心理论已被广泛应用于学术界、工业界和政府的研究之中。


更具体地说，统计学可以分为*描述性统计*（descriptive statistics）和*统计推断*（statistical inference）。前者侧重于概括并展示一组观测数据的特征，这组数据称为*样本*（sample）。样本取自*总体*（population），总体表示我们实验所关注的相似个体、物品或事件的全集。与描述性统计不同，*统计推断*在“样本分布能在某种程度上复现总体分布”这一假设的基础上，进一步从给定的*样本*推断总体的特征。


你可能会问：“机器学习与统计学的本质区别是什么？”从根本上说，统计学关注推断问题。这类问题包括对变量之间关系建模（如因果推断），以及检验模型参数在统计上是否显著（如 A/B 测试）。相比之下，机器学习强调作出准确的预测，而不需要显式地编程并理解每个参数的功能。


本节中，我们将介绍三种统计推断方法：评估与比较估计量、进行假设检验，以及构造置信区间。这些方法可以帮助我们推断给定总体的特征，即真实参数$\theta$。为简洁起见，我们假设给定总体的真实参数$\theta$是一个标量值。把它推广到$\theta$为向量或张量的情形是直接的，因此我们在讨论中不再赘述。



## 评估与比较估计量

在统计学中，*估计量*（estimator）是用于估计真实参数$\theta$的给定样本的函数。我们将$\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$记作在观测到样本{$x_1, x_2, \ldots, x_n$}之后对$\theta$的估计。

我们在 :numref:`sec_maximum_likelihood` 一节中已经见过估计量的简单例子。如果你有若干来自伯努利随机变量的样本，那么该随机变量取值为一的概率的最大似然估计，可以通过统计观测到的 1 的个数并除以样本总数得到。类似地，有一道练习题要求你证明：在给定若干样本的情况下，高斯分布均值的最大似然估计由所有样本的平均值给出。这些估计量几乎永远不会给出参数的真实值，但理想情况下，当样本数量很大时，估计值会接近真实值。

作为一个例子，下面我们展示了均值为零、方差为一的高斯随机变量的真实密度，以及从该高斯分布抽到的若干样本。我们构造了$y$坐标，使每个点都可见，并使它与原始密度的关系更清晰。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch

# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow

# Sample datapoints and create y coordinate
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Compute true density
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

计算参数估计量$\hat{\theta}_n$的方法可以有很多。本节中，我们介绍三种评估与比较估计量的常用方法：均方误差、标准差和统计偏差。


### 均方误差

评估估计量时也许最简单的指标是*均方误差*（mean squared error，MSE）（或$l_2$损失），它可以定义为

$$\textrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

这使我们能够量化偏离真实值的平均平方偏差。MSE 总是非负的。如果你读过 :numref:`sec_linear_regression`，就会认出它是最常用的回归损失函数。作为评估估计量的度量，其值越接近零，估计量就越接近真实参数$\theta$。


### 统计偏差

MSE 提供了一个自然的度量，但我们很容易想象出多种可能使它变大的不同现象。其中两个具有根本重要性的现象是：由数据集中的随机性引起的估计量波动，以及由估计过程引起的估计量的系统性误差。


首先，让我们度量系统性误差。对估计量$\hat{\theta}_n$，*统计偏差*（statistical bias）的数学刻画可以定义为

$$\textrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

注意，当$\textrm{bias}(\hat{\theta}_n) = 0$时，估计量$\hat{\theta}_n$的期望等于参数的真实值。这时我们说$\hat{\theta}_n$是无偏估计量。一般来说，无偏估计量优于有偏估计量，因为它的期望与真实参数相同。


然而，值得意识到的是，有偏估计量在实践中经常被使用。在某些情形下，若无进一步假设，无偏估计量并不存在，或者难以计算。这看起来像是估计量的一个重大缺陷，但实践中遇到的大多数估计量至少是渐近无偏的，即在可用样本数趋于无穷时偏差趋于零：$\lim_{n \rightarrow \infty} \textrm{bias}(\hat{\theta}_n) = 0$。


### 方差与标准差

其次，让我们度量估计量中的随机性。回想 :numref:`sec_random_variables` 中的内容，*标准差*（standard deviation）（或*标准误*（standard error））定义为方差的平方根。我们可以通过度量估计量的标准差或方差来度量该估计量的波动程度。

$$\sigma_{\hat{\theta}_n} = \sqrt{\textrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

把 :eqref:`eq_var_est` 与 :eqref:`eq_mse_est` 相比较是很重要的。在这个公式中，我们不是与总体真实值$\theta$比较，而是与$E(\hat{\theta}_n)$即样本均值的期望比较。因此，我们测量的不是估计量倾向于偏离真实值多远，而是估计量自身的波动。


### 偏差-方差权衡

从直觉上说，这两个主要成分都会影响均方误差，这一点很清楚。而有些令人震惊的是，我们可以证明均方误差实际上可以*分解*为这两项贡献再加上第三项。也就是说，我们可以把均方误差写成偏差的平方、方差与不可约误差之和。

$$
\begin{aligned}
\textrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \textrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \textrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (\textrm{bias} [\hat{\theta}_n])^2 + \textrm{Var} (\hat{\theta}_n) + \textrm{Var} [\theta].\\
\end{aligned}
$$

我们把上述公式称为*偏差-方差权衡*（bias-variance trade-off）。均方误差可以分为三个误差来源：高偏差带来的误差、高方差带来的误差以及不可约误差。偏差误差常见于简单模型（如线性回归模型），这类模型无法提取特征与输出之间的高维关系。如果一个模型存在高偏差误差，我们常说它*欠拟合*（underfitting）或缺乏*灵活性*（flexibility），如 (:numref:`sec_generalization_basics`) 中所介绍的那样。高方差通常源于过于复杂的模型，这种模型会过拟合训练数据。因此，*过拟合*（overfitting）模型对数据中的微小波动很敏感。如果一个模型存在高方差，我们常说它*过拟合*且缺乏*泛化*（generalization），如 (:numref:`sec_generalization_basics`) 中所介绍的那样。不可约误差来自$\theta$本身的噪声。

### 在代码中评估估计量

由于估计量的标准差已经可以通过对张量`a`简单地调用`a.std()`来实现，我们将跳过它，而实现统计偏差和均方误差。

```{.python .input}
#@tab mxnet
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

为了说明偏差-方差权衡的公式，让我们用$10,000$个样本模拟正态分布$\mathcal{N}(\theta, \sigma^2)$。这里我们取$\theta = 1$、$\sigma = 4$。由于估计量是给定样本的函数，这里我们使用样本均值作为该正态分布$\mathcal{N}(\theta, \sigma^2)$中真实$\theta$的估计量。

```{.python .input}
#@tab mxnet
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

让我们通过计算偏差平方与估计量方差之和来验证这一权衡公式。首先，计算我们估计量的 MSE。

```{.python .input}
#@tab all
mse(samples, theta_true)
```

接下来，我们如下计算$\textrm{Var} (\hat{\theta}_n) + [\textrm{bias} (\hat{\theta}_n)]^2$。如你所见，这两个数值在数值精度范围内一致。

```{.python .input}
#@tab mxnet
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## 进行假设检验


统计推断中最常见的主题是假设检验。虽然假设检验在$20^{th}$世纪初才流行起来，但它的首次使用可以追溯到 1700 年代的 John Arbuthnot。John 追踪了伦敦 80 年的出生记录，并得出结论：每年出生的男性都多于女性。此后，现代显著性检验是 Karl Pearson、William Gosset 和 Ronald Fisher 留下的智慧遗产：Karl Pearson 发明了$p$值和 Pearson 卡方检验，William Gosset 是 Student t 分布之父，而 Ronald Fisher 开创了原假设与显著性检验。

*假设检验*（hypothesis test）是一种评估某些证据、以反对关于总体的默认陈述的方式。我们把默认陈述称为*原假设*（null hypothesis）$H_0$，并试图利用观测数据拒绝它。这里，我们把$H_0$作为统计显著性检验的起点。*备择假设*（alternative hypothesis）$H_A$（或$H_1$）是与原假设相反的陈述。原假设通常以陈述句的形式给出，设定变量之间的关系。它应当尽可能明确地反映问题要点，并且能够用统计学理论来检验。

设想你是一名化学家。在实验室中花费数千小时后，你研制出一种新药，它能大幅提升一个人的数学理解能力。为了展示它的神奇效力，你需要对它进行检验。很自然，你可能需要一些志愿者服用这种药，看看它是否能帮助他们更好地学习数学。你该如何开始呢？

首先，你需要仔细随机选取两组志愿者，使他们在由某些指标衡量的数学理解能力上没有差异。这两组通常称为试验组和对照组。*试验组*（test group）（或*处理组*（treatment group））是一组将接受这种药物的人，而*对照组*（control group）代表被留作基准的一组用户，即除服用该药物之外环境设置完全相同。这样，除处理中自变量的影响之外，所有变量的影响都被最小化了。

其次，在服用一段时间药物后，你需要用相同的指标衡量两组的数学理解能力，例如让志愿者在学习一个新数学公式后做同样的测试。然后，你可以收集他们的表现并比较结果。在这种情况下，我们的原假设将是两组之间没有差异，而备择假设是存在差异。

这仍然不够形式化。还有许多细节你必须仔细考虑。例如，检验他们数学理解能力的合适指标是什么？你的测试需要多少志愿者，你才能有信心宣称药物的有效性？测试应该进行多久？你如何判断两组之间是否存在差异？你只关心平均表现，还是也关心分数的波动范围？如此等等。

这样，假设检验为实验设计以及对观测结果中确定性的推理提供了一个框架。如果我们现在能够证明原假设极不可能为真，那么我们就可以有把握地拒绝它。

为了完整讲述如何运用假设检验，我们现在需要引入一些额外的术语，并把上面的一些概念形式化。


### 统计显著性

*统计显著性*（statistical significance）度量的是在本不应拒绝时错误地拒绝原假设$H_0$的概率，即

$$ \textrm{statistical significance }= 1 - \alpha = 1 - P(\textrm{reject } H_0 \mid H_0 \textrm{ is true} ).$$

它也称为*第一类错误*（type I error）或*假阳性*（false positive）。$\alpha$称为*显著性水平*（significance level），其常用值为$5\%$，即$1-\alpha = 95\%$。显著性水平可以解释为：当我们拒绝一个为真的原假设时，我们愿意承担的风险水平。

:numref:`fig_statistical_significance` 展示了两样本假设检验中观测值以及给定正态分布的概率。如果观测数据样例位于$95\%$阈值之外，那么在原假设下它将是一个极不可能的观测。因此，原假设可能有问题，我们将拒绝它。

![统计显著性。](../img/statistical-significance.svg)
:label:`fig_statistical_significance`


### 统计功效

*统计功效*（statistical power）（或*灵敏度*（sensitivity））度量的是在本应拒绝时拒绝原假设$H_0$的概率，即

$$ \textrm{statistical power }= 1 - \beta = 1 - P(\textrm{ fail to reject } H_0  \mid H_0 \textrm{ is false} ).$$

回想一下，*第一类错误*是在原假设为真时拒绝它而造成的错误，而*第二类错误*（type II error）是在原假设为假时未能拒绝它而导致的错误。第二类错误通常记作$\beta$，因此相应的统计功效为$1-\beta$。


从直觉上，统计功效可以解释为：在期望的统计显著性水平下，我们的检验有多大可能检测出某个最小幅度上的真实差异。$80\%$是常用的统计功效阈值。统计功效越高，我们就越有可能检测出真实差异。

统计功效最常见的用途之一是确定所需样本数。原假设为假时你拒绝它的概率，取决于它为假的程度（称为*效应量*（effect size））以及你拥有的样本数。正如你可能预料的，小的效应量将需要非常大的样本量才能以高概率被检测到。在此简要附录中详细推导超出了范围，但举个例子：如果我们想拒绝“样本来自均值为零、方差为一的高斯分布”这一原假设，并且相信样本均值实际上接近于一，那么我们只需$8$个样本就能以可接受的错误率做到这一点。然而，如果我们认为样本总体的真实均值接近$0.01$，那么就需要将近$80000$的样本量才能检测出这一差异。

我们可以把功效想象成一个水过滤器。在这个类比中，高功效的假设检验就像高质量的水过滤系统，它会尽可能减少水中的有害物质。另一方面，较小的差异就像低质量的水过滤器，一些相对较小的物质很容易从缝隙中漏过。类似地，如果统计功效不够高，那么检验可能捕捉不到较小的差异。


### 检验统计量

*检验统计量*（test statistic）$T(x)$是一个标量，它概括样本数据的某些特征。定义这种统计量的目标是：它应当使我们能够区分不同的分布，并进行假设检验。回想我们的化学家例子，如果我们想表明一个总体的表现优于另一个总体，那么取均值作为检验统计量可能是合理的。检验统计量的不同选择可能带来统计功效差异极大的统计检验。

通常，$T(X)$（即原假设下检验统计量的分布）在原假设下考虑时，至少近似服从一个常见的概率分布，例如正态分布。如果我们能够显式地推导出这样的分布，然后在数据集上度量我们的检验统计量，那么当统计量远远落在我们预期的范围之外时，我们就可以有把握地拒绝原假设。把这一点定量化，就引出了$p$值的概念。


### $p$值

$p$值（或*概率值*（probability value））是：假定原假设为*真*，$T(X)$至少与观测到的检验统计量$T(x)$一样极端的概率，即

$$ p\textrm{-value} = P_{H_0}(T(X) \geq T(x)).$$

如果$p$值小于或等于预先设定并固定的统计显著性水平$\alpha$，我们就可以拒绝原假设。否则，我们将得出结论：我们缺乏拒绝原假设的证据。对给定的总体分布，*拒绝域*（region of rejection）将是包含所有$p$值小于统计显著性水平$\alpha$的点的区间。


### 单侧检验与双侧检验

通常有两类显著性检验：单侧检验和双侧检验。当原假设与备择假设只有一个方向时，适用*单侧检验*（one-sided test）（或*单尾检验*（one-tailed test））。例如，原假设可能陈述真实参数$\theta$小于或等于某个值$c$。备择假设则是$\theta$大于$c$。也就是说，拒绝域只在抽样分布的一侧。与单侧检验相反，当拒绝域位于抽样分布两侧时，适用*双侧检验*（two-sided test）（或*双尾检验*（two-tailed test））。这种情况的一个例子可以是：原假设陈述真实参数$\theta$等于某个值$c$。备择假设则是$\theta$不等于$c$。


### 假设检验的一般步骤

熟悉上述概念之后，让我们过一遍假设检验的一般步骤。

1. 陈述问题并建立原假设$H_0$。
2. 设定统计显著性水平$\alpha$和统计功效（$1 - \beta$）。
3. 通过实验获取样本。所需样本数将取决于统计功效和预期的效应量。
4. 计算检验统计量和$p$值。
5. 根据$p$值和统计显著性水平$\alpha$，决定保留还是拒绝原假设。

要进行假设检验，我们首先定义原假设以及我们愿意承担的风险水平。然后我们计算样本的检验统计量，把检验统计量的极端值作为反对原假设的证据。如果检验统计量落入拒绝域，我们就可以拒绝原假设，接受备择假设。

假设检验适用于多种场景，例如临床试验和 A/B 测试。

## 构造置信区间


在估计参数$\theta$的值时，像$\hat \theta$这样的点估计量用处有限，因为它们不包含任何不确定性的概念。相反，如果我们能给出一个以高概率包含真实参数$\theta$的区间，那就好得多。如果你在一个世纪前对这些想法感兴趣，那么阅读 Jerzy Neyman :cite:`Neyman.1937` 的《Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability》会令你兴奋，他于 1937 年首次引入了置信区间的概念。

要具有实用性，对给定的确定程度而言，置信区间应当尽可能小。让我们看看如何推导它。


### 定义

在数学上，真实参数$\theta$的*置信区间*（confidence interval）是由样本数据计算出的一个区间$C_n$，使得

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

这里$\alpha \in (0, 1)$，而$1 - \alpha$称为该区间的*置信水平*（confidence level）或*覆盖率*（coverage）。这个$\alpha$与我们上文讨论的显著性水平是同一个量。

注意 :eqref:`eq_confidence` 讨论的是变量$C_n$，而不是固定的$\theta$。为强调这一点，我们写作$P_{\theta} (C_n \ni \theta)$，而不是$P_{\theta} (\theta \in C_n)$。

### 解释

人们很容易把$95\%$置信区间解释为“你可以有$95\%$把握认为真实参数落在其中”的区间，然而遗憾的是，这并不正确。真实参数是固定的，随机的是区间。因此，更好的解释是：如果你按照这一程序生成大量置信区间，那么其中$95\%$的区间会包含真实参数。

这听起来可能过于拘泥细节，但它对结果的解释可能有实际影响。特别是，我们可能通过构造一些我们*几乎确定*不包含真实值的区间来满足 :eqref:`eq_confidence`，只要这种情况发生得足够少即可。作为本节的结尾，我们给出三个诱人但错误的说法。关于这些观点的深入讨论可参见 :citet:`Morey.Hoekstra.Rouder.ea.2016`。

* **谬误 1**。置信区间窄，意味着我们能精确地估计该参数。
* **谬误 2**。置信区间内的数值比区间外的数值更可能是真实值。
* **谬误 3**。某个特定的、观测到的$95\%$置信区间包含真实值的概率是$95\%$。

不用说，置信区间是微妙的对象。然而，只要你把解释讲清楚，它们就能成为强大的工具。

### 一个高斯例子

让我们讨论最经典的例子：均值未知、方差未知的高斯分布均值的置信区间。假设我们从高斯分布$\mathcal{N}(\mu, \sigma^2)$中收集$n$个样本$\{x_i\}_{i=1}^n$。我们可以如下计算均值和方差的估计量：

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\textrm{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

如果我们现在考虑随机变量

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

我们就得到一个服从众所周知的分布的随机变量，该分布称为*自由度为*$n-1$*的 Student t 分布*。

这个分布得到了非常充分的研究，例如人们知道当$n\rightarrow \infty$时它近似为标准高斯分布，因此通过查高斯分布累积分布函数的数值表，我们可以得出结论：$T$的取值至少有$95\%$的时间落在区间$[-1.96, 1.96]$内。对有限的$n$，区间需要稍大一些，但这些区间是众所周知的，并且已在表格中预先计算好。

因此，我们可以得出结论：对大$n$，

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

把两边同乘以$\hat\sigma_n/\sqrt{n}$，再加上$\hat\mu_n$进行整理，我们得到

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

因此我们知道，我们已找到了$95\%$置信区间：
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

可以肯定地说，:eqref:`eq_gauss_confidence` 是统计学中最常用的公式之一。让我们通过实现它来结束对统计学的讨论。为简单起见，我们假定处于渐近情形。较小的$N$值应当包含通过程序或$t$分布表得到的正确`t_star`值。

```{.python .input}
#@tab mxnet
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.

# Number of samples
N = 1000

# Sample dataset
samples = torch.normal(0, 1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Number of samples
N = 1000

# Sample dataset
samples = tf.random.normal((N,), 0, 1)

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## 小结

* 统计学关注推断问题，而深度学习强调作出准确的预测，无需显式地编程与理解。
* 有三种常见的统计推断方法：评估与比较估计量、进行假设检验，以及构造置信区间。
* 有三种最常见的估计量：统计偏差、标准差和均方误差。
* 置信区间是真实总体参数的一个估计范围，我们可以根据给定样本构造它。
* 假设检验是一种评估某些证据、以反对关于总体的默认陈述的方式。


## 练习

1. 设$X_1, X_2, \ldots, X_n \overset{\textrm{iid}}{\sim} \textrm{Unif}(0, \theta)$，其中“iid”代表*独立同分布*（independent and identically distributed）。考虑$\theta$的如下估计量：
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * 求$\hat{\theta}.$的统计偏差、标准差和均方误差。
    * 求$\tilde{\theta}.$的统计偏差、标准差和均方误差。
    * 哪个估计量更好？
1. 对于引言中的化学家例子，你能推导出进行双侧假设检验的 5 个步骤吗？给定统计显著性水平$\alpha = 0.05$和统计功效$1 - \beta = 0.8$。
1. 对$100$个独立生成的数据集，用$N=2$和$\alpha = 0.5$运行置信区间代码，并画出所得的区间（此时`t_star = 1.0`）。你会看到若干非常短的区间，它们离包含真实均值$0$还差得很远。这是否与置信区间的解释相矛盾？你是否觉得用短区间来表明高精度估计是合适的？

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/1103)
:end_tab:
