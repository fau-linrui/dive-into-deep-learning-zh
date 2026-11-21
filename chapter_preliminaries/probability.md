```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 概率与统计
:label:`sec_prob`

无论如何，
机器学习都与不确定性有关。
在监督学习中，我们希望
给定已知的某样东西（*特征*）
来预测未知的某样东西（*目标值*）。
根据我们的目标，
我们可能试图预测
目标值最可能的值。
或者，我们可能预测与目标值
期望距离最小的值。
而有时，我们不仅希望
预测一个具体的值，
还希望*量化我们的不确定性*。
例如，给定描述某位病人的
一些特征，
我们可能想知道他在接下来一年中
心脏病发作的*可能性有多大*。
在无监督学习中，
我们常常关心不确定性。
要判断一组测量是否异常，
知道在感兴趣的总体中
观测到某些值的可能性有多大是有帮助的。
此外，在强化学习中，
我们希望开发出
能在各种环境中智能行动的智能体。
这需要对以下问题进行推理：
环境可能如何变化，
以及在响应每个可用动作时
可能遇到什么奖励。

*概率*（probability）是
研究不确定性下推理的数学领域。
给定某个过程的概率模型，
我们就可以对各种事件的可能性进行推理。
用概率来描述
可重复事件
（如抛硬币）的频率，
是相当没有争议的。
事实上，*频率学派*（frequentist）学者坚持
一种*只*适用于此类可重复事件的
概率解释。
相比之下，*贝叶斯学派*（Bayesian）学者
更广泛地使用概率的语言，
来形式化不确定性下的推理。
贝叶斯概率
有两个独特之处：
（i）对不可重复的事件
赋予信念程度，
例如，一座大坝
*垮塌的概率*是多少？；
（ii）主观性。虽然贝叶斯
概率为人们应当如何
根据新证据更新信念提供了明确的规则，
但它允许不同的个体
从不同的*先验*（prior）信念出发。
*统计学*（statistics）帮助我们进行反向推理：
从数据的收集与组织开始，
反推我们可能对生成数据的过程
得出什么推断。
每当我们分析一个数据集、寻找模式，
希望这些模式可能刻画更广泛的总体时，
我们都在运用统计思维。
许多课程、专业、论文、职业、院系、
公司和机构都致力于
概率与统计的研究。
本节虽然只是略窥门径，
但我们会提供
你开始构建模型所需的基础。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import random
import jax
from jax import numpy as jnp
import numpy as np
```

## 一个简单的例子：抛硬币

设想我们打算抛一枚硬币，
并想量化看到正面
（相对于反面）的可能性有多大。
如果硬币是*公平的*，
那么两种结果
（正面和反面）
的可能性相同。
此外，如果我们打算抛这枚硬币$n$次，
那么我们*期望*看到的
正面比例，应当恰好等于
*期望的*反面比例。
看待这一点的一种直观方式
是利用对称性：
对于每一种可能的结果，
即出现$n_\textrm{h}$个正面和$n_\textrm{t} = (n - n_\textrm{h})$个反面，
都存在一个同样可能的结果，
即出现$n_\textrm{t}$个正面和$n_\textrm{h}$个反面。
注意，只有当平均而言
我们期望看到$1/2$的抛掷出现正面、
$1/2$出现反面时，这才可能。
当然，如果你以每次$n=1000000$次抛掷
反复做这个实验，
你可能永远也见不到
恰好$n_\textrm{h} = n_\textrm{t}$的那一次试验。


在形式上，量$1/2$称为*概率*（probability），
在这里它刻画了
任意一次抛掷出现正面的确定性程度。
概率为感兴趣的结果（称为*事件*（event））
赋予介于$0$和$1$之间的分数。
这里感兴趣的事件是$\textrm{heads}$，
我们用$P(\textrm{heads})$表示相应的概率。
概率为$1$表示绝对的确定性
（想象一枚两面都是正面的假硬币），
概率为$0$表示不可能
（例如，如果两面都是反面）。
频率$n_\textrm{h}/n$和$n_\textrm{t}/n$不是概率，
而是*统计量*（statistics）。
概率是*理论上的*量，
构成了数据生成过程的基础。
这里，概率$1/2$
是硬币本身的一种属性。
相比之下，统计量是*经验性的*量，
由观测数据的函数计算得到。
我们对概率量和统计量的兴趣
不可分割地交织在一起。
我们常常设计特殊统计量，称为*估计量*（estimator），
它在给定数据集的情况下产生模型参数
（例如概率）的*估计*（estimate）。
此外，当这些估计量满足
一个称为*一致性*（consistency）的良好性质时，
我们的估计将收敛
到相应的概率。
反过来，这些推断出的概率
又告诉我们，未来可能遇到的来自同一总体的数据
具有怎样的统计性质。

假设我们偶然得到一枚真实的硬币，
但不知道它真实的$P(\textrm{heads})$。
要用统计方法研究这个量，
我们需要（i）收集一些数据；
（ii）设计一个估计量。
这里的数据获取很容易：
我们可以抛很多次硬币，
并记录所有结果。
在形式上，从某个潜在随机过程中
抽取实现值称为*采样*（sampling）。
正如你可能已经猜到的，
一种自然的估计量
是观测到的*正面*次数
与总抛掷次数之比。

现在，假设这枚硬币实际上是公平的，
即$P(\textrm{heads}) = 0.5$。
要模拟公平硬币的抛掷，
我们可以调用任何随机数生成器。
要抽取概率为$0.5$的事件的样本，
有一些简单的方法。
例如，Python的`random.random`
产生区间$[0,1]$中的数，
它落在任意子区间$[a, b] \subset [0,1]$中的概率
等于$b-a$。
因此，通过检验返回的浮点数是否大于`0.5`，
我们可以以`0.5`的概率分别得到`0`和`1`：

```{.python .input}
%%tab all
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

更一般地，我们可以模拟
从任何具有有限个可能结果的变量
（例如抛硬币或掷骰子）中进行多次抽取，
方法是调用multinomial函数，
把第一个参数设为抽取次数，
第二个参数设为与每个可能结果
相关联的概率列表。
要模拟公平硬币的十次抛掷，
我们指定概率向量`[0.5, 0.5]`，
把索引0解释为正面、
索引1解释为反面。
该函数返回一个向量，
其长度等于可能结果的数目（这里为2），
其中第一个分量告诉我们
正面出现的次数，
第二个分量告诉我们
反面出现的次数。

```{.python .input}
%%tab mxnet
fair_probs = [0.5, 0.5]
multinomial(100, fair_probs)
```

```{.python .input}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab jax
fair_probs = [0.5, 0.5]
# jax.random does not have multinomial distribution implemented
np.random.multinomial(100, fair_probs)
```

每次运行这个采样过程时，
你都会得到一个可能与前一次结果
不同的新随机值。
除以抛掷次数，
就得到了数据中每种结果的*频率*（frequency）。
注意，这些频率
和它们所要估计的概率一样，
之和为$1$。

```{.python .input}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab jax
np.random.multinomial(100, fair_probs) / 100
```

这里，尽管我们模拟的硬币是公平的
（概率`[0.5, 0.5]`是我们自己设定的），
正面和反面的计数却可能并不相同。
这是因为我们只抽取了相对少量的样本。
如果我们没有自己实现这个模拟，
而只看到结果，
我们怎么知道这枚硬币是稍微不公平，
还是与$1/2$的可能偏差
只是小样本量造成的假象呢？
让我们看看模拟10,000次抛掷时会发生什么。

```{.python .input}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

一般来说，对于重复事件
（如抛硬币）的平均值，
随着重复次数的增加，
我们的估计保证会收敛
到真实的潜在概率。
这一现象的数学表述
称为*大数定律*（law of large numbers），
而*中心极限定理*（central limit theorem）
告诉我们，在许多情况下，
随着样本量$n$的增大，
这些误差应当以$(1/\sqrt{n})$的速率下降。
让我们通过研究估计值
如何随着抛掷次数从1增加到10,000而变化，
来获得更多的直观认识。

```{.python .input}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(1, fair_probs, size=10000).astype(np.float32)
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

每条实线对应于硬币的两个取值之一，
给出在每一组实验之后，
硬币出现该取值的估计概率。
黑色虚线给出了真实的潜在概率。
随着我们通过进行更多实验获得更多数据，
这些曲线会收敛到真实概率。
你可能已经开始看出
统计学家们所关注的
更高级问题的轮廓：
这种收敛有多快？
如果我们已经测试过
同一工厂生产的许多硬币，
又该如何把这些信息纳入进来？

##  更正式的处理

我们已经走了相当远：提出
一个概率模型、
生成合成数据、
运行统计估计量、
从经验上评估收敛性，
并报告误差指标（检查偏差）。
然而，要想走得更远，
我们需要更加精确。


在处理随机性时，
我们把可能结果的集合记为$\mathcal{S}$，
并称之为*样本空间*（sample space）或*结果空间*（outcome space）。
这里，每个元素都是一个不同的可能*结果*（outcome）。
在抛一枚硬币的情况下，
$\mathcal{S} = \{\textrm{heads}, \textrm{tails}\}$。
对于单个骰子，$\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$。
抛两枚硬币时，可能的结果为
$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails}), (\textrm{tails}, \textrm{heads}),  (\textrm{tails}, \textrm{tails})\}$。
*事件*（event）是样本空间的子集。
例如，事件“第一次抛硬币出现正面”
对应于集合$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails})\}$。
只要随机实验的结果$z$
满足$z \in \mathcal{A}$，事件$\mathcal{A}$就发生了。
对于掷一次骰子，我们可以定义事件
“看到$5$”（$\mathcal{A} = \{5\}$）
和“看到奇数”（$\mathcal{B} = \{1, 3, 5\}$）。
在这种情况下，如果骰子掷出$5$，
我们会说$\mathcal{A}$和$\mathcal{B}$都发生了。
另一方面，如果$z = 3$，
那么$\mathcal{A}$没有发生，
但$\mathcal{B}$发生了。


*概率*函数把事件
映射为实数值${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$。
在给定样本空间$\mathcal{S}$中，
事件$\mathcal{A}$的概率记为$P(\mathcal{A})$，
它具有以下性质：

* 任何事件$\mathcal{A}$的概率都是非负实数，即$P(\mathcal{A}) \geq 0$；
* 整个样本空间的概率为$1$，即$P(\mathcal{S}) = 1$；
* 对于任意可数的事件序列$\mathcal{A}_1, \mathcal{A}_2, \ldots$，如果它们*互斥*（即对所有$i \neq j$都有$\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$），那么其中任一事件发生的概率等于它们各自概率之和，即$P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$。

概率论的这些公理
由 :citet:`Kolmogorov.1933` 提出，
可以用来快速推导出许多重要的结论。
例如，由此可以直接得出：
任何事件$\mathcal{A}$*或*其补集$\mathcal{A}'$发生的概率为1
（因为$\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$）。
我们还可以证明$P(\emptyset) = 0$，
因为$1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$。
因此，任何事件$\mathcal{A}$
*和*其补集$\mathcal{A}'$同时发生的概率
为$P(\mathcal{A} \cap \mathcal{A}') = 0$。
通俗地说，这告诉我们不可能事件
发生的概率为零。



## 随机变量

当我们谈到诸如骰子掷出奇数、
或第一次抛硬币出现正面这样的事件时，
我们已经在使用*随机变量*（random variable）的概念。
在形式上，随机变量是从
某个潜在样本空间
到一组（可能很多个）值的映射。
你可能会想，随机变量
与样本空间有何不同，
因为二者都是结果的集合。
重要的是，随机变量可以比
原始的样本空间粗糙得多。
即使潜在的样本空间是无限的，
例如$0$和$1$之间线段上的点，
我们也可以定义像“大于0.5”这样的二值随机变量。
此外，多个随机变量
可以共享同一个潜在样本空间。
例如，“我家的警报是否响起”
和“我家是否被盗”
都是二值随机变量，
它们共享同一个潜在样本空间。
因此，知道一个随机变量取的值，
可以告诉我们关于另一个随机变量
可能取值的一些信息。
知道警报响了，
我们可能会怀疑家里很可能被盗了。


随机变量取得的每个值
都对应于潜在样本空间的一个子集。
因此，随机变量$X$
取值为$v$（记为$X=v$）这一发生情况是一个*事件*，
而$P(X=v)$表示它的概率。
有时这种记号会显得笨拙，
在上下文清楚时我们可以滥用记号。
例如，我们可以用$P(X)$宽泛地指
$X$的*分布*（distribution），即
告诉我们$X$取任一给定值的概率的函数。
另一些时候，我们会写出
$P(X,Y) = P(X) P(Y)$这样的表达式，
作为对随机变量$X$和$Y$可取的所有值
都成立的一个陈述的简写，也就是说，
对所有$i,j$都有$P(X=i \textrm{ and } Y=j) = P(X=i)P(Y=j)$。
还有些时候，当随机变量从上下文中
可以清楚地看出时，我们滥用记号写成$P(v)$。
由于概率论中的事件是样本空间中结果的集合，
我们可以指定随机变量取值的范围。
例如，$P(1 \leq X \leq 3)$表示事件$\{1 \leq X \leq 3\}$的概率。


注意，*离散*（discrete）随机变量，
如抛硬币或掷骰子，
与*连续*（continuous）随机变量，
如从总体中随机抽取的
某人的体重和身高，
之间存在微妙的差别。
对于后者，我们很少真正关心
某人的确切身高。
此外，如果我们进行足够精确的测量，
就会发现地球上没有两个人
身高完全相同。
事实上，测量足够精细的话，
你醒来时和入睡时的身高都不会相同。
问某人身高恰好为
1.801392782910287192米的精确概率
是没有多大意义的。
相反，我们通常更关心的，
是能够说出某人的身高
是否落在某个给定区间内，
比如在1.79米到1.81米之间。
在这些情况下，我们处理的是概率*密度*（density）。
身高恰好为1.80米的
概率为零，但密度非零。
要算出分配给某个区间的概率，
我们必须对该区间上的密度
取*积分*（integral）。

## 多个随机变量

你可能已经注意到，我们在上一节中
都无法回避涉及
多个随机变量之间相互作用的陈述
（回想一下$P(X,Y) = P(X) P(Y)$）。
机器学习的大部分内容
都关心这类关系。
这里，样本空间将是
感兴趣的总体，
比如与某企业有交易的顾客、
互联网上的照片，
或者生物学家所知的蛋白质。
每个随机变量将表示
不同属性的（未知）值。
每当我们从总体中抽取一个个体时，
我们观测到每个随机变量的一个实现值。
由于随机变量取的值
对应于样本空间的子集，
而这些子集可能重叠、部分重叠
或完全不相交，
所以知道一个随机变量取的值，
会使我们更新信念，
关于另一个随机变量的哪些值可能出现。
如果一位病人走进医院，
我们观察到他有呼吸困难
并且丧失了嗅觉，
那么我们会认为他感染COVID-19的可能性，
比他没有呼吸困难、
嗅觉完全正常时更大。


在处理多个随机变量时，
我们可以为这些变量能够联合取到的
每一种取值组合构造事件。
为每种组合（例如$A=a$和$B=b$）
赋予概率的概率函数，
称为*联合概率*（joint probability）函数，
它简单地返回分配给
样本空间中相应子集交集的概率。
随机变量$A$和$B$
分别取值$a$和$b$这一事件
所对应的*联合概率*，
记为$P(A = a, B = b)$，
其中的逗号表示“并且”。
注意，对任意值$a$和$b$，
都有

$$P(A=a, B=b) \leq P(A=a) \textrm{ and } P(A=a, B=b) \leq P(B = b),$$

因为要使$A=a$和$B=b$发生，
$A=a$必须发生，
*并且*$B=b$也必须发生。
有趣的是，联合概率
告诉了我们从概率意义上
关于这些随机变量所能知道的一切，
并且可以用来导出许多其他有用的量，
包括恢复各个分布$P(A)$和$P(B)$。
要恢复$P(A=a)$，我们只需
对随机变量$B$能取的所有值$v$
求和$P(A=a, B=v)$：
$P(A=a) = \sum_v P(A=a, B=v)$。


比值$\frac{P(A=a, B=b)}{P(A=a)} \leq 1$
非常重要。
它称为*条件概率*（conditional probability），
并通过“$\mid$”符号表示：

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

它告诉我们，一旦以$A=a$发生为条件，
与事件$B=b$相关联的新概率。
我们可以把这个条件概率看作
只把注意力限制在与$A=a$相关联的
样本空间子集上，
然后重新规范化，
使所有概率之和为1。
条件概率实际上就是普通的概率，
因此只要我们把所有项
都以同一事件为条件、
从而把注意力限制在同一样本空间上，
它就满足所有公理。
例如，对于不相交的事件
$\mathcal{B}$和$\mathcal{B}'$，有
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$。


利用条件概率的定义，
我们可以推导出著名的*贝叶斯定理*（Bayes' theorem）。
根据构造，我们有$P(A, B) = P(B\mid A) P(A)$
和$P(A, B) = P(A\mid B) P(B)$。
把两个等式结合起来，得到
$P(B\mid A) P(A) = P(A\mid B) P(B)$，因此

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$





这个简单的等式具有深远的意义，
因为它使我们能够颠倒条件作用的方向。
如果我们知道如何估计$P(B\mid A)$、$P(A)$和$P(B)$，
那么就可以估计$P(A\mid B)$。
我们常常发现直接估计其中一项比较容易，
而估计另一项比较困难，
这时贝叶斯定理就可以帮上忙。
例如，如果我们知道某种疾病症状的发生率，
以及该疾病和症状各自的总体发生率，
就可以根据症状确定
某人患该疾病的可能性。
在某些情况下，我们可能无法直接获得$P(B)$，
例如症状的发生率。
这时，贝叶斯定理的一个简化版本就派上了用场：

$$P(A \mid B) \propto P(B \mid A) P(A).$$

由于我们知道$P(A \mid B)$必须规范化为$1$，
即$\sum_a P(A=a \mid B) = 1$，
我们可以用它来计算

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

在贝叶斯统计中，我们把观察者看作
持有一些（主观的）先验信念——
关于可用假设合理性的信念，
编码在*先验*（prior）$P(H)$中——
以及一个*似然函数*（likelihood function），
它说明在给定各类假设$P(E \mid H)$的情况下，
观测到所收集证据的任一取值的可能性有多大。
于是，贝叶斯定理被解释为：
告诉我们如何根据可用的证据$E$
更新初始的*先验*$P(H)$，
从而得到*后验*（posterior）信念
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$。
通俗地说，这可以表述为
“后验等于先验乘以似然，再除以证据”。
现在，由于证据$P(E)$对所有假设都相同，
我们只需对所有假设进行规范化即可。

注意，$\sum_a P(A=a \mid B) = 1$还允许我们
对随机变量进行*边缘化*（marginalize）。也就是说，我们可以从像$P(A, B)$这样的联合分布中去掉某些变量。毕竟，我们有

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

*独立性*（independence）是另一个根本重要的概念，
它构成了统计学中
许多重要思想的支柱。
简而言之，如果以$A$的取值为条件
不会对与$B$相关联的概率分布
造成任何改变，反之亦然，
那么两个变量就是*独立的*（independent）。
更正式地说，独立性（记为$A \perp B$）
要求$P(A \mid B) = P(A)$，从而要求
$P(A,B) = P(A \mid B) P(B) = P(A) P(B)$。
独立性往往是一个合适的假设。
例如，如果随机变量$A$
表示抛一枚公平硬币的结果，
而随机变量$B$
表示抛另一枚硬币的结果，
那么知道$A$是否出现正面，
不应影响$B$出现正面的概率。


当独立性在我们从某个潜在分布中
连续抽取数据时成立
（这使我们能够得出有力的统计结论），
或者当它在我们数据中的各个变量之间成立、
使我们能够使用编码了这种独立结构的
更简单模型时，
独立性尤其有用。
另一方面，估计随机变量之间的依赖关系
往往正是学习的目的。
我们之所以关心估计给定症状下患病的概率，
正是因为我们认为疾病和症状
*不是*独立的。


注意，由于条件概率是真正的概率，
独立性和依赖性的概念也适用于它们。
两个随机变量$A$和$B$
在给定第三个变量$C$时是*条件独立的*（conditionally independent），
当且仅当$P(A, B \mid C) = P(A \mid C)P(B \mid C)$。
有趣的是，两个变量可能总体上是独立的，
但在以第三个变量为条件时就变得不独立。
当两个随机变量$A$和$B$
对应于某个第三个变量$C$的原因时，
这种情况经常发生。
例如，骨折和肺癌在一般人群中可能是独立的，
但如果以住院为条件，
我们可能会发现骨折与肺癌负相关。
这是因为骨折
*解释掉了*（explains away）某人为什么住院，
从而降低了他们因患肺癌而住院的概率。


反过来，两个不独立的随机变量
在以第三个变量为条件时也可能变得独立。
当两个原本无关的事件
有共同原因时，这种情况经常发生。
在小学生中，鞋码和阅读水平高度相关，
但如果以年龄为条件，
这种相关性就消失了。

## 一个例子
:label:`subsec_probability_hiv_app`

让我们来检验一下所学的技能。
假设一位医生给病人做HIV检测。
这项检测相当准确，如果病人健康却被报告为患病，
那么只有1%的概率出错，
也就是说，健康病人中有1%的检测结果为阳性。
此外，如果病人确实感染了HIV，
这项检测从不会漏检。
我们用$D_1 \in \{0, 1\}$表示诊断结果
（$0$表示阴性，$1$表示阳性），
用$H \in \{0, 1\}$表示HIV感染状况。

| 条件概率 | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 \mid H)$        |     1 |  0.01 |
| $P(D_1 = 0 \mid H)$        |     0 |  0.99 |

注意，各列之和均为1（但行之和不是），
因为它们都是条件概率。
让我们计算检测结果为阳性时
病人感染HIV的概率，
即$P(H = 1 \mid D_1 = 1)$。
直观上，这取决于该疾病的常见程度，
因为它会影响误报的数量。
假设该人群基本上没有这种疾病，例如$P(H=1) = 0.0015$。
要应用贝叶斯定理，我们需要应用边缘化来确定

$$\begin{aligned}
P(D_1 = 1)
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

这让我们得到

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

换句话说，尽管检测相当准确，
病人实际感染HIV的概率只有13.06%。
正如我们所见，概率可能违背直觉。
病人在收到如此可怕的消息后应该怎么做呢？
病人很可能会请医生
再做一次检测以弄清楚情况。
第二次检测具有不同的特性，
而且不如第一次那么好。

| 条件概率 | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1 \mid H)$          |  0.98 |  0.03 |
| $P(D_2 = 0 \mid H)$          |  0.02 |  0.97 |

不幸的是，第二次检测结果也是阳性。
让我们在假设条件独立的前提下，
计算应用贝叶斯定理所需的各个概率：

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
& = P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)
=& 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
& = P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)
=& 0.98.
\end{aligned}
$$

现在我们可以应用边缘化，得到
两项检测结果均为阳性的概率：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

最后，在两项检测均为阳性的情况下，
病人感染HIV的概率为

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

也就是说，第二次检测让我们
更有信心地相信情况并非一切正常。
尽管第二次检测的准确率
明显低于第一次，
但它仍然显著改进了我们的估计。
两项检测彼此条件独立这一假设，
对于我们能够给出更准确的估计至关重要。
考虑一个极端情形：
我们把同一个检测做两次。
在这种情况下，我们预期两次结果相同，
因此再跑一次相同的检测不会带来任何额外的信息。
敏锐的读者可能已经注意到，
这一诊断过程就像一个隐藏在明处的分类器：
随着我们获得更多特征（检测结果），
我们判断病人是否健康的能力会提高。


## 期望

通常，做决策不仅需要看
分配给各个事件的概率，
还需要把它们组合成有用的聚合量，
从而为我们提供指导。
例如，当随机变量取连续的标量值时，
我们往往关心*平均*而言应该期望什么值。
这个量正式地称为*期望*（expectation）。
如果我们进行投资，
第一个关心的量
可能就是我们能期望的回报，
即对所有可能结果求平均
（并按相应的概率加权）。
例如，假设一项投资有50%的概率
完全失败，
有40%的概率带来2$\times$的回报，
有10%的概率带来10$\times$的回报10$\times$。
要计算期望回报，
我们对所有回报求和，每一项
乘以它发生的概率。
这得到期望
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$。
因此期望回报是1.8$\times$。


一般来说，随机变量$X$的*期望*（expectation）（或平均值）
定义为

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

同样，对于密度，我们得到$E[X] = \int x \;dp(x)$。
有时我们关心$x$的某个函数的期望值。
我们可以这样计算这些期望：

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ and } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

分别对应离散概率和密度。
回到上面投资的例子，
$f$可能是与回报相关联的*效用*（utility）（幸福感）。
行为经济学家早就注意到，
相对于基准而言，
人们从损失金钱中感受到的负效用，
要大于从赚到一美元中获得的效用。
此外，金钱的价值往往呈次线性。
拥有10万美元与拥有零美元，
可能意味着有钱付房租、吃得好、
享受优质医疗，
与无家可归、饱受煎熬之间的差别。
另一方面，拥有20万美元
相对于拥有10万美元所带来的收益，
则没有那么惊人。
这样的推理引出了
那句老生常谈：“金钱的效用是对数的”。


如果与全部损失相关联的效用是$-1$，
而与$1$、$2$和$10$的回报相关联的效用
分别为$1$、$2$和$4$，
那么投资的期望幸福感
将是$0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$
（期望效用损失为30%）。
如果这确实是你的效用函数，
那么你最好还是把钱存在银行里。

对于财务决策，
我们可能还想衡量
一项投资有多*冒险*。
这里，我们不仅关心期望值，
还关心实际值相对于该值
*波动*的程度。
注意，我们不能直接取
实际值与期望值之差的期望。
这是因为差的期望
等于期望之差，
即$E[X - E[X]] = E[X] - E[E[X]] = 0$。
不过，我们可以考察
该差的任何非负函数的期望。
随机变量的*方差*（variance）通过考察
*平方*差的期望值来计算：

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

这里的等式是通过展开
$(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
并对每一项取期望得到的。
方差的平方根是另一个
有用的量，称为*标准差*（standard deviation）。
虽然它与方差
传达相同的信息（二者可以互相计算），
但标准差有一个很好的性质：
它与随机变量所表示的原始量
具有相同的单位。

最后，随机变量的函数
的方差类似地定义为

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$

回到我们的投资例子，
现在可以计算该投资的方差。
它由$0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$给出。
无论从哪个意义上说，这都是一项有风险的投资。
注意，按照数学惯例，均值和方差
通常记为$\mu$和$\sigma^2$。
当我们用它来参数化高斯分布时
尤其如此。

正如我们为*标量*随机变量
引入期望和方差一样，
我们也可以为向量值随机变量这样做。
期望很容易，因为我们可以按元素地应用它。
例如，$\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
的坐标为$\mu_i = E_{\mathbf{x} \sim P}[x_i]$。
*协方差*（covariance）要复杂一些。
我们通过取随机变量与其均值之差的
*外积*的期望来定义它们：

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

矩阵$\boldsymbol{\Sigma}$称为协方差矩阵。
要看出它的作用，一种简单的方法是
考虑某个与$\mathbf{x}$大小相同的向量$\mathbf{v}$。
由此可得

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$

因此，$\boldsymbol{\Sigma}$使我们能够
通过简单的矩阵乘法，
计算$\mathbf{x}$的任何线性函数的方差。
非对角元素告诉我们
各坐标之间的相关程度：
值为0表示不相关，
而较大的正值
意味着它们相关性更强。

## 讨论

在机器学习中，有许多事情是不确定的！
给定输入时，标签的值可能是不确定的。
参数的估计值可能是不确定的。
我们甚至可能不确定部署时到达的数据
是否与训练数据来自同一分布。

所谓*偶然不确定性*（aleatoric uncertainty），
我们指的是问题本身固有的不确定性，
它源于观测变量
未能解释的真实随机性。
所谓*认知不确定性*（epistemic uncertainty），
我们指的是关于模型参数的不确定性，
是那种我们有望通过收集更多数据
来减少的不确定性。
我们可能对硬币出现正面的概率
存在认知不确定性，
但即使我们知道了这个概率，
对于未来任何一次抛掷的结果，
我们仍然存在偶然不确定性。
无论我们观察某人抛一枚公平硬币多久，
我们对下一次抛掷出现正面的把握
永远不会高于或低于50%。
这些术语来自力学建模
（关于[不确定性量化](https://en.wikipedia.org/wiki/Uncertainty_quantification)这一方面的综述，参见例如 :citet:`Der-Kiureghian.Ditlevsen.2009` ）。
然而值得注意的是，
这些术语构成了一种轻微的语言滥用。
*认知*（epistemic）一词指任何与*知识*有关的东西，
因此在哲学意义上，
所有不确定性都是认知的。


我们看到，从某个未知概率分布中采样数据，
可以为我们提供可用于估计
数据生成分布参数的信息。
话虽如此，做到这一点的速率可能相当慢。
在我们抛硬币的例子（以及许多其他例子）中，
我们能做的最好的就是设计估计量，
使其以$1/\sqrt{n}$的速率收敛，
其中$n$是样本量（例如抛掷次数）。
这意味着，从10次观测增加到1000次观测
（通常是非常容易做到的事），
我们能把不确定性降低十倍，
而接下来的1000次观测帮助相对较小，
只能带来1.41倍的降低。
这是机器学习的一个持久特征：
尽管常常存在容易获得的收益，
但要想进一步提升，
需要非常大量的数据，
而且往往还需要巨大的计算量。
关于这一事实在大规模语言模型上的经验性综述，参见 :citet:`Revels.Lubin.Papamarkou.2016` 。

我们还改进了用于统计建模的语言和工具。
在此过程中，我们学习了条件概率，
以及统计学中最重要的等式之一——贝叶斯定理。
它是一个有效的工具，可以把数据所传达的信息解耦为：
通过似然项$P(B \mid A)$来处理
观测$B$与参数选择$A$的契合程度，
以及通过先验概率$P(A)$来衡量
$A$的某一特定选择最初有多合理。
特别地，我们看到了如何应用这一规则
为诊断结果赋予概率，
依据既是检测的功效，
*也*是疾病本身的流行率（即我们的先验）。

最后，我们引入了第一组
关于特定概率分布效应的非平凡问题，
即期望和方差。
虽然对于一个概率分布，
除了线性期望和二次期望之外还有许多其他量，
但这两个量已经提供了
关于该分布可能行为的大量知识。
例如，[切比雪夫不等式](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)
指出$P(|X - \mu| \geq k \sigma) \leq 1/k^2$，
其中$\mu$是期望，$\sigma^2$是分布的方差，
$k > 1$是我们选定的置信参数。
它告诉我们，从某个分布中抽取的值
以至少50%的概率
落在以期望为中心、
区间为$[-\sqrt{2} \sigma, \sqrt{2} \sigma]$的范围内。




## 练习

1. 举一个例子，说明观测更多数据可以将关于结果的不确定性降低到任意低的水平。
1. 举一个例子，说明观测更多数据只能把不确定性降低到某个程度，之后就不再降低。解释为什么会这样，以及你预期这个临界点出现在哪里。
1. 我们通过实验演示了抛硬币时向均值收敛的现象。计算抽取$n$个样本后，对出现正面的概率的估计的方差。
    1. 该方差如何随观测数量的变化而变化？
    1. 用切比雪夫不等式给出偏离期望的界。
    1. 它与中心极限定理有什么关系？
1. 假设我们从均值为零、方差为单位的概率分布中抽取$m$个样本$x_i$。计算平均值$z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$。我们可以对每个$z_m$独立地应用切比雪夫不等式吗？为什么不能？
1. 给定两个概率分别为$P(\mathcal{A})$和$P(\mathcal{B})$的事件，计算$P(\mathcal{A} \cup \mathcal{B})$和$P(\mathcal{A} \cap \mathcal{B})$的上下界。提示：用[维恩图](https://en.wikipedia.org/wiki/Venn_diagram)画出这种情形。
1. 假设我们有一个随机变量序列，比如$A$、$B$和$C$，其中$B$只依赖于$A$，$C$只依赖于$B$，你能简化联合概率$P(A, B, C)$吗？提示：这是一个[马尔可夫链](https://en.wikipedia.org/wiki/Markov_chain)。
1. 在 :numref:`subsec_probability_hiv_app` 中，假设两项检测的结果并不独立。特别地，假设任一检测单独的假阳性率为10%、假阴性率为1%。也就是说，假设$P(D =1 \mid H=0) = 0.1$，且$P(D = 0 \mid H=1) = 0.01$。此外，假设对于$H = 1$（已感染）的情况，检测结果是条件独立的，即$P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$，但对于健康病人，结果之间通过$P(D_1 = D_2 = 1 \mid H=0) = 0.02$相耦合。
    1. 根据你目前掌握的信息，给出给定$H=0$时$D_1$和$D_2$的联合概率表。
    1. 推导一次检测结果为阳性后病人患病的概率（$H=1$）。你可以假设与之前相同的基准概率$P(H=1) = 0.0015$。
    1. 推导两次检测结果均为阳性后病人患病的概率（$H=1$）。
1. 假设你是一家投资银行的资产经理，你可以选择一些股票$s_i$进行投资。你的投资组合中各只股票的权重$\alpha_i$之和需要为$1$。这些股票的平均回报为$\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$，协方差为$\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$。
    1. 计算给定投资组合$\boldsymbol{\alpha}$的期望回报。
    1. 如果你想让投资组合的回报最大化，应该如何选择投资？
    1. 计算投资组合的*方差*。
    1. 构造一个优化问题：在把方差约束在上界之内的同时最大化回报。这就是获得诺贝尔奖的[马科维茨投资组合](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013`。要解决它，你需要一个二次规划求解器，这远远超出了本书的范围。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/198)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17971)
:end_tab:
