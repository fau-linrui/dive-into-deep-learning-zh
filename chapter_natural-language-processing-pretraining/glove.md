# 全局向量的词嵌入（GloVe）
:label:`sec_glove`


上下文窗口内的词-词共现
可能携带丰富的语义信息。
例如，
在大型语料库中，
单词"solid"与"ice"共现的可能性
比与"steam"更大，
而单词"gas"
与"steam"共现的频率
可能比与"ice"更高。
此外，
这种共现的全局语料库统计量
是可以预先计算的：
这可以带来更高效的训练。
为了利用整个语料库中的
统计信息
进行词嵌入，
让我们首先回顾 :numref:`subsec_skip-gram` 中的跳元模型，
但用共现计数等全局语料库统计量
来重新解释它。

## 带全局语料库统计的跳元模型
:label:`subsec_skipgram-global`

用 $q_{ij}$ 表示
跳元模型中的条件概率
$P(w_j\mid w_i)$，
即给定词 $w_i$ 时词 $w_j$ 的条件概率，
我们有

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},$$

其中，
对于任意索引 $i$，
向量 $\mathbf{v}_i$ 和 $\mathbf{u}_i$
分别表示词 $w_i$
作为中心词和上下文词时的向量，
而 $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$
是词表的索引集。

考虑词 $w_i$
在语料库中可能出现多次。
在整个语料库中，
所有以 $w_i$ 为中心词的上下文词
构成一个词索引的*多重集*（multiset）$\mathcal{C}_i$，
它*允许同一元素有多个实例*。
对于任意元素，
其实例数称为它的*重数*（multiplicity）。
举例说明，
假设词 $w_i$ 在语料库中出现两次，
在两个上下文窗口中
以 $w_i$ 为中心词的上下文词索引
分别为
$k, j, m, k$ 和 $k, l, k, j$。
于是，多重集 $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$，其中
元素 $j, k, l, m$
的重数分别为 2、4、1、1。

现在用 $x_{ij}$ 表示
元素 $j$ 在
多重集 $\mathcal{C}_i$ 中的重数。
这就是整个语料库中
在同一上下文窗口内
词 $w_j$（作为上下文词）
与词 $w_i$（作为中心词）的
全局共现计数。
利用这样的全局语料库统计量，
跳元模型的损失函数
等价于

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

我们进一步用 $x_i$
表示 $w_i$ 作为中心词时
其上下文窗口中所有上下文词的数目，
它等价于 $|\mathcal{C}_i|$。
令 $p_{ij}$ 为
给定中心词 $w_i$ 时生成
上下文词 $w_j$ 的条件概率 $x_{ij}/x_i$，
则 :eqref:`eq_skipgram-x_ij`
可以改写为

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

在 :eqref:`eq_skipgram-p_ij` 中，$-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ 计算的是
全局语料库统计的条件分布 $p_{ij}$
与模型预测的
条件分布 $q_{ij}$
之间的
交叉熵。
如上所述，这一损失
还以 $x_i$ 加权。
最小化
:eqref:`eq_skipgram-p_ij` 中的损失函数
将使
预测的条件分布
接近
由全局语料库统计得到的
条件分布。


虽然交叉熵损失函数
常用于
度量概率分布之间的距离，
但在这里可能不是一个好的选择。
一方面，正如我们在 :numref:`sec_approx_train` 中提到的，
正确规范化 $q_{ij}$
的代价是对整个词表求和，
这在计算上可能非常昂贵。
另一方面，
来自大型语料库的大量稀有事件
在交叉熵损失下
往往会被赋予
过大的权重。

## GloVe 模型

鉴于此，
*GloVe* 模型在基于平方损失的跳元模型上做了三处改动 :cite:`Pennington.Socher.Manning.2014`：

1. 使用不是概率分布的变量 $p'_{ij}=x_{ij}$ 和 $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$，并对二者取对数，于是平方损失项为 $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$。
2. 为每个词 $w_i$ 增加两个标量模型参数：中心词偏置 $b_i$ 和上下文词偏置 $c_i$。
3. 用权重函数 $h(x_{ij})$ 替换每个损失项的权重，其中 $h(x)$ 在区间 $[0, 1]$ 上递增。

综合起来，训练 GloVe 就是最小化以下损失函数：

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

对于权重函数，一个建议的选择是：
如果 $x < c$（例如 $c = 100$），则 $h(x) = (x/c) ^\alpha$（例如 $\alpha = 0.75$）；否则 $h(x) = 1$。
在这种情况下，
由于 $h(0)=0$，
为了计算效率，可以省略任何 $x_{ij}=0$ 的平方损失项。
例如，
当使用小批量随机梯度下降进行训练时，
在每次迭代中，
我们随机采样一个由*非零* $x_{ij}$ 组成的小批量
来计算梯度
并更新模型参数。
请注意，这些非零的 $x_{ij}$ 是预先计算好的
全局语料库统计量；
因此，该模型被称为 GloVe，
即*全局向量*（Global Vectors）。

应当强调，
如果词 $w_i$ 出现在
词 $w_j$ 的上下文窗口中，那么*反之亦然*。
因此，$x_{ij}=x_{ji}$。
与拟合非对称条件概率
$p_{ij}$ 的 word2vec 不同，
GloVe 拟合的是对称的 $\log \, x_{ij}$。
因此，在 GloVe 模型中，任意词的中心词向量
与上下文词向量在数学上是等价的。
然而在实践中，由于初始化值不同，
同一个词在训练后
在这两个向量中可能仍会得到不同的值：
GloVe 将它们相加作为输出向量。



## 从共现概率的比值理解 GloVe


我们也可以从另一个角度来理解 GloVe 模型。
沿用 :numref:`subsec_skipgram-global` 中的记号，
令 $p_{ij} \stackrel{\textrm{def}}{=} P(w_j \mid w_i)$ 为语料库中给定 $w_i$ 作为中心词时生成上下文词 $w_j$ 的条件概率。
:numref:`tab_glove`
列出了给定单词"ice"和"steam"时
基于大型语料库统计量的
若干共现概率
及其比值。


:来自大型语料库的词-词共现概率及其比值（改编自 :citet:`Pennington.Socher.Manning.2014` 中的表 1）
:label:`tab_glove`

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \textrm{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\textrm{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|



从 :numref:`tab_glove` 中可以观察到以下几点：

* 对于与"ice"相关但与"steam"无关的词 $w_k$，例如 $w_k=\textrm{solid}$，我们预期其共现概率的比值较大，例如 8.9。
* 对于与"steam"相关但与"ice"无关的词 $w_k$，例如 $w_k=\textrm{gas}$，我们预期其共现概率的比值较小，例如 0.085。
* 对于与"ice"和"steam"都相关的词 $w_k$，例如 $w_k=\textrm{water}$，我们预期其共现概率的比值接近 1，例如 1.36。
* 对于与"ice"和"steam"都无关的词 $w_k$，例如 $w_k=\textrm{fashion}$，我们预期其共现概率的比值接近 1，例如 0.96。




可以看出，共现概率的比值
可以直观地表达
词与词之间的关系。
因此，我们可以设计一个
关于三个词向量的函数
来拟合这个比值。
对于 $w_i$ 为中心词、
$w_j$ 和 $w_k$ 为上下文词时
共现概率的比值 ${p_{ij}}/{p_{ik}}$，
我们希望用某个函数 $f$
来拟合这个比值：

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

在 $f$ 的许多可能设计中，
我们只在下文中选取一个合理的选择。
由于共现概率的比值
是一个标量，
我们要求 $f$ 是一个标量函数，例如
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$。
交换 :eqref:`eq_glove-f` 中的词索引
$j$ 和 $k$，
必须满足
$f(x)f(-x)=1$，
因此一种可能是 $f(x)=\exp(x)$，
即，

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

现在让我们选取
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$，
其中 $\alpha$ 是一个常数。
由于 $p_{ij}=x_{ij}/x_i$，两边取对数后可得 $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$。
我们可以使用额外的偏置项来拟合 $- \log\, \alpha + \log\, x_i$，例如中心词偏置 $b_i$ 和上下文词偏置 $c_j$：

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

用权重度量 :eqref:`eq_glove-square` 的
平方误差，
就得到 :eqref:`eq_glove-loss` 中
的 GloVe 损失函数。



## 小结

* 跳元模型可以用词-词共现计数等全局语料库统计量来解释。
* 在度量两个概率分布之间的差异时，交叉熵损失可能不是一个好的选择，尤其是对于大型语料库而言。GloVe 使用平方损失来拟合预先计算的全局语料库统计量。
* 在 GloVe 中，对于任意词，其中心词向量和上下文词向量在数学上是等价的。
* GloVe 可以从词-词共现概率的比值来解释。


## 练习

1. 如果词 $w_i$ 和 $w_j$ 在同一上下文窗口中共同出现，我们如何利用它们在文本序列中的距离来重新设计计算条件概率 $p_{ij}$ 的方法？提示：参见 GloVe 论文 :cite:`Pennington.Socher.Manning.2014` 的第 4.2 节。
1. 对于任意词，它的中心词偏置和上下文词偏置在 GloVe 中在数学上等价吗？为什么？


[讨论](https://discuss.d2l.ai/t/385)
